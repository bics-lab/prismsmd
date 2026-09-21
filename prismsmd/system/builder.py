"""System building with CosolvKit.

CosolvKit is used only to create the system; MD is not run in OpenMM. What CosolvKit
cannot provide is patched in afterwards by this module:

* virtual repulsion between probes (:mod:`prismsmd.system.repulsion`), since
  ``add_repulsive_forces`` builds an OpenMM force that never reaches a GROMACS topology;
* staged position restraints (:mod:`prismsmd.system.posre`);
* the missing water, filled in with ``gmx solvate`` (:mod:`prismsmd.system.solvate`);
* the ``.ndx``, which CosolvKit does not emit (:mod:`prismsmd.system.index`).

``save_topology`` is not used either -- it always applies hydrogen mass
repartitioning -- so the topology is saved here.

Atom names cannot be relied on: OpenFF rebuilds the probe atom names as
``f"{element}{serial}x"``, so the mol2 names do not exist in the system. Probe atoms are
identified by index, and :func:`verify_probe_atoms_in_gro` checks the element sequence
and the bond lengths.
"""

from __future__ import annotations

import io
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import ProbeConfig, ProtocolConfig, TargetConfig
from ..probeprep import NM_TO_ANGSTROM
from .charges import apply_reference_charges
from .index import write_index_ndx
from .pdbprep import (
    normalise_protonation_names,
    separate_water_chain,
    variants_for_topology,
)
from .posre import embed_position_restraints_multi
from .repulsion import add_virtual_site_to_gro, add_virtual_site_to_top
from .solvate import WaterFillResult, fill_water_with_gmx

__all__ = [
    "BOND_LENGTH_RANGE_ANGSTROM",
    "DEFAULT_FORCEFIELDS",
    "XH_BOND_LENGTH_RANGE_ANGSTROM",
    "BuildResult",
    "ProbeGeometryError",
    "build_system",
    "verify_probe_atoms_in_gro",
]

#: Force fields passed to CosolvKit: ff14SB + TIP3P + GAFF2.
#: The keys must be uppercase engine names, because ``_parametrize_system`` looks
#: them up as ``forcefields[engine]``.
DEFAULT_FORCEFIELDS: dict[str, list[str]] = {
    "AMBER": ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"],
    "GROMACS": ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"],
    "CHARMM": ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"],
    "OPENMM": ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"],
    # "gaff" resolves to gaff-2.11 = GAFF2. The default "espaloma" is a different
    # force field, so the choice is always stated explicitly.
    "small_molecules": ["gaff"],
}


@dataclass
class BuildResult:
    """What :func:`build_system` produced, for the provenance record."""

    top: Path
    gro: Path
    probe_resname: str
    n_probe_molecules: int
    box_volume_nm3: float
    index: Path | None = None
    #: Number of water molecules added by ``gmx solvate`` (``None`` if no filling).
    n_water_added: int | None = None
    #: Density of the system after filling (g/cm^3).
    density_g_cm3: float | None = None
    #: Mapping from solvent-box to ``.top`` water atom names, decided by reading the
    #: ``.top``. A non-identity ``water_reorder`` means a reordering happened.
    water_atom_name_map: tuple[tuple[str, str], ...] | None = None
    #: Position after remapping -> position on the solvent-box side. Identity is ``(0, 1, 2)``.
    water_reorder: tuple[int, ...] | None = None
    #: Single-point potential energy of the assembled system (kJ/mol).
    initial_potential_kj: float | None = None
    #: Closest approach between two different probe molecules (A). Recorded for
    #: diagnosis; it is not what the build is judged on.
    probe_gap_a: float | None = None
    #: The seed the probe placement was pinned to, which is what lets this system be
    #: rebuilt; :func:`prismsmd.seeds.run_seed` derives it from the run's identity.
    #: Probe placement is reproducible from it; the water count is not, because
    #: OpenMM's addSolvent has its own randomness.
    placement_seed: int | None = None


def build_system(
    target: TargetConfig,
    probe: ProbeConfig,
    protocol: ProtocolConfig,
    outdir: str | Path,
    *,
    seed: int = 0,
    forcefields: dict[str, list[str]] | None = None,
    clean_protein: bool = True,
    fill_water: bool = True,
    check_clash: bool = True,
    periodic_safe_placement_enabled: bool = True,
    gmx: str | Sequence[str] | None = None,
) -> BuildResult:
    """Build a cosolvent system with CosolvKit, save it in GROMACS format, then patch it.

    Args:
        target: the receptor and its box.
        probe: the probe to fill the box with.
        protocol: the MD protocol, which supplies the restraint ladder, the repulsion
            parameters and the charge source.
        outdir: where the generated files are written.
        seed: pins the probe placement, making the system rebuildable.
        forcefields: overrides :data:`DEFAULT_FORCEFIELDS`.
        clean_protein: run the receptor through PDBFixer before building.
        fill_water: use ``gmx solvate`` to add the water CosolvKit could not fit
            (:mod:`prismsmd.system.solvate`). False is for diagnosis only.
        check_clash: refuse a system whose single-point energy says it cannot be
            minimised (:mod:`prismsmd.system.clash`). False is for diagnosis only.
        periodic_safe_placement_enabled: patch CosolvKit's placement check so
            molecules cannot cross the box faces or overlap their own periodic images
            (:mod:`prismsmd.system.placement_pbc`). False reproduces the unpatched
            behaviour, for diagnosis only.
        gmx: the ``gmx`` command. ``None`` uses the environment variable ``MSMD_GMX``,
            or ``"gmx"`` if that is unset too.

    Raises:
        ImportError: when cosolvkit / openmm are not installed.
        FileNotFoundError: when ``fill_water=True`` but ``gmx`` cannot be found.
        prismsmd.system.solvate.WaterFillError: when the density, box or water name
            remapping are inconsistent after filling.
    """
    from .clash import assert_system_is_minimisable
    from .placement_pbc import periodic_safe_placement
    from .placement_seed import deterministic_placement

    try:
        import openmm.unit as openmmunit
        import parmed
        from cosolvkit.cosolvent_system import CosolventSystem
        from cosolvkit.utils import fix_pdb
        from openmm import app
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "cosolvkit / openmm are required:\n"
            "  pip install cosolvkit openmm openmmforcefields openff-toolkit pdbfixer parmed"
        ) from exc

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if probe.mol_file is None:
        raise ValueError(
            f"probe {probe.cid}: CosolvKit needs a mol/sdf."
            " Set mol_file in probes.yaml"
            " (CosolvKit does not accept mol2)."
        )
    if not probe.smiles:
        # CosolvKit's parametrisation only looks at smiles and swallows exceptions, so
        # without smiles the probe is silently dropped and the run fails far away with
        # a seemingly unrelated "no template" error. Stop here instead.
        raise ValueError(
            f"probe {probe.cid}: CosolvKit's parametrisation only looks at smiles."
            " With mol_file alone the molecule is silently dropped."
            " Add smiles to probes.yaml."
        )

    # --- Tidy the receptor ---
    # fix_pdb takes file objects, not paths; pdbxfile is only for passing a .cif.
    #
    # The seeding covers fix_pdb, not just build(): placement fills the empty voxels
    # around the receptor, so the hydrogens fix_pdb places move the probes.
    fixed_pdb = outdir / "receptor_fixed.pdb"
    with deterministic_placement(seed), periodic_safe_placement(
            enabled=periodic_safe_placement_enabled):
        if clean_protein:
            # Move water to a different chain ID first (see prismsmd.system.pdbprep):
            # PDBFixer only adds the terminal OXT to the last residue of a chain, and
            # crystal water sharing the protein's chain ID hides that residue.
            pdb_text, water_chain = separate_water_chain(Path(target.pdb).read_text())
            if water_chain is not None:
                print(f"[build] separated crystal water into chain {water_chain}"
                      f" (so PDBFixer adds the C-terminal OXT)")
            # Protonation states are stated, not inferred: Amber residue names are
            # renamed to the standard ones PDBFixer keeps (it deletes residues it does
            # not recognise, LYN among them), then each state is restored explicitly
            # through addHydrogens(variants=...).
            pdb_text, wanted_variants = normalise_protonation_names(pdb_text)
            if wanted_variants:
                from collections import Counter
                summary = Counter(wanted_variants.values())
                print(f"[build] protonation states to restore: {dict(summary)}")
            topology, positions = fix_pdb(
                pdbfile=io.StringIO(pdb_text),
                pdbxfile=None,
                keep_heterogens=False,
            )
            modeller = app.Modeller(topology, positions)
            if wanted_variants:
                # Existing hydrogens have to go first: addHydrogens only adds, so the
                # hydrogens fix_pdb placed from the standard template would stay and
                # the variant would have no effect.
                hydrogens = [a for a in modeller.topology.atoms()
                             if a.element is not None and a.element.symbol == "H"]
                if hydrogens:
                    modeller.delete(hydrogens)
                variant_list = variants_for_topology(modeller.topology, wanted_variants)
                modeller.addHydrogens(
                    app.ForceField(*(forcefields or DEFAULT_FORCEFIELDS)["AMBER"]),
                    variants=variant_list,
                )
            topology, positions = modeller.topology, modeller.positions
            with fixed_pdb.open("w") as fh:
                app.PDBFile.writeFile(topology, positions, fh)
        else:
            shutil.copy2(target.pdb, fixed_pdb)
            pdbf = app.PDBFile(str(fixed_pdb))
            topology, positions = pdbf.topology, pdbf.positions
            modeller = app.Modeller(topology, positions)

        # --- CosolvKit ---
        # The dict is expanded into CosolventMolecule(**c), so the keys are that
        # class's argument names. Both smiles and mol_filename are required.
        cosolvents = [
            {
                "name": probe.name,
                "resname": probe.residue_name,
                "smiles": probe.smiles,
                "mol_filename": probe.mol_file,
                "concentration": probe.concentration,
            }
        ]
        system = CosolventSystem(
            cosolvents=cosolvents,
            forcefields=forcefields or DEFAULT_FORCEFIELDS,
            simulation_format="GROMACS",
            modeller=modeller,
            padding=protocol.padding_angstrom * openmmunit.angstrom,
            # radius is None because a target is given; the two are mutually exclusive.
            radius=None,
        )
        # build() is what places the probes. Halton and np.random.choice must both be
        # pinned for it to be reproducible; see prismsmd.system.placement_seed.
        system.build(solvent_smiles="H2O", n_solvent_molecules=None, neutralize=True)

    ck_topology = system.modeller.topology
    ck_positions = system.modeller.positions
    omm_system = system.forcefield.createSystem(
        ck_topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=protocol.nonbonded_cutoff_angstrom * openmmunit.angstrom,
        removeCMMotion=False,
        rigidWater=False,
        # hydrogenMass is not passed (no HMR)
    )
    structure = parmed.openmm.topsystem.load_topology(
        ck_topology, omm_system, ck_positions
    )
    structure.save(str(outdir / "system.top"), overwrite=True)
    structure.save(str(outdir / "system.gro"), overwrite=True)

    # Refuse a system that cannot be minimised, before it costs a job.
    if check_clash:
        clash_energy, clash_gap = assert_system_is_minimisable(
            omm_system, ck_positions,
            gro_text=(outdir / "system.gro").read_text(),
            probe_resname=probe.residue_name,
        )
    else:
        from .clash import min_intermolecular_gap, single_point_energy
        clash_energy = single_point_energy(omm_system, ck_positions)
        clash_gap = min_intermolecular_gap((outdir / "system.gro").read_text(),
                                           probe.residue_name)
        print(f"[build] check_clash=False: energy {clash_energy:.4e} kJ/mol, "
              f"probe gap {clash_gap.any_gap_a:.3f} A (not enforced)")

    raw_top = outdir / "system.top"
    raw_gro = outdir / "system.gro"
    if not raw_top.exists() or not raw_gro.exists():
        raise RuntimeError(f"parmed did not produce {raw_top} / {raw_gro}")

    # --- Water filling (CosolvKit's output is short of water) ---
    # Before the VIS patch and the restraints: gmx solvate knows nothing about a
    # massless virtual site, and restraint atom numbers are local to the moleculetype,
    # so water appended at the end does not affect them.
    fill: WaterFillResult | None = None
    if fill_water:
        fill = fill_water_with_gmx(
            raw_gro,
            raw_top,
            out_gro=outdir / "solvated.gro",
            out_top=outdir / "solvated.top",
            gmx=gmx,
        )
        raw_gro = outdir / "solvated.gro"
        raw_top = outdir / "solvated.top"

    n_probe = _count_residues(raw_gro, probe.residue_name)

    verify_probe_atoms_in_gro(raw_gro, probe)

    top_string = raw_top.read_text()
    gro_string = raw_gro.read_text()

    # --- Patch 0: reference (RESP) probe charges ---
    # Before the VIS patch: the mol2 has no virtual site, and the substitution refuses
    # a topology whose atom count does not match.
    if protocol.probe_charges == "resp":
        # The charges come from probe.mol2_file, the same file the atom order was
        # resolved against; a second path here would let the two diverge.
        if not probe.mol2_file:
            raise ValueError(
                f"probe_charges=resp, but probe {probe.cid} has no mol2_file"
                " in probes.yaml"
            )
        top_string, sub = apply_reference_charges(
            top_string, probe.residue_name, Path(probe.mol2_file).read_text()
        )
        print(f"[build] probe_charges=resp: {sub.resname} {sub.n_atoms} atoms, "
              f"max |dq| = {sub.max_abs_delta:.4f} e, "
              f"total {sub.total_before:+.6f} -> {sub.total_after:+.6f}")
    else:
        print(f"[build] probe_charges={protocol.probe_charges} "
              "(keeping the AM1-BCC charges CosolvKit/GAFF assigned)")

    # --- Patch 1: virtual repulsion ---
    if protocol.repulsion.enabled:
        top_string = add_virtual_site_to_top(
            top_string,
            [probe.residue_name],
            rmin_angstrom=protocol.repulsion.rmin_angstrom,
            epsilon_kcal=protocol.repulsion.epsilon_kcal,
            site_name=protocol.repulsion.site_name,
        )
        gro_string = add_virtual_site_to_gro(
            gro_string, probe.residue_name, site_name=protocol.repulsion.site_name
        )

    # --- Patch 2: staged position restraints ---
    # One block per moleculetype: parmed splits a multimer into one moleculetype per
    # chain, and the atom numbers in a restraint block must be local to its own.
    heavy_per_mol, ca_per_mol = _protein_atom_ids_per_molecule(
        gro_string, top_string, [probe.residue_name]
    )
    top_string = embed_position_restraints_multi(
        top_string,
        heavy_per_mol,
        prefix="POSRES",
        strengths_kcal=protocol.posres_strengths_kcal,
    )
    top_string = embed_position_restraints_multi(
        top_string,
        ca_per_mol,
        prefix="CAPOSRES",
        strengths_kcal=protocol.caposres_strengths_kcal,
    )

    out_top = outdir / "input.top"
    out_gro = outdir / "input.gro"
    out_top.write_text(top_string)
    out_gro.write_text(gro_string)

    # --- .ndx ---
    # Groups matching the mdp's tc_grps = Non-Water Water. Built from the final gro
    # including the VIS virtual site, since the coupling groups must cover every atom.
    index = write_index_ndx(outdir / "index.ndx", gro_string, [probe.residue_name])

    return BuildResult(
        top=out_top,
        gro=out_gro,
        probe_resname=probe.residue_name,
        n_probe_molecules=n_probe,
        box_volume_nm3=float(getattr(system, "box_volume", 0.0)),
        index=index,
        n_water_added=None if fill is None else fill.n_added_molecules,
        density_g_cm3=None if fill is None else fill.density_g_cm3,
        water_atom_name_map=None if fill is None else fill.atom_name_map,
        water_reorder=None if fill is None else fill.reorder,
        placement_seed=seed,
        initial_potential_kj=clash_energy,
        probe_gap_a=None if clash_gap is None else clash_gap.any_gap_a,
    )


_SOLVENT_RESNAMES = {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "NA+", "CL-", "K", "MG", "ZN"}


def _iter_gro_atoms(gro_string: str):
    """Yield the atom lines of a ``.gro`` as ``(resi, resn, name, index, (x, y, z))``.

    Coordinates stay in nm, the ``.gro`` unit. Fixed width
    ``resid(5) resname(5) atomname(5) atomnum(5) x(8) y(8) z(8)``.
    """
    lines = gro_string.split("\n")
    natoms = int(lines[1].strip())
    for line in lines[2 : 2 + natoms]:
        yield (
            int(line[0:5]),          # resi
            line[5:10].strip(),      # resn
            line[10:15].strip(),     # name
            int(line[15:20]),        # index
            (
                float(line[20:28]),  # x (nm)
                float(line[28:36]),  # y (nm)
                float(line[36:44]),  # z (nm)
            ),
        )


def _count_residues(gro_path: Path, resname: str) -> int:
    """Count the molecules of ``resname`` in a ``.gro``.

    Residue numbers wrap at 99999 and are not unique, so a molecule is counted as a
    run of consecutive identical residue numbers.
    """
    count = 0
    prev: int | None = None
    for resi, resn, _name, _idx, _xyz in _iter_gro_atoms(gro_path.read_text()):
        if resn != resname:
            prev = None
            continue
        if resi != prev:
            count += 1
            prev = resi
    return count


class ProbeGeometryError(ValueError):
    """Raised when the probes in the generated ``.gro`` are inconsistent with the sdf.

    The typical cause is a mismatch between the atom order of the topology (from
    ``Molecule.from_smiles``) and of the coordinates (from the sdf).
    """


#: Allowed range for intramolecular bonds (A). Loose enough to reject only chemically
#: impossible values: it exists to detect swapped coordinates, not to judge force-field
#: quality. The bounds bracket the standard covalent lengths from O-H (0.96) to C-S
#: (1.82), and stay clear of hydrogen bonds (>= 1.6) and nonbonded neighbours (>= 2.5).
BOND_LENGTH_RANGE_ANGSTROM = (0.90, 1.90)

#: Tighter range for X-H bonds, the longest of which is C-H at 1.09 A.
XH_BOND_LENGTH_RANGE_ANGSTROM = (0.85, 1.25)

#: Per-element-pair overrides, for pairs whose bonds fall outside the default range.
#: X-H pairs belong here too, since :data:`XH_BOND_LENGTH_RANGE_ANGSTROM` is calibrated
#: on C/N/O-H and a bond to a heavier element is longer; :func:`_bond_length_range`
#: therefore consults this table before the X-H range.
BOND_LENGTH_RANGE_OVERRIDES: dict[frozenset, tuple[float, float]] = {
    frozenset(("S", "S")): (1.80, 2.30),    # S-S 2.05
    frozenset(("C", "Br")): (1.75, 2.10),   # C-Br 1.94
    frozenset(("C", "I")): (1.95, 2.30),    # C-I 2.14
    frozenset(("S", "H")): (1.25, 1.45),    # S-H 1.34
    frozenset(("P", "H")): (1.30, 1.50),    # P-H 1.42
}


def _bond_length_range(el_a: str, el_b: str) -> tuple[float, float]:
    # The override table is consulted first, so a pair listed there always wins,
    # including an X-H pair.
    override = BOND_LENGTH_RANGE_OVERRIDES.get(frozenset((el_a, el_b)))
    if override is not None:
        return override
    if "H" in (el_a, el_b):
        return XH_BOND_LENGTH_RANGE_ANGSTROM
    return BOND_LENGTH_RANGE_ANGSTROM


def verify_probe_atoms_in_gro(
    gro_path: Path,
    probe: ProbeConfig,
    *,
    max_molecules_checked: int | None = None,
    site_name: str = "VIS",
) -> None:
    """Verify that the probe molecules in the generated ``.gro`` match the sdf.

    Three checks, none of which relies on atom names:

    1. The element sequence, built from the atom-name prefixes of the probe residues in
       the ``.gro`` and compared position by position against the sdf's.
    2. The bond lengths, measured from the ``.gro`` coordinates for each sdf bond and
       checked against :data:`BOND_LENGTH_RANGE_ANGSTROM` /
       :data:`XH_BOND_LENGTH_RANGE_ANGSTROM`. This catches a permutation whose element
       sequence happens to match, such as swapping two atoms of the same element.
    3. The element at each mapped index against the type recorded for it, rejecting for
       example a carbon labelled ``O_A``.

    Args:
        gro_path: the generated ``.gro``.
        probe: the probe definition, which supplies the sdf and the mapped indices.
        max_molecules_checked: how many probe molecules to check; ``None`` means all.
        site_name: atom name of the virtual repulsion site, excluded from the check
            should it already be present.

    Raises:
        ProbeGeometryError: when the element sequence or a bond length does not match.
    """
    import numpy as np

    from ..probeprep import element_from_atom_name, read_sdf_atoms

    if probe.mol_file is None:
        raise ProbeGeometryError(
            f"probe {probe.cid}: without mol_file (sdf) the element sequence cannot be checked."
            " Add mol_file to probes.yaml."
        )
    expected_elements, bonds = read_sdf_atoms(probe.mol_file)

    gro_string = Path(gro_path).read_text()
    molecules = _probe_molecules_from_gro(
        gro_string, probe.residue_name, site_name=site_name
    )
    if not molecules:
        raise ProbeGeometryError(
            f"probe {probe.cid}: residue {probe.residue_name} is not in the .gro"
        )

    box_nm = _gro_box_nm(gro_string)
    checked = molecules if max_molecules_checked is None else molecules[
        :max_molecules_checked
    ]

    for mol_no, (resi, names, coords_nm) in enumerate(checked, start=1):
        if len(names) != len(expected_elements):
            raise ProbeGeometryError(
                f"probe {probe.cid}: molecule #{mol_no} in the .gro (residue number {resi})"
                f" has {len(names)} atoms, the sdf has {len(expected_elements)}.\n"
                f"  .gro atom names   : {names}\n"
                f"  sdf element seq   : {expected_elements}"
            )
        try:
            actual = [element_from_atom_name(n) for n in names]
        except ValueError as exc:
            raise ProbeGeometryError(
                f"probe {probe.cid}: cannot determine the element from the .gro atom names"
                f" (residue number {resi}, atom names {names}): {exc}"
            ) from exc
        if actual != expected_elements:
            diff = [
                f"[{i}] .gro {names[i]!r} ({actual[i]}) != sdf {expected_elements[i]}"
                for i in range(len(actual))
                if actual[i] != expected_elements[i]
            ]
            raise ProbeGeometryError(
                f"probe {probe.cid} ({probe.residue_name}): the .gro element sequence does not"
                f" match the sdf (residue number {resi}).\n  " + "\n  ".join(diff) + "\n"
                f"  .gro : {actual}\n"
                f"  sdf  : {expected_elements}\n"
                "The atom order of the topology (from Molecule.from_smiles) and of the"
                " coordinates (from the sdf) disagree.\n"
                "Rebuild the sdf with prismsmd.probeprep.prepare_probe_sdf(),"
                " which reorders it into from_smiles order."
            )

        xyz = np.asarray(coords_nm, dtype=float)
        for i, j in bonds:
            d_nm = _distance_nm(xyz[i], xyz[j], box_nm)
            d = d_nm * NM_TO_ANGSTROM
            lo, hi = _bond_length_range(expected_elements[i], expected_elements[j])
            if not (lo <= d <= hi):
                raise ProbeGeometryError(
                    f"probe {probe.cid} ({probe.residue_name}): abnormal bond length"
                    f" (residue number {resi}, molecule #{mol_no}).\n"
                    f"  bond [{i}]{names[i]}({expected_elements[i]})"
                    f" - [{j}]{names[j]}({expected_elements[j]})"
                    f" = {d:.3f} A, allowed range {lo}-{hi} A\n"
                    "Most likely the coordinates are in a different atom order than the"
                    " topology: bond lengths break even for a permutation whose element"
                    " sequence happens to match.\n"
                    "Rebuild the sdf with prismsmd.probeprep.prepare_probe_sdf()."
                )

    # Is the element at each mapped index the one the sdf has?
    for index, xstype in (probe.atom_indices or {}).items():
        if not (0 <= index < len(expected_elements)):
            raise ProbeGeometryError(
                f"probe {probe.cid}: the assigned index {index} is out of range"
                f" for an atom count of {len(expected_elements)}"
            )
        want = xstype.split("_")[0]
        got = expected_elements[index]
        if want.capitalize() != got.capitalize():
            raise ProbeGeometryError(
                f"probe {probe.cid}: the type {xstype} at index {index} requires element"
                f" {want}, but the sdf has {got}."
                " The type assignment and the sdf atom order disagree."
            )


def _probe_molecules_from_gro(
    gro_string: str, resname: str, *, site_name: str = "VIS"
) -> list[tuple[int, list[str], list[tuple[float, float, float]]]]:
    """Extract the probe molecules from a ``.gro`` as ``(residue number, names, coords)``.

    Coordinates are in nm. Residue numbers wrap at 99999, so a molecule is a run of
    consecutive identical residue numbers.
    """
    out: list[tuple[int, list[str], list[tuple[float, float, float]]]] = []
    prev: int | None = None
    for resi, resn, name, _idx, xyz in _iter_gro_atoms(gro_string):
        if resn != resname:
            prev = None
            continue
        if resi != prev:
            out.append((resi, [], []))
            prev = resi
        if name == site_name:
            continue          # the virtual repulsion site is not an atom of the molecule
        out[-1][1].append(name)
        out[-1][2].append(xyz)
    return out


def _gro_box_nm(gro_string: str) -> tuple[float, float, float] | None:
    """Return the box vectors at the end of a ``.gro`` (nm), or ``None`` if not rectangular.

    Used to measure bond lengths under the minimum-image convention, so a molecule
    wrapped at the edge of the box is still measured correctly.
    """
    lines = gro_string.split("\n")
    try:
        natoms = int(lines[1].strip())
        fields = lines[2 + natoms].split()
    except (IndexError, ValueError):
        return None
    if len(fields) < 3:
        return None
    try:
        box = tuple(float(v) for v in fields[:3])
    except ValueError:
        return None
    # Nonzero triclinic components (from the 4th onward) mean the minimum image
    # cannot be simplified this way.
    if len(fields) > 3 and any(abs(float(v)) > 1e-9 for v in fields[3:]):
        return None
    return box  # type: ignore[return-value]


def _distance_nm(a, b, box_nm) -> float:
    """Distance between two points (nm), using the minimum image when the box is known."""
    import numpy as np

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if box_nm is not None:
        box = np.asarray(box_nm, dtype=float)
        ok = box > 0
        d[ok] -= box[ok] * np.round(d[ok] / box[ok])
    return float(np.linalg.norm(d))


def _protein_atom_ids(
    gro_string: str, probe_resnames: Sequence[str] = ()
) -> tuple[list[int], list[int]]:
    """Return the 1-based atom numbers of the protein heavy atoms and of the Ca atoms.

    Anything that is not water, an ion or a probe counts as protein. The numbers are
    system-wide running numbers; :func:`_protein_atom_ids_per_molecule` converts them
    to the moleculetype-local ones the restraint blocks need.

    Raises:
        ValueError: if solvent appears before the protein, which would break the
            ordering assumption the caller relies on.
    """
    heavy: list[int] = []
    ca: list[int] = []
    known_solvent = {s.upper() for s in _SOLVENT_RESNAMES}
    probes = {p.upper() for p in probe_resnames}
    first_non_protein: int | None = None
    for i, (_resi, resn, name, _idx, _xyz) in enumerate(
        _iter_gro_atoms(gro_string), start=1
    ):
        key = resn.upper()
        if key in known_solvent or key in probes:
            if first_non_protein is None:
                first_non_protein = i
            continue
        if first_non_protein is not None:
            raise ValueError(
                f"a protein atom (#{i}, {resn} {name}) appeared after a non-protein atom "
                f"(#{first_non_protein}). Position restraint atom numbers must be local to"
                " the moleculetype, so they cannot be embedded correctly in this ordering."
                " Check the ordering of the topology."
            )
        if name.startswith("H"):
            continue
        heavy.append(i)
        if name == "CA":
            ca.append(i)
    if not heavy:
        raise ValueError("no protein heavy atom found at all")
    return heavy, ca


def _protein_atom_ids_per_molecule(
    gro_string: str, top_string: str, probe_resnames: Sequence[str] = ()
) -> tuple[list[list[int]], list[list[int]]]:
    """Split the protein atom numbers into moleculetype-local lists.

    ``[ position_restraints ]`` numbers atoms within one moleculetype, and parmed
    splits a multimer into one moleculetype per chain, so the boundaries are read from
    the topology and each chain gets its own block with its own local numbering.

    Returns:
        ``(heavy_per_molecule, ca_per_molecule)``, one list per moleculetype in the
        order they are defined in the topology, with 1-based local numbers. A
        moleculetype with no protein atom gets an empty list.
    """
    from .solvate import parse_molecules_section, parse_moleculetypes

    types = parse_moleculetypes(top_string)
    order = parse_molecules_section(top_string)

    # [ molecules ] is a sequence of (name, count) and atom numbers run through it in
    # order, so build the span [start, end] each entry occupies.
    spans: list[tuple[str, int, int]] = []
    pos = 1
    for name, count in order:
        if name not in types:
            raise ValueError(
                f"[ molecules ] names {name!r}, which has no [ moleculetype ]."
                " Position restraint numbers cannot be assigned."
            )
        n_atoms = types[name].n_atoms
        for _ in range(count):
            spans.append((name, pos, pos + n_atoms - 1))
            pos += n_atoms

    heavy, ca = _protein_atom_ids(gro_string, probe_resnames)
    if not heavy:
        return [], []

    # Ordered by definition order, which is what embed_position_restraints'
    # molecule_index counts. A moleculetype used several times is defined once.
    defined = list(types.keys())
    heavy_by_type: dict[str, list[int]] = {k: [] for k in defined}
    ca_by_type: dict[str, list[int]] = {k: [] for k in defined}

    def assign(ids: Sequence[int], bucket: dict[str, list[int]]) -> None:
        i = 0
        for name, start, end in spans:
            while i < len(ids) and ids[i] < start:
                i += 1                      # before this span = already handled
            while i < len(ids) and ids[i] <= end:
                bucket[name].append(ids[i] - start + 1)   # to a local number
                i += 1
        if i != len(ids):
            raise ValueError(
                f"atom number {ids[i]} falls in no moleculetype's span"
                f" (the system has {pos - 1} atoms). The .gro and .top disagree."
            )

    assign(heavy, heavy_by_type)
    assign(ca, ca_by_type)
    return ([heavy_by_type[k] for k in defined],
            [ca_by_type[k] for k in defined])
