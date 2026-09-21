"""Config file schema and loading.

The config is split in three, and the pieces are bundled into a single
:class:`RunConfig`:

  probes.yaml      probe definitions and the atoms to map (target independent)
  targets.yaml     per-target PDB and docking box
  protocol_*.yaml  MD protocol (target and probe independent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .gfe import GFEParams
from .grid import GridSpec

__all__ = [
    "MDStep",
    "ProbeConfig",
    "ProtocolConfig",
    "RepulsionConfig",
    "RunConfig",
    "TargetConfig",
    "load_probes",
    "load_protocol",
    "load_targets",
    "load_yaml",
]


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file and return its top-level mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        # Reports invalid config content rather than a type error, hence ValueError.
        raise ValueError(f"{path}: the top level is not a mapping")  # noqa: TRY004
    return data


@dataclass(frozen=True)
class ProbeConfig:
    """Definition of one probe."""

    cid: str
    name: str
    smiles: str | None = None
    mol_file: str | None = None          # .mol / .sdf (for CosolvKit)
    mol2_file: str | None = None         # RESP charges and GAFF atom types
    resname: str | None = None
    concentration: float = 1.0           # M
    #: ``{mol2 atom name: element}`` of the atoms to map. Display labels only:
    #: CosolvKit / OpenFF rebuild atom names, so these cannot be matched against
    #: the generated system.
    atoms: dict[str, str] = field(default_factory=dict)
    #: ``{0-based index within the probe molecule: element}``, the primary
    #: representation. The index follows the atom order of ``mol_file`` (sdf)
    #: = ``Molecule.from_smiles(smiles)``.
    atom_indices: dict[int, str] = field(default_factory=dict)
    #: Atom order and per-atom identity (:class:`prismsmd.probeorder.ProbeAtomOrder`).
    #: ``None`` when no mol2 has been read.
    atom_order: Any = None
    #: False keeps the definition but builds no maps for it.
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.smiles is None and self.mol_file is None:
            raise ValueError(
                f"probe {self.cid}: either smiles or mol_file is required"
            )

    @property
    def residue_name(self) -> str:
        """Residue name of the probe, defaulting to its cid."""
        return self.resname or self.cid

    @property
    def mapped_atoms(self) -> list[str]:
        """Names of the atoms that get a PMAP. Display labels."""
        return sorted(self.atoms)

    @property
    def mapped_atom_indices(self) -> list[int]:
        """Indices of the atoms that get a PMAP (0-based, ascending)."""
        return sorted(self.atom_indices)


def load_probes(
    path: str | Path, *, resolve_order: bool = True
) -> dict[str, ProbeConfig]:
    """Read probes.yaml and return the probe definitions.

    The atom order is lined up with the mol2 by graph isomorphism
    (:mod:`prismsmd.probeorder`). The atoms to map come from ``map_atoms`` (mol2 atom
    names) in the yaml; omitted, every non-hydrogen atom is mapped.

    Args:
        path: the probes.yaml to read.
        resolve_order: False skips reading the mol2, for tests without rdkit.
            Production uses True.
    """
    from .probeorder import resolve_probe_atom_order

    cfg = load_yaml(path)
    probes: dict[str, ProbeConfig] = {}
    base = Path(path).parent

    for cid, spec in (cfg.get("probes") or {}).items():
        cid = str(cid)
        spec = spec or {}
        mol_file = spec.get("mol_file")
        mol2_file = spec.get("mol2_file")
        mol2_abs = str((base / mol2_file).resolve()) if mol2_file else None
        enabled = bool(spec.get("enabled", True))
        smiles = spec.get("smiles")
        map_atoms = spec.get("map_atoms")

        order = None
        atoms: dict[str, str] = {}
        atom_indices: dict[int, str] = {}
        if enabled and resolve_order and mol2_abs:
            if not smiles:
                raise ValueError(
                    f"probe {cid}: an active probe requires smiles.\n"
                    "CosolvKit builds the topology from Molecule.from_smiles(smiles) and"
                    " the coordinates from mol_file (sdf), so smiles is needed to line up"
                    " the atom order."
                )
            order = resolve_probe_atom_order(mol2_abs, str(smiles), probe_id=cid)
            names = (
                [str(a) for a in map_atoms]
                if map_atoms is not None
                else [order.mol2_names[i] for i in order.heavy_indices()]
            )
            for name in names:
                i = order.index_of(name)
                if order.elements[i].upper() == "H":
                    raise ValueError(f"probe {cid}: map_atoms lists a hydrogen ({name})")
                atoms[name] = order.elements[i]
                atom_indices[i] = order.elements[i]
        elif enabled and map_atoms:
            # mol2 not read (tests): keep the labels, the element is unknown.
            atoms = {str(a): "" for a in map_atoms}

        probes[cid] = ProbeConfig(
            cid=cid,
            name=str(spec.get("name", cid)),
            smiles=smiles,
            mol_file=str((base / mol_file).resolve()) if mol_file else None,
            mol2_file=mol2_abs,
            resname=spec.get("resname"),
            concentration=float(
                spec.get("concentration", cfg.get("default_concentration", 1.0))
            ),
            atoms=atoms,
            atom_indices=atom_indices,
            atom_order=order,
            enabled=enabled,
        )
    return probes


@dataclass(frozen=True)
class TargetConfig:
    """Definition of one target."""

    key: str
    name: str
    pdb: str
    docking_center: tuple[float, float, float] | None = None
    docking_size: tuple[float, float, float] | None = None
    spacing: float = 0.375
    ssbonds: tuple[tuple[int, int], ...] = ()
    rmsd_residues: str | None = None      # cpptraj :resnum selection
    vina_map_dir: str | None = None
    notes: str | None = None

    def grid_spec(self, spacing: float | None = None) -> GridSpec:
        """Return the GridSpec of the docking box."""
        if self.docking_center is None or self.docking_size is None:
            raise ValueError(
                f"target {self.key}: docking_center / docking_size are unset."
            )
        return GridSpec.from_size(
            self.docking_center, self.docking_size, spacing or self.spacing
        )


def load_targets(path: str | Path) -> dict[str, TargetConfig]:
    """Read targets.yaml and return the target definitions."""
    cfg = load_yaml(path)
    base = Path(path).parent
    out: dict[str, TargetConfig] = {}
    for key, spec in (cfg.get("targets") or {}).items():
        spec = spec or {}
        center = spec.get("docking_center")
        size = spec.get("docking_size")
        pdb = spec.get("pdb")
        if pdb is None:
            raise ValueError(f"target {key}: no 'pdb'")
        out[str(key)] = TargetConfig(
            key=str(key),
            name=str(spec.get("name", key)),
            pdb=str((base / pdb).resolve()) if not str(pdb).startswith("/") else str(pdb),
            docking_center=tuple(float(v) for v in center) if center else None,  # type: ignore[arg-type]
            docking_size=tuple(float(v) for v in size) if size else None,  # type: ignore[arg-type]
            spacing=float(spec.get("spacing", cfg.get("default_spacing", 0.375))),
            ssbonds=tuple(tuple(int(x) for x in b) for b in (spec.get("ssbonds") or [])),  # type: ignore[misc]
            rmsd_residues=spec.get("rmsd_residues"),
            vina_map_dir=spec.get("vina_map_dir"),
            notes=spec.get("notes"),
        )
    return out


@dataclass(frozen=True)
class RepulsionConfig:
    """Virtual repulsion between probes. The config states Rmin (A); the sigma and
    epsilon GROMACS wants are derived from it."""

    enabled: bool = True
    rmin_angstrom: float = 20.0
    epsilon_kcal: float = 1.0e-6
    site_name: str = "VIS"

    @property
    def sigma_nm(self) -> float:
        """The sigma (nm) written to GROMACS, derived from Rmin."""
        from .system.repulsion import rmin_angstrom_to_sigma_nm

        return rmin_angstrom_to_sigma_nm(self.rmin_angstrom)

    @property
    def epsilon_kj(self) -> float:
        """The epsilon (kJ/mol) written to GROMACS, derived from the kcal value."""
        from .system.repulsion import epsilon_kcal_to_kj

        return epsilon_kcal_to_kj(self.epsilon_kcal)


@dataclass(frozen=True)
class MDStep:
    """One step of the MD sequence."""

    name: str
    type: str                 # minimization / heating / equilibration / production
    nsteps: int
    define: str = ""
    title: str = ""
    nstlog: int = 500
    nstxtcout: int = 5000
    nstenergy: int = 5000
    pcoupl: str = "no"
    initial_temp: float | None = None
    target_temp: float | None = None


@dataclass(frozen=True)
class ProtocolConfig:
    """The MD protocol: integrator settings, restraint ladder and step sequence."""

    dt: float = 0.002
    temperature: float = 300.0
    pressure: float = 1.0
    pbc: str = "xyz"
    seed: int | None = None
    steps: tuple[MDStep, ...] = ()
    #: Restraint strengths in kcal/mol/A^2. The define names use the x100 notation
    #: of :mod:`prismsmd.system.posre` (POSRES1000 = 10 kcal/mol/A^2).
    posres_strengths_kcal: tuple[float, ...] = (10, 5, 2, 1, 0.5, 0.2, 0.1, 0.01, 0)
    caposres_strengths_kcal: tuple[float, ...] = (10, 5, 2, 1, 0.5, 0.2, 0.1, 0)
    repulsion: RepulsionConfig = RepulsionConfig()
    n_runs: int = 20
    padding_angstrom: float = 12.0
    #: Nonbonded cutoff (A) at build time (``forcefield.createSystem``). The mdp does
    #: not state rvdw / rcoulomb, so the GROMACS default of 1.0 nm applies and this
    #: default matches it. Change both together.
    nonbonded_cutoff_angstrom: float = 10.0
    #: ``compressed-x-grps``: the atom group written to xtc. ``None`` or empty means
    #: unset, i.e. all atoms. ``"Non-Water"`` cuts the size sharply at the cost of any
    #: later water analysis. The group must exist in the ``index.ndx`` passed to
    #: grompp (written by :mod:`prismsmd.system.index`).
    compressed_x_grps: str | None = None
    #: Where the probe's partial charges come from: ``"am1bcc"`` for the values the
    #: CosolvKit / GAFF template generator assigns, ``"resp"`` for the RESP charges
    #: in the probe's mol2.
    probe_charges: str = "am1bcc"

    @property
    def production_step(self) -> MDStep:
        """The last production step of the sequence."""
        prods = [s for s in self.steps if s.type == "production"]
        if not prods:
            raise ValueError("no production step is defined")
        return prods[-1]

    @property
    def step_names(self) -> tuple[str, ...]:
        """The step names, in the order they run."""
        return tuple(s.name for s in self.steps)

    def step(self, name: str) -> MDStep:
        """The step of that name."""
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"no step named {name!r}; defined: {list(self.step_names)}")

    def step_ns(self, name: str) -> float:
        """Simulated time of that step (ns). Minimisation counts as 0."""
        s = self.step(name)
        return 0.0 if s.type == "minimization" else s.nsteps * self.dt / 1000.0

    @property
    def total_ns(self) -> float:
        """Simulated time of the whole sequence (ns)."""
        return sum(self.step_ns(n) for n in self.step_names)


#: Accepted values of ``probe_charges``, enumerated so a typo is not silently
#: turned into the default.
PROBE_CHARGE_SOURCES = ("am1bcc", "resp")


def _probe_charges(value, path) -> str:
    v = str(value).lower()
    if v not in PROBE_CHARGE_SOURCES:
        raise ValueError(
            f"{path}: probe_charges must be one of {PROBE_CHARGE_SOURCES}, got '{value}'"
        )
    return v


def load_protocol(path: str | Path) -> ProtocolConfig:
    """Read a protocol yaml and return the protocol configuration."""
    cfg = load_yaml(path)
    general = cfg.get("general") or {}
    rep = cfg.get("repulsion") or {}

    steps: list[MDStep] = []
    for i, spec in enumerate(cfg.get("sequence") or []):
        spec = dict(spec)
        steps.append(
            MDStep(
                name=str(spec.get("name", f"step{i + 1}")),
                type=str(spec["type"]),
                nsteps=int(spec["nsteps"]),
                define=str(spec.get("define", "") or ""),
                title=str(spec.get("title", "")),
                nstlog=int(spec.get("nstlog", 500)),
                nstxtcout=int(spec.get("nstxtcout", 5000)),
                nstenergy=int(spec.get("nstenergy", 5000)),
                pcoupl=str(spec.get("pcoupl", "no")),
                initial_temp=spec.get("initial_temp"),
                target_temp=spec.get("target_temp"),
            )
        )
    if not steps:
        raise ValueError(f"{path}: 'sequence' is empty")

    names = [s.name for s in steps]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate step names: {names}")

    return ProtocolConfig(
        dt=float(general.get("dt", 0.002)),
        temperature=float(general.get("temperature", 300.0)),
        pressure=float(general.get("pressure", 1.0)),
        pbc=str(general.get("pbc", "xyz")),
        seed=general.get("seed"),
        steps=tuple(steps),
        posres_strengths_kcal=tuple(
            float(x)
            for x in general.get(
                "posres_strengths_kcal", (10, 5, 2, 1, 0.5, 0.2, 0.1, 0.01, 0)
            )
        ),
        caposres_strengths_kcal=tuple(
            float(x)
            for x in general.get(
                "caposres_strengths_kcal", (10, 5, 2, 1, 0.5, 0.2, 0.1, 0)
            )
        ),
        repulsion=RepulsionConfig(
            enabled=bool(rep.get("enabled", True)),
            rmin_angstrom=float(rep.get("rmin_angstrom", 20.0)),
            epsilon_kcal=float(rep.get("epsilon_kcal", 1.0e-6)),
            site_name=str(rep.get("site_name", "VIS")),
        ),
        n_runs=int(general.get("n_runs", 20)),
        padding_angstrom=float(general.get("padding_angstrom", 12.0)),
        nonbonded_cutoff_angstrom=float(
            general.get("nonbonded_cutoff_angstrom", 10.0)
        ),
        probe_charges=_probe_charges(general.get("probe_charges", "am1bcc"), path),
        compressed_x_grps=(
            None if not general.get("compressed_x_grps")
            else str(general["compressed_x_grps"])
        ),
    )


@dataclass
class RunConfig:
    """Everything needed for one run."""

    target: TargetConfig
    probes: dict[str, ProbeConfig]
    protocol: ProtocolConfig
    gfe: GFEParams = GFEParams()   # noqa: RUF009 (frozen, so sharing it is safe)
    aggregation: str = "ave"
    pmap_box_size: float = 80.0
    pmap_spacing: float = 1.0
    pmap_valid_distance: float = 5.0
    pmap_snapshot: str = "1-last:1"
    #: Atom name in the reference structure used to fix the grid centre ("CA" or "*").
    pmap_center_selection: str = "CA"
    #: Maximum RMSD from the reference structure (A); above it,
    #: ``TrajectoryNotAlignedError``. A gate on a forgotten fit, not a measure of
    #: accuracy.
    pmap_fit_tolerance: float = 10.0
    #: Clipping, then scaling, in series. Each can be switched independently.
    gfe_clip: bool = True
    gfe_scale: bool = True
    #: How the bulk reference density is determined: ``"counts"`` uses the selected
    #: atom count over (box volume - excluded volume), ``"molar"`` computes it from
    #: the nominal concentration.
    gfe_bulk: str = "counts"
    workdir: str = "./output"
    overwrite: str = "error"

    @classmethod
    def load(
        cls,
        *,
        target_key: str,
        targets_yaml: str | Path,
        probes_yaml: str | Path,
        protocol_yaml: str | Path,
        analysis_yaml: str | Path | None = None,
    ) -> RunConfig:
        """Build a RunConfig from the target, probe, protocol and analysis yamls."""
        targets = load_targets(targets_yaml)
        if target_key not in targets:
            raise KeyError(
                f"target {target_key!r} is not in {targets_yaml}."
                f" Defined: {sorted(targets)}"
            )
        probes = load_probes(probes_yaml)
        protocol = load_protocol(protocol_yaml)

        kwargs: dict[str, Any] = {}
        # probes.yaml is the primary source for concentration. Assumes every probe
        # with mapped atoms shares it (otherwise build GFEParams per probe).
        in_use = [p for p in probes.values() if p.atoms]
        default_molar = in_use[0].concentration if in_use else 1.0
        gfe_params = GFEParams(molar=default_molar)
        if analysis_yaml is not None:
            acfg = load_yaml(analysis_yaml)
            g = acfg.get("gfe") or {}
            clip = g.get("clip_max", 3.0)
            gfe_params = GFEParams(
                molar=float(g.get("molar", default_molar)),
                temperature=float(g.get("temperature", 300.0)),
                clip_max=None if clip in (None, "none", "null") else float(clip),
                floor_probability=float(g.get("floor_probability", 1e-10)),
            )
            p = acfg.get("pmap") or {}
            kwargs.update(
                aggregation=str(acfg.get("aggregation", "ave")),
                pmap_box_size=float(p.get("box_size", 80.0)),
                pmap_spacing=float(p.get("spacing", 1.0)),
                pmap_valid_distance=float(p.get("valid_distance", 5.0)),
                pmap_snapshot=str(p.get("snapshot", "1-last:1")),
                pmap_center_selection=str(p.get("center_selection", "CA")),
                pmap_fit_tolerance=float(p.get("fit_tolerance", 10.0)),
                gfe_clip=bool(g.get("clip", True)),
                gfe_scale=bool(g.get("scale", True)),
                gfe_bulk=str(g.get("bulk", "counts")),
            )
            if kwargs["gfe_bulk"] not in ("counts", "molar"):
                raise ValueError(
                    f"gfe.bulk must be 'counts' or 'molar': {kwargs['gfe_bulk']!r}"
                )
            if "workdir" in acfg:
                kwargs["workdir"] = str(acfg["workdir"])
            if "overwrite" in acfg:
                kwargs["overwrite"] = str(acfg["overwrite"])

        return cls(
            target=targets[target_key],
            probes=probes,
            protocol=protocol,
            gfe=gfe_params,
            **kwargs,
        )

    def pmap_grid_spec(self, reference_pdb: str | Path | None = None) -> GridSpec:
        """Determine the fixed grid of the PMAP from the config and a reference structure.

        Omitting the reference structure uses the target's PDB. The same target with
        the same config always gives the same grid.
        """
        from .pmap_cosolvkit import pmap_grid_spec_from_reference

        return pmap_grid_spec_from_reference(
            reference_pdb if reference_pdb is not None else self.target.pdb,
            box_size=self.pmap_box_size,
            spacing=self.pmap_spacing,
            selection=self.pmap_center_selection,
        )

    def probes_in_use(self) -> list[ProbeConfig]:
        """Return only the probes that have atoms to map."""
        return [p for p in self.probes.values() if p.atoms]

    def to_dict(self) -> dict:
        """Return a summary of the configuration as a plain dict."""
        return {
            "target": self.target.key,
            "probes": {p.cid: p.mapped_atoms for p in self.probes_in_use()},
            "aggregation": self.aggregation,
            "gfe": {
                "bulk": self.gfe_bulk,
                "molar": self.gfe.molar,
                "bulk_probability_from_molar": self.gfe.bulk_probability,
                "clip": self.gfe_clip,
                "clip_max": self.gfe.clip_max,
                "scale": self.gfe_scale,
            },
            "n_runs": self.protocol.n_runs,
        }
