"""Reject a built system that cannot be minimised.

A single-point energy of the assembled system decides: anything whose forces are not
finite is refused. :func:`min_intermolecular_gap` reports the closest approach between
two probe molecules for the record; it is not a gate.
"""

from __future__ import annotations

import dataclasses

__all__ = [
    "MAX_INITIAL_POTENTIAL_KJ",
    "ProbeClashError",
    "ProbeGap",
    "assert_system_is_minimisable",
    "min_intermolecular_gap",
    "single_point",
    "single_point_energy",
]

#: Potential energy (kJ/mol) above which the system is declared unbuildable. A
#: backstop for a system whose forces happen to stay finite; the main test is the
#: force (see :func:`assert_system_is_minimisable`).
MAX_INITIAL_POTENTIAL_KJ = 1.0e12


class ProbeClashError(RuntimeError):
    """Raised when two probe molecules overlap badly enough that MD cannot start."""


@dataclasses.dataclass(frozen=True)
class ProbeGap:
    """Closest approach between two different probe molecules."""

    #: Smallest distance over all atom pairs (A).
    any_gap_a: float
    #: Smallest distance over heavy-atom pairs only (A). ``inf`` if there are none.
    heavy_gap_a: float
    #: The atom pair achieving ``any_gap_a``, as ``(resid_a, name_a, resid_b, name_b)``.
    closest: tuple[str, str, str, str] | None
    n_molecules: int
    n_atoms: int


def _cell_list_min(xyz, mol, box, *, cell: float, mask=None):
    """Smallest inter-molecular distance, via a cell list with minimum image.

    ``cell`` must exceed the distance being looked for; the 27-cell neighbourhood is
    scanned, so a genuine minimum below ``cell`` is always found.
    """
    import numpy as np

    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        xyz, mol = xyz[keep], mol[keep]
    if len(xyz) < 2:
        return float("inf"), None

    idx = np.floor(xyz / cell).astype(int)
    ncell = np.maximum(np.floor(box / cell).astype(int), 1)
    idx %= ncell
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(map(tuple, idx)):
        buckets.setdefault(key, []).append(i)

    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    best, where = float("inf"), None
    for key, mine in buckets.items():
        neigh: set[int] = set()
        base = np.array(key)
        for off in offsets:
            neigh.update(buckets.get(tuple((base + off) % ncell), ()))
        cand = np.fromiter(neigh, dtype=int)
        if not len(cand):
            continue
        for i in mine:
            j = cand[mol[cand] != mol[i]]
            if not len(j):
                continue
            d = xyz[j] - xyz[i]
            d -= box * np.round(d / box)          # minimum image
            r = np.sqrt((d * d).sum(axis=1))
            k = int(np.argmin(r))
            if r[k] < best:
                best, where = float(r[k]), (int(i), int(j[k]))
    return best, where


def min_intermolecular_gap(gro_text: str, probe_resname: str, *,
                           cell: float = 4.0) -> ProbeGap:
    """Closest approach between two different probe molecules in a ``.gro``.

    Args:
        gro_text: contents of the ``.gro``.
        probe_resname: residue name of the probe. Water, ions and the protein are
            ignored; the failure being hunted is probe-on-probe.
        cell: cell-list width in A. Must be larger than the distances of interest.

    Returns:
        :class:`ProbeGap`. ``inf`` gaps mean there was nothing to compare.
    """
    import numpy as np

    from .builder import _iter_gro_atoms

    xs, mols, names, resids, heavy = [], [], [], [], []
    for resi, resn, name, _index, xyz in _iter_gro_atoms(gro_text):
        if resn != probe_resname:
            continue
        xs.append(xyz)
        mols.append(f"{resi}{resn}")
        names.append(name)
        resids.append(f"{resi}{resn}")
        # CosolvKit/OpenFF rewrite probe names as f"{element}{serial}x", so the first
        # letter is the element. VIS (the virtual site) carries no LJ term of its own
        # and is excluded from both tests.
        heavy.append(not name.startswith("H") and name != "VIS")

    if not xs:
        return ProbeGap(float("inf"), float("inf"), None, 0, 0)

    xyz = np.asarray(xs) * 10.0                   # nm -> A
    _, mol = np.unique(np.asarray(mols), return_inverse=True)
    lines = gro_text.split("\n")
    natoms = int(lines[1].strip())
    box = np.asarray([float(v) for v in lines[2 + natoms].split()[:3]]) * 10.0

    is_vis = np.asarray([n == "VIS" for n in names])
    any_gap, where = _cell_list_min(xyz, mol, box, cell=cell, mask=~is_vis)
    heavy_gap, _ = _cell_list_min(xyz, mol, box, cell=cell,
                                  mask=np.asarray(heavy))

    closest = None
    if where is not None:
        # where indexes the masked array, so recover the original positions.
        kept = np.flatnonzero(~is_vis)
        a, b = int(kept[where[0]]), int(kept[where[1]])
        closest = (resids[a], names[a], resids[b], names[b])
    return ProbeGap(any_gap, heavy_gap, closest, int(mol.max()) + 1, len(xs))


def single_point(omm_system, positions, *,
                 platform_name: str = "CPU") -> tuple[float, float]:
    """Potential energy (kJ/mol) and largest atomic force (kJ/mol/nm) of the system.

    Args:
        omm_system: the ``openmm.System`` built for this system.
        positions: its positions (an OpenMM Quantity).
        platform_name: ``"CPU"`` by default.

    Returns:
        ``(energy_kj, max_force)``. Either is ``inf`` when OpenMM reports a
        non-finite value.
    """
    import math

    import numpy as np
    import openmm
    import openmm.unit as openmmunit

    # An integrator is required to make a Context even though no step is taken.
    integrator = openmm.VerletIntegrator(0.001 * openmmunit.picoseconds)
    try:
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            context = openmm.Context(omm_system, integrator, platform)
        except Exception:                       # noqa: BLE001 - platform availability
            # Fall back rather than fail: a missing CPU plugin must not stop a build.
            context = openmm.Context(omm_system, integrator)
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getForces=True)
        value = state.getPotentialEnergy().value_in_unit(
            openmmunit.kilojoule_per_mole)
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
            openmmunit.kilojoule_per_mole / openmmunit.nanometer), dtype=float)
        norms = np.sqrt((forces * forces).sum(axis=1))
        fmax = float(norms.max()) if norms.size else 0.0
        del context
    finally:
        del integrator
    energy = float("inf") if not math.isfinite(value) else float(value)
    fmax = float("inf") if not math.isfinite(fmax) else fmax
    return energy, fmax


def single_point_energy(omm_system, positions, *, platform_name: str = "CPU") -> float:
    """Potential energy only, for callers that do not need the force."""
    return single_point(omm_system, positions, platform_name=platform_name)[0]


def assert_system_is_minimisable(
    omm_system,
    positions,
    *,
    gro_text: str | None = None,
    probe_resname: str | None = None,
    max_potential_kj: float = MAX_INITIAL_POTENTIAL_KJ,
) -> tuple[float, ProbeGap | None]:
    """Raise if the assembled system cannot be minimised.

    The test is whether the forces are finite, not whether the energy is small.

    Args:
        omm_system: the ``openmm.System``.
        positions: its positions.
        gro_text: if given, used to name the offending atoms in the message; the
            decision does not depend on it.
        probe_resname: needed alongside ``gro_text``.
        max_potential_kj: backstop, see :data:`MAX_INITIAL_POTENTIAL_KJ`.

    Returns:
        ``(energy_kj, gap)``. ``gap`` is ``None`` when ``gro_text`` was not supplied.

    Raises:
        ProbeClashError: when the forces are not finite, or the energy exceeds the
            backstop.
    """
    import math

    energy, fmax = single_point(omm_system, positions)
    gap = None
    if gro_text is not None and probe_resname is not None:
        gap = min_intermolecular_gap(gro_text, probe_resname)

    detail = ""
    if gap is not None and gap.closest is not None:
        a_res, a_name, b_res, b_name = gap.closest
        detail = (f" Closest probe atoms of different molecules: "
                  f"{a_res} {a_name} -- {b_res} {b_name} at {gap.any_gap_a:.3f} A"
                  f" (heavy-atom pairs: {gap.heavy_gap_a:.3f} A).")

    if not math.isfinite(fmax):
        raise ProbeClashError(
            f"the assembled system has a non-finite force (energy {energy:.4e} "
            f"kJ/mol), which minimisation cannot recover from."
            f"{detail} Rebuild with the next placement seed (prismsmd.seeds.seed_for_attempt)."
        )
    if energy >= max_potential_kj:
        raise ProbeClashError(
            f"the assembled system has a potential energy of {energy:.4e} kJ/mol "
            f"with finite forces, at or above the backstop of "
            f"{max_potential_kj:.1e}.{detail} Rebuild with the next placement seed (prismsmd.seeds.seed_for_attempt)."
        )
    return energy, gap
