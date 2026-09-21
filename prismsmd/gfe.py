"""PMAP -> GFE conversion.

Two stages are applied in series:

1. ``GFE = -RT ln(P / P_bulk)``, with ``P <= 0`` replaced by a floor probability and
   the result clipped from above (Raman et al., *J. Chem. Inf. Model.* **53** (2013)
   3384).
2. Points with ``GFE < 0`` are linearly mapped onto the negative-value distribution
   of a reference (Vina) map.

Each stage can be switched off independently.

The bulk occupancy is the number of probe atoms selected for the map divided by
(box volume - excluded volume), passed in through ``bulk_probability``
(:class:`prismsmd.volume.BulkReference`). :func:`bulk_number_density` derives it from
the nominal molar concentration instead, for when those volumes are unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import constants as C

from .grid import Grid

__all__ = [
    "GFEParams",
    "bulk_number_density",
    "bulk_number_density_from_atom_count",
    "clip_gfe",
    "energy_from_probability",
    "gas_constant_kcal",
    "pmap_to_gfe",
    "rt_kcal",
    "scale_negatives_to_reference",
]


def gas_constant_kcal() -> float:
    """Return the gas constant R in kcal/mol/K."""
    return C.R / C.calorie / C.kilo


def rt_kcal(temperature: float = 300.0) -> float:
    """Return RT in kcal/mol."""
    return gas_constant_kcal() * temperature


def bulk_number_density(molar: float) -> float:
    """Convert a concentration (mol/L) into a bulk number density (per A^3)."""
    if molar <= 0:
        raise ValueError(f"molar must be positive: {molar!r}")
    return molar / C.liter * C.N_A * C.angstrom**3


def bulk_number_density_from_atom_count(
    n_atoms: int, box_volume: float, excluded_volume: float
) -> float:
    """Bulk number density (per A^3) as ``n_atoms / (box_volume - excluded_volume)``.

    ``n_atoms`` is the total number of atoms the map's atom selection matched, over
    every probe molecule in the system.
    """
    from .volume import BulkReference

    return BulkReference(
        n_atoms=int(n_atoms),
        box_volume=float(box_volume),
        excluded_volume=float(excluded_volume),
    ).number_density


@dataclass(frozen=True)
class GFEParams:
    """Parameters of the GFE conversion.

    Attributes:
        molar: nominal molar concentration of the probe (M), used only by the
            alternative bulk occupancy :func:`bulk_number_density`.
        temperature: temperature (K).
        clip_max: upper bound on GFE (kcal/mol); ``None`` disables clipping.
        floor_probability: lower bound on probability, to avoid log(0).
    """

    molar: float = 1.0
    temperature: float = 300.0
    clip_max: float | None = 3.0
    floor_probability: float = 1e-10

    @property
    def rt(self) -> float:
        return rt_kcal(self.temperature)

    @property
    def bulk_probability(self) -> float:
        """Bulk occupancy (per A^3), derived from :attr:`molar`."""
        return bulk_number_density(self.molar)


def clip_gfe(values: np.ndarray, clip_max: float | None) -> np.ndarray:
    """Clip GFE from above at ``clip_max``; ``None`` returns the values unchanged."""
    if clip_max is None:
        return values
    return np.where(values > clip_max, clip_max, values)


def pmap_to_gfe(
    pmap: Grid,
    params: GFEParams = GFEParams(),  # noqa: B008 (frozen, so sharing it is safe)
    *,
    reference: Grid | None = None,
    clip: bool = True,
    scale: bool = True,
    bulk_probability: float | None = None,
) -> Grid:
    """Convert a PMAP into GFE, clipping and then scaling.

    Args:
        pmap: the occupancy grid.
        params: temperature, concentration and clip bound.
        reference: the Vina map used for scaling. Required when ``scale=True``.
        clip: clip at ``params.clip_max``.
        scale: linearly map negative values onto the negative-value distribution of
            ``reference``.
        bulk_probability: the bulk number density (per A^3), given explicitly. Only
            when it is ``None`` is it computed from ``params.molar`` instead.
    """
    p_bulk = params.bulk_probability if bulk_probability is None else bulk_probability
    if p_bulk <= 0:
        raise ValueError(f"bulk occupancy must be positive: {p_bulk}")

    p = np.where(pmap.values <= 0, params.floor_probability, pmap.values)
    gfe = -params.rt * np.log(p / p_bulk)

    if clip:
        gfe = clip_gfe(gfe, params.clip_max)

    if scale:
        if reference is None:
            raise ValueError("scale=True requires a reference (Vina map)")
        pmap.spec.assert_compatible(reference.spec)
        gfe = scale_negatives_to_reference(gfe, reference.values)

    return pmap.copy_with(gfe)


def scale_negatives_to_reference(
    values: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Map the negative values onto the mean and SD of the reference's negatives."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)

    neg_mask = values < 0
    ref_neg = reference[reference < 0]
    if not neg_mask.any() or ref_neg.size == 0:
        raise ValueError(
            f"not enough negative values: gfe={int(neg_mask.sum())}, reference={ref_neg.size}"
        )

    x = values[neg_mask]
    sd = x.std()
    if sd == 0:
        raise ValueError("the SD of the negative GFE values is 0, cannot scale")
    scaled = (x - x.mean()) / sd * ref_neg.std() + ref_neg.mean()

    out = values.copy()
    out[neg_mask] = scaled
    return out


def energy_from_probability(
    probability: float,
    params: GFEParams = GFEParams(clip_max=None),  # noqa: B008 (frozen, so immutable)
) -> float:
    """GFE for a single point, before clipping and scaling."""
    if probability <= 0:
        return math.inf
    value = -params.rt * math.log(probability / params.bulk_probability)
    if params.clip_max is not None:
        value = min(value, params.clip_max)
    return value
