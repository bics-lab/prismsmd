"""Rigid-body superposition (Kabsch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Superposition", "kabsch"]


@dataclass(frozen=True)
class Superposition:
    """Rigid transform ``x -> x @ rotation + translation``."""

    rotation: np.ndarray      # (3, 3)
    translation: np.ndarray   # (3,)
    rmsd: float

    def apply(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        return coords @ self.rotation + self.translation

    def inverse_apply(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        return (coords - self.translation) @ self.rotation.T


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> Superposition:
    """Return the rigid transform that puts ``mobile`` onto ``reference``.

    ``mobile`` and ``reference`` are (N, 3) with point i matching point i.
    Reflections are excluded.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError(
            f"point sets have different shapes: {mobile.shape} vs {reference.shape}"
        )
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("point sets must be (N, 3)")
    if len(mobile) < 3:
        raise ValueError(
            f"superposition needs at least 3 points (got {len(mobile)})"
        )

    mob_c = mobile.mean(axis=0)
    ref_c = reference.mean(axis=0)
    m = mobile - mob_c
    r = reference - ref_c

    cov = m.T @ r
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rot = (vt.T @ correction @ u.T).T   # transposed to match the `x @ rot` convention

    translation = ref_c - mob_c @ rot
    diff = (mobile @ rot + translation) - reference
    rmsd = float(np.sqrt((diff**2).sum() / len(mobile)))
    return Superposition(rotation=rot, translation=translation, rmsd=rmsd)
