"""Make CosolvKit's probe placement depend only on a seed.

Placement draws on ``scipy.stats.qmc.Halton``, which is seeded from OS entropy unless
told otherwise, and on ``numpy.random``'s legacy global state; both are pinned by
:func:`deterministic_placement`. PDBFixer's hydrogen placement, upstream of it, also
moves the probes, so the context manager is entered before the receptor is tidied
rather than around ``build()``.

Only the ``qmc`` attribute of the cosolvkit module object is swapped, for the duration
of the ``with`` block; ``scipy.stats.qmc`` itself is left alone.
"""

from __future__ import annotations

import contextlib
import random as _random

__all__ = ["PlacementSeedError", "deterministic_placement"]


class PlacementSeedError(RuntimeError):
    """Raised when placement cannot be made deterministic."""


class _SeededQmc:
    """Stand-in for ``scipy.stats.qmc`` that gives ``Halton`` a fixed seed.

    Every other attribute is forwarded to the real module untouched.
    """

    def __init__(self, real, seed: int):
        self._real = real
        self._seed = seed

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Halton(self, *args, **kwargs):  # noqa: N802 - mirrors scipy's name
        # setdefault, not overwrite: if a future CosolvKit passes its own seed,
        # respect it rather than silently overriding.
        kwargs.setdefault("seed", self._seed)
        return self._real.Halton(*args, **kwargs)


@contextlib.contextmanager
def deterministic_placement(seed: int, *, module=None):
    """Pin every randomness CosolvKit's placement uses, for the block's duration.

    Args:
        seed: the run's seed.
        module: the module whose ``qmc`` attribute to swap. Defaults to
            ``cosolvkit.cosolvent_system``; injectable for testing.

    Raises:
        PlacementSeedError: if the module does not expose ``qmc.Halton``.

    The previous states of ``numpy.random`` and ``random`` are saved and restored,
    so callers outside the block are unaffected.
    """
    import numpy as np

    if module is None:
        try:
            from cosolvkit import cosolvent_system as module  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise PlacementSeedError(
                "cosolvkit is not importable, so placement cannot be seeded"
            ) from exc

    real_qmc = getattr(module, "qmc", None)
    if real_qmc is None or not hasattr(real_qmc, "Halton"):
        raise PlacementSeedError(
            f"{module.__name__} does not expose qmc.Halton. CosolvKit must have "
            "changed how it builds the Halton sampler; placement seeding was "
            "verified against the line 'sampler = qmc.Halton(d=3)'."
        )

    np_state = np.random.get_state()
    py_state = _random.getstate()
    try:
        module.qmc = _SeededQmc(real_qmc, seed)
        np.random.seed(seed)
        _random.seed(seed)
        yield
    finally:
        module.qmc = real_qmc
        np.random.set_state(np_state)
        _random.setstate(py_state)
