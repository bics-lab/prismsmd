"""Grid definition and OpenDX I/O.

This module is the single home of the grid definition; the origin formula and the
axis-order conversions are not written anywhere else.

Grid values are always held as a ``numpy.ndarray`` of shape ``(nx, ny, nz)``. The
orderings the file formats use stay confined to the conversion helpers below:

OpenDX      : C order ``grid[x][y][z]``, i.e. z varies fastest
AutoDock map: x varies fastest (``for z: for y: for x:``)
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DXFormatError",
    "Grid",
    "GridSpec",
    "read_dx",
    "write_dx",
]


class DXFormatError(ValueError):
    """Raised when an OpenDX file is not shaped as expected."""


@dataclass(frozen=True)
class GridSpec:
    """Geometry of a grid.

    Attributes:
        center: centre coordinate of the grid (A), as in an AutoDock map's ``CENTER``.
        dims: number of grid points along each axis, ``(nx, ny, nz)``. An AutoDock
            map's ``NELEMENTS`` is this minus one.
        spacing: grid spacing (A).
    """

    center: tuple[float, float, float]
    dims: tuple[int, int, int]
    spacing: float

    def __post_init__(self) -> None:
        if len(self.center) != 3:
            raise ValueError(f"center must have 3 elements: {self.center!r}")
        if len(self.dims) != 3:
            raise ValueError(f"dims must have 3 elements: {self.dims!r}")
        if any(int(d) < 2 for d in self.dims):
            raise ValueError(f"dims must be at least 2 on every axis: {self.dims!r}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive: {self.spacing!r}")
        object.__setattr__(self, "center", tuple(float(c) for c in self.center))
        object.__setattr__(self, "dims", tuple(int(d) for d in self.dims))
        object.__setattr__(self, "spacing", float(self.spacing))

    @property
    def nelements(self) -> tuple[int, int, int]:
        """AutoDock NELEMENTS (= number of grid points - 1)."""
        return tuple(d - 1 for d in self.dims)  # type: ignore[return-value]

    @property
    def origin(self) -> tuple[float, float, float]:
        """Coordinate of the lowest-corner grid point.

        The single definition: ``origin = center - (dims - 1) * spacing / 2``,
        consistent with the Vina .map header (CENTER / NELEMENTS / SPACING).
        """
        return tuple(
            self.center[i] - (self.dims[i] - 1) * self.spacing / 2.0 for i in range(3)
        )  # type: ignore[return-value]

    @property
    def box_size(self) -> tuple[float, float, float]:
        """Edge length spanned by the grid (= nelements * spacing)."""
        return tuple((self.dims[i] - 1) * self.spacing for i in range(3))  # type: ignore[return-value]

    @property
    def npoints(self) -> int:
        """Total number of grid points."""
        return self.dims[0] * self.dims[1] * self.dims[2]

    def axis_coordinates(self, axis: int) -> np.ndarray:
        """Return the grid-point coordinates along one axis."""
        return self.origin[axis] + np.arange(self.dims[axis]) * self.spacing

    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """gridData-compatible edges (boundaries when grid points are cell centres)."""
        return tuple(  # type: ignore[return-value]
            self.origin[i] - self.spacing / 2.0
            + np.arange(self.dims[i] + 1) * self.spacing
            for i in range(3)
        )

    @classmethod
    def from_size(
        cls,
        center: Sequence[float],
        size: Sequence[float],
        spacing: float,
    ) -> GridSpec:
        """Build a GridSpec by deriving the point count from the box edge lengths."""
        dims = tuple(cls.nelements_for(s, spacing) + 1 for s in size)
        return cls(center=tuple(center), dims=dims, spacing=spacing)  # type: ignore[arg-type]

    @staticmethod
    def nelements_for(size: float, spacing: float) -> int:
        """Edge length -> NELEMENTS, rounded up to an even number as AutoDock requires."""
        if size <= 0:
            raise ValueError(f"size must be positive: {size!r}")
        n = int(math.ceil(round(size / spacing, 9)))
        if n % 2:
            n += 1
        return n

    def to_dict(self) -> dict:
        """Return the geometry as a plain dict."""
        return {
            "center": list(self.center),
            "dims": list(self.dims),
            "spacing": self.spacing,
            "nelements": list(self.nelements),
            "origin": [round(v, 6) for v in self.origin],
        }

    def assert_compatible(self, other: GridSpec, tol: float = 1e-6) -> None:
        """Raise unless the two grids can be treated as identical."""
        if self.dims != other.dims:
            raise ValueError(f"dims differ: {self.dims} vs {other.dims}")
        if abs(self.spacing - other.spacing) > tol:
            raise ValueError(f"spacing differs: {self.spacing} vs {other.spacing}")
        for i in range(3):
            if abs(self.origin[i] - other.origin[i]) > tol:
                raise ValueError(
                    f"origin differs: {self.origin} vs {other.origin}"
                )


@dataclass
class Grid:
    """Grid spec plus values. Values have shape (nx, ny, nz)."""

    spec: GridSpec
    values: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != self.spec.dims:
            raise ValueError(
                f"values shape {self.values.shape} differs from dims {self.spec.dims}"
            )

    def copy_with(self, values: np.ndarray) -> Grid:
        """Return a new Grid with the same spec and the given values."""
        return Grid(spec=self.spec, values=np.asarray(values, dtype=float))


_RE_COUNTS = re.compile(
    r"^object\s+1\s+class\s+gridpositions\s+counts\s+(\d+)\s+(\d+)\s+(\d+)", re.IGNORECASE
)
_RE_ORIGIN = re.compile(r"^origin\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)
_RE_DELTA = re.compile(r"^delta\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)
_RE_ITEMS = re.compile(r"^object\s+3\s+class\s+array\b.*?items\s+(\d+)\s+data\s+follows", re.IGNORECASE)

#: Prefixes of the epilogue lines that mark the end of the data section.
_EPILOGUE_PREFIXES = ("attribute", "object", "component", "end")


def read_dx(path: str | Path) -> Grid:
    """Read an OpenDX file.

    The header's ``items N`` is trusted and reading stops after exactly N values, so
    the epilogue gridData writes is not mistaken for grid data.
    """
    path = Path(path)
    counts: tuple[int, int, int] | None = None
    origin: list[float] | None = None
    deltas: list[list[float]] = []
    n_items: int | None = None

    values: list[float] = []

    with path.open() as fh:
        # --- Header ---
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _RE_COUNTS.match(stripped)
            if m:
                counts = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                continue
            m = _RE_ORIGIN.match(stripped)
            if m:
                origin = [float(m.group(i)) for i in (1, 2, 3)]
                continue
            m = _RE_DELTA.match(stripped)
            if m:
                deltas.append([float(m.group(i)) for i in (1, 2, 3)])
                continue
            m = _RE_ITEMS.match(stripped)
            if m:
                n_items = int(m.group(1))
                break

        if counts is None:
            raise DXFormatError(f"{path}: no 'object 1 class gridpositions counts'")
        if origin is None:
            raise DXFormatError(f"{path}: no 'origin'")
        if len(deltas) != 3:
            raise DXFormatError(f"{path}: expected 3 'delta' lines, found {len(deltas)}")
        if n_items is None:
            raise DXFormatError(f"{path}: no 'object 3 class array ... data follows'")

        expected = counts[0] * counts[1] * counts[2]
        if n_items != expected:
            raise DXFormatError(
                f"{path}: items={n_items} does not match the product of counts, {expected}"
            )

        # --- Data section: stop after exactly N values ---
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if low.startswith(_EPILOGUE_PREFIXES):
                break  # reached the epilogue -> data is done
            if stripped.startswith("#"):
                continue
            for tok in stripped.split():
                values.append(float(tok))
            if len(values) >= n_items:
                break

    if len(values) != n_items:
        raise DXFormatError(
            f"{path}: only {len(values)} values could be read (expected {n_items})"
        )

    # spacing comes from the diagonal (off-diagonal entries are required to be 0)
    for i in range(3):
        for j in range(3):
            if i != j and abs(deltas[i][j]) > 1e-9:
                raise DXFormatError(f"{path}: non-orthogonal delta is not supported")
    spacings = [deltas[i][i] for i in range(3)]
    if max(spacings) - min(spacings) > 1e-9:
        raise DXFormatError(f"{path}: spacing differs between axes: {spacings}")
    spacing = spacings[0]

    # Recovering center from origin leaves float noise in the last digits, so round
    # (prevents a .map CENTER line reading "1.9699000000000009")
    center = tuple(
        round(origin[i] + (counts[i] - 1) * spacing / 2.0, 6) for i in range(3)
    )
    spec = GridSpec(center=center, dims=counts, spacing=spacing)  # type: ignore[arg-type]

    # dx is C order (z fastest), so reshape straight to (nx, ny, nz)
    arr = np.asarray(values, dtype=float).reshape(spec.dims)
    return Grid(spec=spec, values=arr)


_DX_HEADER_COMMENT = (
    "# OpenDX density file written by prismsmd.grid.write_dx()\n"
    "# File format: http://opendx.sdsc.edu/docs/html/pages/usrgu068.htm#HDREDF\n"
    "# Data are embedded in the header and tied to the grid positions.\n"
    "# Data is written in C array order: In grid[x,y,z] the axis z is fastest\n"
    "# varying, then y, then finally x, i.e. z is the innermost loop.\n"
    "# (Note: the VMD dx-reader chokes on comments below this line)\n"
)


def write_dx(path: str | Path, grid: Grid, *, precision: int = 5) -> Path:
    """Write an OpenDX file, in the same layout as gridData's ``export()``.

    Three numbers per line, tab separated, followed by the epilogue.
    """
    path = Path(path)
    spec = grid.spec
    nx, ny, nz = spec.dims
    o = spec.origin
    flat = grid.values.ravel(order="C")  # z fastest

    fmt = f"%.{precision}f"
    with path.open("w") as fh:
        fh.write(_DX_HEADER_COMMENT)
        fh.write(f"object 1 class gridpositions counts  {nx} {ny} {nz}\n")
        fh.write(f"origin {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}\n")
        fh.write(f"delta  {spec.spacing} 0 0\n")
        fh.write(f"delta  0 {spec.spacing} 0\n")
        fh.write(f"delta  0 0 {spec.spacing}\n")
        fh.write(f"object 2 class gridconnections counts  {nx} {ny} {nz}\n")
        fh.write(
            f'object 3 class array type "double" rank 0 items {flat.size} data follows\n'
        )
        for i in range(0, flat.size, 3):
            fh.write("\t".join(fmt % v for v in flat[i : i + 3]) + "\n")
        fh.write('attribute "dep" string "positions"\n')
        fh.write('object "density" class field\n')
        fh.write('component "positions" value 1\n')
        fh.write('component "connections" value 2\n')
        fh.write('component "data" value 3\n')
    return path


def dx_flat_to_array(values: Iterable[float], dims: Sequence[int]) -> np.ndarray:
    """Flat dx data (z fastest) -> (nx, ny, nz)."""
    arr = np.asarray(list(values), dtype=float)
    nx, ny, nz = dims
    if arr.size != nx * ny * nz:
        raise ValueError(f"cannot reshape {arr.size} values into {dims}")
    return arr.reshape((nx, ny, nz))


def array_to_dx_flat(arr: np.ndarray) -> np.ndarray:
    """(nx, ny, nz) -> flat dx data (z fastest)."""
    return np.asarray(arr, dtype=float).ravel(order="C")


def map_flat_to_array(values: Iterable[float], dims: Sequence[int]) -> np.ndarray:
    """Flat AutoDock map data (x fastest) -> (nx, ny, nz)."""
    arr = np.asarray(list(values), dtype=float)
    nx, ny, nz = dims
    if arr.size != nx * ny * nz:
        raise ValueError(f"cannot reshape {arr.size} values into {dims}")
    return arr.reshape((nz, ny, nx)).transpose(2, 1, 0)


def array_to_map_flat(arr: np.ndarray) -> np.ndarray:
    """(nx, ny, nz) -> flat AutoDock map data (x fastest)."""
    return np.asarray(arr, dtype=float).transpose(2, 1, 0).ravel(order="C")
