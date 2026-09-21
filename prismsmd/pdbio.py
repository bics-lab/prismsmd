"""Minimal PDB reader (single- and multi-model), returning numpy arrays.

The fixed-column offsets are defined once, in :data:`PDB_COLUMNS`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PDB_COLUMNS",
    "Frame",
    "atom_coord",
    "iter_models",
    "parse_pdb_string",
    "read_pdb",
]

#: Fixed-column offsets of an ATOM/HETATM record (0-based, end-exclusive),
#: per the PDB format specification v3.3.
PDB_COLUMNS = {
    "atom_name": (12, 16),
    "res_name": (17, 20),
    "chain": (21, 22),
    "res_seq": (22, 26),
    "x": (30, 38),
    "y": (38, 46),
    "z": (46, 54),
    "element": (76, 78),
}


def atom_coord(line: str) -> tuple[float, float, float]:
    """Return the (x, y, z) of an ATOM/HETATM line, in Angstrom."""
    return (
        float(line[slice(*PDB_COLUMNS["x"])]),
        float(line[slice(*PDB_COLUMNS["y"])]),
        float(line[slice(*PDB_COLUMNS["z"])]),
    )


@dataclass
class Frame:
    """The atoms of one snapshot. All arrays have the same length."""

    atom_name: np.ndarray   # <U4
    res_name: np.ndarray    # <U4
    res_seq: np.ndarray     # int
    chain: np.ndarray       # <U1
    coord: np.ndarray       # (N, 3) float

    def __len__(self) -> int:
        return len(self.atom_name)

    def select(self, mask: np.ndarray) -> Frame:
        """Return a Frame holding only the atoms where ``mask`` is true."""
        return Frame(
            atom_name=self.atom_name[mask],
            res_name=self.res_name[mask],
            res_seq=self.res_seq[mask],
            chain=self.chain[mask],
            coord=self.coord[mask],
        )

    def residue_keys(self) -> np.ndarray:
        """Return a per-atom key identifying its residue: ``chain|res_seq|res_name``."""
        return np.array(
            [f"{c}|{s}|{n}" for c, s, n in zip(self.chain, self.res_seq, self.res_name, strict=True)]
        )

    def alpha_carbons(self) -> Frame:
        """Return a Frame holding only the Ca atoms."""
        return self.select(self.atom_name == "CA")

    def by_resname(self, resname: str) -> Frame:
        """Return a Frame holding only atoms of residues with this name."""
        return self.select(self.res_name == resname)

    def exclude_resname(self, resname: str) -> Frame:
        """Return a Frame holding every atom except residues with this name."""
        return self.select(self.res_name != resname)


def _parse_atom_lines(lines: Sequence[str]) -> Frame:
    names, resnames, resseqs, chains, coords = [], [], [], [], []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        names.append(line[slice(*PDB_COLUMNS["atom_name"])].strip())
        resnames.append(line[slice(*PDB_COLUMNS["res_name"])].strip())
        chains.append(line[slice(*PDB_COLUMNS["chain"])].strip() or " ")
        resseqs.append(int(line[slice(*PDB_COLUMNS["res_seq"])]))
        coords.append(atom_coord(line))
    return Frame(
        atom_name=np.array(names, dtype="<U4"),
        res_name=np.array(resnames, dtype="<U4"),
        res_seq=np.array(resseqs, dtype=int),
        chain=np.array(chains, dtype="<U1"),
        coord=np.array(coords, dtype=float).reshape(-1, 3),
    )


def parse_pdb_string(text: str) -> list[Frame]:
    """Parse a PDB string into a list of Frames, one per MODEL.

    A file without MODEL records gives a single Frame. Raises if no atoms.
    """
    frames: list[Frame] = []
    buf: list[str] = []
    saw_model = False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            saw_model = True
            buf = []
            continue
        if line.startswith("ENDMDL"):
            frames.append(_parse_atom_lines(buf))
            buf = []
            continue
        buf.append(line)
    if buf and (not saw_model or any(
        ln.startswith(("ATOM", "HETATM")) for ln in buf
    )):
        frame = _parse_atom_lines(buf)
        if len(frame) > 0:
            frames.append(frame)
    if not frames:
        raise ValueError("the PDB contains no atoms at all")
    return frames


def iter_models(path: str | Path) -> Iterator[Frame]:
    """Yield the models of a PDB file one at a time."""
    path = Path(path)
    buf: list[str] = []
    saw_atom = False
    with path.open() as fh:
        for line in fh:
            if line.startswith("MODEL"):
                buf = []
                saw_atom = False
                continue
            if line.startswith("ENDMDL"):
                if saw_atom:
                    yield _parse_atom_lines(buf)
                buf = []
                saw_atom = False
                continue
            if line.startswith(("ATOM", "HETATM")):
                saw_atom = True
            buf.append(line)
    if saw_atom:
        yield _parse_atom_lines(buf)


def read_pdb(path: str | Path) -> Frame:
    """Read a PDB file and return its first model."""
    return next(iter_models(path))
