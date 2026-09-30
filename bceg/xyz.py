"""Multi-xyz I/O. Energies are in Hartree, written as the first float of the comment line."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.211386246
BOHR_TO_ANGSTROM = 0.529177210903

_FLOAT = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")


@dataclass
class Conformer:
    coords: np.ndarray  # (n_atoms, 3), Angstrom
    energy: float | None = None  # Hartree
    info: dict[str, Any] = field(default_factory=dict)  # extra key/values written to the comment line


def write_xyz(path: str | Path, elements: list[str], conformers: list[Conformer]) -> None:
    lines = []
    for c in conformers:
        comment = [] if c.energy is None else [f"{c.energy:.10f}"]
        comment += [f"{k}={v}" for k, v in c.info.items()]
        lines.append(str(len(elements)))
        lines.append(" ".join(comment))
        lines.extend(f"{el:<2} {x:15.8f} {y:15.8f} {z:15.8f}" for el, (x, y, z) in zip(elements, c.coords))
    Path(path).write_text("\n".join(lines) + "\n")


def read_xyz(path: str | Path) -> tuple[list[str], list[Conformer]]:
    """Read a (multi-)xyz file; the first float on each comment line is taken as the energy."""
    lines = Path(path).read_text().splitlines()
    elements: list[str] = []
    confs = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        m = _FLOAT.search(lines[i + 1])
        block = [ln.split() for ln in lines[i + 2 : i + 2 + n]]
        els = [b[0].capitalize() for b in block]
        if elements and els != elements:
            raise ValueError(f"{path}: inconsistent atom order between frames")
        elements = els
        coords = np.array([[float(v) for v in b[1:4]] for b in block])
        confs.append(Conformer(coords, float(m.group()) if m else None))
        i += 2 + n
    return elements, confs
