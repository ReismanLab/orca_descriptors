"""Minimal xtb driver (used to pre-optimise start geometries)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from bceg.external import find_executable, run
from bceg.xyz import Conformer, read_xyz, write_xyz

OPT_LEVELS = ("crude", "sloppy", "loose", "lax", "normal", "tight", "vtight", "extreme")
METHOD_FLAGS = {"gfnff": ["--gfnff"], "gfn0": ["--gfn", "0"], "gfn1": ["--gfn", "1"], "gfn2": ["--gfn", "2"]}


def xtb_optimize(elements: list[str], coords: np.ndarray, charge: int, uhf: int, workdir: Path,
                 method: str = "gfn2", level: str = "normal", executable: str = "xtb",
                 threads: int = 1) -> tuple[np.ndarray, float | None]:
    """Optimise one geometry with xtb; returns (coords, energy in Hartree)."""
    if method not in METHOD_FLAGS:
        raise ValueError(f"Unknown xtb method {method!r}; choose from {sorted(METHOD_FLAGS)}")
    if level not in OPT_LEVELS:
        raise ValueError(f"Unknown xtb --opt level {level!r}; choose from {OPT_LEVELS}")
    exe = find_executable(executable)
    workdir.mkdir(parents=True, exist_ok=True)
    write_xyz(workdir / "start.xyz", elements, [Conformer(coords)])
    cmd = [exe, "start.xyz", "--opt", level, *METHOD_FLAGS[method], "--chrg", str(charge), "--uhf", str(uhf)]
    proc = run(cmd, workdir, threads=threads, log_name="xtb.out")
    out = workdir / "xtbopt.xyz"
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"xtb pre-optimisation failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}")
    els, confs = read_xyz(out)
    if els != elements:
        raise RuntimeError("xtb changed the atom order")
    energy = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", proc.stdout)
    return confs[0].coords, float(energy[-1]) if energy else confs[0].energy
