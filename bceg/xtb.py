"""Minimal xtb driver (used to pre-optimise start geometries)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from bceg.external import ProgramNotFound, find_executable, program_version, run
from bceg.xyz import Conformer, read_xyz, write_xyz

OPT_LEVELS = ("crude", "sloppy", "loose", "lax", "normal", "tight", "vtight", "extreme")
METHOD_FLAGS = {"gfnff": ["--gfnff"], "gfn0": ["--gfn", "0"], "gfn1": ["--gfn", "1"], "gfn2": ["--gfn", "2"]}


@lru_cache(maxsize=None)
def _version_of(exe: str) -> str | None:
    """``--version`` line of an absolute path (one subprocess per binary, per process)."""
    return program_version([exe, "--version"])


def xtb_identity(executable: str = "xtb") -> tuple[str | None, str | None]:
    """(resolved path, version line) of the xtb ``executable`` names, or (None, None).

    Both are needed: ``xtb_executable`` is a resource key, so a different build never
    trips the params.json guard, and two builds of the same release report the same
    version string -- the path is what distinguishes a `module load xtb` from the
    conda one when both shadow `xtb` on PATH. Resolution is repeated on every call
    (only the version subprocess is cached, keyed by path), so a name is never
    reported against a binary it no longer resolves to.
    """
    try:
        exe = find_executable(executable)
    except ProgramNotFound:
        return None, None
    return exe, _version_of(exe)


def fix_input(workdir: Path, fixed: list[int] | None) -> list[str]:
    """Write ``xcontrol`` fixing the atoms ``fixed`` (0-based; xtb counts from 1) and return the ``--input``
    arguments that read it; nothing without fixed atoms.

    ``$fix`` holds the atoms exactly only with the L-BFGS (or FIRE) optimiser: xtb 6.7.1's default ANCopt
    ignores it without a word (measured on bipy-NiH2: the "fixed" first shell moved by 2 A). Restraining
    the distances within the group (``$constrain atoms:``) does not hold a plane or a linear angle either:
    bending out of the plane changes those distances only to second order."""
    if not fixed:
        return []
    (workdir / "xcontrol").write_text("$fix\n   atoms: " + ",".join(str(i + 1) for i in fixed)
                                      + "\n$opt\n   engine=lbfgs\n$end\n")
    return ["--input", "xcontrol"]


def xtb_optimize(elements: list[str], coords: np.ndarray, charge: int, uhf: int, workdir: Path,
                 method: str = "gfn2", level: str = "normal", executable: str = "xtb",
                 threads: int = 1, fixed: list[int] | None = None) -> tuple[np.ndarray, float | None]:
    """Optimise one geometry with xtb, the atoms ``fixed`` (0-based) held in place; returns (coords,
    energy in Hartree)."""
    if method not in METHOD_FLAGS:
        raise ValueError(f"Unknown xtb method {method!r}; choose from {sorted(METHOD_FLAGS)}")
    if level not in OPT_LEVELS:
        raise ValueError(f"Unknown xtb --opt level {level!r}; choose from {OPT_LEVELS}")
    exe = find_executable(executable)
    workdir.mkdir(parents=True, exist_ok=True)
    write_xyz(workdir / "start.xyz", elements, [Conformer(coords)])
    cmd = [exe, "start.xyz", "--opt", level, *METHOD_FLAGS[method], "--chrg", str(charge), "--uhf", str(uhf)]
    cmd += fix_input(workdir, fixed)
    proc = run(cmd, workdir, threads=threads, log_name="xtb.out")
    out = workdir / "xtbopt.xyz"
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"xtb pre-optimisation failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}")
    els, confs = read_xyz(out)
    if els != elements:
        raise RuntimeError("xtb changed the atom order")
    energy = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", proc.stdout)
    return confs[0].coords, float(energy[-1]) if energy else confs[0].energy
