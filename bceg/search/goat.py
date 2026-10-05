"""ORCA GOAT global conformer optimizer (``! GOAT <method>``)."""

from __future__ import annotations

import re
from pathlib import Path

from bceg.external import find_executable, resolve_threads, run
from bceg.search.base import CONSTRAINT_DEFAULTS, START_DEFAULTS, START_RESOURCE_KEYS, SearchResult, Searcher
from bceg.structure import Structure
from bceg.xyz import Conformer, read_xyz, write_xyz


class Goat(Searcher):
    name = "goat"
    defaults = {
        "method": "XTB",  # level of the GOAT run (simple-input keyword)
        "keywords": "",  # additional simple-input keywords
        "goat_block": {},  # key -> value lines inside %goat ... end
        **CONSTRAINT_DEFAULTS,  # exact: ORCA's %geom Constraints, no force constant
        **START_DEFAULTS,
        "seed_offset": 0,
        "nprocs": None,  # None/0: SLURM_CPUS_PER_TASK (or all CPUs outside SLURM)
        "maxcore": 2000,
        "orca_cmd": "orca",  # use the absolute path for parallel runs
    }
    resource_keys = {"nprocs", "maxcore", "orca_cmd"} | START_RESOURCE_KEYS
    exact_constraints_only = True

    def build_input(self, structure: Structure, constraints: dict | None = None) -> str:
        p = self.params
        lines = [f"! GOAT {p['method']} {p['keywords']}".rstrip(),
                 f"%pal nprocs {resolve_threads(p['nprocs'])} end", f"%maxcore {p['maxcore']}"]
        if p["goat_block"]:
            lines += ["%goat"] + [f"  {k} {v}" for k, v in p["goat_block"].items()] + ["end"]
        if constraints:
            lines += ["%geom", "  Constraints"] + [f"    {c}" for c in geom_constraints(constraints)] + ["  end", "end"]
        lines.append(f"* xyzfile {structure.charge} {structure.multiplicity} input.xyz")
        return "\n".join(lines) + "\n"

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        exe = find_executable(self.params["orca_cmd"])
        seed = self.seed(replicate)
        coords, start = self.start_geometry(structure, seed, workdir)
        constraints = self.resolve_constraints(structure, coords)
        write_xyz(workdir / "input.xyz", structure.elements, [Conformer(coords)])
        (workdir / "goat.inp").write_text(self.build_input(structure, constraints))

        proc = run([exe, "goat.inp"], workdir, threads=1, log_name="goat.out")
        out = workdir / "goat.finalensemble.xyz"
        logs = [workdir / "goat.inp", workdir / "start_geometry.xyz", workdir / "goat.out"]
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"GOAT failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}")
        elements, confs = read_xyz(out)
        if elements != structure.elements:
            raise RuntimeError("GOAT output atom order differs from the input")
        version = re.search(r"Program Version\s+(\S+)", proc.stdout)
        return SearchResult(
            confs,
            versions={"orca": version.group(1) if version else "unknown"},
            details={"seed": seed, "start_geometry": start} | ({"constraints": constraints} if constraints else {}),
            log_files=logs,
        )


ORCA_CONSTRAINT_TYPE = {"bonds": "B", "angles": "A", "dihedrals": "D"}


def geom_constraints(constraints: dict) -> list[str]:
    """``%geom Constraints`` lines for resolved constraints (bceg.constraints.resolve): ``{C i C}`` per
    frozen atom, ``{B i j value C}`` (Angstrom), ``{A ...}`` and ``{D ...}`` (degrees). ORCA counts atoms
    from 0, as bceg does."""
    lines = [f"{{ C {i} C }}" for i in constraints["freeze"]]
    for kind, letter in ORCA_CONSTRAINT_TYPE.items():
        for c in constraints[kind]:
            lines.append(f"{{ {letter} {' '.join(map(str, c['atoms']))} {c['value']:.4f} C }}")
    return lines
