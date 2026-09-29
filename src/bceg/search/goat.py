"""ORCA GOAT global conformer optimizer (``! GOAT <method>``)."""

from __future__ import annotations

import re
from pathlib import Path

from bceg.external import find_executable, run
from bceg.search.base import START_DEFAULTS, SearchResult, Searcher
from bceg.structure import Structure
from bceg.xyz import Conformer, read_xyz, write_xyz


class Goat(Searcher):
    name = "goat"
    defaults = {
        "method": "XTB",  # level of the GOAT run (simple-input keyword)
        "keywords": "",  # additional simple-input keywords
        "goat_block": {},  # key -> value lines inside %goat ... end
        **START_DEFAULTS,
        "seed_offset": 0,
        "nprocs": 8,
        "maxcore": 2000,
        "orca_cmd": "orca",  # use the absolute path for parallel runs
    }
    resource_keys = {"nprocs", "maxcore", "orca_cmd"}

    def build_input(self, structure: Structure) -> str:
        p = self.params
        lines = [f"! GOAT {p['method']} {p['keywords']}".rstrip(),
                 f"%pal nprocs {p['nprocs']} end", f"%maxcore {p['maxcore']}"]
        if p["goat_block"]:
            lines += ["%goat"] + [f"  {k} {v}" for k, v in p["goat_block"].items()] + ["end"]
        lines.append(f"* xyzfile {structure.charge} {structure.multiplicity} input.xyz")
        return "\n".join(lines) + "\n"

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        exe = find_executable(self.params["orca_cmd"])
        seed = self.seed(replicate)
        coords, start = self.start_geometry(structure, seed)
        write_xyz(workdir / "input.xyz", structure.elements, [Conformer(coords)])
        (workdir / "goat.inp").write_text(self.build_input(structure))

        proc = run([exe, "goat.inp"], workdir, threads=1, log_name="goat.out")
        out = workdir / "goat.finalensemble.xyz"
        logs = [workdir / "goat.inp", workdir / "input.xyz", workdir / "goat.out"]
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"GOAT failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}")
        elements, confs = read_xyz(out)
        if elements != structure.elements:
            raise RuntimeError("GOAT output atom order differs from the input")
        version = re.search(r"Program Version\s+(\S+)", proc.stdout)
        return SearchResult(
            confs,
            versions={"orca": version.group(1) if version else "unknown"},
            details={"seed": seed, "start_geometry": start},
            log_files=logs,
        )
