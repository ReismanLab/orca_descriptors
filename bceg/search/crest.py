"""CREST (>= 3.1) driven by a TOML input.

Default: iMTD-GC at GFN2-xTB. ``refine: sp`` / ``refine: opt`` adds an inline
refinement level (CREST composite workflow); its default level of theory is
UMA through CREST's ``mlip`` calculator (needs ``fmlip-relay-server`` on PATH):
https://crest-lab.github.io/crest-docs/page/examples/composite.html
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bceg.external import find_executable, program_version, resolve_threads, run
from bceg.search.base import CONSTRAINT_DEFAULTS, START_DEFAULTS, START_RESOURCE_KEYS, SearchResult, Searcher
from bceg.structure import Structure
from bceg.xyz import BOHR_TO_ANGSTROM, Conformer, read_xyz, write_xyz

UMA_LEVEL = {"method": "mlip", "mlip_backend": "uma", "mlip_uma_task": "omol", "mlip_uma_model": "uma-s-1p2"}


class Crest(Searcher):
    name = "crest"
    defaults = {
        "runtype": "imtd-gc",
        "workhorse": "gfn2",  # method name, or a full [[calculation.level]] mapping
        "workhorse_extra": {},  # extra keys for the workhorse level
        "refine": None,  # None | "sp" | "opt" (| "add" | "freq")
        "refine_level": dict(UMA_LEVEL),  # level used for the inline refinement
        "toml_extra": {},  # deep-merged into the TOML document (e.g. {"cregen": {"ewin": 6.0}})
        **CONSTRAINT_DEFAULTS,
        **START_DEFAULTS,
        "seed_offset": 0,
        "threads": None,  # None/0: SLURM_CPUS_PER_TASK (or all CPUs outside SLURM)
        "refine_threads": None,  # threads per refinement-level instance
        "mlip_device": "cpu",  # device of the mlip levels; "cuda" sends the job to the gpu profile when an mlip level is used
        "executable": "crest",
    }
    resource_keys = {"threads", "refine_threads", "mlip_device", "executable"} | START_RESOURCE_KEYS

    def resource_class(self) -> str:
        p = self.params
        uses_mlip = (p["refine"] and p["refine_level"].get("method") == "mlip") or (
            isinstance(p["workhorse"], dict) and p["workhorse"].get("method") == "mlip"
        )
        return "gpu" if uses_mlip and p["mlip_device"] != "cpu" else "cpu"

    def build_input(self, structure: Structure, constraints: dict | None = None) -> dict[str, Any]:
        p = self.params
        electronic = {"charge": structure.charge, "uhf": structure.uhf}
        workhorse = {"method": p["workhorse"]} if isinstance(p["workhorse"], str) else dict(p["workhorse"])
        levels = [{**workhorse, **p["workhorse_extra"], **electronic}]
        if p["refine"]:
            level = {**p["refine_level"], **electronic, "refine": p["refine"]}
            if level.get("method") == "mlip":
                level["mlip_device"] = p["mlip_device"]
            if p["refine_threads"]:
                level["threads"] = p["refine_threads"]
            levels.append(level)
        if workhorse.get("method") == "mlip":
            levels[0]["mlip_device"] = p["mlip_device"]
        doc = {"input": "input.xyz", "runtype": p["runtype"], "threads": resolve_threads(p["threads"]),
               "calculation": {"level": levels}}
        if constraints:
            doc["calculation"].update(constraint_toml(constraints))
            if constraints["freeze"]:
                doc["dynamics"] = dict(FROZEN_DYNAMICS)
        return _deep_merge(doc, p["toml_extra"])

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        exe = find_executable(self.params["executable"])
        seed = self.seed(replicate)
        coords, start = self.start_geometry(structure, seed, workdir)
        constraints = self.resolve_constraints(structure, coords)
        write_xyz(workdir / "input.xyz", structure.elements, [Conformer(coords)])
        (workdir / "input.toml").write_text(to_toml(self.build_input(structure, constraints)))

        frozen = bool(constraints and constraints["freeze"])
        proc = run([exe, "--input", "input.toml", *(FROZEN_FLAGS if frozen else [])], workdir, threads=resolve_threads(self.params["threads"]), log_name="crest.out")
        out = workdir / "crest_conformers.xyz"
        logs = [workdir / "input.toml", workdir / "start_geometry.xyz", workdir / "crest.out"]
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"CREST failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}")
        elements, confs = read_xyz(out)
        if elements != structure.elements:
            raise RuntimeError("CREST output atom order differs from the input")
        return SearchResult(
            confs,
            versions={"crest": program_version([exe, "--version"]) or "unknown"},
            details={"seed": seed, "start_geometry": start} | ({"constraints": constraints} if constraints else {}),
            log_files=logs,
        )


# SHAKE holds bond lengths by moving both atoms of a bond, frozen or not (CREST 3.0.2 constrains all
# bonds but those to a metal), so frozen atoms bonded to free ones drift during the MD and the
# optimisations keep them where it left them. Off whenever atoms are frozen, with a time step the MD
# survives without it (CREST keeps its 5 fs for GFN2, which blows up); toml_extra can override both.
FROZEN_DYNAMICS = {"shake": 0, "tstep": 2.0}
# The genetic crossing (GC) mixes the Cartesian coordinates of aligned conformers, frozen atoms
# included (up to 0.02 A on a Ni first shell); the TOML input has no key for it, so a flag.
FROZEN_FLAGS = ["-nocross"]

CREST_CONSTRAINT_TYPE = {"bonds": "bond", "angles": "angle", "dihedrals": "dihedral"}


def constraint_toml(constraints: dict) -> dict[str, Any]:
    """``[calculation]`` keys for resolved constraints (bceg.constraints.resolve): ``freeze`` (exact; an
    atom list, 1-based) and one ``[[calculation.constraints]]`` table per bond, angle or dihedral. CREST
    keeps references in atomic units, so distances go in Bohr and angles in radians; without ``fc``
    CREST's default force constant applies."""
    out: dict[str, Any] = {}
    if constraints["freeze"]:
        out["freeze"] = ",".join(str(i + 1) for i in constraints["freeze"])
    tables = []
    for kind, crest_type in CREST_CONSTRAINT_TYPE.items():
        for c in constraints[kind]:
            ref = c["value"] / BOHR_TO_ANGSTROM if kind == "bonds" else math.radians(c["value"])
            table = {"type": crest_type, "atoms": [i + 1 for i in c["atoms"]], "ref": round(ref, 6)}
            if c["force_constant"] is not None:
                table["fc"] = c["force_constant"]
            tables.append(table)
    if tables:
        out["constraints"] = tables
    return out


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"Cannot write {v!r} to TOML")


def to_toml(doc: dict, prefix: str = "") -> str:
    """Minimal TOML writer: scalars, arrays, tables and arrays of tables."""
    scalars, tables, arrays = [], [], []
    for k, v in doc.items():
        if v is None:
            continue
        if isinstance(v, dict):
            tables.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            arrays.append((k, v))
        else:
            scalars.append(f"{k} = {_toml_value(v)}")
    out = "\n".join(scalars) + ("\n" if scalars else "")
    for k, v in tables:
        out += f"\n[{prefix}{k}]\n" + to_toml(v, f"{prefix}{k}.")
    for k, items in arrays:
        for item in items:
            out += f"\n[[{prefix}{k}]]\n" + to_toml(item, f"{prefix}{k}.")
    return out
