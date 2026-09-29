"""Single source of truth for the results folder layout.

results/<transformation>/<InChIKey>/<method>/
    raw_<i>.xyz            ensemble search output, replicate i
    raw_<i>_<lot>.xyz      post-search optimised at level of theory <lot>
    params.json            search hyper-parameter log
    params_<lot>.json      post-optimisation hyper-parameter log
    run_<i>.json           per-replicate record (search)
    run_<i>_<lot>.json     per-replicate record (post-optimisation)
    logs/                  inputs and stdout of the programs
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def check_name(name: str, what: str) -> str:
    if not _SAFE.match(name):
        raise ValueError(f"Invalid {what} name {name!r}: use letters, digits and _ . + - only")
    return name


@dataclass(frozen=True)
class MethodDir:
    path: Path

    def raw(self, replicate: int) -> Path:
        return self.path / f"raw_{replicate}.xyz"

    def refined(self, replicate: int, lot: str) -> Path:
        return self.path / f"raw_{replicate}_{lot}.xyz"

    def params(self, lot: str | None = None) -> Path:
        return self.path / ("params.json" if lot is None else f"params_{lot}.json")

    def run_record(self, replicate: int, lot: str | None = None) -> Path:
        return self.path / (f"run_{replicate}.json" if lot is None else f"run_{replicate}_{lot}.json")

    @property
    def logs(self) -> Path:
        return self.path / "logs"


def method_dir(results_dir: str | Path, transformation: str, inchikey: str, method: str) -> MethodDir:
    return MethodDir(Path(results_dir) / check_name(transformation, "transformation") / inchikey / check_name(method, "method"))
