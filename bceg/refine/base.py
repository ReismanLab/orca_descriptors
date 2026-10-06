from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bceg.params import Configurable
from bceg.structure import Structure
from bceg.xyz import Conformer


@dataclass
class RefineResult:
    conformers: list[Conformer]
    versions: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


POOL_RESERVE_GB = 2.0  # the job's own process besides its model copy: search results, CREST's leftovers, slack


class Refiner(Configurable):
    worker_mem_gb: float | None = None  # memory of one process holding the model; None: no estimate

    def pool_mem_gb(self, cpus: int) -> float | None:
        """Memory a job of ``cpus`` CPUs needs for this refiner's CPU pool: one model copy per worker
        (``workers``, else one per CPU) plus one for the job's own process (it may have loaded the
        model for a pre-refinement) and ``POOL_RESERVE_GB``. None on a GPU or without an estimate."""
        if not self.worker_mem_gb or self.params.get("device", "cpu") != "cpu":
            return None
        workers = self.params.get("workers") or cpus
        return (int(workers) + 1) * self.worker_mem_gb + POOL_RESERVE_GB
    @property
    def lot(self) -> str:
        """Level-of-theory label used in ``raw_<i>_<lot>.xyz``."""
        raise NotImplementedError

    def run(self, structure: Structure, conformers: list[Conformer], workdir: Path) -> RefineResult:
        raise NotImplementedError
