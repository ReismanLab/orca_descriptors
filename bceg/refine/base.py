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


#: every refiner takes the conformer search's constraints (``post_optimization: constrained``): the atoms they
#: name (bceg.constraints.held_atoms) stay fixed in Cartesian coordinates during the optimisation
REFINER_CONSTRAINT_DEFAULTS = {"constraints": {}}

POOL_RESERVE_GB = 2.0  # the job's own process besides its model copy: search results, CREST's leftovers, slack


class Refiner(Configurable):
    worker_mem_gb: float | None = None  # memory of one process holding the model; None: no estimate

    def __init__(self, **params) -> None:
        super().__init__(**params)
        if self.params.get("constraints"):  # RDKit only then: the pool's workers start faster without it
            from bceg import constraints as cons
            cons.validate(self.params["constraints"], f"{self.name}.constraints")

    def compared_params(self) -> dict:
        """No ``constraints`` key when there are none, so the hash of an unconstrained refinement is unchanged."""
        out = super().compared_params()
        if "constraints" in out and not out["constraints"]:
            del out["constraints"]
        return out

    def fixed_atoms(self, structure: Structure) -> list[int]:
        """The atoms held fixed (0-based), none without constraints."""
        spec = self.params.get("constraints")
        if not spec:
            return []
        from bceg import constraints as cons
        return cons.held_atoms(spec, structure.mol)

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
