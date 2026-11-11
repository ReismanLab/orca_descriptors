from __future__ import annotations

from dataclasses import dataclass, field, replace
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

#: the ASE refiners' stop for an optimisation that no longer goes down (bceg.refine.ase_opt): off by default,
#: and then left out of the hash, so the refinements written before it existed keep theirs
STALL_DEFAULTS = {"stall_steps": None, "stall_energy": 0.001}

#: the spin state of the refinement: ``None`` (default) keeps the transformation's multiplicity (stage 1);
#: a number re-optimises at that multiplicity instead, e.g. a septet ``continue_from`` of a quintet search.
#: Left out of the hash while ``None``, so the refinements written before it existed keep theirs.
SPIN_DEFAULTS = {"multiplicity": None}

POOL_RESERVE_GB = 2.0  # the job's own process besides its model copy: search results, CREST's leftovers, slack


class Refiner(Configurable):
    worker_mem_gb: float | None = None  # memory of one process holding the model; None: no estimate
    resumable = False  # run() takes ``checkpoint=(path, key)`` and resumes from it (bceg.refine.checkpoint)

    def __init__(self, **params) -> None:
        super().__init__(**params)
        steps = self.params.get("stall_steps")
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
            raise ValueError(f"{self.name}.stall_steps must be a positive number of steps or null, got {steps!r}")
        if "stall_energy" in self.params and not self.params["stall_energy"] > 0:
            raise ValueError(f"{self.name}.stall_energy must be > 0 eV, got {self.params['stall_energy']!r}")
        mult = self.params.get("multiplicity")
        if mult is not None and (isinstance(mult, bool) or not isinstance(mult, int) or mult < 1):
            raise ValueError(f"{self.name}.multiplicity must be a positive integer or null, got {mult!r}")
        if self.params.get("constraints"):  # RDKit only then: the pool's workers start faster without it
            from bceg import constraints as cons
            cons.validate(self.params["constraints"], f"{self.name}.constraints")

    def compared_params(self) -> dict:
        """No ``constraints`` key when there are none, and no ``stall_*`` keys while ``stall_steps`` is off, so
        the hash of a refinement that does not use them is unchanged."""
        out = super().compared_params()
        if "constraints" in out and not out["constraints"]:
            del out["constraints"]
        if out.get("stall_steps") is None:
            out.pop("stall_steps", None)
            out.pop("stall_energy", None)
        if "multiplicity" in out and out["multiplicity"] is None:
            del out["multiplicity"]
        return out

    def for_structure(self, structure: Structure) -> Structure:
        """``structure`` at this refinement's spin state: unchanged without ``multiplicity``, else a copy with
        that multiplicity (checked against the electron count, as for the transformation's)."""
        mult = self.params.get("multiplicity")
        if mult is None or mult == structure.multiplicity:
            return structure
        return replace(structure, multiplicity=mult)

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
