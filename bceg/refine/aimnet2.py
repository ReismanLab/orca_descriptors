"""Post-search geometry optimisation with AIMNet2 (the ``aimnet`` package) through ASE."""

from __future__ import annotations

from pathlib import Path

from bceg.external import resolve_threads
from bceg.layout import check_name
from bceg.refine.ase_opt import optimize_conformers, package_version
from bceg.refine.base import RefineResult, Refiner
from bceg.refine.uma import load_env_runtime
from bceg.structure import Structure
from bceg.xyz import Conformer

_NETWORK_MODULES = ("requests", "urllib", "urllib3", "http", "ssl", "socket")


def _is_download_failure(exc: BaseException) -> bool:
    """True when ``exc`` or one of its causes comes out of the network stack."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if (type(exc).__module__ or "").split(".")[0] in _NETWORK_MODULES or isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class AIMNet2(Refiner):
    name = "aimnet2"
    defaults = {
        "model": "aimnet2",  # registry alias: aimnet2 (wB97M-D3), aimnet2-2025, aimnet2-nse (open shell), aimnet2-pd, aimnet2-rxn
        "optimizer": "LBFGS",  # LBFGS | BFGS | FIRE
        "fmax": 0.05,  # eV/Angstrom
        "max_steps": 500,
        "keep_unconverged": True,  # unconverged conformers are kept and flagged converged=False
        "lot": None,  # label override; defaults to the model name
        "device": "cpu",  # "cuda" sends the step (and, chained, its whole job) to the gpu profile
        "workers": None,  # None/0: one process per CPU of the allocation on cpu (at most 8 outside SLURM), 1 on cuda
        "threads": None,  # None/0: torch threads per worker, the allocation's CPUs shared between the workers
    }
    resource_keys = {"device", "workers", "threads"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._base = None

    @property
    def lot(self) -> str:
        return check_name(self.params["lot"] or self.params["model"], "level of theory")

    def resource_class(self) -> str:
        return "cpu" if self.params["device"] == "cpu" else "gpu"

    def base_calculator(self):
        """The loaded model (``aimnet.calculators.AIMNet2Calculator``), created once per refiner."""
        if self._base is None:
            load_env_runtime()  # the env's libstdc++ before torch's
            import torch

            # torch sizes its pool from the node's cores, not from the job's cgroup (see UMA.calculator)
            torch.set_num_threads(resolve_threads(self.params["threads"]))
            from aimnet.calculators import AIMNet2Calculator

            try:
                self._base = AIMNet2Calculator(self.params["model"], device=self.params["device"])
            except Exception as exc:
                if _is_download_failure(exc):
                    raise RuntimeError(
                        f"AIMNet2 weights for {self.params['model']!r} are not in the cache and the node could not "
                        f"download them ({type(exc).__name__}). Compute nodes have no internet access: fetch them once "
                        f"on a login node into a shared cache, then point the jobs at it:\n"
                        f"    export AIMNET_CACHE_DIR=<shared cache>   # e.g. /groups/ReismanLab/aimnet_cache\n"
                        f"    aimnet download {self.params['model']}\n"
                        f"and put `export AIMNET_CACHE_DIR=<shared cache>` in slurm.setup."
                    ) from exc
                raise
        return self._base

    def calculator(self, structure: Structure):
        """ASE calculator for ``structure``: charge and multiplicity are fixed per molecule."""
        from aimnet.calculators import AIMNet2ASE

        base = self.base_calculator()
        if structure.multiplicity != 1 and not getattr(base, "is_nse", True):
            raise ValueError(
                f"aimnet2: {self.params['model']!r} is a closed-shell model and would ignore multiplicity "
                f"{structure.multiplicity} ({structure.smiles}); use an open-shell model such as aimnet2-nse")
        return AIMNet2ASE(base, charge=structure.charge, mult=structure.multiplicity)

    make_calculator = calculator  # the hook of the shared ASE loop

    def run(self, structure: Structure, conformers: list[Conformer], workdir: Path) -> RefineResult:
        return optimize_conformers(
            self, structure, conformers, workdir,
            versions={"aimnet": package_version("aimnet"), "ase": package_version("ase")},
        )
