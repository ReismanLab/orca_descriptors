"""Post-search geometry optimisation with UMA (fairchem) through ASE."""

from __future__ import annotations

from pathlib import Path

from bceg.external import resolve_threads
from bceg.layout import check_name
from bceg.refine.ase_opt import optimize_conformers, package_version
from bceg.refine.base import RefineResult, Refiner
from bceg.structure import Structure
from bceg.xyz import Conformer


def load_env_runtime() -> None:
    """Load the conda environment's C++ runtime before any pip wheel does.

    pip-installed torch links ``libstdc++.so.6`` without a conda RPATH, so imported first it binds the
    process to the system copy (EL9: too old for the env's ICU, ``CXXABI_1.3.15`` missing) and every later
    conda library fails to import. Importing ``sqlite3`` (conda-built, RPATH into ``$CONDA_PREFIX/lib``)
    first puts the env's libstdc++ in the process; torch then reuses it. Harmless outside conda.
    """
    import sqlite3  # noqa: F401


def weights_cache_dir() -> str | None:
    """Where the UMA checkpoints live: FAIRCHEM_CACHE_DIR, else HF_HUB_CACHE, else fairchem's default
    (~/.cache/fairchem). One shared, offline cache then serves every user's jobs."""
    import os
    return os.environ.get("FAIRCHEM_CACHE_DIR") or os.environ.get("HF_HUB_CACHE") or None


def point_fairchem_cache(cache_dir: str | None) -> None:
    """Make fairchem download / look up its checkpoints in ``cache_dir``.

    fairchem ignores HF_HUB_CACHE: its ``CACHE_DIR`` (``~/.cache/fairchem``) is read from the
    ``FAIRCHEM_CACHE_DIR`` environment variable when ``fairchem.core._config`` is imported, and
    ``get_predict_unit`` does not forward its own ``cache_dir`` argument to the checkpoint download.
    So the variable is set here, before fairchem is imported (the refiner imports it lazily); when
    fairchem is already in the process the module constants are rebound as a fallback.
    """
    if not cache_dir:
        return
    import os
    import sys

    os.environ["FAIRCHEM_CACHE_DIR"] = cache_dir
    if "fairchem.core._config" in sys.modules:
        sys.modules["fairchem.core._config"].CACHE_DIR = cache_dir
    if "fairchem.core.calculate.pretrained_mlip" in sys.modules:
        sys.modules["fairchem.core.calculate.pretrained_mlip"].CACHE_DIR = cache_dir


class UMA(Refiner):
    name = "uma"
    defaults = {
        "model": "uma-s-1p2",
        "task": "omol",
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
    # uma-s-1p2 on CPU, measured on the cluster: ~2.0 GB per process up to ~100 atoms, then roughly
    # +15 kB x atoms^2 (+0.7 GB at 200 atoms, +1.6 GB at 300); 3.6 GB holds molecules up to ~300 atoms.
    worker_mem_gb = 3.6

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._calc = None

    @property
    def lot(self) -> str:
        return check_name(self.params["lot"] or self.params["model"], "level of theory")

    def resource_class(self) -> str:
        return "cpu" if self.params["device"] == "cpu" else "gpu"

    def calculator(self):
        if self._calc is None:
            load_env_runtime()                      # the env's libstdc++ before torch's
            cache_dir = weights_cache_dir()
            point_fairchem_cache(cache_dir)         # FAIRCHEM_CACHE_DIR before fairchem is imported
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip

            # torch sizes its thread pool from the node's core count, not from the cgroup SLURM put us
            # in, so an 8-core allocation on a 64-core node would oversubscribe eightfold. This runs
            # in-process, so run()'s OMP_NUM_THREADS never applies to it.
            torch.set_num_threads(resolve_threads(self.params["threads"]))
            try:
                kwargs = {"cache_dir": cache_dir} if cache_dir else {}
                predictor = pretrained_mlip.get_predict_unit(self.params["model"], device=self.params["device"], **kwargs)
            except Exception as exc:  # huggingface_hub raises its own hierarchy; the cause is in the message
                name = type(exc).__name__
                if "huggingface" in repr(type(exc)).lower() or "hf_hub" in str(exc).lower() or "Hub" in str(exc):
                    raise RuntimeError(
                        f"UMA weights for {self.params['model']!r} are not in the Hugging Face cache and the node "
                        f"could not download them ({name}). Compute nodes have no usable internet access: fetch the "
                        f"weights once on a login node into the same cache (now: {cache_dir or 'fairchem default ~/.cache/fairchem'}), "
                        f"then run the jobs offline:\n"
                        f"    export FAIRCHEM_CACHE_DIR=<shared cache>   # e.g. /groups/ReismanLab/hf_cache/hub\n"
                        f"    hf auth login                         # once per user, the weights are gated\n"
                        f"    orcadesc fetch-weights --model {self.params['model']}\n"
                        f"and put `export FAIRCHEM_CACHE_DIR=<shared cache>` and `export HF_HUB_OFFLINE=1` in slurm.setup."
                    ) from exc
                raise
            self._calc = FAIRChemCalculator(predictor, task_name=self.params["task"])
        return self._calc

    def make_calculator(self, structure: Structure):
        """The hook of the shared ASE loop (charge and spin travel in ``atoms.info`` with UMA)."""
        return self.calculator()

    def run(self, structure: Structure, conformers: list[Conformer], workdir: Path) -> RefineResult:
        return optimize_conformers(
            self, structure, conformers, workdir,
            versions={"fairchem-core": package_version("fairchem-core"), "ase": package_version("ase")},
        )
