"""Post-search geometry optimisation with UMA (fairchem) through ASE."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from bceg.external import resolve_threads
from bceg.layout import check_name
from bceg.refine.base import RefineResult, Refiner
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_EV, Conformer


def load_env_runtime() -> None:
    """Load the conda environment's C++ runtime before any pip wheel does.

    pip-installed torch links ``libstdc++.so.6`` without a conda RPATH, so imported first it binds the
    process to the system copy (EL9: too old for the env's ICU, ``CXXABI_1.3.15`` missing) and every later
    conda library fails to import. Importing ``sqlite3`` (conda-built, RPATH into ``$CONDA_PREFIX/lib``)
    first puts the env's libstdc++ in the process; torch then reuses it. Harmless outside conda.
    """
    import sqlite3  # noqa: F401


def _version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


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
        "threads": None,  # None/0: SLURM_CPUS_PER_TASK (or all CPUs outside SLURM)
    }
    resource_keys = {"device", "threads"}

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
            import sqlite3  # noqa: F401  -- before torch: see load_env_runtime()
            load_env_runtime()
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip

            # torch sizes its thread pool from the node's core count, not from the cgroup SLURM put us
            # in, so an 8-core allocation on a 64-core node would oversubscribe eightfold. This runs
            # in-process, so run()'s OMP_NUM_THREADS never applies to it.
            torch.set_num_threads(resolve_threads(self.params["threads"]))
            try:
                predictor = pretrained_mlip.get_predict_unit(self.params["model"], device=self.params["device"])
            except Exception as exc:  # huggingface_hub raises its own hierarchy; the cause is in the message
                name = type(exc).__name__
                if "huggingface" in repr(type(exc)).lower() or "hf_hub" in str(exc).lower() or "Hub" in str(exc):
                    raise RuntimeError(
                        f"UMA weights for {self.params['model']!r} are not in the Hugging Face cache and the node "
                        f"could not download them ({name}). Compute nodes have no usable internet access: fetch the "
                        f"weights once on a login node with the same HF_HUB_CACHE, then run the jobs offline:\n"
                        f"    export HF_HUB_CACHE=<shared cache>   # e.g. /groups/ReismanLab/hf_cache/hub\n"
                        f"    huggingface-cli login                 # once per user, the weights are gated\n"
                        f"    python -c \"from fairchem.core import pretrained_mlip; "
                        f"pretrained_mlip.get_predict_unit('{self.params['model']}', device='cpu')\"\n"
                        f"and put `export HF_HUB_CACHE=<shared cache>` and `export HF_HUB_OFFLINE=1` in slurm.setup."
                    ) from exc
                raise
            self._calc = FAIRChemCalculator(predictor, task_name=self.params["task"])
        return self._calc

    def run(self, structure: Structure, conformers: list[Conformer], workdir: Path) -> RefineResult:
        from ase import Atoms, optimize

        opt_cls = getattr(optimize, self.params["optimizer"])
        calc = self.calculator()
        out, n_unconverged = [], 0
        for i, conf in enumerate(conformers):
            atoms = Atoms(symbols=structure.elements, positions=conf.coords)
            atoms.info.update(charge=structure.charge, spin=structure.multiplicity)
            atoms.calc = calc
            opt = opt_cls(atoms, logfile=str(workdir / f"opt_{i}.log"))
            opt.run(fmax=self.params["fmax"], steps=self.params["max_steps"])
            fmax = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
            converged = fmax <= self.params["fmax"]
            n_unconverged += not converged
            if not converged and not self.params["keep_unconverged"]:
                continue
            energy = atoms.get_potential_energy() / HARTREE_TO_EV
            out.append(Conformer(atoms.get_positions(), energy,
                                 {"converged": converged, "steps": opt.nsteps, "fmax": round(fmax, 5), "source_index": i}))
        return RefineResult(
            out,
            versions={"fairchem-core": _version("fairchem-core"), "ase": _version("ase")},
            details={"n_input": len(conformers), "n_unconverged": n_unconverged},
        )
