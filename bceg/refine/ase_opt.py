"""The ASE optimisation loop shared by the calculator-based refiners (UMA, AIMNet2)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from bceg.refine.base import RefineResult
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_EV, Conformer


def package_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def optimize_conformers(calc, structure: Structure, conformers: list[Conformer], workdir: Path, *,
                        optimizer: str, fmax: float, max_steps: int, keep_unconverged: bool,
                        versions: dict[str, Any]) -> RefineResult:
    """Optimise every conformer with ``calc`` (an ASE calculator); energies come back in Hartree.

    ``optimizer`` names a class of ``ase.optimize`` (LBFGS, BFGS, FIRE); one ``opt_<i>.log`` per
    conformer is written to ``workdir``. A conformer whose residual force exceeds ``fmax`` after
    ``max_steps`` is kept with ``converged=False`` or dropped, after ``keep_unconverged``.
    """
    from ase import Atoms, optimize

    opt_cls = getattr(optimize, optimizer)
    out, n_unconverged = [], 0
    for i, conf in enumerate(conformers):
        atoms = Atoms(symbols=structure.elements, positions=conf.coords)
        atoms.info.update(charge=structure.charge, spin=structure.multiplicity)
        atoms.calc = calc
        opt = opt_cls(atoms, logfile=str(workdir / f"opt_{i}.log"))
        opt.run(fmax=fmax, steps=max_steps)
        residual = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
        converged = residual <= fmax
        n_unconverged += not converged
        if not converged and not keep_unconverged:
            continue
        energy = atoms.get_potential_energy() / HARTREE_TO_EV
        out.append(Conformer(atoms.get_positions(), energy,
                             {"converged": converged, "steps": opt.nsteps, "fmax": round(residual, 5), "source_index": i}))
    return RefineResult(out, versions=versions, details={"n_input": len(conformers), "n_unconverged": n_unconverged})
