"""Shared driver for the refiners that optimise each conformer in its own ``xtb`` subprocess.

``xtb`` (GFN-n) and ``gxtb`` (g-xTB) run the same program with different method flags, so the loop
lives here and each refiner only supplies its command line: this module is to :mod:`bceg.refine.xtb`
and :mod:`bceg.refine.gxtb` what :mod:`bceg.refine.ase_opt` is to the UMA and AIMNet2 refiners.

Each conformer is optimised in its own subfolder ``conf_<i>/`` of the work directory, because xtb
always writes ``charges``, ``wbo`` and ``xtbopt.xyz`` under fixed names. The optimised geometry is
``xtbopt.xyz``; a run that stops before convergence leaves its last step in ``xtblast.xyz``, which is
what the conformer is then taken from. Energies come from the ``TOTAL ENERGY`` line, falling back to
the first float of the xyz comment line.

There is no checkpoint: these refiners restart from the first conformer (``resumable`` stays False).
"""

from __future__ import annotations

import re
from pathlib import Path

from bceg.external import find_executable, resolve_threads, run
from bceg.layout import check_name
from bceg.refine.base import REFINER_CONSTRAINT_DEFAULTS, RefineResult, Refiner
from bceg.structure import Structure
from bceg.xtb import OPT_LEVELS, fix_input, xtb_identity
from bceg.xyz import Conformer, read_xyz, write_xyz

_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_CYCLES = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS")

#: the parameters every xtb-driven refiner takes; a subclass adds its own method keys
XTB_REFINER_DEFAULTS = {
    "level": "normal",  # xtb --opt level: crude | sloppy | loose | lax | normal | tight | vtight | extreme
    "cycles": None,  # optimiser cycles per conformer (xtb --cycles); None: xtb's own default
    "keep_unconverged": True,  # unconverged conformers are kept and flagged converged=False
    "lot": None,  # label used in raw_<i>_<lot>.xyz
    "executable": "xtb",  # a name on PATH or an absolute path
    "threads": None,  # None/0: SLURM_CPUS_PER_TASK (or all CPUs outside SLURM)
    **REFINER_CONSTRAINT_DEFAULTS,
}


class XtbRefiner(Refiner):
    """One ``xtb`` run per conformer. Subclasses define ``name``, ``defaults`` and :meth:`command`."""

    resource_keys = {"executable", "threads"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.validate()  # at construction, so a bad value fails when the config is read, not on the node

    @property
    def lot(self) -> str:
        return check_name(self.params["lot"], "level of theory")

    def command(self, structure: Structure, exe: str) -> list[str]:
        """The argument list optimising ``start.xyz`` in the conformer's folder."""
        raise NotImplementedError

    def check_output(self, stdout: str, exe: str) -> None:
        """Raise if the program is not the one this refiner asked for; nothing by default."""

    def validate(self) -> None:
        """Raise on a parameter the command line would otherwise pass to xtb unchecked.

        Called from ``__init__``: ``bceg.config.Config.from_raw`` builds every refiner while the config
        is read, so a wrong value is reported there rather than by the job that would have used it."""
        if self.params["level"] not in OPT_LEVELS:
            raise ValueError(f"{self.name}: unknown --opt level {self.params['level']!r}; choose from {OPT_LEVELS}")

    def opt_flags(self) -> list[str]:
        """``--opt <level>``. ``--cycles`` is appended by :meth:`run`, after the method's own flags."""
        return ["--opt", self.params["level"]]

    def run(self, structure: Structure, conformers: list[Conformer], workdir: Path) -> RefineResult:
        exe = find_executable(self.params["executable"])
        _, version = xtb_identity(exe)
        threads = resolve_threads(self.params["threads"])
        fixed = self.fixed_atoms(structure)
        out, n_unconverged = [], 0
        for i, conf in enumerate(conformers):
            sub = workdir / f"conf_{i}"
            sub.mkdir(parents=True, exist_ok=True)
            write_xyz(sub / "start.xyz", structure.elements, [Conformer(conf.coords)])
            cmd = self.command(structure, exe)
            if self.params["cycles"]:
                cmd += ["--cycles", str(self.params["cycles"])]
            proc = run(cmd + fix_input(sub, fixed), sub, threads=threads, log_name="xtb.out")
            self.check_output(proc.stdout, exe)
            optimized = sub / "xtbopt.xyz"
            converged = proc.returncode == 0 and optimized.exists() and "FAILED TO CONVERGE" not in proc.stdout
            geometry = optimized if optimized.exists() else sub / "xtblast.xyz"  # xtb leaves the last step there
            if not geometry.exists():
                raise RuntimeError(f"{self.name} optimisation of conformer {i} failed "
                                   f"(exit {proc.returncode}):\n{proc.stdout[-2000:]}")
            n_unconverged += not converged
            if not converged and not self.params["keep_unconverged"]:
                continue
            elements, confs = read_xyz(geometry)
            if elements != structure.elements:
                raise RuntimeError(f"{geometry}: xtb changed the atom order")
            energies = _ENERGY.findall(proc.stdout)
            energy = float(energies[-1]) if energies else confs[0].energy  # the xyz comment carries the last energy
            if energy is None:
                raise RuntimeError(f"{self.name} gave no energy for conformer {i} (see {sub / 'xtb.out'})")
            info = {"converged": converged, "source_index": i}
            cycles = _CYCLES.search(proc.stdout)
            if cycles:
                info["cycles"] = int(cycles.group(1))
            out.append(Conformer(confs[0].coords, energy, info))
        return RefineResult(
            out,
            versions={"xtb": version or "unknown"},
            details={"executable": exe, "n_input": len(conformers), "n_unconverged": n_unconverged}
            | ({"fixed_atoms": fixed} if fixed else {}),
        )
