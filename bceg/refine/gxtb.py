"""Post-search geometry optimisation with g-xTB (Grimme group), CPU only.

g-xTB is shipped as a statically linked build of ``xtb`` (6.7.1 with a modified tblite,
https://github.com/grimme-lab/g-xTB) that runs the method when ``--gxtb`` is given: there is no
separate program and no parameter file. The driver follows ``bceg/xtb.py``: one ``xtb`` subprocess
per conformer, each in its own subfolder of the work directory, energies taken from the output.
"""

from __future__ import annotations

import re
from pathlib import Path

from bceg.external import find_executable, resolve_threads, run
from bceg.layout import check_name
from bceg.refine.base import RefineResult, Refiner
from bceg.structure import Structure
from bceg.xtb import OPT_LEVELS, xtb_identity
from bceg.xyz import Conformer, read_xyz, write_xyz

_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_CYCLES = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS")
# a standard xtb warns about the flag it does not know and carries on with GFN2: that is not a g-xTB result
_UNKNOWN_FLAG = re.compile(r"unknown (?:option|argument|flag)[^\n]*gxtb", re.IGNORECASE)


class GXTB(Refiner):
    name = "gxtb"
    defaults = {
        "level": "normal",  # xtb --opt level: crude | sloppy | loose | lax | normal | tight | vtight | extreme
        "cycles": None,  # optimiser cycles per conformer (xtb --cycles); None: xtb's own default
        "keep_unconverged": True,  # unconverged conformers are kept and flagged converged=False
        "lot": "gxtb",  # label used in raw_<i>_<lot>.xyz
        "executable": "xtb",  # the g-xTB build of xtb: a name on PATH or an absolute path
        "threads": None,  # None/0: SLURM_CPUS_PER_TASK (or all CPUs outside SLURM)
    }
    resource_keys = {"executable", "threads"}

    @property
    def lot(self) -> str:
        return check_name(self.params["lot"], "level of theory")

    def command(self, structure: Structure, exe: str) -> list[str]:
        """``--uhf`` only for open shells: with the flag g-xTB always runs unrestricted, even for 0."""
        cmd = [exe, "start.xyz", "--gxtb", "--opt", self.params["level"], "--chrg", str(structure.charge)]
        if structure.uhf:
            cmd += ["--uhf", str(structure.uhf)]
        if self.params["cycles"]:
            cmd += ["--cycles", str(self.params["cycles"])]
        return cmd

    def run(self, structure: Structure, conformers: list[Conformer], workdir: Path) -> RefineResult:
        if self.params["level"] not in OPT_LEVELS:
            raise ValueError(f"gxtb: unknown --opt level {self.params['level']!r}; choose from {OPT_LEVELS}")
        exe = find_executable(self.params["executable"])
        _, version = xtb_identity(exe)
        threads = resolve_threads(self.params["threads"])
        out, n_unconverged = [], 0
        for i, conf in enumerate(conformers):
            sub = workdir / f"conf_{i}"
            sub.mkdir(parents=True, exist_ok=True)
            write_xyz(sub / "start.xyz", structure.elements, [Conformer(conf.coords)])
            proc = run(self.command(structure, exe), sub, threads=threads, log_name="xtb.out")
            if _UNKNOWN_FLAG.search(proc.stdout):
                raise RuntimeError(
                    f"{exe} does not know --gxtb: it is not the g-xTB build of xtb (https://github.com/grimme-lab/g-xTB). "
                    f"Put that binary first on PATH or set post_optimization.params.executable to its path.")
            optimized = sub / "xtbopt.xyz"
            converged = proc.returncode == 0 and optimized.exists() and "FAILED TO CONVERGE" not in proc.stdout
            geometry = optimized if optimized.exists() else sub / "xtblast.xyz"  # xtb leaves the last step there
            if not geometry.exists():
                raise RuntimeError(f"g-xTB optimisation of conformer {i} failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}")
            n_unconverged += not converged
            if not converged and not self.params["keep_unconverged"]:
                continue
            elements, confs = read_xyz(geometry)
            if elements != structure.elements:
                raise RuntimeError(f"{geometry}: xtb changed the atom order")
            energies = _ENERGY.findall(proc.stdout)
            energy = float(energies[-1]) if energies else confs[0].energy  # the xyz comment carries the last energy
            if energy is None:
                raise RuntimeError(f"g-xTB gave no energy for conformer {i} (see {sub / 'xtb.out'})")
            info = {"converged": converged, "source_index": i}
            cycles = _CYCLES.search(proc.stdout)
            if cycles:
                info["cycles"] = int(cycles.group(1))
            out.append(Conformer(confs[0].coords, energy, info))
        return RefineResult(
            out,
            versions={"xtb": version or "unknown"},
            details={"executable": exe, "n_input": len(conformers), "n_unconverged": n_unconverged},
        )
