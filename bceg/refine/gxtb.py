"""Post-search geometry optimisation with g-xTB (Grimme group), CPU only.

g-xTB is shipped as a statically linked build of ``xtb`` (6.7.1 with a modified tblite,
https://github.com/grimme-lab/g-xTB) that runs the method when ``--gxtb`` is given: there is no
separate program and no parameter file. One ``xtb`` subprocess per conformer, through the shared
driver in :mod:`bceg.refine.xtb_opt`.
"""

from __future__ import annotations

import re

from bceg.refine.xtb_opt import XTB_REFINER_DEFAULTS, XtbRefiner
from bceg.structure import Structure

# a standard xtb warns about the flag it does not know and carries on with GFN2: that is not a g-xTB result
_UNKNOWN_FLAG = re.compile(r"unknown (?:option|argument|flag)[^\n]*gxtb", re.IGNORECASE)


class GXTB(XtbRefiner):
    name = "gxtb"
    defaults = {**XTB_REFINER_DEFAULTS, "lot": "gxtb"}

    def command(self, structure: Structure, exe: str) -> list[str]:
        """``--uhf`` only for open shells: with the flag g-xTB always runs unrestricted, even for 0."""
        cmd = [exe, "start.xyz", "--gxtb", *self.opt_flags(), "--chrg", str(structure.charge)]
        if structure.uhf:
            cmd += ["--uhf", str(structure.uhf)]
        return cmd

    def check_output(self, stdout: str, exe: str) -> None:
        if _UNKNOWN_FLAG.search(stdout):
            raise RuntimeError(
                f"{exe} does not know --gxtb: it is not the g-xTB build of xtb (https://github.com/grimme-lab/g-xTB). "
                f"Put that binary first on PATH or set post_optimization.params.executable to its path.")
