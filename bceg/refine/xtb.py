"""Post-search geometry optimisation with GFN-n xtb (GFN2-xTB and friends), CPU only.

The cheap counterpart of the UMA and g-xTB refiners: one ``xtb --opt`` run per conformer, through the
shared driver in :mod:`bceg.refine.xtb_opt`. Its purpose is the first half of a cascade -- the
searchers that embed rather than optimise (``rdkit_etkdg``, ``loqi``, ``molassembler``) hand on
geometries that GFN2-xTB cleans up for a fraction of the cost of the better method, which then
continues from that lot (``post_optimization.continue_from``, docs/stages/4_post_optimization.md).

This is the same program and the same method as ``pre_refinement.method: xtb`` (stage 2), which
pre-optimises the one start geometry a search begins from; here every conformer the search produced is
optimised instead.
"""

from __future__ import annotations

from bceg.layout import check_name
from bceg.refine.xtb_opt import XTB_REFINER_DEFAULTS, XtbRefiner
from bceg.structure import Structure
from bceg.xtb import METHOD_FLAGS


class XTB(XtbRefiner):
    name = "xtb"
    defaults = {
        **XTB_REFINER_DEFAULTS,
        "gfn": "gfn2",  # gfn2 | gfn1 | gfn0 | gfnff (bceg.xtb.METHOD_FLAGS)
    }

    @property
    def lot(self) -> str:
        """The method name unless ``lot`` says otherwise, so two GFN levels never collide."""
        return check_name(self.params["lot"] or self.params["gfn"], "level of theory")

    def validate(self) -> None:
        super().validate()
        if self.params["gfn"] not in METHOD_FLAGS:
            raise ValueError(f"xtb: unknown method {self.params['gfn']!r}; choose from {sorted(METHOD_FLAGS)}")

    def command(self, structure: Structure, exe: str) -> list[str]:
        """``--uhf`` is always passed: unlike g-xTB, xtb reads it for closed shells too (0 is a no-op)."""
        return [exe, "start.xyz", *METHOD_FLAGS[self.params["gfn"]], *self.opt_flags(),
                "--chrg", str(structure.charge), "--uhf", str(structure.uhf)]
