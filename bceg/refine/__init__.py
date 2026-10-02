"""Post-search refiners. Register new ones in ``REFINERS``."""

from bceg.refine.aimnet2 import AIMNet2
from bceg.refine.base import RefineResult, Refiner
from bceg.refine.gxtb import GXTB
from bceg.refine.uma import UMA

REFINERS: dict[str, type[Refiner]] = {cls.name: cls for cls in (UMA, GXTB, AIMNet2)}

__all__ = ["REFINERS", "RefineResult", "Refiner"]
