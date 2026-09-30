"""Post-search refiners. Register new ones in ``REFINERS``."""

from bceg.refine.base import RefineResult, Refiner
from bceg.refine.uma import UMA

REFINERS: dict[str, type[Refiner]] = {cls.name: cls for cls in (UMA,)}

__all__ = ["REFINERS", "RefineResult", "Refiner"]
