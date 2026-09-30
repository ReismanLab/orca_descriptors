"""Conformer ensemble searchers. Register new ones in ``SEARCHERS``."""

from bceg.search.base import SEED_RULE, SearchResult, Searcher
from bceg.search.crest import Crest
from bceg.search.goat import Goat
from bceg.search.loqi import LoQI
from bceg.search.molassembler import Molassembler
from bceg.search.rdkit_etkdg import RDKitETKDG

SEARCHERS: dict[str, type[Searcher]] = {cls.name: cls for cls in (Crest, Goat, Molassembler, RDKitETKDG, LoQI)}

__all__ = ["SEARCHERS", "SEED_RULE", "SearchResult", "Searcher"]
