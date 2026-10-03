"""Every parameter the code accepts is documented in its stage file under docs/stages/.

A new key in a searcher's or refiner's ``defaults``, in ``START_DEFAULTS`` or in one of the config key
sets must get a line in the stage document, or this fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bceg import constraints
from bceg.refine import REFINERS
from bceg.refine.uma import UMA
from bceg.search import SEARCHERS
from bceg.search.base import START_DEFAULTS

from desc_utils.submit import config, template

DOCS = Path(__file__).resolve().parents[1] / "docs" / "stages"
README = Path(__file__).resolve().parents[1] / "README.md"


def _tokens(name: str) -> set[str]:
    return set(re.findall(r"`([^`\s]+)`", (DOCS / name).read_text()))


START_KEYS = set(START_DEFAULTS)
# the bceg test-suite registers stand-in searchers and refiners (tests/bceg/conftest.py); only shipped ones are documented
SHIPPED = {name: cls for name, cls in SEARCHERS.items() if not cls.__module__.startswith("tests")}
SHIPPED_REFINERS = {name: cls for name, cls in REFINERS.items() if not cls.__module__.startswith("tests")}
CASES = [
    ("1_transformation.md", config.TRANSFORMATION_KEYS | {"none", "ni_x2.py", "x", "not_applicable"}),
    ("2_pre_refinement.md", config.PRE_REFINEMENT_KEYS | set(config.XTB_PARAMS) | START_KEYS
     | set(config.PRE_REFINEMENT_METHODS) | (set(UMA.defaults) - {"lot", "keep_unconverged", "workers"})),
    ("3_conformer_search.md", config.SEARCH_KEYS | set(SHIPPED) | constraints.TOP_KEYS | constraints.ENTRY_KEYS
     | {k for s in SHIPPED.values() for k in s.defaults if k not in START_KEYS}),
    ("4_post_optimization.md", config.POST_OPT_KEYS | set(SHIPPED_REFINERS) | {"null"}
     | {k for r in SHIPPED_REFINERS.values() for k in r.defaults}),
    ("5_pruning.md", config.PRUNING_KEYS | {"prepared", "skipped_energy_filter", "skipped_no_energy", "null"}),
    ("6_dft.md", template.ALL_KEYS | {"orca", "dft.calc_param_template", "dft.level_of_theory", "slurm.orca",
                                      "slurm.orca_cmd", "condense", "optimization", "frequency", "singlepoint",
                                      "nmr", "oxidation", "reduction"}),
]


@pytest.mark.parametrize("doc, keys", CASES, ids=[c[0] for c in CASES])
def test_stage_doc_names_every_parameter(doc, keys):
    missing = sorted(k for k in keys if k not in _tokens(doc))
    assert not missing, f"{doc} does not mention: {missing}"


def test_readme_links_every_stage_document():
    text = README.read_text()
    for doc in sorted(p.name for p in DOCS.glob("*.md")):
        assert f"docs/stages/{doc}" in text, f"README does not link docs/stages/{doc}"
    for stage in config.STAGES:
        assert f"`{stage}`" in text


def test_compound_steps_documented_match_the_template():
    steps = template.compound_steps({**{k: True for k in template.BOOL_KEYS}, **{k: "" for k in template.STRING_KEYS}}, 0, 1)
    tokens = _tokens("6_dft.md")
    assert set(steps) <= tokens
