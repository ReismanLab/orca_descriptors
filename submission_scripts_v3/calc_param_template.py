"""Render calcParam.inp from calcParam.inp.tmpl using workflow_inputs.yaml's
dft.level_of_theory block, plus crest.charge / crest.spin_multiplicity so
CREST and DFT can never silently disagree on what species is being computed.

Uses string.Template rather than a general template engine (Jinja2, etc.):
this file has exactly one thing that needs substituting -- the
`Variable X = ...;` declarations -- and ORCA's own `&{X}` runtime-substitution
syntax must be left completely alone, so a full templating language would be
solving a problem this doesn't have.
"""
import os
from string import Template

import workflow_inputs as wi

# Booleans control which Compound steps run at all (rendered as ORCA's
# lowercase true/false literals, not Python's True/False).
BOOL_KEYS = {"conf_search", "opt_geom", "freq_calc", "nmr_calc", "vertical_redox"}
# Strings are method/basis/solvent/population-analysis choices.
STRING_KEYS = {"goat_method", "opt_method", "opt_solvent", "svp_method",
               "property_solvent", "nmr_method", "nbo_run"}
ALL_LOT_KEYS = BOOL_KEYS | STRING_KEYS


def render(cfg, template_path=None):
    """Return (rendered_text, level_of_theory_dict_for_metadata).

    The returned dict is exactly what should go into metadata.json's
    dft.level_of_theory -- the values that produced this render, not a
    re-parse of the rendered text, so there's no risk of the two drifting.
    """
    template_path = template_path or wi.get(cfg, "dft.calc_param_template")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"ORCA Compound template not found: {template_path}")

    lot = wi.get(cfg, "dft.level_of_theory")
    missing = ALL_LOT_KEYS - lot.keys()
    if missing:
        raise KeyError(
            f"workflow_inputs.yaml's dft.level_of_theory is missing: "
            f"{', '.join(sorted(missing))}"
        )

    substitutions = {}
    for key in BOOL_KEYS:
        substitutions[key] = "true" if lot[key] else "false"
    for key in STRING_KEYS:
        substitutions[key] = lot[key]
    substitutions["charge"] = wi.get(cfg, "crest.charge")
    substitutions["multiplicity"] = wi.get(cfg, "crest.spin_multiplicity")

    with open(template_path) as f:
        template = Template(f.read())
    # substitute() (not safe_substitute()) so an unrendered $placeholder left
    # over from a template edit fails loudly instead of being written
    # verbatim into an ORCA input file, where it would just cause a cryptic
    # ORCA parse error three steps later.
    rendered = template.substitute(substitutions)

    recorded = {k: lot[k] for k in ALL_LOT_KEYS}
    return rendered, recorded
