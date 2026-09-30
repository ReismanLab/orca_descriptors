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
# Mappings: extra_basis = {element: basis}, rendered as a %basis block in the
# optimization and frequency steps. Optional in the YAML (default: none).
MAP_KEYS = {"extra_basis"}
ALL_LOT_KEYS = BOOL_KEYS | STRING_KEYS | MAP_KEYS
REQUIRED_LOT_KEYS = BOOL_KEYS | STRING_KEYS


def extra_basis_of(lot):
    """The extra_basis mapping of a level_of_theory block, validated ({} if absent)."""
    eb = lot.get("extra_basis") or {}
    if not isinstance(eb, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in eb.items()):
        raise TypeError("dft.level_of_theory.extra_basis must be a mapping element -> basis, e.g. {Ni: def2-tzvp}")
    return dict(eb)


def extra_basis_block(eb):
    """ORCA %basis block for the overrides, or "" when there are none."""
    if not eb:
        return ""
    lines = ["", "    %basis"] + [f'        NewGTO {el} "{basis}" end' for el, basis in eb.items()] + ["    end"]
    return "\n".join(lines)


def extra_basis_label(eb):
    """Flat text for the metadata / consistency checks: "Ni def2-tzvp; Pd def2-tzvp", "" when none."""
    return "; ".join(f"{el} {basis}" for el, basis in eb.items())


def compound_steps(cfg):
    """Level of theory of every compound step, as it will actually run, keyed
    by the step's %base name. Written to metadata.json under dft.compound_steps
    so the record says per job which method, solvent, basis overrides, charge
    and multiplicity were used, not just the flat list of YAML values.
    (Keywords fixed in the template, TightSCF, HIRSHFELD CHELPG, %elprop, are
    covered by dft.calc_param_md5.)"""
    lot = wi.get(cfg, "dft.level_of_theory")
    charge = wi.get(cfg, "crest.charge")
    mult = wi.get(cfg, "crest.spin_multiplicity")
    eb = extra_basis_label(extra_basis_of(lot))
    sp = {"method": lot["svp_method"], "solvent": lot["property_solvent"], "population_analysis": lot["nbo_run"]}
    return {
        "conf_search": {"runs": bool(lot["conf_search"]), "method": lot["goat_method"], "solvent": "",
                        "charge": charge, "multiplicity": mult},
        "optimization": {"runs": bool(lot["opt_geom"]), "method": lot["opt_method"], "solvent": lot["opt_solvent"],
                         "extra_basis": eb, "charge": charge, "multiplicity": mult},
        "frequency": {"runs": bool(lot["freq_calc"]), "method": lot["opt_method"], "solvent": lot["opt_solvent"],
                      "extra_basis": eb, "charge": charge, "multiplicity": mult},
        "singlepoint": {"runs": True, **sp, "charge": charge, "multiplicity": mult},
        "nmr": {"runs": bool(lot["nmr_calc"]), "method": lot["nmr_method"], "solvent": lot["property_solvent"],
                "charge": charge, "multiplicity": mult},
        "oxidation": {"runs": bool(lot["vertical_redox"]), **sp, "charge": charge + 1, "multiplicity": mult + 1},
        "reduction": {"runs": bool(lot["vertical_redox"]), **sp, "charge": charge - 1, "multiplicity": mult + 1},
    }


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
    missing = REQUIRED_LOT_KEYS - lot.keys()
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
    eb = extra_basis_of(lot)
    substitutions["extra_basis_block"] = extra_basis_block(eb)

    with open(template_path) as f:
        template = Template(f.read())
    # substitute() (not safe_substitute()) so an unrendered $placeholder left
    # over from a template edit fails loudly instead of being written
    # verbatim into an ORCA input file, where it would just cause a cryptic
    # ORCA parse error three steps later.
    rendered = template.substitute(substitutions)

    recorded = {k: lot[k] for k in REQUIRED_LOT_KEYS}
    recorded["extra_basis"] = eb
    return rendered, recorded
