"""Render ``calcParam.inp`` from the ORCA Compound template and the ``dft.level_of_theory`` block.

The template fixes the STRUCTURE of the compound job (which steps exist, how they chain); the YAML
values fix WHAT each step computes. ``string.Template`` is used on purpose: ORCA's own ``&{X}``
substitutions must be left alone, and ``substitute()`` (not ``safe_substitute``) makes an unrendered
``$placeholder`` fail here rather than three ORCA steps later.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from string import Template

# Booleans control which Compound steps run (rendered as ORCA's lowercase true/false).
BOOL_KEYS = {"conf_search", "opt_geom", "freq_calc", "nmr_calc", "vertical_redox"}
# Strings are method/basis/solvent/population-analysis choices.
STRING_KEYS = {"goat_method", "opt_method", "opt_solvent", "svp_method", "property_solvent", "nmr_method", "nbo_run"}
# extra_basis = {element: basis}, rendered as a %basis block in the optimization and frequency steps.
MAP_KEYS = {"extra_basis"}
REQUIRED_KEYS = BOOL_KEYS | STRING_KEYS
ALL_KEYS = REQUIRED_KEYS | MAP_KEYS


def validate_level_of_theory(lot: dict) -> dict:
    """Return a normalised copy of ``lot``; raise on missing or unknown keys."""
    if not isinstance(lot, dict):
        raise TypeError("dft.level_of_theory must be a mapping")
    missing = REQUIRED_KEYS - lot.keys()
    if missing:
        raise KeyError(f"dft.level_of_theory is missing: {', '.join(sorted(missing))}")
    unknown = set(lot) - ALL_KEYS
    if unknown:
        raise KeyError(f"dft.level_of_theory: unknown keys {sorted(unknown)}")
    out = {k: lot[k] for k in REQUIRED_KEYS}
    out["extra_basis"] = extra_basis_of(lot)
    return out


def extra_basis_of(lot: dict) -> dict:
    eb = lot.get("extra_basis") or {}
    if not isinstance(eb, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in eb.items()):
        raise TypeError("dft.level_of_theory.extra_basis must be a mapping element -> basis, e.g. {Ni: def2-tzvp}")
    return dict(eb)


def extra_basis_block(eb: dict) -> str:
    """ORCA %basis block for the overrides, "" when there are none."""
    if not eb:
        return ""
    lines = ["", "    %basis"] + [f'        NewGTO {el} "{basis}" end' for el, basis in eb.items()] + ["    end"]
    return "\n".join(lines)


def extra_basis_label(eb: dict) -> str:
    return "; ".join(f"{el} {basis}" for el, basis in eb.items())


def render(template_path: str | Path, lot: dict, charge: int, multiplicity: int,
           nprocs: int, maxcore_mb: int) -> str:
    """The text of ``calcParam.inp`` for one molecule."""
    lot = validate_level_of_theory(lot)
    subs: dict[str, object] = {k: ("true" if lot[k] else "false") for k in BOOL_KEYS}
    subs.update({k: lot[k] for k in STRING_KEYS})
    subs["extra_basis_block"] = extra_basis_block(lot["extra_basis"])
    subs["charge"] = int(charge)
    subs["multiplicity"] = int(multiplicity)
    subs["nprocs"] = int(nprocs)
    subs["maxcore"] = int(maxcore_mb)
    return Template(Path(template_path).read_text()).substitute(subs)


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def compound_steps(lot: dict, charge: int, multiplicity: int) -> dict:
    """Level of theory of every compound step as it will actually run, keyed by the step's %base name.

    Recorded in metadata.json so the record says per step which method, solvent, basis overrides,
    charge and multiplicity were used. Keywords fixed in the template (TightSCF, HIRSHFELD CHELPG,
    %elprop, ...) are covered by the MD5 of the rendered input.
    """
    lot = validate_level_of_theory(lot)
    eb = extra_basis_label(lot["extra_basis"])
    base = {"charge": charge, "multiplicity": multiplicity}
    sp = {"method": lot["svp_method"], "solvent": lot["property_solvent"], "population_analysis": lot["nbo_run"]}
    return {
        "conf_search": {"runs": bool(lot["conf_search"]), "method": lot["goat_method"], "solvent": "", **base},
        "optimization": {"runs": bool(lot["opt_geom"]), "method": lot["opt_method"], "solvent": lot["opt_solvent"],
                         "extra_basis": eb, **base},
        "frequency": {"runs": bool(lot["freq_calc"]), "method": lot["opt_method"], "solvent": lot["opt_solvent"],
                      "extra_basis": eb, **base},
        "singlepoint": {"runs": True, **sp, **base},
        "nmr": {"runs": bool(lot["nmr_calc"]), "method": lot["nmr_method"], "solvent": lot["property_solvent"], **base},
        "oxidation": {"runs": bool(lot["vertical_redox"]), **sp, "charge": charge + 1, "multiplicity": multiplicity + 1},
        "reduction": {"runs": bool(lot["vertical_redox"]), **sp, "charge": charge - 1, "multiplicity": multiplicity + 1},
    }


def maxcore_mb(mem_per_cpu: str | None, mem: str | None = None, ntasks: int = 1, fraction: float = 0.75) -> int:
    """ORCA %maxcore (MB per process) from the SLURM allocation: 75 % of the memory per task.

    ``mem_per_cpu`` wins; else ``mem`` divided by ``ntasks``. Units K/M/G/T, default MB (SLURM's rule).
    """
    spec = mem_per_cpu
    divide = 1
    if spec is None:
        if mem is None:
            return 2000
        spec, divide = mem, max(int(ntasks), 1)
    spec = str(spec).strip().upper()
    factor = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}.get(spec[-1], None)
    number = float(spec[:-1]) if factor is not None else float(spec)
    mb = number * (factor if factor is not None else 1) / divide
    return max(int(mb * fraction), 256)
