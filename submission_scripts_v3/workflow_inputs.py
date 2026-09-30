"""Shared loader for the user-editable workflow_inputs.yaml file.

Every "chosen" pipeline parameter -- CREST energy window, UMA model, DFT
energy cutoff, SLURM resources, conda env names -- lives in one YAML file the
user edits, instead of being hardcoded across crest.py, uma_opt_confs.py,
launch_orca_all.py, and launch_workflow.sh.

Values the pipeline DISCOVERS rather than the user CHOOSES (rdkit_version,
crest_version, orca_module, calc_param_md5, ...) are not read from here --
they only ever live in <inchikey>/metadata.json, written by metadata_utils.py.

Usage from a Python stage script:

    import workflow_inputs as wi
    cfg = wi.load_inputs()
    ewin = wi.get(cfg, "crest.ewin_kcal_mol")

Usage from bash (launch_workflow.sh, submit_all_smiles.py's sbatch calls):
see get_input.py, a one-value-per-call CLI wrapper around this module.
"""

import os
import sys

try:
    import yaml
except ImportError:
    sys.exit(
        "ERROR: PyYAML is required to read workflow_inputs.yaml "
        "(pip install pyyaml --break-system-packages, or add pyyaml to "
        "the conda env this pipeline runs in)."
    )

DEFAULT_FILENAME = "workflow_inputs.yaml"
_SENTINEL = object()


def default_path():
    """$WORKFLOW_INPUTS if set, else ./workflow_inputs.yaml (cwd at call time)."""
    return os.environ.get("WORKFLOW_INPUTS", DEFAULT_FILENAME)


def load_inputs(path=None):
    """Parse workflow_inputs.yaml into a nested dict."""
    path = path or default_path()
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: inputs file not found: {path}\n"
            f"Set $WORKFLOW_INPUTS to point at your copy, or place "
            f"workflow_inputs.yaml in the directory the pipeline is run from."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def get(cfg, dotted_path, default=_SENTINEL):
    """Dotted-path lookup, e.g. get(cfg, 'crest.ewin_kcal_mol').

    Raises KeyError (with the full path for context) if the key is missing
    and no default was given -- a silently-missing pipeline parameter is
    worse than a loud one.
    """
    node = cfg
    parts = dotted_path.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            if default is not _SENTINEL:
                return default
            raise KeyError(
                f"'{dotted_path}' not found in inputs file "
                f"(missing at '{'.'.join(parts[:i + 1])}')"
            )
        node = node[part]
    return node
