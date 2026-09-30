#!/usr/bin/env python3
"""Print one value out of workflow_inputs.yaml, for use from bash:

    EWIN=$(python "$SCRIPT_DIR/get_input.py" dft.ewin_kcal_mol)

Exists because the cluster can't be assumed to have jq/yq installed, but
every node already has to have Python for the rest of this pipeline anyway.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workflow_inputs as wi

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: get_input.py <dotted.path.to.value>")
    cfg = wi.load_inputs()
    value = wi.get(cfg, sys.argv[1])
    # YAML booleans/numbers -> plain text bash can consume directly.
    if isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)
