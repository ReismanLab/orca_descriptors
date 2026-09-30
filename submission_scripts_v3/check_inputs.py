"""Cross-check workflow_inputs.yaml against already-recorded metadata.json
files, and (when there's a human at a terminal to ask) confirm whether to
proceed anyway.

Two checks, both run for every ligand about to be (re)submitted:
  1. compute_mismatches()       -- this ligand's OWN metadata.json, if it has
                                    one, vs the current config.
  2. compute_batch_mismatches() -- every OTHER ligand's metadata.json already
                                    on disk in this batch directory, vs the
                                    current config. This is the one that
                                    matters for a brand new ligand: it has no
                                    history of its own to disagree with, but
                                    submitting it under settings that differ
                                    from the rest of the batch is exactly the
                                    inconsistency aggregate_metadata.py would
                                    otherwise only catch after the compute is
                                    already spent.

Only fields belonging to a stage that has ALREADY run are compared -- a
field's presence in metadata.json is what marks its stage as having run, so
e.g. crest.* is only checked against directories where metadata already has
a non-empty "crest" block.

Used by:
  - submit_all_smiles.py, interactively, before submitting each ligand's job
    (this is the one place in the pipeline with a real TTY attached)
  - launch_workflow.sh, as a non-interactive safety net right before the
    "build" stage
"""
import argparse
import hashlib
import os
import sys

from rdkit import Chem

import metadata_utils
import workflow_inputs as wi
import calc_param_template as cpt

_MISSING = object()
_NOT_GIVEN = object()  # distinct from None: None means "rendering failed, don't flag"

# (label, metadata dotted-path, workflow_inputs.yaml dotted-path, kind)
# kind "bool" normalizes both sides to a boolean before comparing, so legacy
# metadata that recorded a raw CLI flag string (e.g. "--noreftopo") instead
# of a literal boolean still compares correctly against today's config.
_LOT_CHECKS = [
    (f"dft.level_of_theory.{k}", f"dft.level_of_theory.{k}", f"dft.level_of_theory.{k}",
     "bool" if k in cpt.BOOL_KEYS else "raw")
    for k in sorted(cpt.ALL_LOT_KEYS)
]

SIMPLE_CHECKS = [
    ("crest.charge", "crest.charge", "crest.charge", "raw"),
    ("crest.spin_multiplicity", "crest.spin_multiplicity", "crest.spin_multiplicity", "raw"),
    ("crest.ewin_kcal_mol", "crest.ewin_kcal_mol", "crest.ewin_kcal_mol", "raw"),
    ("crest.n_conf", "crest.n_conf", "crest.n_conf", "raw"),
    ("crest.quick", "crest.quick", "crest.quick", "bool"),
    ("crest.norotmd", "crest.norotmd", "crest.norotmd", "bool"),
    ("crest.noreftopo", "crest.noreftopo", "crest.noreftopo", "bool"),
    ("uma.model_name", "uma.model_name", "uma.model_name", "raw"),
    ("uma.fmax", "uma.fmax", "uma.fmax", "raw"),
    ("uma.task_name", "uma.task_name", "uma.task_name", "raw"),
] + _LOT_CHECKS


def _get(d, dotted_path):
    node = d
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _as_bool_ish(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "0")
    return bool(v)


def _raw_equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return round(float(a), 6) == round(float(b), 6)
        except (TypeError, ValueError):
            return a == b
    return a == b


def _render_expected_md5(cfg):
    """MD5 of what calcParam.inp.tmpl + today's config would render to, or
    None if it can't be rendered right now (missing template, bad YAML,
    etc.) -- callers treat None as "skip this check" rather than a mismatch.
    Split out so submit_all_smiles.py can render once per run and hand the
    same hash to every ligand's check instead of re-rendering per directory.
    """
    template_path = wi.get(cfg, "dft.calc_param_template", default=None)
    if not template_path or not os.path.exists(template_path):
        return None
    try:
        rendered, _ = cpt.render(cfg, template_path=template_path)
        return hashlib.md5(rendered.encode()).hexdigest()
    except Exception:
        return None


def compute_mismatches(cfg, metadata, _expected_calc_param_md5=_NOT_GIVEN):
    """Return a list of (label, recorded_value, expected_value) tuples."""
    if not metadata:
        return []

    if _expected_calc_param_md5 is _NOT_GIVEN:
        _expected_calc_param_md5 = _render_expected_md5(cfg)

    mismatches = []

    for label, meta_path, cfg_path, kind in SIMPLE_CHECKS:
        recorded = _get(metadata, meta_path)
        if recorded is _MISSING:
            continue  # this stage hasn't run yet for this molecule
        try:
            expected = wi.get(cfg, cfg_path)
        except KeyError:
            continue  # inputs file doesn't define this field at all
        equal = _as_bool_ish(recorded) == _as_bool_ish(expected) if kind == "bool" \
            else _raw_equal(recorded, expected)
        if not equal:
            mismatches.append((label, recorded, expected))

    # conformer_gen: optimize on/off, and if on, which xtb GFN level
    recorded_preopt = _get(metadata, "conformer_gen.xtb_preopt")
    if recorded_preopt is not _MISSING:
        expected_optimize = wi.get(cfg, "conformer_gen.optimize", default=None)
        if _as_bool_ish(recorded_preopt) != _as_bool_ish(expected_optimize):
            mismatches.append(("conformer_gen.optimize", recorded_preopt, expected_optimize))
        elif recorded_preopt:
            recorded_method = _get(metadata, "conformer_gen.xtb_preopt_method")
            expected_level = wi.get(cfg, "conformer_gen.xtb_gfn_level", default=None)
            expected_method = f"GFN{expected_level}-xTB" if expected_level is not None else None
            if recorded_method is not _MISSING and recorded_method != expected_method:
                mismatches.append(("conformer_gen.xtb_gfn_level", recorded_method, expected_method))

    # DFT energy window lives per-conformer in metadata, single value in config
    conformers = _get(metadata, "dft.conformers")
    if isinstance(conformers, dict):
        for rec in conformers.values():
            if isinstance(rec, dict) and "ewin_kcal_mol" in rec:
                expected_ewin = wi.get(cfg, "dft.ewin_kcal_mol", default=None)
                if not _raw_equal(rec["ewin_kcal_mol"], expected_ewin):
                    mismatches.append(("dft.ewin_kcal_mol", rec["ewin_kcal_mol"], expected_ewin))
                break  # same value for every conformer; one check is enough

    # Rendered-template fingerprint -- catches edits to calcParam.inp.tmpl's
    # STRUCTURE, not just to the level_of_theory VALUES already checked above.
    recorded_md5 = _get(metadata, "dft.calc_param_md5")
    if recorded_md5 is not _MISSING and _expected_calc_param_md5 is not None:
        if recorded_md5 != _expected_calc_param_md5:
            mismatches.append(("dft.calc_param_template (rendered)", recorded_md5, _expected_calc_param_md5))

    return mismatches


def find_batch_metadata(batch_root="."):
    """{directory_name: metadata_dict} for every subdirectory of batch_root
    that has a metadata.json -- i.e. every ligand that's been through the
    pipeline at all so far, regardless of how far it got."""
    found = {}
    for name in sorted(os.listdir(batch_root)):
        path = os.path.join(batch_root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.json")):
            found[name] = metadata_utils.load_metadata(path)
    return found


def compute_batch_mismatches(cfg, batch_root=".", exclude_inchikey=None,
                              _expected_calc_param_md5=_NOT_GIVEN):
    """{directory_name: [mismatches]} for every OTHER already-run ligand in
    batch_root whose recorded settings disagree with workflow_inputs.yaml
    right now. This is what catches a brand new ligand about to be submitted
    under different settings than the rest of an in-progress batch, since a
    new ligand has no metadata of its own yet for compute_mismatches() to
    compare against."""
    if _expected_calc_param_md5 is _NOT_GIVEN:
        _expected_calc_param_md5 = _render_expected_md5(cfg)

    results = {}
    for name, metadata in find_batch_metadata(batch_root).items():
        if name == exclude_inchikey:
            continue
        mismatches = compute_mismatches(cfg, metadata, _expected_calc_param_md5=_expected_calc_param_md5)
        if mismatches:
            results[name] = mismatches
    return results


def print_mismatches(inchi_key, smiles, mismatches):
    print(f"\nWARNING: {inchi_key} ({smiles}) already has results recorded, but "
          f"workflow_inputs.yaml now disagrees with what was actually used:\n",
          file=sys.stderr)
    width = max(len(m[0]) for m in mismatches)
    for label, recorded, expected in mismatches:
        print(f"  {label.ljust(width)}   recorded: {recorded!r:<30} now: {expected!r}",
              file=sys.stderr)
    print("\nProceeding will use the NEW settings for any stage that hasn't "
          "completed yet; stages that already ran keep their existing results "
          "(nothing here recomputes retroactively).\n", file=sys.stderr)


def print_batch_mismatches(batch_mismatches):
    print("\nWARNING: workflow_inputs.yaml disagrees with settings already used "
          "by OTHER ligands already run in this batch:\n", file=sys.stderr)
    width = max(len(label) for mismatches in batch_mismatches.values() for label, _, _ in mismatches)
    for name, mismatches in sorted(batch_mismatches.items()):
        for label, recorded, expected in mismatches:
            print(f"  [{name}] {label.ljust(width)}   recorded: {recorded!r:<30} now: {expected!r}",
                  file=sys.stderr)
    print("\nSubmitting under the current settings means this batch will contain "
          "molecules computed under different settings -- aggregate_metadata.py's "
          "consistency check will flag this later if you proceed now.\n", file=sys.stderr)


def confirm_or_skip(smiles, cfg=None, assume_yes=False, non_interactive_ok=False,
                     batch_root=".", batch_mismatches=None):
    """Returns (proceed: bool, had_mismatches: bool).

    proceed=False means the caller should NOT submit/run anything further for
    this molecule. had_mismatches tells the caller whether a disagreement was
    found at all (submit_all_smiles.py uses this to pass FORCE_INPUT_MISMATCH
    down to the SLURM job it's about to submit, once a human has confirmed).

    batch_mismatches lets a caller that's looping over many ligands (like
    submit_all_smiles.py) pass in a dict already computed once via
    compute_batch_mismatches(), instead of every call re-scanning the whole
    batch directory. Left as None, it's computed here.
    """
    cfg = cfg or wi.load_inputs()
    inchi_key = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
    expected_md5 = _render_expected_md5(cfg)

    own_metadata = metadata_utils.load_metadata(inchi_key) if os.path.isdir(inchi_key) else {}
    own_mismatches = compute_mismatches(cfg, own_metadata, _expected_calc_param_md5=expected_md5)

    if batch_mismatches is None:
        batch_mismatches = compute_batch_mismatches(
            cfg, batch_root=batch_root, exclude_inchikey=inchi_key,
            _expected_calc_param_md5=expected_md5)
    else:
        batch_mismatches = {k: v for k, v in batch_mismatches.items() if k != inchi_key}

    if not own_mismatches and not batch_mismatches:
        return True, False

    if own_mismatches:
        print_mismatches(inchi_key, smiles, own_mismatches)
    if batch_mismatches:
        print_batch_mismatches(batch_mismatches)

    if assume_yes:
        print("--yes given: proceeding despite the differences above.", file=sys.stderr)
        return True, True

    if not sys.stdin.isatty():
        if non_interactive_ok:
            print("Proceeding: an operator already confirmed this "
                  "(FORCE_INPUT_MISMATCH set).", file=sys.stderr)
            return True, True
        if os.path.isdir(inchi_key):
            # A brand new ligand (own_mismatches empty, only batch_mismatches
            # populated) has no directory of its own yet to record this into.
            metadata_utils.update_metadata(inchi_key, {
                "steps": {"input_check": {
                    "status": "failed",
                    "reason": "workflow_inputs.yaml disagrees with recorded metadata "
                              "(this ligand's own record and/or other ligands already "
                              "run in this batch); no terminal attached to confirm",
                }},
            })
        print(f"ERROR: {inchi_key} has unresolved input disagreements and no "
              f"terminal is attached to ask. Re-run through submit_all_smiles.py "
              f"(interactive), or set FORCE_INPUT_MISMATCH=1 if this was already "
              f"confirmed elsewhere.", file=sys.stderr)
        return False, True

    answer = input("Continue anyway? [y/N]: ").strip().lower()
    return answer in ("y", "yes"), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lig", required=True, help="SMILES string of the molecule")
    parser.add_argument("--batch-root", default=".",
                         help="Directory containing every <inchikey>/ in this batch. Default: .")
    parser.add_argument("--yes", action="store_true",
                         help="Proceed automatically even if settings differ")
    parser.add_argument("--force-non-interactive", action="store_true",
                         help="Set when FORCE_INPUT_MISMATCH=1 is present -- an operator "
                              "already confirmed this via submit_all_smiles.py")
    args = parser.parse_args()

    proceed, _ = confirm_or_skip(args.lig, assume_yes=args.yes, batch_root=args.batch_root,
                                  non_interactive_ok=args.force_non_interactive)
    sys.exit(0 if proceed else 1)


if __name__ == "__main__":
    main()
