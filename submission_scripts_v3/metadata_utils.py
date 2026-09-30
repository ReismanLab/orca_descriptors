"""Shared helpers for reading/writing the per-molecule <inchikey>/metadata.json
provenance file used across the conformer generation / CREST / UMA / ORCA
pipeline (see launch_workflow.sh).

update_metadata() is safe to call concurrently -- e.g. several ORCA SLURM
jobs for different conformers of the same molecule finishing at the same
time -- via an flock on the metadata file itself.
"""

import copy
import fcntl
import json
import os

# Terminal outcomes that mean a stage / conformer finished its work. Defined
# here rather than in aggregate_metadata.py so the summary's notion of "this
# run is fine" and submit_all_smiles.py's notion of "don't resubmit this
# ligand" can never drift apart.
OK_STEP_STATUSES = {"completed", "skipped"}
OK_CONFORMER_STATUSES = {"completed", "skipped_energy_filter"}
PENDING_STEP_STATUSES = {"running"}
PENDING_CONFORMER_STATUSES = {"submitted"}


def _default_path(inchikey_dir):
    return os.path.join(inchikey_dir, "metadata.json")


def _deep_merge(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def update_metadata(inchikey_dir, updates, path=None):
    """Deep-merge `updates` into <inchikey_dir>/metadata.json and return the
    resulting document. Dict values are merged key-by-key (recursively);
    non-dict values overwrite. Creates the file if it doesn't exist yet.

    If the merge is a no-op -- every key in `updates` already holds that
    value -- the file is left completely untouched, mtime included. Stages
    that resume over an already-finished molecule re-derive the same values
    and would otherwise rewrite the record of a run they didn't perform.
    """
    path = path or _default_path(inchikey_dir)

    if not os.path.exists(path):
        with open(path, "a"):
            pass

    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = f.read()
            data = json.loads(raw) if raw.strip() else {}
            merged = _deep_merge(copy.deepcopy(data), updates)
            if merged == data and raw.strip():
                return data
            f.seek(0)
            f.truncate()
            json.dump(merged, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return merged


def run_state(inchikey_dir, path=None):
    """Classify one molecule directory as (state, reason).

    'absent'      - never built; nothing has run for this ligand
    'in_progress' - a stage is still marked running, or ORCA jobs are queued
    'incomplete'  - a stage or conformer failed, or nothing was produced
    'complete'    - every stage finished and every conformer reached a
                    terminal-good status

    Note that 'in_progress' is inferred from what was written down, not from
    squeue: a job killed at walltime leaves its stage at "running" forever,
    so a ligand stuck in that state needs --force to resubmit.
    """
    if not os.path.isdir(inchikey_dir):
        return "absent", "no directory yet"

    meta = load_metadata(inchikey_dir, path)
    if not meta:
        return "incomplete", "no metadata.json"

    steps = meta.get("steps") or {}
    running = [n for n in sorted(steps)
               if (steps[n] or {}).get("status") in PENDING_STEP_STATUSES]
    if running:
        return "in_progress", f"stage(s) still marked running: {', '.join(running)}"
    failed = [n for n in sorted(steps)
              if (steps[n] or {}).get("status") not in OK_STEP_STATUSES]
    if failed:
        return "incomplete", f"stage(s) not completed: {', '.join(failed)}"

    conformers = (meta.get("dft") or {}).get("conformers") or {}
    if not conformers:
        return "incomplete", "no DFT conformers recorded"

    pending = [c for c in sorted(conformers)
               if (conformers[c] or {}).get("status") in PENDING_CONFORMER_STATUSES]
    if pending:
        return "in_progress", f"{len(pending)} ORCA job(s) still pending"
    bad = [c for c in sorted(conformers)
           if (conformers[c] or {}).get("status") not in OK_CONFORMER_STATUSES]
    if bad:
        return "incomplete", f"conformer(s) not completed: {', '.join(bad)}"

    return "complete", f"{len(conformers)} conformer(s) done"


def load_metadata(inchikey_dir, path=None):
    path = path or _default_path(inchikey_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            raw = f.read()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return json.loads(raw) if raw.strip() else {}
