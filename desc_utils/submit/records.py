"""Per-molecule ``metadata.json`` and the library's ``molecule_registry.json``.

``update_json`` deep-merges under an flock, so the ORCA tasks of one molecule's conformers can finish
at the same time and all land in the same file. A merge that changes nothing leaves the file alone.

Statuses
--------
Ensemble stage of a molecule (``metadata["ensemble_stage"]``): written by ``submit`` as ``queued`` with the
bceg job ids when stages 1-4 were just queued for it; bceg's own run records take over from there.
ORCA stage of a molecule (``metadata["orca_stage"]["status"]``):
  ``queued``            prepare task submitted, ensemble not looked at yet
  ``prepared``          ensemble split into conf_N/, ORCA jobs submitted (or nothing to submit)
  ``ensemble_missing``  bceg has not produced the ensemble (yet); ``ensemble_failed``; ``not_applicable``
  ``ensemble_changed``  the ensemble's params.json hash differs from the one this library was built on
  ``failed``            the prepare task did not run to its end (``reason: cancelled`` after ``cancel``)
Conformer (``metadata["conformers"][conf]["status"]``):
  ``skipped_energy_filter``, ``skipped_no_energy``, ``prepared``, ``submitted``, ``running``,
  ``completed``, ``failed``, ``timeout``
``cancel`` writes ``failed`` with ``reason: cancelled`` (and ``cancelled_at``): a cancelled step is a failed
step, and ``submit`` relaunches both alike.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path

METADATA = "metadata.json"
REGISTRY = "molecule_registry.json"

OK_CONFORMER = {"completed", "skipped_energy_filter", "skipped_no_energy"}
PENDING_CONFORMER = {"submitted", "running"}
BAD_CONFORMER = {"failed", "timeout", "prepared"}
STAGE_WAITING = {"queued"}
STAGE_BLOCKED = {"ensemble_missing", "ensemble_failed", "not_applicable", "ensemble_changed", "failed"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _deep_merge(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            raw = f.read()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return json.loads(raw) if raw.strip() else {}


def update_json(path: str | Path, updates: dict) -> dict:
    """Deep-merge ``updates`` into the JSON file at ``path`` (created if absent); return the document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
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
            json.dump(merged, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return merged


# ----------------------------------------------------------------------------- molecule records
def metadata_path(library_dir: Path, inchikey: str) -> Path:
    return Path(library_dir) / inchikey / METADATA


def load_metadata(library_dir: Path, inchikey: str) -> dict:
    return load_json(metadata_path(library_dir, inchikey))


def update_metadata(library_dir: Path, inchikey: str, updates: dict) -> dict:
    return update_json(metadata_path(library_dir, inchikey), updates)


def molecule_state(meta: dict) -> tuple[str, str]:
    """Classify one molecule as (state, reason).

    ``absent``       nothing recorded for this molecule
    ``in_progress``  prepare task queued, or ORCA jobs submitted or running
    ``incomplete``   ensemble unavailable, or a conformer failed, timed out or was never submitted
    ``complete``     every conformer completed or skipped by the energy filter

    Inferred from what was written down, not from squeue: a job killed before it recorded its end
    leaves its conformer at ``running``, and the molecule needs ``--force`` to be resubmitted.
    """
    if not meta:
        return "absent", "nothing recorded"
    stage = (meta.get("orca_stage") or {}).get("status")
    conformers = meta.get("conformers") or {}
    if stage in STAGE_BLOCKED:
        return "incomplete", f"{stage}: {(meta.get('orca_stage') or {}).get('reason', '')}".rstrip(": ")
    if stage in STAGE_WAITING and not conformers:
        return "in_progress", "prepare task queued"
    if not conformers:
        return "incomplete", "no conformer recorded"
    statuses = {c: (v or {}).get("status") for c, v in conformers.items()}
    pending = sorted(c for c, s in statuses.items() if s in PENDING_CONFORMER)
    if pending:
        return "in_progress", f"{len(pending)} ORCA job(s) submitted or running"
    bad = sorted(c for c, s in statuses.items() if s not in OK_CONFORMER)
    if bad:
        return "incomplete", "conformer(s) not completed: " + ", ".join(f"{c}={statuses[c]}" for c in bad)
    n_done = sum(1 for s in statuses.values() if s == "completed")
    return "complete", f"{n_done} conformer(s) completed, {len(statuses) - n_done} skipped"


# ----------------------------------------------------------------------------- registry
def registry_path(library_dir: Path) -> Path:
    return Path(library_dir) / REGISTRY


def load_registry(library_dir: Path) -> dict:
    return load_json(registry_path(library_dir))


def register_molecule(library_dir: Path, inchikey: str, entry: dict) -> dict:
    """Merge ``entry`` under ``inchikey``; stamp ``registered_at`` once; keep the first submitted SMILES
    and record other spellings under ``also_submitted_as`` (a WARNING when they are not even the same
    canonical SMILES: a genuine InChIKey collision)."""
    existing = load_registry(library_dir).get(inchikey) or {}
    entry = dict(entry)
    if not existing:
        entry.setdefault("registered_at", now())
    old, new = existing.get("smiles"), entry.get("smiles")
    if old and new and new != old:
        entry["smiles"] = old
        entry["also_submitted_as"] = sorted(set(existing.get("also_submitted_as") or []) | {new})
        old_can, new_can = existing.get("canonical_smiles"), entry.get("canonical_smiles")
        if old_can and new_can and old_can != new_can:
            print(f"WARNING: {inchikey} was first submitted as {old!r} ({old_can}) and now as {new!r} "
                  f"({new_can}); keeping the first, recording the second under also_submitted_as", file=sys.stderr)
        elif new_can:
            entry["canonical_smiles"] = old_can or new_can
    return update_json(registry_path(library_dir), {inchikey: entry})
