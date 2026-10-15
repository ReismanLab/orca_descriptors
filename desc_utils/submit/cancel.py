"""``cancel``: stop every SLURM job of one library where it stands, and record what was cut short as
``status: failed, reason: cancelled`` so that the next ``submit`` relaunches it like any failure.

The jobs are found two ways, and the union is cancelled:
  * the records: ``ensemble_stage.bceg_job_ids`` and ``orca_stage.prepare_job_id`` of a stage still
    ``queued``, ``slurm_array_job_id`` of a conformer ``submitted`` or ``running``;
  * squeue: every job of the user whose batch script lies under ``<library_dir>/_jobs/`` (the bceg
    arrays, the prepare array and the ORCA arrays of every submission of this library), which also
    finds an ORCA array a prepare task submitted but was killed before recording.
Without squeue (a laptop, the tests) the record ids are taken as still alive.

What is marked, with ``reason: cancelled`` and ``cancelled_at``:
  * conformers ``submitted`` / ``running`` -> ``failed`` (written before ``scancel``; a running ORCA task
    that receives the TERM finds its job id in ``_jobs/cancelled.json`` and records ``failed``, not ``timeout``);
  * ``orca_stage`` ``queued`` -> ``failed``;
  * ``ensemble_stage`` ``queued`` whose bceg jobs were cancelled -> ``failed``;
  * bceg run records (search and post-optimisation) still ``running`` in a cancelled job -> ``failed``.
    A bceg task that had not started has no record and is simply pending again.
Completed conformers and finished bceg replicates are left alone.
"""

from __future__ import annotations

from pathlib import Path

from bceg.provenance import read_record, write_record

from desc_utils.submit import records, slurm
from desc_utils.submit.config import Config
from desc_utils.submit.source import Ensembles

REASON = "cancelled"
CANCELLED_FILE = "cancelled.json"


def cancelled_path(library_dir: Path) -> Path:
    return Path(library_dir) / "_jobs" / CANCELLED_FILE


def was_cancelled(library_dir: Path, job_ids: list[str | None]) -> bool:
    """True when one of ``job_ids`` (array id or element id) was cancelled by ``cancel``."""
    done = records.load_json(cancelled_path(library_dir))
    return any(j and str(j) in done for j in job_ids)


def _library_jobs(cfg: Config) -> list[dict] | None:
    """This library's jobs in squeue ({array_job_id, job_id, script}), None when squeue is unavailable."""
    jobs = slurm.queued_jobs()
    if jobs is None:
        return None
    root = (cfg.library_dir / "_jobs").resolve()
    return [j for j in jobs if Path(j["script"]).resolve().is_relative_to(root)]


def cancel_library(cfg: Config, dry_run: bool = False) -> dict:
    """Cancel the library's jobs and mark what they leave unfinished. Returns a summary."""
    lib = cfg.library_dir
    source = Ensembles(cfg)
    molecules = source.molecules()
    metas = {m.inchikey: records.load_metadata(lib, m.inchikey) for m in molecules}

    recorded: set[str] = set()
    for meta in metas.values():
        ens, orca = meta.get("ensemble_stage") or {}, meta.get("orca_stage") or {}
        if ens.get("status") == "queued":
            recorded |= {str(j) for j in ens.get("bceg_job_ids") or []}
        if orca.get("status") == "queued" and orca.get("prepare_job_id"):
            recorded.add(str(orca["prepare_job_id"]))
        for c in (meta.get("conformers") or {}).values():
            if (c or {}).get("status") in records.PENDING_CONFORMER and c.get("slurm_array_job_id"):
                recorded.add(str(c["slurm_array_job_id"]))

    live = _library_jobs(cfg)
    if live is None:
        to_cancel, alive = sorted(recorded), set(recorded)        # no squeue: trust the records
    else:
        to_cancel = sorted({j["array_job_id"] for j in live})
        alive = {j["array_job_id"] for j in live} | {j["job_id"] for j in live}

    summary = {"job_ids": to_cancel, "conformers": 0, "orca_stages": 0, "ensemble_stages": 0, "bceg_records": 0,
               "molecules": []}
    stamp = {"reason": REASON, "cancelled_at": records.now()}

    # 1. conformers first, and the cancelled ids, so a running ORCA task sees both when its TERM arrives
    updates: dict[str, dict] = {}
    for key, meta in metas.items():
        confs = {c: {"status": "failed", **stamp} for c, v in (meta.get("conformers") or {}).items()
                 if (v or {}).get("status") in records.PENDING_CONFORMER}
        if confs:
            updates[key] = {"conformers": confs}
            summary["conformers"] += len(confs)
    if not dry_run:
        if to_cancel:
            records.update_json(cancelled_path(lib), {j: stamp["cancelled_at"] for j in sorted(alive | set(to_cancel))})
        for key, upd in updates.items():
            records.update_metadata(lib, key, upd)

    # 2. scancel
    if to_cancel and not dry_run:
        slurm.scancel(to_cancel)

    # 3. the stages still waiting
    for key, meta in metas.items():
        upd = updates.setdefault(key, {})
        stage_upd: dict = {}
        if (meta.get("orca_stage") or {}).get("status") == "queued":
            stage_upd["orca_stage"] = {"status": "failed", **stamp}
            summary["orca_stages"] += 1
        ens = meta.get("ensemble_stage") or {}
        if ens.get("status") == "queued" and {str(j) for j in ens.get("bceg_job_ids") or []} & alive:
            stage_upd["ensemble_stage"] = {"status": "failed", **stamp}
            summary["ensemble_stages"] += 1
        if stage_upd and not dry_run:
            records.update_metadata(lib, key, stage_upd)
        upd.update(stage_upd)

    # 4. bceg's run records left at "running" by a cancelled job
    bceg_marked: dict[str, list[str]] = {}
    for mol in molecules:
        md = source.method_dir(mol.inchikey)
        for i in source.replicates:
            for path in (md.run_record(i), md.run_record(i, source.lot) if source.lot else None):
                rec = read_record(path) if path is not None else None
                if not rec or rec.get("status") != "running" or str(rec.get("slurm_job_id")) not in alive:
                    continue
                if not dry_run:
                    write_record(path, rec | {"status": "failed", "error": REASON, **stamp})
                bceg_marked.setdefault(mol.inchikey, []).append(path.name)
                summary["bceg_records"] += 1

    for mol in molecules:
        upd, runs = updates.get(mol.inchikey) or {}, bceg_marked.get(mol.inchikey) or []
        parts = []
        if upd.get("conformers"):
            parts.append(f"{len(upd['conformers'])} conformer(s)")
        if "orca_stage" in upd:
            parts.append("prepare")
        if "ensemble_stage" in upd:
            parts.append("ensemble")
        if runs:
            parts.append("bceg " + ", ".join(runs))
        if parts:
            summary["molecules"].append(mol.inchikey)
            print(f"  cancel  {mol.inchikey}  {'; '.join(parts)}")
    return summary


def print_cancel(cfg: Config, dry_run: bool = False) -> dict:
    s = cancel_library(cfg, dry_run=dry_run)
    verb = "would cancel" if dry_run else "cancelled"
    print(f"{verb} {len(s['job_ids'])} job(s)" + (f" ({', '.join(s['job_ids'])})" if s["job_ids"] else "")
          + f"; {len(s['molecules'])} molecule(s): {s['conformers']} conformer(s), {s['orca_stages']} prepare, "
          f"{s['ensemble_stages']} ensemble stage(s), {s['bceg_records']} bceg run record(s) marked failed (reason: cancelled)")
    if not dry_run and s["molecules"]:
        print("Relaunch with: orcadesc submit <config> (no --force needed)")
    return s
