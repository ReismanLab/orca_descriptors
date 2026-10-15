"""``status``: one row per molecule of the input list with one status per stage, from the records
(not from squeue).

Stage columns ``transformation``, ``pre_refinement``, ``search``, ``post_opt``, ``pruning``, ``dft``
take the values ``queued``, ``running``, ``done``, ``failed``, ``n/a`` (the pipeline has no such step
for this molecule), ``-`` (nothing recorded and nothing queued) or ``<k>/<n>`` (k of the n replicates
done, the others without a record yet); ``dft`` and ``pruning`` show ``off`` when ``orca: off``.
Stages 1-4 come from bceg's ``run_<i>[_<lot>].json`` (with the ``phase`` the search task recorded)
and from the ``ensemble_stage: queued`` mark ``submit`` leaves; stages 5-6 from ``orca_stage`` and
the conformer records.

The ``transformation`` column is not a record: ``status`` applies the transformation script as it is
now. A molecule the script accepts today but did not when ``submit`` ran (nothing recorded, nothing
queued) is reported in the state ``not submitted``; one the script does not apply to is ``n/a``.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from desc_utils.submit import records
from desc_utils.submit.config import Config
from desc_utils.submit.source import Ensembles

STAGE_COLUMNS = ["transformation", "pre_refinement", "search", "post_opt", "pruning", "dft"]
QUEUED, RUNNING, DONE, FAILED, NA, NONE, OFF = "queued", "running", "done", "failed", "n/a", "-", "off"
PARTIAL = "partial"       # aggregate_records: some replicates done, the others without a record
PRINT_NAMES = {"transformation": "transf", "pre_refinement": "preopt"}


def _partial(rec: dict | None) -> str | None:
    """``<k>/<n>`` for a folded record in the ``partial`` state, else None."""
    if rec and rec.get("status") == PARTIAL:
        return f"{rec['n_done']}/{rec['n_total']}"
    return None


def stage_statuses(progress: dict, meta: dict, orca_on: bool) -> dict[str, str]:
    """One status per stage for one molecule. ``progress`` is ``Ensembles.progress(mol)``."""
    queued_ens = (meta.get("ensemble_stage") or {}).get("status") == QUEUED
    t = progress["transformation"]
    out = {"transformation": t if t in (DONE, FAILED) else NA}
    if t != DONE:
        return out | {k: NA for k in STAGE_COLUMNS[1:]}

    search, refine = progress["search_record"], progress["refine_record"]
    s_status = (search or {}).get("status")
    phase = (search or {}).get("phase")
    partial = _partial(search)

    # 2. pre-refinement: part of the search task, told apart by the phase it recorded
    if not progress["has_preopt"]:
        pre = NA
    elif search is None:
        pre = QUEUED if queued_ens else NONE
    elif s_status == RUNNING:
        pre = RUNNING if phase == "pre_refinement" else DONE
    elif s_status in (DONE, PARTIAL):
        pre = DONE
    elif s_status == FAILED:
        pre = FAILED if phase in ("pre_refinement", None) else DONE
    elif s_status == "not_applicable":
        pre = NA
    else:
        pre = s_status

    # 3. conformer search
    if search is None:
        se = QUEUED if queued_ens else NONE
    elif s_status == RUNNING:
        se = QUEUED if phase == "pre_refinement" else RUNNING
    elif s_status == DONE:
        se = DONE
    elif s_status == PARTIAL:
        se = partial
    elif s_status == FAILED:
        se = NONE if phase == "pre_refinement" else FAILED
    elif s_status == "not_applicable":
        se = NA
    else:
        se = s_status

    # 4. post-optimisation
    if not progress["has_refiner"]:
        po = NA
    elif refine is None:
        po = QUEUED if (se in (QUEUED, RUNNING) or (se in (DONE, partial) and queued_ens)) else NONE
    elif refine.get("status") == PARTIAL:
        po = _partial(refine)
    else:
        po = refine.get("status") or NONE

    # 5. pruning and 6. dft, from the ORCA stage records
    stage = (meta.get("orca_stage") or {}).get("status")
    confs = meta.get("conformers") or {}
    counts = Counter((v or {}).get("status") for v in confs.values())
    if not orca_on:
        pr = dft = OFF
    else:
        pr = {"prepared": DONE, QUEUED: QUEUED, "ensemble_changed": FAILED, FAILED: FAILED}.get(stage, NONE)
        if not confs:
            dft = QUEUED if stage in (QUEUED, "prepared") else NONE
        elif counts[RUNNING]:
            dft = RUNNING
        elif counts["submitted"]:
            dft = QUEUED
        elif all(st in records.OK_CONFORMER for st in counts):
            dft = DONE
        else:
            dft = FAILED
    return out | {"pre_refinement": pre, "search": se, "post_opt": po, "pruning": pr, "dft": dft}


def molecule_rows(cfg: Config) -> list[dict]:
    source = Ensembles(cfg)
    rows = []
    for mol in source.molecules():
        meta = records.load_metadata(cfg.library_dir, mol.inchikey)
        state, reason = records.molecule_state(meta)
        progress = source.progress(mol)
        stages = stage_statuses(progress, meta, cfg.orca_on)
        if progress.get("error"):
            reason = f"ensemble failed: {progress['error']}"
        elif state == "absent" and stages["transformation"] == NA:
            state, reason = NA, f"transformation {source.transformation.name} does not apply"
        elif state == "absent" and stages["transformation"] == DONE:
            state, reason = "not submitted", "nothing recorded and nothing queued: run submit"
        elif state == "in_progress" and not (meta.get("conformers") or {}):
            reason = "ensemble: " + ", ".join(f"{PRINT_NAMES.get(k, k)} {stages[k]}" for k in STAGE_COLUMNS[1:4])
        confs = meta.get("conformers") or {}
        counts = Counter((v or {}).get("status") for v in confs.values())
        n_to_run = sum(1 for v in confs.values() if (v or {}).get("status") not in ("skipped_energy_filter", "skipped_no_energy"))
        n_conf = progress["n_conformers"]
        if n_conf is None and (meta.get("orca_stage") or {}).get("n_conformers_in_ensemble") is not None:
            n_conf = meta["orca_stage"]["n_conformers_in_ensemble"]
        rows.append({"inchikey": mol.inchikey, "name": mol.name or "", **stages,
                     "n_conf": "" if n_conf is None else n_conf,
                     "n_dft": f"{n_to_run}/{len(confs)}" if confs else "",
                     "dft_done": f"{counts['completed']}/{n_to_run}" if confs else "",
                     "state": state, "reason": reason, "smiles": mol.input_smiles,
                     "_counts": dict(counts)})
    return rows


def status_table(cfg: Config) -> pd.DataFrame:
    df = pd.DataFrame(molecule_rows(cfg))
    if df.empty:
        return df
    return df.sort_values(["state", "inchikey"]).reset_index(drop=True)


def print_status(cfg: Config) -> pd.DataFrame:
    df = status_table(cfg)
    if df.empty:
        print("No molecules in the input list.")
        return df
    shown = df.drop(columns=["smiles", "reason", "_counts"]).rename(columns=PRINT_NAMES)
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.max_colwidth", 60):
        print(shown.to_string(index=False))
        print()
        for col in STAGE_COLUMNS:
            print(f"{PRINT_NAMES.get(col, col):>14}: " + ", ".join(f"{v} {n}" for v, n in sorted(Counter(df[col]).items())))
        total = Counter()
        for c in df["_counts"]:
            total.update(c)
        if total:
            print(f"{'dft jobs':>14}: " + ", ".join(f"{k} {v}" for k, v in sorted(total.items())))
        print(f"{'molecules':>14}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(df['state']).items())))
        failed = df[(df["state"] == "incomplete") | (df[STAGE_COLUMNS] == FAILED).any(axis=1)]
        if not failed.empty:
            print("\nneeds a look:")
            for _, r in failed.drop_duplicates("inchikey").iterrows():
                print(f"  {r['inchikey']}  {r['reason']}")
    return df
