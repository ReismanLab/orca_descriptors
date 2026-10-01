"""``status``: one row per molecule of the input list, from the records (not from squeue): bceg's search
and post-optimisation records for the ensemble, then the ORCA stage and its conformer counts."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from desc_utils.submit import records
from desc_utils.submit.config import Config
from desc_utils.submit.source import make_source

CONFORMER_COLUMNS = ["completed", "running", "submitted", "prepared", "failed", "timeout",
                     "skipped_energy_filter", "skipped_no_energy"]


def molecule_rows(cfg: Config) -> list[dict]:
    source = make_source(cfg.source)
    rows = []
    for mol in source.molecules():
        meta = records.load_metadata(cfg.library_dir, mol.inchikey)
        state, reason = records.molecule_state(meta)
        progress = source.progress(mol)
        if state == "in_progress" and not (meta.get("conformers") or {}):
            # the ORCA stage is waiting on bceg: say where bceg stands
            reason = f"ensemble: search {progress['search']}, post-optimisation {progress['post_opt']}"
        if progress.get("error"):
            reason = f"ensemble failed: {progress['error']}"
        counts = Counter((v or {}).get("status") for v in (meta.get("conformers") or {}).values())
        rows.append({"inchikey": mol.inchikey, "name": mol.name or "", "smiles": mol.input_smiles,
                     "search": progress["search"], "post_opt": progress["post_opt"],
                     "n_conf": progress["n_conformers"] if progress["n_conformers"] is not None else "",
                     "state": state, "orca_stage": (meta.get("orca_stage") or {}).get("status", ""),
                     **{c: counts.get(c, 0) for c in CONFORMER_COLUMNS}, "reason": reason})
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
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.max_colwidth", 60):
        print(df.drop(columns=["smiles", "reason"]).to_string(index=False))
        print()
        print("ensembles (bceg): search", dict(Counter(df["search"])), "| post-optimisation", dict(Counter(df["post_opt"])))
        print("molecules per state:  ", dict(Counter(df["state"])))
        waiting = df[(df["state"] == "in_progress") & (df["orca_stage"] == "queued")]
        if not waiting.empty:
            print(f"\n{len(waiting)} molecule(s) waiting for their ensemble (ORCA prepare queued after bceg)")
        bad = df[(df["state"] == "incomplete") | df["search"].isin(["failed"]) | df["post_opt"].isin(["failed"])]
        if not bad.empty:
            print("\nneeds a look:")
            for _, r in bad.drop_duplicates("inchikey").iterrows():
                print(f"  {r['inchikey']}  {r['reason']}")
    return df
