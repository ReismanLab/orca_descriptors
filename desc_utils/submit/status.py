"""``status``: one row per molecule of the input list, from the records (not from squeue)."""

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
        counts = Counter((v or {}).get("status") for v in (meta.get("conformers") or {}).values())
        rows.append({"inchikey": mol.inchikey, "name": mol.name or "", "smiles": mol.input_smiles,
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
        print("molecules per state:", dict(Counter(df["state"])))
        bad = df[df["state"] == "incomplete"]
        if not bad.empty:
            print("\nincomplete:")
            for _, r in bad.iterrows():
                print(f"  {r['inchikey']}  {r['reason']}")
    return df
