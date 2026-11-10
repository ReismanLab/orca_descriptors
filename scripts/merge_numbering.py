"""Concatenate the numbering tables of several libraries into one, for a pooled extraction.

A numbering table's ``path`` is relative to the repository root, and ``run_extraction`` resolves it
with ``conf_key(path, repo_root)`` rather than against ``data.root`` - so the rows of several
libraries can sit in one table and the extraction finds every conformer where it actually is. That
is all a pooled library needs; no spec key has to name several roots.

A ``source_library`` column is added, because once the rows are pooled the only record of which
search found a conformer is its path, and a column is what the per-conformer table can carry
through to the modelling app.

    python scripts/merge_numbering.py A.csv B.csv ... --out merged.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def library_of(path: str) -> str:
    """``DATA/<library>/<molecule>/conf_N`` -> ``<library>``."""
    parts = Path(str(path)).parts
    return parts[1] if len(parts) > 3 and parts[0] == "DATA" else (parts[0] if parts else "")


def merge(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8")
        df["source_library"] = df["path"].map(library_of)
        df["source_numbering_table"] = p.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    dup = out["path"].duplicated()
    if dup.any():
        raise SystemExit(
            f"{int(dup.sum())} conformer path(s) appear in more than one table, e.g. "
            f"{out.loc[dup, 'path'].iloc[0]}. Pooling them would weight those conformers twice in "
            f"every Boltzmann average. Give each library its own root and merge once.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tables", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    out = merge(args.tables)
    out.to_csv(args.out, index=False, encoding="utf-8")
    counts = out["source_library"].value_counts().to_dict()
    print(f"wrote {args.out}  rows={len(out)}  molecules={out['inchikey'].nunique()}")
    for lib, n in counts.items():
        print(f"    {lib:34s} {n:7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
