"""List the molecules of a computation folder: InChIKey and number of conformer folders.

Expects the campaign layout ``<computation_folder>/<inchikey>/conf_*/``. Standard library only.

Usage:
    python3 scripts-for-orca-computations/list_molecules.py <computation_folder> [-o molecules.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("computation_folder", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("molecules.csv"))
    args = parser.parse_args()

    mol_dirs = sorted(p for p in args.computation_folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    rows = [{"inchikey": d.name, "n_conformers": sum(1 for c in d.glob("conf_*") if c.is_dir())} for d in mol_dirs]
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["inchikey", "n_conformers"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} molecules written to {args.output}")


if __name__ == "__main__":
    main()
