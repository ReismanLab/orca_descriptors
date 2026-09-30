#!/usr/bin/env python3
"""Query molecule_registry.json: which molecule is behind a folder name.

Folders are named by InChIKey, a hash that cannot be turned back into a
SMILES. The registry (written by submit_all_smiles.py and the build stage)
maps every InChIKey of the batch to the SMILES as submitted, its line in
smiles.txt, the canonical SMILES, the InChI and the RDKit molecule that
became the first xyz (a mol block with explicit hydrogens and the embedded
3D coordinates).

    python lookup_molecule.py --list                    # InChIKey and submitted SMILES, one per line
    python lookup_molecule.py --inchikey <KEY>          # the whole entry (add --mol-block for the geometry)
    python lookup_molecule.py --smiles 'CC(C)P(C(C)C)C(C)C'   # InChIKey(s) of a molecule, any SMILES spelling
    python lookup_molecule.py --csv molecule_registry.csv     # flat table without the mol blocks

Standard library plus RDKit (only for --smiles). Run from the batch
directory, or give --batch-root.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metadata_utils

CSV_COLUMNS = ["inchikey", "smiles", "smiles_line", "canonical_smiles", "inchi", "formal_charge",
               "has_3d_coordinates", "registered_at", "built_at", "also_submitted_as"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-root", default=".", help="Batch directory holding molecule_registry.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--inchikey")
    g.add_argument("--smiles")
    g.add_argument("--csv", metavar="OUT")
    ap.add_argument("--mol-block", action="store_true", help="With --inchikey: include the mol block")
    args = ap.parse_args()

    reg = metadata_utils.load_registry(args.batch_root)
    if not reg:
        sys.exit(f"ERROR: no {metadata_utils.REGISTRY_FILENAME} in {os.path.abspath(args.batch_root)}")

    if args.list:
        for key in sorted(reg):
            print(f"{key}  {reg[key].get('smiles', '')}")
        return

    if args.inchikey:
        entry = reg.get(args.inchikey)
        if entry is None:
            sys.exit(f"ERROR: {args.inchikey} not in the registry")
        if not args.mol_block:
            entry = {k: v for k, v in entry.items() if k != "mol_block"}
        print(json.dumps(entry, indent=2, sort_keys=True))
        return

    if args.smiles:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(args.smiles)
        if mol is None:
            sys.exit(f"ERROR: invalid SMILES {args.smiles!r}")
        canonical = Chem.MolToSmiles(mol)
        hits = [k for k, e in reg.items()
                if e.get("canonical_smiles") == canonical or e.get("smiles") == args.smiles
                or args.smiles in (e.get("also_submitted_as") or [])]
        if not hits:
            sys.exit(f"no entry for {args.smiles!r} (canonical {canonical}); "
                     f"expected InChIKey {Chem.rdinchi.MolToInchiKey(mol)}")
        for k in sorted(hits):
            print(f"{k}  {reg[k].get('smiles', '')}")
        return

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for key in sorted(reg):
            row = {c: reg[key].get(c, "") for c in CSV_COLUMNS if c != "inchikey"}
            row["inchikey"] = key
            if isinstance(row["also_submitted_as"], list):
                row["also_submitted_as"] = ";".join(row["also_submitted_as"])
            w.writerow(row)
    print(f"wrote {len(reg)} molecules to {args.csv}")


if __name__ == "__main__":
    main()
