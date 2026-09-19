"""Command-line entry point for the spec-driven descriptor pipeline.

    python -m desc_utils.run validate    --spec specs/lnih2_pyox.yaml
    python -m desc_utils.run number      --spec S [--data ROOT] [--out CSV] [--backend manual --csv CSV] [--strict]
    python -m desc_utils.run extract     --spec S --numbering CSV [--families orca.xyz,descripytor,...] [--out CSV]
    python -m desc_utils.run aggregate   --spec S --per-conformer CSV [--out CSV] [--no-filter]
    python -m desc_utils.run all         --spec S [--data ROOT] [--numbering CSV]
    python -m desc_utils.run to-schema-b --spec S --numbering CSV [--row N]

Every subcommand is a thin wrapper around a library function so notebooks can call the
same functions directly. All paths are resolved relative to the repository root (the
parent of ``desc_utils/``) unless absolute.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from desc_utils.spec import Spec, SpecError, load_spec

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path | None, default: str | Path | None = None) -> Path | None:
    if path is None:
        path = default
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load(args) -> Spec:
    try:
        spec = load_spec(_resolve(args.spec))
    except SpecError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    if getattr(args, "data", None):
        spec = Spec(**{**spec.__dict__, "data": {**spec.data, "root": args.data}})
    return spec


def _aggregated_name(spec: Spec) -> str:
    return spec.output["aggregated_csv"].format(date=datetime.now().strftime("%Y%m%d"), name=spec.name)


# --------------------------------------------------------------------------- commands

def cmd_validate(args) -> int:
    spec = _load(args)
    families = []
    if spec.orca:
        families += [f"orca.{k}" for k in spec.orca]
    if spec.descripytor:
        families.append("descripytor")
    print(f"OK  {spec.path}")
    print(f"    name={spec.name}  labels={spec.labels}  numbering={spec.numbering['backend']}")
    print(f"    families={families}")
    return 0


def cmd_number(args) -> int:
    from desc_utils.numbering import build_numbering_table, write_numbering_table

    spec = _load(args)
    backend_override = None
    if args.backend:
        from desc_utils.numbering.base import make_backend

        cfg = dict(spec.numbering, backend=args.backend)
        if args.backend == "manual":
            cfg["manual"] = {"csv": str(_resolve(args.csv)), "key_column": "path", "index_base": 1}
        backend_override = make_backend(cfg, spec.labels)
    # always write the table (even when --strict fails) so the offending rows can be inspected
    df = build_numbering_table(spec, repo_root=REPO_ROOT, backend=backend_override,
                               strict=False, regenerate_sdf=args.regenerate_sdf)
    out = _resolve(args.out, spec.output["numbering_csv"])
    write_numbering_table(df, out)
    status = df["numbering_status"].value_counts().to_dict()
    print(f"wrote {out}  rows={len(df)}  status={status}")
    bad = df[df["numbering_status"] != "ok"]
    if len(bad):
        print(bad[["path", "numbering_status", "numbering_notes"]].head(20).to_string(), file=sys.stderr)
        if args.strict:
            print(f"--strict: {len(bad)} conformer(s) not numbered cleanly", file=sys.stderr)
            return 1
    return 0


def cmd_extract(args) -> int:
    from desc_utils.extract import run_extraction
    from desc_utils.numbering import read_numbering_table

    spec = _load(args)
    df_num = read_numbering_table(_resolve(args.numbering, spec.output["numbering_csv"]), spec)
    families = args.families.split(",") if args.families else None
    df = run_extraction(spec, df_num, repo_root=REPO_ROOT, families=families,
                        overwrite_xyz_json=args.overwrite_xyz_json)
    out = _resolve(args.out, spec.output["per_conformer_csv"])
    df.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(df)}  columns={df.shape[1]}")
    return 0


def cmd_aggregate(args) -> int:
    from desc_utils.aggregate import aggregate_conformers, apply_filter

    spec = _load(args)
    df = pd.read_csv(_resolve(args.per_conformer, spec.output["per_conformer_csv"]))
    agg = aggregate_conformers(df, spec)
    if spec.filter.get("enabled", True) and not args.no_filter:
        agg, report = apply_filter(agg, spec)
        print("filter:", {k: (len(v) if isinstance(v, list) else v) for k, v in report.items()})
    out = _resolve(args.out, _aggregated_name(spec))
    agg.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(agg)}  columns={agg.shape[1]}")
    return 0


def cmd_all(args) -> int:
    from desc_utils.aggregate import aggregate_conformers, apply_filter
    from desc_utils.extract import run_extraction
    from desc_utils.numbering import build_numbering_table, read_numbering_table, write_numbering_table

    spec = _load(args)
    if args.numbering:
        df_num = read_numbering_table(_resolve(args.numbering), spec)
    else:
        df_num = build_numbering_table(spec, repo_root=REPO_ROOT)
        write_numbering_table(df_num, _resolve(spec.output["numbering_csv"]))
    df = run_extraction(spec, df_num, repo_root=REPO_ROOT)
    df.to_csv(_resolve(spec.output["per_conformer_csv"]), index=False)
    agg = aggregate_conformers(df, spec)
    if spec.filter.get("enabled", True):
        agg, _ = apply_filter(agg, spec)
    out = _resolve(_aggregated_name(spec))
    agg.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(agg)}  columns={agg.shape[1]}")
    return 0


def cmd_to_schema_b(args) -> int:
    from desc_utils.extract.descripytor import to_schema_b
    from desc_utils.numbering import read_numbering_table

    spec = _load(args)
    df_num = read_numbering_table(_resolve(args.numbering, spec.output["numbering_csv"]), spec)
    row = df_num.iloc[args.row]
    print(json.dumps(to_schema_b(spec.descripytor, row), indent=2))
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m desc_utils.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--spec", required=True, help="spec YAML/JSON file")

    sp = sub.add_parser("validate", help="check a spec file"); common(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("number", help="build the atom numbering table"); common(sp)
    sp.add_argument("--data", help="override data.root")
    sp.add_argument("--out", help="output CSV (default: output.numbering_csv)")
    sp.add_argument("--backend", choices=["smarts", "manual", "first_n", "molalign"], help="override numbering.backend")
    sp.add_argument("--csv", help="numbering CSV for --backend manual")
    sp.add_argument("--strict", action="store_true", help="exit non-zero unless every row has status ok")
    sp.add_argument("--regenerate-sdf", action="store_true", help="rebuild .sdf files from the xyz even if present")
    sp.set_defaults(func=cmd_number)

    sp = sub.add_parser("extract", help="compute per-conformer descriptors"); common(sp)
    sp.add_argument("--numbering", help="numbering CSV (default: output.numbering_csv)")
    sp.add_argument("--families", help="comma list, e.g. orca.xyz,orca.charges,orca.nbo,orca.redox,descripytor")
    sp.add_argument("--out", help="output CSV (default: output.per_conformer_csv)")
    sp.add_argument("--overwrite-xyz-json", action="store_true", help="recompute morfeus descriptors instead of reading xyz.json")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("aggregate", help="Boltzmann-aggregate conformers and filter"); common(sp)
    sp.add_argument("--per-conformer", help="per-conformer CSV (default: output.per_conformer_csv)")
    sp.add_argument("--out", help="output CSV (default: output.aggregated_csv)")
    sp.add_argument("--no-filter", action="store_true")
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("all", help="number -> extract -> aggregate -> filter"); common(sp)
    sp.add_argument("--data", help="override data.root")
    sp.add_argument("--numbering", help="use an existing numbering CSV instead of rebuilding it")
    sp.set_defaults(func=cmd_all)

    sp = sub.add_parser("to-schema-b", help="print the DescriPyTor JSON for one conformer"); common(sp)
    sp.add_argument("--numbering", help="numbering CSV (default: output.numbering_csv)")
    sp.add_argument("--row", type=int, default=0)
    sp.set_defaults(func=cmd_to_schema_b)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
