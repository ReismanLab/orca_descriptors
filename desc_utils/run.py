"""Command-line entry point for the spec-driven descriptor pipeline.

    python -m desc_utils.run validate    --spec specs/lnih2_pyox.yaml
    python -m desc_utils.run number      --spec S [--data ROOT] [--out CSV] [--backend manual --csv CSV] [--strict] [--jobs N]
    python -m desc_utils.run extract     --spec S --numbering CSV [--families orca.xyz,descripytor,...] [--out CSV] [--jobs N]
    python -m desc_utils.run aggregate   --spec S --per-conformer CSV [--out CSV] [--filter | --no-filter]
    python -m desc_utils.run all         --spec S [--data ROOT] [--numbering CSV] [--filter | --no-filter] [--jobs N]
                                         (never stops for a bad conformer: it warns and lists them in output.check_csv)
    python -m desc_utils.run to-schema-b --spec S --numbering CSV [--row N]
    python -m desc_utils.run condense    --spec S [--data ROOT] [--overwrite] [--jobs N] [--nbo-full all]

``--jobs N`` runs the per-conformer work of ``number`` and ``extract`` (and both inside ``all``) over N
processes; ``--jobs 0`` uses every core. The tables are identical to a serial run.

Every subcommand is a thin wrapper around a library function so notebooks can call the
same functions directly. All paths are resolved relative to the repository root (the
parent of ``desc_utils/``) unless absolute. ``{date}`` and ``{name}`` in the ``output`` names of the
spec are filled in at run time, so the tables of one day never overwrite those of another.
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


def _use_filter(spec: Spec, args) -> bool:
    """The noise filter is off unless the spec enables it or --filter is given; --no-filter always wins."""
    if getattr(args, "no_filter", False):
        return False
    return bool(getattr(args, "filter", False) or spec.filter.get("enabled", False))


def _output_name(spec: Spec, key: str) -> str:
    """``output.<key>`` of the spec with ``{date}`` (today) and ``{name}`` filled in."""
    return spec.output[key].format(date=datetime.now().strftime("%Y%m%d"), name=spec.name)


def _aggregated_name(spec: Spec) -> str:
    return _output_name(spec, "aggregated_csv")


def _existing_output(spec: Spec, key: str) -> Path:
    """The file ``output.<key>`` names for today if it exists, else the most recent one matching the pattern.

    Lets ``extract`` and ``to-schema-b`` find the numbering table of a previous day without ``--numbering``,
    and ``aggregate`` the per-conformer table without ``--per-conformer``.
    Falls back to today's name (which the caller will then report as missing) when nothing matches.
    """
    today = _resolve(_output_name(spec, key))
    if today.exists() or "{date}" not in spec.output[key]:
        return today
    pattern = spec.output[key].replace("{date}", "*").format(name=spec.name)
    candidates = [p for p in today.parent.glob(Path(pattern).name) if p.is_file()]
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"using {latest} (most recent {key})")
        return latest
    return today


def _report_check(df: pd.DataFrame, df_num: pd.DataFrame, spec: Spec) -> Path:
    """Warn about the conformers that need a look and write them to ``output.check_csv``.

    The run is never stopped for them: their descriptors are simply missing or NaN, and the
    aggregation flags the molecule (``all_conformers_completed``). The file is only there when
    there is something to look at; today's copy from an earlier run is removed otherwise.
    """
    from desc_utils.extract import conformers_to_check

    path = _resolve(_output_name(spec, "check_csv"))
    bad = conformers_to_check(df, df_num)
    if len(bad) == 0:
        if path.exists():
            path.unlink()
        msg = f"all {len(df)} conformers numbered"
        if "orca_job_status" in df.columns:
            unknown = int((df["orca_job_status"] == "unknown").sum())
            msg += (f"; ORCA job status unknown for {unknown} (condensed before the status was recorded)" if unknown
                    else " and their ORCA jobs completed")
        print(msg)
        return path
    bad.to_csv(path, index=False)
    counts = bad["reason"].value_counts().to_dict()
    print(f"WARNING: {len(bad)} conformer(s) need checking, see {path}  {counts}", file=sys.stderr)
    print(bad.drop(columns=[c for c in ["numbering_notes"] if c in bad.columns]).head(20).to_string(index=False),
          file=sys.stderr)
    return path


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
                               strict=False, regenerate_sdf=args.regenerate_sdf, jobs=args.jobs)
    out = _resolve(args.out, _output_name(spec, "numbering_csv"))
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


def _report_empty_columns(df, spec) -> None:
    """Name the requested descriptor columns that came out empty for every conformer.

    `aggregate` drops all-NaN columns, so a family that produced nothing simply does not appear in
    the spreadsheet and the run looks clean. That is how a g-xTB library was extracted with all 21
    Fukui columns empty - `descriptors.orca.redox.prefixes: [cm5]` cannot fill on g-xTB output,
    which has no Hirshfeld charges to derive CM5 from, while the IP/EA/hardness columns of the same
    family filled normally and hid it. Saying so at extraction time is the only cheap check.
    """
    import numpy as np

    meta = {"path", "xyz_name", "inchi", "inchikey", "smiles", "smiles_noH", "ligand_type",
            "ligand_class", "numbering_status", "numbering_notes", "numbering_backend",
            "orca_job_status", "n_conformer_folders", *spec.labels}
    cols = [c for c in df.columns if c not in meta and not str(c).startswith(("orca_run", "orca_step", "orca_nprocs", "orca_cpu"))]
    empty = [c for c in cols if df[c].isna().all()]
    if not empty:
        return
    print(f"NOTE: {len(empty)} of {len(cols)} descriptor column(s) are empty for every conformer "
          f"and will not reach the spreadsheet: "
          f"{', '.join(map(str, empty[:12]))}{' ...' if len(empty) > 12 else ''}",
          file=sys.stderr)


def _apply_hemisphere(spec, df_num, args, repo_root):
    """BUNNY's N1/N2 convention, applied to the numbering table before anything reads it.

    Before the prune, so the record covers every conformer rather than the survivors, and
    long before extraction, because every site-labelled descriptor is computed from these
    columns. The re-labelled table is written back over the numbering csv: a stage that
    read the un-corrected one would silently describe different atoms under the same name.
    """
    cfg = spec.hemisphere
    if not cfg.get("enabled"):
        return df_num
    from desc_utils import hemisphere as hs

    df_num, record = hs.relabel(df_num, cfg, molecule_column=spec.data["molecule_id_column"],
                                geometry_file=spec.data["geometry_file"])
    out = _resolve(_output_name(spec, "numbering_csv")).with_name(
        _output_name(spec, "numbering_csv").replace("atom_numbering_results", "hemisphere"))
    record.to_csv(out, index=False, encoding="utf-8")
    # Written back, not kept in memory: the numbering table on disk is what the modelling
    # app reads to mark the site atom in 3D, so a table saying N1 is one atom beside a
    # descriptor computed on the other is the silent disagreement this pass exists to end.
    # Idempotent - once the bulkier donor IS N2 the frame flips with it, north wins, and a
    # second pass swaps nothing.
    table = _resolve(_output_name(spec, "numbering_csv"))
    if table.exists():
        df_num.to_csv(table, index=False, encoding="utf-8")
    info = hs.summary(record)
    print(f"hemisphere: {info['n_swapped']} of {info['n_decided']} molecules swapped "
          f"({info['n_skipped_by_class']} kept their class convention, "
          f"{info['n_undecided'] - info['n_skipped_by_class']} undecided); "
          f"record in {out}")
    if info["n_within_0p5_percent"]:
        # a symmetric ligand's two donors ARE the same atom, so the winner is noise and
        # the labels are arbitrary - harmless there, but not something to discover later
        print(f"hemisphere: {info['n_within_0p5_percent']} molecule(s) decided on a margin "
              f"under 0.5 %Vbur (smallest {info['smallest_margin']:.3f}); on a symmetric "
              f"ligand the two donors are equivalent and the label is arbitrary")
    return df_num


def _apply_prune(spec, df_num, args, repo_root):
    """The prune block, between numbering and extraction, with its record written beside the tables.

    Pruning here rather than after extraction is not only cheaper: `aggregate` would otherwise have
    to be told which rows to ignore, and a per-conformer table holding rows the library does not
    describe is a table two stages can disagree about.
    """
    cfg = spec.prune
    if not cfg.get("enabled"):
        return df_num
    from desc_utils import prune as pr

    jobs = getattr(args, "jobs", 1)
    keep, record = pr.prune(df_num, cfg, molecule_column=spec.data["molecule_id_column"],
                            geometry_file=spec.data["geometry_file"], jobs=jobs)
    out = _resolve(_output_name(spec, "pruning_csv"))
    record.to_csv(out, index=False, encoding="utf-8")
    info = pr.summary(record, int(cfg.get("max_automorphisms", pr.MAX_AUTOMORPHISMS)))
    print(f"prune: {info['n_kept']} of {info['n_conformers']} conformers kept "
          f"(window {cfg.get('energy_window_kcal_mol')} kcal/mol dropped {info['n_above_energy_window']}, "
          f"RMSD {cfg.get('rmsd_threshold_angstrom')} A dropped {info['n_duplicates']}, "
          f"{info['n_without_energy']} had no energy); record in {out}")
    if info["molecules_at_automorphism_cap"]:
        # their heavy-atom graph has more symmetries than were tried, so their count is an upper
        # bound and can move by a conformer or two with the RDKit version
        print(f"prune: {len(info['molecules_at_automorphism_cap'])} molecule(s) reached the "
              f"{cfg.get('max_automorphisms')}-automorphism cap: "
              f"{', '.join(info['molecules_at_automorphism_cap'])}")
    return keep


def cmd_extract(args) -> int:
    from desc_utils.extract import run_extraction
    from desc_utils.numbering import read_numbering_table

    spec = _load(args)
    # every row, including the badly numbered ones: run_extraction keeps their metadata and skips the engines
    df_num = read_numbering_table(_resolve(args.numbering) if args.numbering else _existing_output(spec, "numbering_csv"),
                                  spec, usable=False)
    df_num = _apply_hemisphere(spec, df_num, args, REPO_ROOT)
    df_num = _apply_prune(spec, df_num, args, REPO_ROOT)
    families = args.families.split(",") if args.families else None
    df = run_extraction(spec, df_num, repo_root=REPO_ROOT, families=families,
                        overwrite_xyz_json=args.overwrite_xyz_json, jobs=args.jobs)
    out = _resolve(args.out, _output_name(spec, "per_conformer_csv"))
    df.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(df)}  columns={df.shape[1]}")
    _report_empty_columns(df, spec)
    _report_check(df, df_num, spec)
    return 0


def cmd_aggregate(args) -> int:
    from desc_utils.aggregate import aggregate_conformers, apply_filter

    spec = _load(args)
    df = pd.read_csv(_resolve(args.per_conformer) if args.per_conformer else _existing_output(spec, "per_conformer_csv"))
    agg = aggregate_conformers(df, spec)
    if _use_filter(spec, args):
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
        df_num = read_numbering_table(_resolve(args.numbering), spec, usable=False)
    else:
        df_num = build_numbering_table(spec, repo_root=REPO_ROOT, jobs=args.jobs)
        num_out = _resolve(_output_name(spec, "numbering_csv"))
        write_numbering_table(df_num, num_out)
        print(f"wrote {num_out}  rows={len(df_num)}  status={df_num['numbering_status'].value_counts().to_dict()}")
    df_num = _apply_hemisphere(spec, df_num, args, REPO_ROOT)
    df_num = _apply_prune(spec, df_num, args, REPO_ROOT)
    df = run_extraction(spec, df_num, repo_root=REPO_ROOT, jobs=args.jobs)
    per_conf = _resolve(_output_name(spec, "per_conformer_csv"))
    df.to_csv(per_conf, index=False)
    print(f"wrote {per_conf}  rows={len(df)}  columns={df.shape[1]}")
    _report_empty_columns(df, spec)
    agg = aggregate_conformers(df, spec)
    if _use_filter(spec, args):
        agg, _ = apply_filter(agg, spec)
    out = _resolve(_aggregated_name(spec))
    agg.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(agg)}  columns={agg.shape[1]}")
    _report_check(df, df_num, spec)
    return 0


def cmd_to_schema_b(args) -> int:
    from desc_utils.extract.descripytor import to_schema_b
    from desc_utils.numbering import read_numbering_table

    spec = _load(args)
    df_num = read_numbering_table(_resolve(args.numbering) if args.numbering else _existing_output(spec, "numbering_csv"), spec)
    row = df_num.iloc[args.row]
    print(json.dumps(to_schema_b(spec.descripytor, row), indent=2))
    return 0


def cmd_condense(args) -> int:
    """Write ``<conf>/<data.condensed_file>`` for every conformer under ``data.root``.

    The same producer runs on the cluster without this environment:
    ``PYTHONPATH=<repo> python -m desc_utils.condense --root <library>``.
    """
    from desc_utils.condense import CONDENSED_FILENAME, condense_tree, iter_conformer_dirs

    spec = _load(args)
    name = spec.data.get("condensed_file") or CONDENSED_FILENAME
    dirs = iter_conformer_dirs(_resolve(spec.data["root"]), spec.data.get("conformer_glob", "conf_*"))
    if not dirs:
        print(f"no conformer folders under {spec.data['root']}", file=sys.stderr)
        return 1
    full = "all" if args.nbo_full.strip() == "all" else tuple(int(x) for x in args.nbo_full.split(",") if x.strip())
    summary = condense_tree(dirs, name=name, overwrite=args.overwrite, jobs=args.jobs, nbo_full_blocks=full)
    return 1 if summary["failed"] else 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m desc_utils.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--spec", required=True, help="spec YAML/JSON file")

    def jobs(sp):
        sp.add_argument("-j", "--jobs", type=int, default=1,
                        help="processes for the per-conformer work (default 1; 0 = every core)")

    sp = sub.add_parser("validate", help="check a spec file"); common(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("number", help="build the atom numbering table"); common(sp)
    sp.add_argument("--data", help="override data.root")
    sp.add_argument("--out", help="output CSV (default: output.numbering_csv)")
    sp.add_argument("--backend", choices=["smarts", "manual", "first_n", "molalign"], help="override numbering.backend")
    sp.add_argument("--csv", help="numbering CSV for --backend manual")
    sp.add_argument("--strict", action="store_true", help="exit non-zero unless every row has status ok")
    sp.add_argument("--regenerate-sdf", action="store_true", help="rebuild .sdf files from the xyz even if present")
    jobs(sp)
    sp.set_defaults(func=cmd_number)

    sp = sub.add_parser("extract", help="compute per-conformer descriptors"); common(sp)
    sp.add_argument("--numbering", help="numbering CSV (default: output.numbering_csv of today, else the most recent one)")
    sp.add_argument("--families", help="comma list, e.g. orca.xyz,orca.charges,orca.nbo,orca.redox,descripytor")
    sp.add_argument("--out", help="output CSV (default: output.per_conformer_csv)")
    sp.add_argument("--overwrite-xyz-json", action="store_true", help="recompute morfeus descriptors instead of reading xyz.json")
    jobs(sp)
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("aggregate", help="Boltzmann-aggregate conformers and filter"); common(sp)
    sp.add_argument("--per-conformer", help="per-conformer CSV (default: output.per_conformer_csv of today, else the most recent one)")
    sp.add_argument("--out", help="output CSV (default: output.aggregated_csv)")
    sp.add_argument("--filter", action="store_true", help="apply the noise filter of the spec's `filter` block (off by default)")
    sp.add_argument("--no-filter", action="store_true", help="never filter, even if the spec enables it")
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("all", help="number -> extract -> aggregate -> filter"); common(sp)
    sp.add_argument("--data", help="override data.root")
    sp.add_argument("--numbering", help="use an existing numbering CSV instead of rebuilding it")
    sp.add_argument("--filter", action="store_true", help="apply the noise filter of the spec's `filter` block (off by default)")
    sp.add_argument("--no-filter", action="store_true", help="never filter, even if the spec enables it")
    jobs(sp)
    sp.set_defaults(func=cmd_all)

    sp = sub.add_parser("to-schema-b", help="print the DescriPyTor JSON for one conformer"); common(sp)
    sp.add_argument("--numbering", help="numbering CSV (default: output.numbering_csv of today, else the most recent one)")
    sp.add_argument("--row", type=int, default=0)
    sp.set_defaults(func=cmd_to_schema_b)

    sp = sub.add_parser("condense", help="write <conf>/orca_condensed.json from the raw ORCA files"); common(sp)
    sp.add_argument("--data", help="override data.root")
    sp.add_argument("--overwrite", action="store_true", help="rewrite existing condensed files")
    sp.add_argument("--jobs", type=int, default=1, help="parallel workers")
    sp.add_argument("--nbo-full", default="0", help="NBO block indices stored with every table, or 'all' (default 0)")
    sp.set_defaults(func=cmd_condense)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
