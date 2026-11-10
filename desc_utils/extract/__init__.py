"""Per-conformer descriptor extraction driven by a spec and a numbering table.

Two engines:

* ``orca``        ORCA property JSON / NBO output / morfeus on the optimised geometry
                  (``desc_utils.extract.orca``, wrapping the functions in ``data_extraction``)
* ``descripytor`` DescriPyTor ``Molecules`` families on the condensed JSON file
                  (``desc_utils.extract.descripytor``)

:func:`run_extraction` joins their outputs on the conformer key ``path``.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from desc_utils.parallel import parallel_map, resolve_jobs
from desc_utils.spec import Spec

ALL_FAMILIES = ("orca.xyz", "orca.charges", "orca.nbo", "orca.redox", "descripytor")
#: Families that read ORCA output (property JSON, NBO, condensed file). ``orca.xyz`` is not one of them: it
#: only needs the geometry file, so a library optimised with UMA can be described without any ORCA job.
ORCA_OUTPUT_FAMILIES = ("orca.charges", "orca.nbo", "orca.redox", "descripytor")


def needs_orca_output(spec: Spec, families: Iterable[str] | None = None) -> bool:
    """Whether the requested families (default: every family of the spec) read ORCA output at all.

    When they do not (geometry-only run on UMA structures), the ORCA job bookkeeping is skipped: no
    ``orca_job_status`` column, no "ORCA job missing" entry in the check report, no completeness flag.
    """
    present = set()
    if spec.orca:
        present.update(f"orca.{k}" for k in spec.orca)
    if spec.descripytor:
        present.add("descripytor")
    wanted = set(families) if families is not None else present
    return bool(present & wanted & set(ORCA_OUTPUT_FAMILIES))


def _meta_columns(spec: Spec, df_num: pd.DataFrame) -> list[str]:
    keep = ["path", "xyz_name", spec.data["molecule_id_column"], spec.numbering["energy"]["column"],
            "smiles", "smiles_noH", "ligand_type", "ligand_class", "numbering_status"] + spec.labels
    return [c for c in dict.fromkeys(keep) if c in df_num.columns]


#: per-conformer ORCA cost columns (``job.timings`` of the condensed file); ``orca_step_<step>_s`` joins them
COST_COLUMNS = ("orca_run_time_s", "orca_nprocs", "orca_cpu_hours")


def cost_columns_of(job: dict) -> dict:
    """The cost columns of one conformer from its ``job`` entry (empty when the entry has no timings).

    ``orca_run_time_s`` is ORCA's total run time, ``orca_nprocs`` its ``%pal`` setting, ``orca_cpu_hours``
    their product in hours (the core-hours the job cost, assuming the allocation matched ``nprocs``),
    ``orca_step_<step>_s`` the wall time of every compound step (``optimization``, ``frequency``, ...).
    """
    timings = (job or {}).get("timings") or {}
    if not timings:
        return {}
    run, nprocs = timings.get("total_run_time_s"), timings.get("nprocs")
    out = {"orca_run_time_s": run, "orca_nprocs": nprocs,
           "orca_cpu_hours": round(run * nprocs / 3600, 5) if run is not None and nprocs else None}
    for step, st in (timings.get("steps") or {}).items():
        out[f"orca_step_{step}_s"] = st.get("wall_s")
    return out


def job_status_columns(paths: Iterable[str], spec: Spec, repo_root: Path) -> pd.DataFrame:
    """Bookkeeping columns per conformer, so nothing downstream mistakes a dead job for a small library,
    and what the job cost, so accuracy can be weighed against price.

    ``orca_job_status``: ``completed`` / ``running`` / ``failed`` as recorded in the condensed file
    (``unknown`` for files condensed before the status existed, ``missing`` when the folder has neither a
    condensed file nor ``calcParam.out``). ``n_conformer_folders``: how many ``conformer_glob`` folders the
    molecule folder holds on disk, finished or not; NaN for the flat layout. The aggregation compares
    the two to flag molecules whose conformer set is incomplete. Then the cost columns of
    :func:`cost_columns_of` (NaN when the condensed file predates the timings).
    """
    from desc_utils.condense import conformer_job_entry

    condensed = spec.data.get("condensed_file")
    glob = spec.data["conformer_glob"]
    per_molecule = spec.data["layout"] != "flat"
    folder_counts: dict[Path, int] = {}
    rows = []
    for key in paths:
        conf_dir = Path(key) if Path(key).is_absolute() else Path(repo_root) / key
        job = conformer_job_entry(conf_dir, condensed) if conf_dir.is_dir() else {"status": "missing"}
        row = {"path": key, "orca_job_status": job["status"], "n_conformer_folders": np.nan}
        if per_molecule and conf_dir.is_dir():
            mol = conf_dir.parent
            if mol not in folder_counts:
                folder_counts[mol] = sum(1 for p in mol.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob))
            row["n_conformer_folders"] = folder_counts[mol]
        rows.append(row | cost_columns_of(job))
    df = pd.DataFrame(rows, columns=["path", "orca_job_status", "n_conformer_folders"] if not rows else None)
    for c in COST_COLUMNS:   # always present, so a table without timings has the same shape
        if c not in df.columns:
            df[c] = np.nan
    return df


#: ``orca_job_status`` values that do not call for a look: the job finished, or the file predates the status
_JOB_OK = ("completed", "unknown")
CHECK_COLUMNS = ("path", "reason", "numbering_status", "numbering_notes", "orca_job_status")


def conformers_to_check(df: pd.DataFrame, df_num: pd.DataFrame | None = None) -> pd.DataFrame:
    """The conformers of a per-conformer table a human should look at, with why.

    A row is listed when its ``numbering_status`` is not ``ok`` (its descriptors are missing from the
    table) or when its ``orca_job_status`` says the ORCA job is running, failed or absent (its late
    descriptors are NaN). ``df_num`` (the numbering table) supplies ``numbering_notes`` when the
    per-conformer table does not carry them. Empty when nothing needs a look.
    """
    out = df.copy()
    if "numbering_notes" not in out.columns and df_num is not None and "numbering_notes" in df_num.columns:
        notes = df_num[["path", "numbering_notes"]].copy()
        notes["path"] = notes["path"].map(lambda p: str(p).replace("\\", "/").rstrip("/"))
        out["_key"] = out["path"].map(lambda p: str(p).replace("\\", "/").rstrip("/"))
        out = out.merge(notes.rename(columns={"path": "_key"}).drop_duplicates("_key"), on="_key", how="left").drop(columns="_key")
    reasons = []
    for _, r in out.iterrows():
        why = []
        ns = r.get("numbering_status")
        if isinstance(ns, str) and ns != "ok":
            why.append(f"numbering {ns}")
        js = r.get("orca_job_status")
        if isinstance(js, str) and js not in _JOB_OK:
            why.append(f"ORCA job {js}")
        reasons.append("; ".join(why))
    out["reason"] = reasons
    out = out[out["reason"] != ""]
    cols = [c for c in CHECK_COLUMNS if c in out.columns]
    return out[cols].reset_index(drop=True)


# Worker-side state for ``run_extraction(jobs > 1)``: set once per process by the pool initializer.
_WORKER: dict = {}


def _init_worker(spec: Spec, repo_root: Path, orca_fams: list[str], want_dp: bool, overwrite_xyz_json: bool) -> None:
    _WORKER.update(spec=spec, repo_root=repo_root, orca_fams=orca_fams, want_dp=want_dp,
                   overwrite_xyz_json=overwrite_xyz_json)
    if want_dp:  # import DescriPyTor once per worker rather than on the first conformer
        from desc_utils.extract.descripytor import _import_molecules

        _import_molecules(repo_root)


def _work(rows: pd.DataFrame) -> dict[str, list[dict]]:
    """Every engine on a slice of the numbering table, as records per family.

    NBO is left raw (post-processed library-wide by the caller). Records rather than frames: the raw
    NBO rows of a library span tens of thousands of distinct columns, and turning them into row dicts
    here, in parallel, is what lets the parent build the table as cheaply as the serial path does.
    """
    w = _WORKER
    frames: dict[str, pd.DataFrame] = {}
    if w["orca_fams"]:
        from desc_utils.extract.orca import orca_family_frames

        frames.update(orca_family_frames(rows, w["spec"], w["repo_root"], families=w["orca_fams"],
                                         overwrite_xyz_json=w["overwrite_xyz_json"], progress=False, postprocess=False))
    if w["want_dp"]:
        from desc_utils.extract.descripytor import extract_descripytor

        frames["descripytor"] = extract_descripytor(rows, w["spec"].descripytor, w["repo_root"], progress=False,
                                                    condensed_file=w["spec"].data.get("condensed_file"))
    return {k: f.to_dict("records") for k, f in frames.items()}


def _gather(records: list[list[dict]]) -> pd.DataFrame:
    """Stack per-conformer records the way the serial extractors build their frames: from a dict of row dicts.

    ``pd.concat`` of one-row frames would keep e.g. a lone ``None`` as an object column where the
    serial path infers a float NaN column; the same constructor gives the same dtypes and NaN pattern.
    """
    rows = [rec for recs in records for rec in recs]
    return pd.DataFrame.from_dict(dict(enumerate(rows)), orient="index")


#: DescriPyTor columns that repeat an ORCA-side column (morfeus geometry or the NBO parser) under another
#: name. On LNiH2_library_002 (349 conformers) every pair was identical to the last digit, so the
#: DescriPyTor copy is dropped. DescriPyTor Sterimol (``L_``/``B1_``/``B5_``) is NOT listed: it uses other
#: radii and cuts the substituent at the bond, so it differs from morfeus ``Sterimol_*`` (r 0.995-0.99999,
#: offsets of ~0.1 A on B and ~0.5 A on L; on Ni-H the two measure different fragments).
_DUPLICATE_PATTERNS = (
    (re.compile(r"dist_([^-]+)-([^-]+)"), ("Bond_{0}_{1}", "Bond_{1}_{0}")),
    (re.compile(r"angle_([^-]+)-([^-]+)-([^-]+)"), ("Angle_{0}_{1}_{2}", "Angle_{2}_{1}_{0}")),
    (re.compile(r"charge_NBO_([^-]+)"), ("npa_{0}_charge",)),
)


#: Direction-free families: (column prefix, label separator, number of atoms, sign of the reversed column).
#: ``A-B`` and ``B-A`` (``A-B-C`` and ``C-B-A``, ...) hold the same value, or its negative for the signed
#: NBO charge difference. Sterimol (morfeus and DescriPyTor) is direction dependent and is never listed.
_REVERSIBLE = (
    ("Bond_", "_", 2, 1), ("dist_", "-", 2, 1),
    ("vib_freq_", "-", 2, 1), ("vib_amp_", "-", 2, 1),
    ("charge_NBO_diff_", "-", 2, -1),
    ("Angle_", "_", 3, 1), ("angle_", "-", 3, 1),
    ("Dihed_", "_", 4, 1),
)
_AGG_SUFFIX = re.compile(r"(.+?)(_(?:min|max|lowE|boltz))")


def _parse_reversible(col: str, labels: Iterable[str]):
    """``(prefix, sep, terms, sign)`` when ``col`` is a direction-free descriptor of known labels, else None."""
    labels = set(labels)
    for prefix, sep, n, sign in _REVERSIBLE:
        if col.startswith(prefix):
            terms = tuple(col[len(prefix):].split(sep))
            if len(terms) == n and all(t in labels for t in terms):
                return prefix, sep, terms, sign
    return None


def _reversed_name(col: str, labels: Iterable[str]) -> tuple[str, int] | None:
    hit = _parse_reversible(col, labels)
    if hit is None:
        return None
    prefix, sep, terms, sign = hit
    return prefix + sep.join(terms[::-1]), sign


def duplicate_twin(col: str, columns: Iterable[str], labels: Iterable[str] = ()) -> tuple[str, int] | None:
    """The column among ``columns`` that holds the same values as ``col``, and the sign relating them.

    Two kinds of twin: the morfeus/NBO column a DescriPyTor column repeats (:data:`_DUPLICATE_PATTERNS`), and,
    when ``labels`` are given, the reversed-order column of a direction-free descriptor (:data:`_REVERSIBLE`).
    An aggregation suffix is carried over (``dist_Ni-H1_boltz`` -> ``Bond_Ni_H1_boltz``); for a negated twin
    ``_min`` and ``_max`` swap.
    """
    columns = set(columns)
    candidates = [(str(col), "")]
    m = _AGG_SUFFIX.fullmatch(str(col))
    if m:
        candidates.append((m.group(1), m.group(2)))
    for b, s in candidates:
        names: list[tuple[str, int]] = []
        for pattern, templates in _DUPLICATE_PATTERNS:
            hit = pattern.fullmatch(b)
            if hit:
                names += [(t.format(*hit.groups()), 1) for t in templates]
                break
        rev = _reversed_name(b, labels)
        if rev is not None:
            names.append(rev)
            # the DescriPyTor copy of a reversed pair: dist_B-A -> Bond_A_B, ...
            for pattern, templates in _DUPLICATE_PATTERNS:
                hit = pattern.fullmatch(rev[0])
                if hit:
                    names += [(t.format(*hit.groups()), rev[1]) for t in templates]
                    break
        for name, sign in names:
            suffix = {"_min": "_max", "_max": "_min"}.get(s, s) if sign < 0 else s
            if name + suffix in columns and name + suffix != col:
                return name + suffix, sign
    return None


def _same(a: pd.Series, b: pd.Series, sign: int, rtol: float, atol: float) -> bool:
    a, b = pd.to_numeric(a, errors="coerce"), sign * pd.to_numeric(b, errors="coerce")
    ok = a.notna()
    return bool((ok == b.notna()).all() and np.isclose(a[ok], b[ok], rtol=rtol, atol=atol).all())


def _ascending_share(df: pd.DataFrame, terms: tuple[str, ...]) -> float:
    """Share of rows whose first atom has a lower index than the last one (NaN when indices are unknown)."""
    if terms[0] not in df.columns or terms[-1] not in df.columns:
        return float("nan")
    first, last = pd.to_numeric(df[terms[0]], errors="coerce"), pd.to_numeric(df[terms[-1]], errors="coerce")
    ok = first.notna() & last.notna()
    return float((first[ok] < last[ok]).mean()) if ok.any() else float("nan")


def drop_duplicate_descriptors(df: pd.DataFrame, labels: Iterable[str] = (), rtol: float = 1e-6, atol: float = 1e-8,
                               quiet: bool = False) -> pd.DataFrame:
    """Drop descriptor columns that repeat another column of the table.

    1. A DescriPyTor column whose morfeus/NBO twin (:data:`_DUPLICATE_PATTERNS`) holds the same values.
    2. Of a direction-free descriptor written both ways (``Bond_C1_C2`` and ``Bond_C2_C1``, see
       :data:`_REVERSIBLE`), the column whose atoms are NOT in ascending index order. The index columns of
       ``labels`` decide; when the order differs between molecules, the orientation that is ascending in
       most rows is kept.

    A pair is only collapsed when both columns share the NaN pattern and agree to ``rtol``/``atol``
    (``numpy.isclose``; older DescriPyTor runs rounded angles at ~1e-7 degrees). A pair that disagrees is
    kept and reported, since that points at a numbering or parsing problem.
    """
    labels = list(labels)
    drop: list[tuple[str, str]] = []
    differ: list[tuple[str, str]] = []
    mixed: list[str] = []

    # 1. DescriPyTor copies of morfeus / NBO columns
    for col in df.columns:
        twin = duplicate_twin(col, df.columns)
        if twin is not None:
            (drop if _same(df[col], df[twin[0]], twin[1], rtol, atol) else differ).append((col, twin[0]))
    df = df.drop(columns=[a for a, _ in drop])

    # 2. reversed orientations of direction-free descriptors
    n_cross, seen = len(drop), set()
    for col in list(df.columns):
        rev = _reversed_name(col, labels)
        if rev is None or rev[0] not in df.columns or col in seen:
            continue
        other, sign = rev
        seen.update((col, other))
        if not _same(df[col], df[other], sign, rtol, atol):
            differ.append((col, other))
            continue
        terms = _parse_reversible(col, labels)[2]
        share = _ascending_share(df, terms)        # share of rows where ``col`` is the ascending one
        if share == share and 0 < share < 1:
            mixed.append(col if share >= 0.5 else other)
        keep, lose = (other, col) if share == share and share < 0.5 else (col, other)
        drop.append((lose, keep))
    df = df.drop(columns=[a for a, _ in drop[n_cross:]])

    if not quiet:
        if differ:
            print(f"[extract] {len(differ)} duplicate-looking column pairs differ, both kept: "
                  + ", ".join(f"{a} vs {b}" for a, b in differ[:5]))
        if n_cross:
            print(f"[extract] dropped {n_cross} DescriPyTor columns identical to morfeus/NBO columns "
                  f"(e.g. {drop[0][0]} = {drop[0][1]})")
        if len(drop) > n_cross:
            print(f"[extract] dropped {len(drop) - n_cross} reversed-order columns of direction-free descriptors "
                  f"(e.g. {drop[n_cross][0]}, kept {drop[n_cross][1]})")
        if mixed:
            print(f"[extract] atom order differs between molecules for {len(mixed)} kept columns "
                  f"(majority orientation kept): {', '.join(mixed[:5])}")
    return df


def _run_parallel(spec: Spec, df_num: pd.DataFrame, out: pd.DataFrame, repo_root: Path, orca_fams: list[str],
                  want_dp: bool, overwrite_xyz_json: bool, jobs: int, progress: bool) -> pd.DataFrame:
    """One task per conformer over ``jobs`` processes, then the same joins as the serial path."""
    from desc_utils.extract.orca import merge_orca_frames, postprocess_nbo

    engines = orca_fams + (["descripytor"] if want_dp else [])
    if progress:
        print(f"[extract] {len(df_num)} conformers over {jobs} processes: {engines}")
    tasks = [df_num.iloc[[i]] for i in range(len(df_num))]
    parts = parallel_map(_work, tasks, jobs, initializer=_init_worker,
                         initargs=(spec, repo_root, orca_fams, want_dp, overwrite_xyz_json),
                         progress=progress, desc="extract")
    keys = list(dict.fromkeys(k for part in parts for k in part))
    frames = {k: _gather([part[k] for part in parts if k in part]) for k in keys}
    if "nbo" in frames:
        frames["nbo"] = postprocess_nbo(frames["nbo"], spec.orca["nbo"])
    if orca_fams:
        orca_df = merge_orca_frames({k: v for k, v in frames.items() if k != "descripytor"}, df_num, repo_root)
        out = out.merge(orca_df, on="path", how="left", suffixes=("", "_orca"))
    if want_dp:
        from desc_utils.extract.descripytor import order_columns

        out = out.merge(order_columns(frames["descripytor"], spec.descripytor), on="path", how="left",
                        suffixes=("", "_descripytor"))
    return out


def run_extraction(spec: Spec, df_num: pd.DataFrame, repo_root: Path, families: Iterable[str] | None = None,
                   overwrite_xyz_json: bool = False, progress: bool = True, jobs: int | None = 1) -> pd.DataFrame:
    """Compute every requested family and return one wide row per conformer.

    ``families`` defaults to every family present in the spec. The numbering table's
    metadata columns (path, molecule id, energy, smiles, atom indices, ...) are carried
    into the output so the aggregation stage needs nothing else, together with the ORCA job
    status of every conformer (:func:`job_status_columns`) whenever a requested family reads ORCA
    output (:func:`needs_orca_output`); a geometry-only run carries no job status.

    ``jobs > 1`` (``0`` = every core) spreads the conformers over that many processes, each running
    every engine on its conformer; the library-wide NBO rank consolidation is then applied once to the
    gathered rows, so the table is identical to a serial run.
    """
    wanted = list(families) if families is not None else list(ALL_FAMILIES)
    unknown = [f for f in wanted if f not in ALL_FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; choose from {ALL_FAMILIES}")

    from desc_utils.data_extraction import conf_key
    from desc_utils.numbering import usable_rows

    df_num = df_num.reset_index(drop=True).copy()
    df_num["path"] = df_num["path"].map(lambda p: conf_key(p, repo_root))  # one join key for every engine
    out = df_num[_meta_columns(spec, df_num)].copy()
    if needs_orca_output(spec, wanted):
        out = out.merge(job_status_columns(list(out["path"]), spec, repo_root), on="path", how="left")

    # Every conformer keeps its metadata row (so the check report can list it), but only the
    # cleanly numbered ones are handed to the engines: a failed row has <NA> where an atom index
    # should be and would crash the first ``int()``. Its descriptor columns stay NaN.
    df_num = usable_rows(df_num, spec, quiet=True)
    if len(df_num) < len(out):
        print(f"[extract] {len(out) - len(df_num)} of {len(out)} conformers not numbered cleanly: "
              "descriptors left NaN, see numbering_status")
    if df_num.empty:
        print("[extract] no conformer is numbered cleanly; nothing to extract")
        return out

    orca_wanted = [f.split(".", 1)[1] for f in wanted if f.startswith("orca.")]
    orca_fams = [f for f in orca_wanted if f in spec.orca] if spec.orca else []
    want_dp = bool(spec.descripytor) and "descripytor" in wanted

    n_jobs = resolve_jobs(jobs)
    if n_jobs > 1 and len(df_num) > 1 and (orca_fams or want_dp):
        out = _run_parallel(spec, df_num, out, repo_root, orca_fams, want_dp, overwrite_xyz_json, n_jobs, progress)
        return drop_duplicate_descriptors(out.loc[:, ~out.columns.duplicated()], spec.labels, quiet=not progress)

    if orca_fams:
        from desc_utils.extract.orca import run_orca

        orca_df = run_orca(df_num, spec, repo_root, families=orca_fams,
                           overwrite_xyz_json=overwrite_xyz_json, progress=progress)
        out = out.merge(orca_df, on="path", how="left", suffixes=("", "_orca"))

    if want_dp:
        from desc_utils.extract.descripytor import extract_descripytor

        dp_df = extract_descripytor(df_num, spec.descripytor, repo_root, progress=progress,
                                    condensed_file=spec.data.get("condensed_file"))
        out = out.merge(dp_df, on="path", how="left", suffixes=("", "_descripytor"))

    # a column produced by two engines under the same name is kept once (first wins); a DescriPyTor column
    # that repeats a morfeus/NBO column under another name is dropped when the values agree
    return drop_duplicate_descriptors(out.loc[:, ~out.columns.duplicated()], spec.labels, quiet=not progress)
