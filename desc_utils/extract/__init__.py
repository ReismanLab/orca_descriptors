"""Per-conformer descriptor extraction driven by a spec and a numbering table.

Two engines:

* ``orca``        ORCA property JSON / NBO output / morfeus on the optimised geometry
                  (``desc_utils.extract.orca``, wrapping the functions in ``data_extraction``)
* ``descripytor`` DescriPyTor ``Molecules`` families on the condensed JSON (or legacy feather) file
                  (``desc_utils.extract.descripytor``)

:func:`run_extraction` joins their outputs on the conformer key ``path``.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from desc_utils.spec import Spec

ALL_FAMILIES = ("orca.xyz", "orca.charges", "orca.nbo", "orca.redox", "descripytor")


def _meta_columns(spec: Spec, df_num: pd.DataFrame) -> list[str]:
    keep = ["path", "xyz_name", spec.data["molecule_id_column"], spec.numbering["energy"]["column"],
            "smiles", "ligand_type", "ligand_class", "numbering_status"] + spec.labels
    return [c for c in dict.fromkeys(keep) if c in df_num.columns]


def job_status_columns(paths: Iterable[str], spec: Spec, repo_root: Path) -> pd.DataFrame:
    """Two bookkeeping columns per conformer, so nothing downstream mistakes a dead job for a small library.

    ``orca_job_status``: ``completed`` / ``running`` / ``failed`` as recorded in the condensed file
    (``unknown`` for files condensed before the status existed, ``missing`` when the folder has neither a
    condensed file nor ``calcParam.out``). ``n_conformer_folders``: how many ``conformer_glob`` folders the
    molecule folder holds on disk, finished or not; NaN for the flat layout. The aggregation compares
    the two to flag molecules whose conformer set is incomplete.
    """
    from desc_utils.condense import conformer_job_status

    condensed = spec.data.get("condensed_file")
    glob = spec.data["conformer_glob"]
    per_molecule = spec.data["layout"] != "flat"
    folder_counts: dict[Path, int] = {}
    statuses, counts = [], []
    for key in paths:
        conf_dir = Path(key) if Path(key).is_absolute() else Path(repo_root) / key
        statuses.append(conformer_job_status(conf_dir, condensed) if conf_dir.is_dir() else "missing")
        if per_molecule and conf_dir.is_dir():
            mol = conf_dir.parent
            if mol not in folder_counts:
                folder_counts[mol] = sum(1 for p in mol.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob))
            counts.append(folder_counts[mol])
        else:
            counts.append(np.nan)
    return pd.DataFrame({"path": list(paths), "orca_job_status": statuses, "n_conformer_folders": counts})


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


def run_extraction(spec: Spec, df_num: pd.DataFrame, repo_root: Path, families: Iterable[str] | None = None,
                   overwrite_xyz_json: bool = False, progress: bool = True) -> pd.DataFrame:
    """Compute every requested family and return one wide row per conformer.

    ``families`` defaults to every family present in the spec. The numbering table's
    metadata columns (path, molecule id, energy, smiles, atom indices, ...) are carried
    into the output so the aggregation stage needs nothing else, together with the ORCA job
    status of every conformer (:func:`job_status_columns`).
    """
    wanted = list(families) if families is not None else list(ALL_FAMILIES)
    unknown = [f for f in wanted if f not in ALL_FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; choose from {ALL_FAMILIES}")

    from desc_utils.data_extraction import conf_key

    df_num = df_num.reset_index(drop=True).copy()
    df_num["path"] = df_num["path"].map(lambda p: conf_key(p, repo_root))  # one join key for every engine
    out = df_num[_meta_columns(spec, df_num)].copy()
    out = out.merge(job_status_columns(list(out["path"]), spec, repo_root), on="path", how="left")

    orca_wanted = [f.split(".", 1)[1] for f in wanted if f.startswith("orca.")]
    if spec.orca and orca_wanted:
        from desc_utils.extract.orca import run_orca

        orca_df = run_orca(df_num, spec, repo_root, families=[f for f in orca_wanted if f in spec.orca],
                           overwrite_xyz_json=overwrite_xyz_json, progress=progress)
        out = out.merge(orca_df, on="path", how="left", suffixes=("", "_orca"))

    if spec.descripytor and "descripytor" in wanted:
        from desc_utils.extract.descripytor import extract_descripytor

        dp_df = extract_descripytor(df_num, spec.descripytor, repo_root, progress=progress,
                                    condensed_file=spec.data.get("condensed_file"))
        out = out.merge(dp_df, on="path", how="left", suffixes=("", "_descripytor"))

    # a column produced by two engines under the same name is kept once (first wins)
    return out.loc[:, ~out.columns.duplicated()]
