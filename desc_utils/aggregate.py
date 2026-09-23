"""Boltzmann aggregation of per-conformer descriptors and the post-aggregation filter.

``aggregate_conformers`` is an exact port of the aggregation cell that used to live in
``make_descriptor_descripytor.ipynb``: for every molecule (group) and every numeric
column it writes ``<col>_min``, ``<col>_max``, ``<col>_lowE`` (value at the lowest-energy
conformer) and ``<col>_boltz`` (Boltzmann-weighted mean at ``temperature_K``).
"""
from __future__ import annotations

import inspect
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from desc_utils.spec import Spec

_CONF_SUFFIX = re.compile(r"/conf_\d+/?$")


def boltzmann_weights(energies: pd.Series, temperature_K: float = 298.15, kB: float = 3.166811563e-6) -> pd.Series:
    """Normalised Boltzmann weights from energies in Hartree; NaN energies get NaN weight."""
    e = pd.to_numeric(energies, errors="coerce")
    if e.notna().sum() == 0:
        return pd.Series(np.nan, index=energies.index)
    w = np.exp(-((e - e.min()) / (kB * temperature_K)))
    return w / np.nansum(w)


def molecule_folder(path: str) -> str:
    """``DATA/lib/<id>/conf_3`` -> ``DATA/lib/<id>`` (also accepts trailing slash / feather file)."""
    p = str(path).replace("\\", "/")
    if p.endswith(".feather"):
        p = p.rsplit("/", 1)[0]
    return _CONF_SUFFIX.sub("", p.rstrip("/"))


#: per-conformer bookkeeping columns that are summarised, never aggregated as descriptors
JOB_COLUMNS = ("n_conformer_folders",)
#: the summary columns ``completeness`` adds to every aggregated row
COMPLETENESS_COLUMNS = ("n_conformers_completed", "n_conformer_folders", "all_conformers_completed")


def completeness(g: pd.DataFrame) -> dict:
    """Conformer-set completeness of one molecule from its per-conformer rows (see :func:`aggregate_frame`)."""
    status = g["orca_job_status"].astype(str)
    n_completed = int((status == "completed").sum())
    folders = pd.to_numeric(g["n_conformer_folders"], errors="coerce") if "n_conformer_folders" in g else pd.Series(dtype=float)
    n_folders = int(folders.max()) if folders.notna().any() else int(len(g))
    n_folders = max(n_folders, len(g))
    if (status == "unknown").any():
        flag = pd.NA
    else:
        flag = bool(n_completed == n_folders)
    return {"n_conformers_completed": n_completed, "n_conformer_folders": n_folders, "all_conformers_completed": flag}


def aggregate_frame(
    df: pd.DataFrame,
    group_by: str,
    energy_column: str,
    temperature_K: float = 298.15,
    kB: float = 3.166811563e-6,
    stats: Sequence[str] = ("min", "max", "lowE", "boltz"),
    carry_first: Iterable[str] = (),
    exclude_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Aggregate every numeric column of ``df`` per ``group_by`` value.

    Groups without a single valid energy are skipped, as in the original notebook cell.
    ``exclude_columns`` are numeric columns that must not be aggregated (label index
    columns, bookkeeping columns).

    When the per-conformer table carries ``orca_job_status`` (see
    ``desc_utils.extract.job_status_columns``) every row also gets ``n_conformers_completed``,
    ``n_conformer_folders`` and the flag ``all_conformers_completed``: True when every conformer
    folder of the molecule holds a completed ORCA job, False when at least one is running, failed
    or absent, and empty (NA) when a status is ``unknown`` (files condensed before the status
    was recorded) so that an old library is not reported as broken.
    """
    if energy_column not in df.columns:
        raise KeyError(f"energy column {energy_column!r} not found; available: {list(df.columns)[:10]}...")
    exclude = set(exclude_columns) | {group_by} | set(JOB_COLUMNS)
    has_status = "orca_job_status" in df.columns
    # Numeric columns with at least one value. Descriptors that are empty for every conformer (e.g.
    # Enthalpy without a thermochemistry step) are dropped here so the library is the same whether the
    # per-conformer table comes from memory (object column of None) or from a CSV (float column of NaN).
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude and df[c].notna().any()]
    carry_first = [c for c in carry_first if c in df.columns]

    rows = []
    for key, g in df.groupby(group_by, dropna=True, sort=True):
        e = pd.to_numeric(g[energy_column], errors="coerce")
        if e.notna().sum() == 0:
            continue
        idx_low = e.idxmin()
        w = boltzmann_weights(e, temperature_K, kB)
        out: dict = {group_by: key, "n_conformers": int(len(g))}
        if has_status:
            out.update(completeness(g))
        for c in carry_first:
            out[c] = g[c].iloc[0]
        for col in numeric_cols:
            v = pd.to_numeric(g[col], errors="coerce")
            if "min" in stats:
                out[f"{col}_min"] = v.min()
            if "max" in stats:
                out[f"{col}_max"] = v.max()
            if "lowE" in stats:
                out[f"{col}_lowE"] = g.loc[idx_low, col]
            if "boltz" in stats:
                valid = v.notna() & w.notna()
                out[f"{col}_boltz"] = np.average(v[valid], weights=w[valid]) if valid.any() else np.nan
        rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_conformers(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Spec-driven wrapper around :func:`aggregate_frame`.

    Adds ``inchi_folder`` (molecule folder derived from ``path``) when a ``path`` column is
    present, excludes the atom-label index columns from aggregation, and carries the
    molecule id plus ``aggregate.carry_first`` columns through.
    """
    cfg = spec.aggregate
    df = df.copy()
    group_by = cfg["group_by"]
    if "path" in df.columns:
        df["inchi_folder"] = df["path"].map(molecule_folder)
        if group_by not in df.columns:
            df[group_by] = df["inchi_folder"].str.rsplit("/", n=1).str[-1]
    if group_by not in df.columns:
        raise KeyError(f"aggregate.group_by column {group_by!r} not found in the per-conformer table")
    carry = list(cfg.get("carry_first", []))
    if "inchi_folder" in df.columns and "inchi_folder" not in carry:
        carry.insert(0, "inchi_folder")
    exclude = set(cfg.get("exclude_columns", [])) | set(spec.labels) | {"n_loaded"}
    agg = aggregate_frame(
        df, group_by=group_by, energy_column=cfg["energy_column"], temperature_K=cfg["temperature_K"],
        kB=cfg["kB_hartree_per_K"], stats=cfg["stats"], carry_first=carry, exclude_columns=exclude,
    )
    # Match the historical column order: inchi_folder, inchi, smiles, ligand_class, then descriptors.
    front = [c for c in ["inchi_folder", group_by] + [c for c in carry if c != "inchi_folder"] if c in agg.columns]
    rest = [c for c in agg.columns if c not in front]
    return agg[front + rest]


def filter_kwargs_from_spec(spec: Spec) -> dict:
    """Translate the spec ``filter`` section into ``filter_numeric_descriptors`` kwargs."""
    from desc_utils.data_extraction import filter_numeric_descriptors

    accepted = set(inspect.signature(filter_numeric_descriptors).parameters)
    kwargs = {k: v for k, v in spec.filter.items() if k in accepted and k != "df"}
    kwargs.setdefault("atom_labels", spec.labels)
    unknown = [k for k in spec.filter if k not in accepted and k != "enabled"]
    if unknown:
        raise KeyError(f"filter: unknown option(s) {unknown}; filter_numeric_descriptors accepts {sorted(accepted)}")
    return kwargs


def apply_filter(agg: pd.DataFrame, spec: Spec) -> tuple[pd.DataFrame, dict]:
    """Run ``filter_numeric_descriptors`` and re-attach the non-numeric metadata columns."""
    from desc_utils.data_extraction import filter_numeric_descriptors

    filtered, report = filter_numeric_descriptors(agg, **filter_kwargs_from_spec(spec))
    meta = [c for c in ["smiles", spec.aggregate["group_by"], "inchi_folder", "ligand_class", "n_conformers",
                        *COMPLETENESS_COLUMNS]
            if c in agg.columns and c not in filtered.columns]
    for c in meta:
        filtered[c] = agg[c].values
    return filtered, report
