"""ORCA-side descriptor families (morfeus geometry, ORCA charges, NBO, redox), spec-driven.

Thin adapters over the functions in ``desc_utils.data_extraction``, which already take
label lists and a numbering table. This module only (a) reads the term lists from the spec,
(b) normalises the conformer key, and (c) applies the NBO rank filter that used to be a
notebook cell.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from desc_utils import data_extraction as de
from desc_utils.data_extraction import conf_key  # noqa: F401  (re-exported join key)
from desc_utils.orca_parse import CONDENSED_FILENAME
from desc_utils.spec import Spec

_HM = re.compile(r"_hm(\d+)_")
_LP = re.compile(r"_lp(\d+)_")


def _to_frame(results: Mapping | pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame.from_dict(results, orient="index")
    if "path" not in df.columns:
        raise KeyError("extractor output has no 'path' column")
    df = df.copy()
    df["path"] = df["path"].map(lambda p: conf_key(p, repo_root))
    return df


def _with_repo_paths(df_num: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """The legacy extractors resolve ``path`` relative to the repo; make that explicit."""
    df = df_num.copy()
    df["path"] = df["path"].map(lambda p: str(Path(p) if Path(p).is_absolute() else (Path(repo_root) / p)))
    return df


def filter_nbo_rank_columns(df: pd.DataFrame, rank_range: Iterable[int] | None, drop_analysis_info: bool = True) -> pd.DataFrame:
    """Keep ``_hmN_`` / ``_lpN_`` columns only for N in ``rank_range``; drop ``analysis_info``.

    ``hmN`` means HOMO-N and ``lpN`` means LUMO+N. Rank-agnostic columns (``_hm_``, ``_lp_``)
    are always kept. ``rank_range=None`` keeps every rank.
    """
    drop: list[str] = []
    allowed = {int(n) for n in rank_range} if rank_range is not None else None
    for c in df.columns:
        hm, lp = _HM.search(c), _LP.search(c)
        if allowed is not None:
            if hm and int(hm.group(1)) not in allowed:
                drop.append(c)
                continue
            if lp and int(lp.group(1)) not in allowed:
                drop.append(c)
                continue
        if drop_analysis_info and "analysis_info" in c:
            drop.append(c)
    return df.drop(columns=drop)


def extract_xyz(df_num: pd.DataFrame, cfg: Mapping, repo_root: Path, overwrite_json: bool = False,
                progress: bool = True) -> pd.DataFrame:
    res = de.get_xyz_descriptors(
        _with_repo_paths(df_num, repo_root), cfg["atoms"], cfg["bonds"], cfg["angles"], cfg["dihedrals"],
        quadrants=cfg["quadrants"] or None,
        overwrite_json=overwrite_json or not cfg.get("use_cached_json", True),
        vbur_radii=cfg.get("vbur_radii"), progress=progress,
    )
    return _to_frame(res, repo_root)


def extract_charges(df_num: pd.DataFrame, cfg: Mapping, repo_root: Path,
                    condensed_file: str | None = CONDENSED_FILENAME) -> pd.DataFrame:
    res = de.get_all_charges(_with_repo_paths(df_num, repo_root), cfg["atoms"], basename=cfg.get("basename", "singlepoint"),
                             condensed_file=condensed_file)
    return _to_frame(res, repo_root)


def postprocess_nbo(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    """The two library-wide steps of the NBO family: rank consolidation, then the rank filter of the spec.

    Applied by :func:`extract_nbo` unless ``postprocess=False``, in which case the caller applies it
    once to the concatenated per-conformer frames (see ``run_extraction(jobs=...)``). The order matters:
    a column dropped by the filter could otherwise change which rank groups are merged.
    """
    df = de.consolidate_nbo_rank_columns(df)
    return filter_nbo_rank_columns(df, cfg.get("homo_lumo_rank_range"), cfg.get("drop_analysis_info", True))


def extract_nbo(df_num: pd.DataFrame, cfg: Mapping, repo_root: Path,
                condensed_file: str | None = CONDENSED_FILENAME, postprocess: bool = True) -> pd.DataFrame:
    res = de.get_all_nbo_properties(
        _with_repo_paths(df_num, repo_root), cfg["atoms"], basename=cfg.get("basename", "singlepoint"),
        nbo_block_index=int(cfg.get("nbo_block_index", 0)), spin=cfg.get("spin", "alpha"),
        condensed_file=condensed_file, consolidate_ranks=False,
    )
    df = _to_frame(res, repo_root)
    return postprocess_nbo(df, cfg) if postprocess else df


def extract_redox(df_num: pd.DataFrame, cfg: Mapping, repo_root: Path,
                  condensed_file: str | None = CONDENSED_FILENAME) -> pd.DataFrame:
    res = de.get_redox_descriptors(
        _with_repo_paths(df_num, repo_root), cfg["atoms"],
        descriptorPrefixes=list(cfg.get("prefixes", ["cm5"])), nbo_block_map=cfg.get("nbo_block_map"),
        condensed_file=condensed_file,
    )
    return _to_frame(res, repo_root)


_RUNNERS = {"xyz": extract_xyz, "charges": extract_charges, "nbo": extract_nbo, "redox": extract_redox}
#: Historical order of the notebook's concat: xyz, charges, redox, nbo.
ORCA_ORDER = ["xyz", "charges", "redox", "nbo"]


def orca_families(spec: Spec, families: Iterable[str] | None = None) -> list[str]:
    """The ORCA families to run, in :data:`ORCA_ORDER`, restricted to ``families`` when given."""
    assert spec.orca is not None, "spec has no descriptors.orca section"
    return [f for f in ORCA_ORDER if f in spec.orca and (families is None or f in families)]


def orca_family_frames(df_num: pd.DataFrame, spec: Spec, repo_root: Path, families: Iterable[str] | None = None,
                       overwrite_xyz_json: bool = False, progress: bool = True,
                       postprocess: bool = True) -> dict[str, pd.DataFrame]:
    """One frame per requested ORCA family, keyed by family name, for the rows of ``df_num``.

    ``postprocess=False`` leaves the NBO frame raw (no rank consolidation, no rank filter) so that
    frames computed on slices of the library can be concatenated and post-processed once.
    """
    condensed_file = spec.data.get("condensed_file")
    frames: dict[str, pd.DataFrame] = {}
    for fam in orca_families(spec, families):
        cfg = spec.orca[fam]
        if progress:
            print(f"[orca] {fam}: {len(df_num)} conformers")
        if fam == "xyz":
            frames[fam] = extract_xyz(df_num, cfg, repo_root, overwrite_json=overwrite_xyz_json, progress=progress)
        elif fam == "nbo":
            frames[fam] = extract_nbo(df_num, cfg, repo_root, condensed_file=condensed_file, postprocess=postprocess)
        else:
            frames[fam] = _RUNNERS[fam](df_num, cfg, repo_root, condensed_file=condensed_file)
    return frames


def merge_orca_frames(frames: Mapping[str, pd.DataFrame], df_num: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Outer-join the family frames on ``path`` in :data:`ORCA_ORDER`; first occurrence of a column wins."""
    ordered = [frames[f] for f in ORCA_ORDER if f in frames]
    if not ordered:
        return pd.DataFrame({"path": df_num["path"].map(lambda p: conf_key(p, repo_root))})
    out = ordered[0]
    for f in ordered[1:]:
        new_cols = ["path"] + [c for c in f.columns if c != "path" and c not in out.columns]
        out = out.merge(f[new_cols], on="path", how="outer")
    return out


def run_orca(df_num: pd.DataFrame, spec: Spec, repo_root: Path, families: Iterable[str] | None = None,
             overwrite_xyz_json: bool = False, progress: bool = True) -> pd.DataFrame:
    """Run the requested ORCA-side families and outer-join them on ``path``.

    Duplicate column names across families (e.g. ``SCF_energy``) keep the first occurrence,
    exactly as the notebook did with ``loc[:, ~columns.duplicated()]``. The charges, NBO and
    redox families read ``<conf>/<data.condensed_file>`` when it exists (``null`` = raw files only).
    """
    frames = orca_family_frames(df_num, spec, repo_root, families, overwrite_xyz_json, progress)
    return merge_orca_frames(frames, df_num, repo_root)
