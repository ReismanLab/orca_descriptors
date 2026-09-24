"""DescriPyTor descriptor families, driven by a label-keyed spec.

The :data:`FAMILIES` table is the single source of truth for which ``Molecules`` method
each DescriPyTor "Schema B" key maps to and how its index-based output columns are
renamed to label-based columns. It mirrors ``feature_steps`` in
``DescriPyTor/M2_data_extractor/data_extractor.py`` (``get_molecules_features_set``) one
to one, but the per-method loop is called directly so thresholds and defaults are explicit
and failures are attributed to one conformer and one family instead of being printed and
swallowed.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from desc_utils.orca_parse import CONDENSED_FILENAME
from desc_utils.spec import DESCRIPYTOR_TERM_ARITY, resolve_terms

_DESCRIPYTOR_DIRNAME = "DescriPyTor"


@dataclass(frozen=True)
class Family:
    key: str                      # Schema B key in the spec
    method: str                   # Molecules method
    arg: str                      # keyword the method takes the index list under
    arity: int
    columns: dict[str, str]       # index-column template -> label-column template
    kwargs: dict[str, str] = field(default_factory=dict)   # spec option -> method kwarg
    flat_single: bool = False     # method takes ONE flat term, not a list of terms


FAMILIES: dict[str, Family] = {
    "charges": Family("charges", "get_charge_df_dict", "atom_indices", 1,
                      {"nbo_atom_{i0}": "charge_NBO_{l0}"}),
    "sterimol": Family("sterimol", "get_sterimol_dict", "atom_indices", 2,
                       {"B1_{i0}-{i1}": "B1_{l0}-{l1}", "B5_{i0}-{i1}": "B5_{l0}-{l1}",
                        "L_{i0}-{i1}": "L_{l0}-{l1}", "loc_B5_{i0}-{i1}": "loc_B5_{l0}-{l1}",
                        "B1_B5_angle_{i0}-{i1}": "B1-B5_angle_{l0}-{l1}"},
                       kwargs={"radii": "radii", "drop_atoms": "drop_atoms"}),
    "bond_length": Family("bond_length", "get_bond_length_dict", "atom_pairs", 2,
                          {"0_bond_length_{i0}-{i1}": "dist_{l0}-{l1}"}),
    "stretching_vibration": Family("stretching_vibration", "get_stretch_vibration_dict", "atom_pairs", 2,
                                   {"Frequency_Stretch_{i0}_{i1}": "vib_freq_{l0}-{l1}",
                                    "Amplitude_Stretch_{i0}_{i1}": "vib_amp_{l0}-{l1}"},
                                   kwargs={"stretch_threshold": "threshold", "upper_stretch_threshold": "upper_threshold"}),
    "bending_vibration": Family("bending_vibration", "get_bend_vibration_dict", "atom_pairs", 2,
                                {"Frequency_Bending_{i0}-{i1}": "vib_freq_bend_{l0}-{l1}",
                                 "Cross_mag_Bending_{i0}-{i1}": "vib_amp_bend_{l0}-{l1}"},
                                kwargs={"bend_threshold": "threshold"}),
    "charge_diff": Family("charge_diff", "get_charge_diff_df_dict", "atom_indices", 2,
                          {"nbo_diff_{i0}-{i1}": "charge_NBO_diff_{l0}-{l1}"}),
    "dipole": Family("dipole", "get_dipole_dict", "atom_indices", 3,
                     {"total_dipole_{i0}-{i1}-{i2}": "dipole_{l0}-{l1}-{l2}",
                      "dipole_x_{i0}-{i1}-{i2}": "dipole_x_{l0}-{l1}-{l2}",
                      "dipole_y_{i0}-{i1}-{i2}": "dipole_y_{l0}-{l1}-{l2}",
                      "dipole_z_{i0}-{i1}-{i2}": "dipole_z_{l0}-{l1}-{l2}"},
                     flat_single=True),
    "bond_angle": Family("bond_angle", "get_bond_angle_dict", "atom_indices", 3,
                         {"0_angle_[{i0}, {i1}, {i2}]": "angle_{l0}-{l1}-{l2}"}),
}
assert {k: f.arity for k, f in FAMILIES.items()} == DESCRIPYTOR_TERM_ARITY

#: Column order of the historical notebook output, used to keep CSVs diff-friendly.
FAMILY_ORDER = ["charges", "sterimol", "bond_length", "stretching_vibration",
                "bending_vibration", "charge_diff", "dipole", "bond_angle"]


# --------------------------------------------------------------------------- DescriPyTor import

def descripytor_dir(repo_root: Path | None = None) -> Path:
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return repo_root / _DESCRIPYTOR_DIRNAME


def _import_molecules(repo_root: Path | None = None):
    """Put the DescriPyTor submodule on ``sys.path`` (once) and return ``Molecules``."""
    d = descripytor_dir(repo_root)
    if not (d / "M2_data_extractor").is_dir():
        raise ImportError(
            f"DescriPyTor not found at {d}. Run `git submodule update --init` from the repository root."
        )
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from M2_data_extractor.data_extractor import Molecules  # noqa: WPS433

    return Molecules


# --------------------------------------------------------------------------- helpers

def _fmt(template: str, idx: tuple[int, ...], labels: tuple[str, ...]) -> str:
    return template.format(**{f"i{k}": v for k, v in enumerate(idx)}, **{f"l{k}": v for k, v in enumerate(labels)})


def label_columns(cfg: Mapping[str, Any]) -> list[str]:
    """Every label-based column the given spec section will produce, in output order."""
    cols: list[str] = []
    for key in FAMILY_ORDER:
        fam = FAMILIES[key]
        terms = cfg.get(key) or []
        for term in terms:
            term = (term,) if isinstance(term, str) else tuple(term)
            for tmpl in fam.columns.values():
                cols.append(_fmt(tmpl, tuple(range(len(term))), term))
    return _legacy_dipole_name(cols, cfg)


def _legacy_dipole_name(cols: list[str], cfg: Mapping[str, Any]) -> list[str]:
    """With exactly one dipole triple the total column is called ``dipole`` (historical name)."""
    if len(cfg.get("dipole") or []) == 1:
        term = cfg["dipole"][0]
        legacy = "dipole_" + "-".join(term)
        cols = ["dipole" if c == legacy else c for c in cols]
    return cols


def to_schema_b(cfg: Mapping[str, Any], row: Mapping[str, Any]) -> dict:
    """Resolve a labelled spec section to DescriPyTor's Schema B dict for ONE conformer.

    The result is what ``build_answers_dict`` in DescriPyTor's descriptor_extraction_toolkit
    consumes (indices are 1-based, lists are native JSON lists).
    """
    out: dict[str, Any] = {}
    for key, fam in FAMILIES.items():
        terms = resolve_terms(cfg.get(key) or [], row)
        if fam.arity == 1:
            out[key] = [t[0] for t in terms]
        else:
            out[key] = [list(t) for t in terms]
    out["stretch_threshold"] = cfg.get("stretch_threshold")
    out["upper_stretch_threshold"] = cfg.get("upper_stretch_threshold")
    out["bend_threshold"] = cfg.get("bend_threshold")
    out["drop_atoms"] = [int(row[l]) for l in cfg.get("drop_atoms") or []]
    out["center_atoms_dipole"] = []
    out["ring_vibration"] = []
    return out


@contextlib.contextmanager
def _quiet_cwd():
    """Silence DescriPyTor's prints and restore the working directory afterwards."""
    cwd = os.getcwd()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------- extraction

def extract_one(mols, cfg: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """Run every configured family on one loaded ``Molecules`` object (one conformer)."""
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key in FAMILY_ORDER:
        fam = FAMILIES[key]
        label_terms = [(t,) if isinstance(t, str) else tuple(t) for t in (cfg.get(key) or [])]
        if not label_terms:
            continue
        idx_terms = resolve_terms(label_terms, row)
        kwargs = {}
        for spec_key, method_kw in fam.kwargs.items():
            if spec_key in cfg and cfg[spec_key] is not None:
                value = cfg[spec_key]
                if spec_key == "drop_atoms":
                    # DescriPyTor's Sterimol treats drop_atoms=[] differently from the default
                    # None (B1 and the B1-B5 angle change), so only pass it when atoms are listed.
                    if not value:
                        continue
                    value = [int(row[l]) for l in value]
                kwargs[method_kw] = value
        method = getattr(mols, fam.method)
        calls = [(t,) for t in idx_terms] if fam.flat_single else [tuple(idx_terms)]
        label_calls = [(t,) for t in label_terms] if fam.flat_single else [tuple(label_terms)]
        for idx_group, label_group in zip(calls, label_calls):
            arg = list(idx_group[0]) if fam.flat_single else [list(t) if fam.arity > 1 else t[0] for t in idx_group]
            try:
                res = method(**{fam.arg: arg, **kwargs})
                first = res.iloc[0] if isinstance(res, pd.DataFrame) and len(res) else None
            except Exception as exc:  # one family failing must not kill the conformer
                notes.append(f"{key}: {type(exc).__name__}: {str(exc)[:120]}")
                first = None
            for idx, lab in zip(idx_group, label_group):
                for in_tmpl, out_tmpl in fam.columns.items():
                    in_col, out_col = _fmt(in_tmpl, idx, lab), _fmt(out_tmpl, idx, lab)
                    if first is not None and in_col in first.index:
                        out[out_col] = first[in_col]
                    else:
                        out[out_col] = np.nan
                        if first is not None:
                            notes.append(f"{key}: column {in_col!r} missing")
    # historical name for the single-dipole total column
    if len(cfg.get("dipole") or []) == 1:
        legacy = "dipole_" + "-".join(cfg["dipole"][0])
        if legacy in out:
            out = {("dipole" if k == legacy else k): v for k, v in out.items()}
    return out, notes


@contextlib.contextmanager
def _molecule_dir(condensed: Path):
    """Folder handed to ``Molecules``.

    ``Molecules`` loads every molecule file it finds in a folder. A temporary folder holding only a
    link to the condensed file is used, so that other molecule-shaped JSON in the conformer folder
    (``singlepoint.json``, ``xyz.json``) can never be loaded as extra molecules.
    """
    with tempfile.TemporaryDirectory(prefix="descripytor_") as td:
        link = Path(td) / condensed.name
        try:
            os.symlink(condensed.resolve(), link)
        except OSError:  # file systems without symlinks
            shutil.copy(condensed, link)
        yield Path(td)


def conformer_dir(row: Mapping[str, Any], repo_root: Path) -> Path:
    p = Path(str(row["path"]))
    return p if p.is_absolute() else (repo_root / p)


def extract_descripytor(df_num: pd.DataFrame, cfg: Mapping[str, Any], repo_root: Path,
                        progress: bool = True, condensed_file: str | None = None) -> pd.DataFrame:
    """Per-conformer DescriPyTor descriptors for every row of the numbering table.

    Each row's ``path`` is a conformer folder; the molecule file DescriPyTor reads is
    ``<folder>/<condensed_file>`` written by ``desc_utils.condense``. A missing file or a failing
    family produces NaN plus a note in ``descripytor_notes``; they never abort the run.
    """
    Molecules = _import_molecules(repo_root)
    condensed_name = condensed_file or CONDENSED_FILENAME
    rows: dict = {}
    iterator = df_num.iterrows()
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=len(df_num), desc="descripytor")
    for i, row in iterator:
        folder = conformer_dir(row, repo_root)
        rec: dict[str, Any] = {"path": row["path"]}
        condensed = folder / condensed_name
        if not condensed.is_file():
            rec["descripytor_notes"] = (f"no {condensed_name} in {folder}. Run `python -m desc_utils.run condense --spec <spec>` "
                                        f"(or on the cluster `python -m desc_utils.condense --conf-dir <folder>`) first")
            for col in label_columns(cfg):
                rec[col] = np.nan
            rows[i] = rec
            continue
        with _quiet_cwd(), _molecule_dir(condensed) as mol_dir:
            try:
                mols = Molecules(molecules_dir_name=str(mol_dir))
            except Exception as exc:
                mols = None
                rec["descripytor_notes"] = f"Molecules failed: {type(exc).__name__}: {str(exc)[:120]}"
            if mols is not None:
                n = len(getattr(mols, "molecules", []))
                if n != 1:
                    rec["descripytor_notes"] = f"expected 1 molecule file in {mol_dir}, loaded {n}"
                vals, notes = extract_one(mols, cfg, row) if n >= 1 else ({c: np.nan for c in label_columns(cfg)}, [])
                rec.update(vals)
                if notes:
                    rec["descripytor_notes"] = "; ".join(filter(None, [rec.get("descripytor_notes"), *notes]))
            else:
                for col in label_columns(cfg):
                    rec[col] = np.nan
        rows[i] = rec
    return order_columns(pd.DataFrame.from_dict(rows, orient="index"), cfg)


def order_columns(out: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """``path``, then the label columns of the spec in output order, then anything else (the notes)."""
    ordered = ["path"] + [c for c in label_columns(cfg) if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    return out[ordered]
