"""Build the atom numbering table: one row per conformer, ``label -> 1-based index`` columns.

    spec = load_spec("specs/lnih2_pyox.yaml")
    df = build_numbering_table(spec, repo_root=Path("."))
    write_numbering_table(df, "atom_numbering_results.csv")

Columns: ``xyz_name, path, <molecule id>, <energy>, smiles, smiles_noH, <labels...>, <annotator columns>,
numbering_backend, numbering_status, numbering_notes``. Index base conversion happens exactly
once, at the end of :func:`build_numbering_table`.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from desc_utils.numbering.base import (
    STATUS_ELEMENT_MISMATCH, STATUS_FAILED, STATUS_INCOMPLETE, STATUS_INCONSISTENT, STATUS_OK,
    ConformerContext, NoMatch, NumberingBackend, NumberingRow, make_backend, validate_elements,
)
from desc_utils.numbering.io import get_sdf_from_xyz, load_mol, scf_energy, smiles_for_mol, xyz_comment_energy
from desc_utils.numbering.ligand_classes import ANNOTATORS
from desc_utils.parallel import parallel_map
from desc_utils.spec import Spec

__all__ = [
    "iter_conformers", "build_numbering_table", "check_consistency", "apply_orient",
    "write_numbering_table", "read_numbering_table", "usable_rows", "make_backend",
]


# --------------------------------------------------------------------------- discovery

def iter_conformers(spec: Spec, repo_root: Path, data_root: Path | None = None) -> Iterator[ConformerContext]:
    """Yield one context per conformer folder that contains the geometry file (sorted, deterministic)."""
    repo_root = Path(repo_root).resolve()
    root = Path(data_root) if data_root else Path(spec.data["root"])
    root = root if root.is_absolute() else repo_root / root
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    geometry = spec.data["geometry_file"]
    glob = spec.data["conformer_glob"]

    def ctx(mol_id: str, conf_dir: Path) -> ConformerContext:
        try:
            rel = conf_dir.resolve().relative_to(repo_root).as_posix() + "/"
        except ValueError:
            rel = conf_dir.resolve().as_posix() + "/"
        xyz = conf_dir / geometry
        return ConformerContext(mol_id, conf_dir.resolve(), rel, xyz, xyz.with_suffix(".sdf"))

    if spec.data["layout"] == "flat":
        for conf_dir in sorted(p for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob)):
            if (conf_dir / geometry).exists():
                yield ctx(conf_dir.name, conf_dir)
        return
    for mol_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for conf_dir in sorted(p for p in mol_dir.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob)):
            if (conf_dir / geometry).exists():
                yield ctx(mol_dir.name, conf_dir)


# --------------------------------------------------------------------------- orientation rules

def apply_orient(mol, indices: dict[str, int], rules: Iterable[dict]) -> dict[str, int]:
    """``{rule: closest, move: [H1, H2], to: [N1, N2]}``: permute the ``move`` labels so that
    ``move[k]`` is the atom (among the current ``move`` atoms) closest to ``to[k]``.
    Generalises the former ``number_H1_and_H2``. Works on 0-based indices."""
    indices = dict(indices)
    conf = mol.GetConformer()
    for rule in rules or []:
        move, to = list(rule["move"]), list(rule["to"])
        if not all(l in indices for l in move + to):
            continue
        pool = [indices[l] for l in move]
        assigned: dict[str, int] = {}
        for k, target_label in enumerate(to):
            t = conf.GetAtomPosition(indices[target_label])
            best = min(pool, key=lambda i: (conf.GetAtomPosition(i) - t).Length())
            assigned[move[k]] = best
            pool.remove(best)
        indices.update(assigned)
    return indices


# --------------------------------------------------------------------------- pipeline

def _number_one(ctx: ConformerContext, spec: Spec, backend: NumberingBackend, regenerate_sdf: bool) -> tuple[NumberingRow, dict]:
    row = NumberingRow(ctx)
    extra: dict = {}
    try:
        sdf = get_sdf_from_xyz(ctx.conf_dir, spec.data["geometry_file"], overwrite=regenerate_sdf)
        mol = load_mol(sdf, notes=row.notes)
    except Exception as exc:
        row.status, row.notes = STATUS_FAILED, [f"read: {type(exc).__name__}: {exc}"]
        return row, extra
    try:
        indices = backend.number(mol, ctx)
    except NoMatch as exc:
        row.status, row.notes = STATUS_FAILED, [str(exc)]
        return row, extra
    except Exception as exc:
        row.status, row.notes = STATUS_FAILED, [f"{backend.name}: {type(exc).__name__}: {exc}"]
        return row, extra
    indices = apply_orient(mol, indices, spec.numbering.get("orient"))
    row.indices = indices
    missing = [l for l in spec.labels if l not in indices]
    if missing:
        row.status = STATUS_INCOMPLETE
        row.notes.append(f"unassigned labels: {missing}")
    problems = validate_elements(mol, indices, spec.atoms)
    if problems:
        row.status = STATUS_ELEMENT_MISMATCH
        row.notes.extend(problems)

    smiles_cfg = spec.numbering["smiles"]
    try:
        strip = [indices[l] for l in smiles_cfg.get("strip", []) if l in indices]
        extra["smiles"] = smiles_for_mol(mol, smiles_cfg["mode"], strip)
        extra["smiles_noH"] = smiles_for_mol(mol, smiles_cfg["mode"], strip, remove_hs=True)
    except Exception as exc:
        extra.setdefault("smiles", None)
        extra["smiles_noH"] = None
        row.notes.append(f"smiles: {type(exc).__name__}: {exc}")
    for name in spec.numbering.get("annotators", []):
        fn = ANNOTATORS.get(name)
        if fn is None:
            row.notes.append(f"unknown annotator {name!r}")
            continue
        try:
            extra.update(fn(mol, indices))
        except Exception as exc:
            row.notes.append(f"annotator {name}: {type(exc).__name__}: {exc}")
    return row, extra


def _record(ctx: ConformerContext, spec: Spec, backend: NumberingBackend, regenerate_sdf: bool) -> dict:
    """One row of the numbering table (0-based indices) for one conformer."""
    energy_cfg = spec.numbering["energy"]
    row, extra = _number_one(ctx, spec, backend, regenerate_sdf)
    rec = {"xyz_name": ctx.xyz_name, "path": ctx.rel_path, spec.data["molecule_id_column"]: ctx.molecule_id}
    try:
        if energy_cfg.get("source", "orca") == "xyz_comment":
            rec[energy_cfg["column"]] = xyz_comment_energy(ctx.conf_dir, spec.data["geometry_file"],
                                                           energy_cfg["key"], energy_cfg["units"])
        else:
            rec[energy_cfg["column"]] = scf_energy(ctx.conf_dir, energy_cfg["basename"], spec.data.get("condensed_file"))
    except Exception as exc:
        rec[energy_cfg["column"]] = np.nan
        row.notes.append(f"energy: {type(exc).__name__}: {str(exc)[:80]}")
    rec["smiles"] = extra.pop("smiles", None)
    rec["smiles_noH"] = extra.pop("smiles_noH", None)
    for label in spec.labels:
        rec[label] = row.indices.get(label, np.nan)
    rec.update(extra)
    rec["numbering_backend"] = backend.name
    rec["numbering_status"] = row.status
    rec["numbering_notes"] = "; ".join(row.notes) if row.notes else ""
    return rec


# Worker-side state for the parallel path: set once per process by the pool initializer.
_WORKER: dict = {}


def _init_worker(spec: Spec, backend: NumberingBackend, regenerate_sdf: bool) -> None:
    _WORKER.update(spec=spec, backend=backend, regenerate_sdf=regenerate_sdf)


def _work(ctx: ConformerContext) -> dict:
    return _record(ctx, _WORKER["spec"], _WORKER["backend"], _WORKER["regenerate_sdf"])


def build_numbering_table(spec: Spec, repo_root: Path, backend: NumberingBackend | None = None,
                          strict: bool = False, regenerate_sdf: bool = False,
                          data_root: Path | None = None, progress: bool = True, jobs: int | None = 1) -> pd.DataFrame:
    """Number every conformer under the data root and return the table (indices in ``index_base``).

    ``jobs > 1`` numbers the conformers over that many processes (``0`` = every core); the table is
    identical to the serial one, rows in the same order. The backend is sent to every worker, so a
    custom backend must be picklable (the built-in ones are).
    """
    backend = backend or make_backend(spec.numbering, spec.labels)
    id_col = spec.data["molecule_id_column"]
    contexts = list(iter_conformers(spec, repo_root, data_root))
    records = parallel_map(_work, contexts, jobs, initializer=_init_worker, initargs=(spec, backend, regenerate_sdf),
                           progress=progress, desc=f"numbering[{backend.name}]")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("no conformer folders with a geometry file were found")
    base = int(spec.numbering.get("index_base", 1))
    for label in spec.labels:
        df[label] = (df[label] + base).astype("Int64")
    if spec.numbering.get("consistency", "per_molecule") == "per_molecule":
        df = check_consistency(df, spec.labels, id_col)
    bad = df[df["numbering_status"] != STATUS_OK]
    if strict and len(bad):
        raise RuntimeError(f"{len(bad)} conformer(s) not numbered cleanly:\n" + bad[["path", "numbering_status", "numbering_notes"]].to_string())
    return df


def check_consistency(df: pd.DataFrame, labels: list[str], group_col: str) -> pd.DataFrame:
    """Mark rows of molecules whose conformers disagree on any label (replaces the old interactive prompt)."""
    df = df.copy()
    ok = df["numbering_status"] == STATUS_OK
    for _, g in df[ok].groupby(group_col):
        if len(g) < 2:
            continue
        if any(g[l].nunique(dropna=True) > 1 for l in labels):
            differing = [l for l in labels if g[l].nunique(dropna=True) > 1]
            df.loc[g.index, "numbering_status"] = STATUS_INCONSISTENT
            df.loc[g.index, "numbering_notes"] = df.loc[g.index, "numbering_notes"].map(
                lambda s: "; ".join(filter(None, [s, f"conformers disagree on {differing}"]))
            )
    return df


# --------------------------------------------------------------------------- IO

def write_numbering_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def usable_rows(df: pd.DataFrame, spec: Spec, require_ok: bool = True, quiet: bool = False) -> pd.DataFrame:
    """The rows of a numbering table whose atom indices can be used for extraction.

    Rows with any missing label index are dropped; when the table has a ``numbering_status``
    column and ``require_ok`` is set, only ``ok`` rows are kept (a molecule whose conformers
    disagree, or a conformer Open Babel could not perceive, has no reliable indices). The label
    columns come back as plain ``int`` so a nullable ``Int64`` never reaches ``int()`` downstream.
    Legacy tables without a status column are accepted as-is. The index is reset.
    """
    missing = [l for l in spec.labels if l not in df.columns]
    if missing:
        raise KeyError(f"numbering table: missing label columns {missing}; the spec declares {spec.labels}")
    n0 = len(df)
    if require_ok and "numbering_status" in df.columns:
        df = df[df["numbering_status"] == STATUS_OK]
    df = df.dropna(subset=spec.labels).copy()
    if len(df) < n0 and not quiet:
        print(f"[numbering] {n0 - len(df)} of {n0} rows skipped (not numbered cleanly)")
    for l in spec.labels:
        df[l] = df[l].astype(int)
    return df.reset_index(drop=True)


def read_numbering_table(path: str | Path, spec: Spec, require_ok: bool = True, usable: bool = True) -> pd.DataFrame:
    """Read a numbering CSV.

    With ``usable`` (default) only the rows fit for extraction come back (see :func:`usable_rows`).
    ``usable=False`` returns every row, validated but unfiltered: what ``run_extraction`` wants, since
    it keeps the metadata of every conformer for the check report and filters the rows itself.
    """
    df = pd.read_csv(path)
    missing = [l for l in spec.labels if l not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing label columns {missing}; the spec declares {spec.labels}")
    if "path" not in df.columns:
        raise KeyError(f"{path}: missing 'path' column")
    if not usable:
        return df.reset_index(drop=True)
    return usable_rows(df, spec, require_ok=require_ok)
