"""``singlepoint.nbo`` -> ``singlepoint_nbo.json``: the NBOs of the single point as an ORCA JSON wavefunction.

ORCA writes the NBOs of the single point step to ``singlepoint.nbo``, a file in .gbw format with the NBOs
in place of the MOs (orbital i is NBO number i of ``orca_condensed.json``). ``orca_2json`` converts it,
but only under a ``.gbw`` name, so the conversion is::

    cp singlepoint.nbo singlepoint_nbo.gbw && orca_2json singlepoint_nbo.gbw && rm singlepoint_nbo.gbw

(FILE.47 is no substitute: it has no NBO coefficients and each NBO step of the compound job overwrites it.)

Two callers: ``run_orca_conformer`` right after ORCA when ``dft.nbo_json`` is on, and ``orcadesc nbo-json``
for conformers that finished before (optionally only the lowest-energy one of each molecule, ~4 MB each).
``orca_2json`` comes with the ORCA module (``module load orca/6.1.0``).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import traceback
from pathlib import Path

from desc_utils.submit import records

NBO_FILE = "singlepoint.nbo"
GBW_COPY = "singlepoint_nbo.gbw"
JSON_FILE = "singlepoint_nbo.json"
CONDENSED = "orca_condensed.json"


def find_orca_2json(orca: str | None = None) -> str | None:
    """``orca_2json`` next to the ``orca`` binary (the same ORCA build), else the one on PATH; None if neither."""
    if orca:
        sibling = Path(orca).resolve().parent / "orca_2json"
        if sibling.exists():
            return str(sibling)
    return shutil.which("orca_2json")


def convert(conf_dir: str | Path, orca_2json: str | None = None, overwrite: bool = False,
            timeout: int = 3600) -> dict:
    """Write ``conf_dir/singlepoint_nbo.json``; returns the record kept in metadata.json. Never raises.

    ``{"ok": True, "at", "size_bytes"}``, ``{"ok": True, "skipped": "exists"}`` (and nothing run) when the
    JSON is there already and not ``overwrite``, or ``{"ok": False, "error", "at"}``. The ``.gbw`` copy is
    removed whatever happens.
    """
    conf_dir = Path(conf_dir)
    out = conf_dir / JSON_FILE
    gbw = conf_dir / GBW_COPY
    try:
        if out.exists() and not overwrite:
            return {"ok": True, "skipped": "exists", "size_bytes": out.stat().st_size, "at": records.now()}
        if not (conf_dir / NBO_FILE).exists():
            return {"ok": False, "error": f"no {NBO_FILE} (single point not run, or without NBO)", "at": records.now()}
        exe = orca_2json or find_orca_2json()
        if not exe:
            return {"ok": False, "error": "orca_2json not found (module load orca/6.1.0)", "at": records.now()}
        shutil.copyfile(conf_dir / NBO_FILE, gbw)
        try:
            proc = subprocess.run([exe, GBW_COPY], cwd=str(conf_dir), capture_output=True, text=True, timeout=timeout)
        finally:
            gbw.unlink(missing_ok=True)
        if proc.returncode != 0 or not out.exists():
            tail = (proc.stdout + proc.stderr)[-1000:]
            return {"ok": False, "error": f"orca_2json exit {proc.returncode}, {JSON_FILE} "
                                          f"{'written' if out.exists() else 'missing'}: {tail}", "at": records.now()}
        return {"ok": True, "size_bytes": out.stat().st_size, "at": records.now()}
    except Exception as exc:  # noqa: BLE001 -- the ORCA outcome is recorded regardless
        gbw.unlink(missing_ok=True)
        return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-2000:], "at": records.now()}


# --------------------------------------------------------------------------- backfill
def conformer_dirs(mol_dir: Path) -> list[Path]:
    """``conf_N`` folders of a molecule, in N order."""
    dirs = [p for p in mol_dir.iterdir() if p.is_dir() and p.name.startswith("conf_") and p.name[5:].isdigit()]
    return sorted(dirs, key=lambda p: int(p.name[5:]))


def molecule_dirs(library_dir: Path) -> list[Path]:
    """Folders of ``library_dir`` that hold ``conf_N`` folders (one per molecule)."""
    return sorted(p for p in library_dir.iterdir() if p.is_dir() and conformer_dirs(p))


def singlepoint_energy(conf_dir: Path) -> float | None:
    """The single point's SCF energy (Hartree) from ``orca_condensed.json``; None without one."""
    try:
        data = json.loads((conf_dir / CONDENSED).read_text())
        e = data["properties"]["energies"]["final_single_point_energy_hartree_orca"]
        e = float(e)
        return e if math.isfinite(e) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def lowest_energy_conformer(mol_dir: Path) -> Path | None:
    """The conformer with the lowest single point energy among those with a ``singlepoint.nbo``
    (the energy ``SCF_energy`` of the specs); None when none has both."""
    scored = [(e, c) for c in conformer_dirs(mol_dir)
              if (c / NBO_FILE).exists() and (e := singlepoint_energy(c)) is not None]
    return min(scored, key=lambda t: t[0])[1] if scored else None


def targets(library_dir: Path, lowest_only: bool = False, inchikeys: list[str] | None = None) -> list[Path]:
    """Conformer folders to convert: every one with a ``singlepoint.nbo``, or the lowest-energy one per molecule."""
    out = []
    for mol in molecule_dirs(Path(library_dir)):
        if inchikeys and mol.name not in inchikeys:
            continue
        if lowest_only:
            best = lowest_energy_conformer(mol)
            if best is not None:
                out.append(best)
        else:
            out += [c for c in conformer_dirs(mol) if (c / NBO_FILE).exists()]
    return out


def backfill(library_dir: str | Path, lowest_only: bool = False, overwrite: bool = False, dry_run: bool = False,
             orca_2json: str | None = None, inchikeys: list[str] | None = None) -> dict[str, int]:
    """Convert the conformers :func:`targets` selects; prints one line each and returns counts by outcome.
    The outcome is added to the molecule's metadata.json (``conformers.<conf>.nbo_json``) when it has one."""
    library_dir = Path(library_dir)
    todo = targets(library_dir, lowest_only=lowest_only, inchikeys=inchikeys)
    counts = {"converted": 0, "exists": 0, "failed": 0, "would_convert": 0}
    exe = orca_2json or find_orca_2json()
    if not dry_run and not exe and todo:
        raise FileNotFoundError("orca_2json not found on PATH: module load orca/6.1.0, or pass --orca-2json")
    for conf in todo:
        rel = conf.relative_to(library_dir)
        if dry_run:
            exists = (conf / JSON_FILE).exists() and not overwrite
            counts["exists" if exists else "would_convert"] += 1
            print(f"{rel}: {'exists' if exists else 'would convert'}")
            continue
        rec = convert(conf, orca_2json=exe, overwrite=overwrite)
        key = "failed" if not rec["ok"] else ("exists" if rec.get("skipped") else "converted")
        counts[key] += 1
        print(f"{rel}: {key}" + ("" if rec["ok"] else f" ({rec['error'][:200]})"))
        if (conf.parent / records.METADATA).exists() and not rec.get("skipped"):
            records.update_metadata(library_dir, conf.parent.name, {"conformers": {conf.name: {"nbo_json": rec}}})
    print(f"{len(todo)} conformer(s): " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    return counts
