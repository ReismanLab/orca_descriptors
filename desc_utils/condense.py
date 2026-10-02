"""Condensed per-conformer ORCA file: one small JSON instead of ~70 MB of raw outputs.

Producer (cluster side, standard library only)::

    PYTHONPATH=/path/to/orca_descriptors python -m desc_utils.condense --conf-dir conf_1
    PYTHONPATH=/path/to/orca_descriptors python -m desc_utils.condense --root /path/LIB --jobs 4
    PYTHONPATH=/path/to/orca_descriptors python -m desc_utils.condense --root /path/LIB --check

writes ``<conf>/orca_condensed.json`` from ``optimization.xyz``, the ``*.property.json`` files,
the GBW dump ``singlepoint.json`` and ``calcParam.out``. Pull the condensed files together with
``optimization.xyz`` (rsync) and the pipeline never needs the raw files.

Consumer (laptop): ``OrcaReader`` and ``desc_utils.data_extraction`` read this file in preference
to the raw ORCA files when it exists (spec key ``data.condensed_file``; ``null`` disables).
DescriPyTor's ``Molecules`` loads the same file through its JSON reader: the top-level
``molecule`` / ``properties`` / ``vibrations`` / ``charges`` sections follow
``DescriPyTor/utils/help_functions.py::json_file_handler``, which ignores the other keys.

Schema version 1 (units in key names, ``null`` = not available)::

    schema_version  1
    producer        tool, created_utc, hostname, python, git_commit, orca_version, nbo_program,
                    conformer_dir, sources{file: {found, bytes, mtime_utc}}
    molecule        atoms[{atom, x, y, z}] (Å, from optimization.xyz), units, n_atoms, charge, multiplicity
    properties      dipole{x, y, z, total} (Debye, singlepoint), polarizability{iso, aniso, eigenvalues} (a.u.),
                    energies{final_single_point_energy_hartree_orca}
    vibrations      modes[{mode_index, frequency_cm-1, ir_intensity_km_per_mol, displacements[{dx, dy, dz}]}]
                    (translations / rotations dropped), source, displacement_decimals, n_dropped_zero_modes, n_imaginary
    charges         nbo (NPA natural charges, NBO block 0), hirshfeld, cm5 (= hirshfeld: the CM5 correction is not implemented); singlepoint, per atom
    states          singlepoint / oxidation / reduction / nmr -> scf_energy_eh, charge, multiplicity, electron counts,
                    atomic_numbers, charges{mulliken, loewdin, chelpg, hirshfeld}, dipole_au, dipole_magnitude_au,
                    polarizability, thermochemistry, nmr{nuc, elems, siso_ppm, saniso_ppm}, mo{energies_eh, occupations}
    nbo             blocks[{index, state_guess, closed_shell, spin_stored, level: full|npa, results}], block_map
                    ``results`` is ``NBOProps.results`` of that block with its list tables packed as
                    {columns, rows}; open-shell rows carry a ``spin`` field (alpha rows before beta rows).
    notes           parser messages emitted while condensing
    job             status: completed | failed | running, detail, checked_utc  (``orca_parse.job_status`` of the
                    folder when it was condensed; files written before 2026-09-23 lack the key -> ``unknown``),
                    timings{total_run_time_s, nprocs, steps{<step>: {index, wall_s, modules{<module>: s}}}}
                    (``orca_parse.compound_step_timings``: what every compound step cost, from ORCA's own
                    timing tables; files written before 2026-10-02 lack it)

A file is written as soon as ``optimization.xyz``, ``singlepoint.property.json`` and one complete NBO
block exist, so a job that died or is still running after the singlepoint gets a file too, with its
missing states as ``null`` and ``job.status`` saying so. Re-running the condenser rebuilds every file
whose stored status is not ``completed`` (when the raw files are still there), and only skips the
completed ones.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fnmatch
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Union

from desc_utils import orca_parse as op
from desc_utils.orca_parse import (  # noqa: F401  (re-exported)
    CONDENSED_FILENAME, JOB_COMPLETED, JOB_FAILED, JOB_NOT_STARTED, JOB_RUNNING, JOB_STATUSES, NBOProps, bohr_to_A,
)

SCHEMA_VERSION = 1
AU_TO_DEBYE = 2.541746
STATES = ("singlepoint", "oxidation", "reduction", "nmr")
#: state -> 0-based NBO block of the compound job (same order as ``DEFAULTS["orca_redox"]["nbo_block_map"]``)
DEFAULT_NBO_BLOCKS = {"singlepoint": 0, "oxidation": 1, "reduction": 2}
#: ``NBOProps.results`` entries that are lists of row dicts (stored columnar on disk)
TABLE_KEYS = ("natural_atomic_orbitals", "natural_bond_orbitals", "bond_orbital_details",
              "nho_directionality", "second_order_perturbation")
#: the tables NBO prints once per spin set in an open-shell block
SPIN_TABLE_KEYS = ("natural_bond_orbitals", "bond_orbital_details", "nho_directionality", "second_order_perturbation")
REQUIRED_TOP_KEYS = ("schema_version", "producer", "molecule", "properties", "vibrations", "charges", "states", "nbo")
#: ``job.status`` of a condensed file written before the key existed, and of a folder that has neither a
#: condensed file nor raw outputs (see :func:`conformer_job_status`)
JOB_UNKNOWN, JOB_MISSING = "unknown", "missing"
_SNIFF_BYTES = 8192  # DescriPyTor's folder scanner needs "molecule" and "atoms" within the first 8192 bytes


class CondenseError(Exception):
    """A conformer folder cannot be condensed (missing or inconsistent inputs)."""


# --------------------------------------------------------------------------- table packing

def pack_table(rows):
    """List of row dicts -> ``{"columns": [...], "rows": [[...], ...]}``; anything else is returned as is."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return rows
    columns: list = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    return {"columns": columns, "rows": [[r.get(c) for c in columns] for r in rows]}


def unpack_table(obj):
    """Inverse of :func:`pack_table`."""
    if isinstance(obj, dict) and set(obj) == {"columns", "rows"}:
        cols = obj["columns"]
        return [dict(zip(cols, row)) for row in obj["rows"]]
    return obj


# --------------------------------------------------------------------------- producer

def _utc(ts=None):
    d = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc) if ts is not None else _dt.datetime.now(_dt.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_meta(path):
    p = Path(path)
    if not p.is_file():
        return {"found": False}
    st = p.stat()
    return {"found": True, "bytes": st.st_size, "mtime_utc": _utc(st.st_mtime)}


def _git_commit(repo_root):
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _producer(conf_dir, text):
    repo_root = Path(__file__).resolve().parent.parent
    files = ["optimization.xyz", "calcParam.out", "frequency.property.json"]
    files += [f"{s}.property.json" for s in STATES] + [f"{s}.json" for s in STATES]
    return {
        "tool": "desc_utils.condense",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "git_commit": _git_commit(repo_root),
        "orca_version": op.orca_version(text) if text else None,
        "nbo_program": op.nbo_program(text) if text else None,
        "conformer_dir": str(Path(conf_dir).resolve()),
        "sources": {name: _file_meta(Path(conf_dir) / name) for name in files},
    }


def _nbo_blocks(out_path, text, full_blocks):
    """One entry per ``Now starting NBO`` block. ``full_blocks``: block indices stored with every table, or "all"."""
    first = NBOProps(out_path, block_index=0, spin="both", content=text)
    n = len(first.nbo_blocks)
    if n == 0:
        started = text.count("Now starting NBO")
        hint = " (NBO started but never returned: aborted ORCA run?)" if started else " (no NBO output at all)"
        raise CondenseError(f"{out_path.name}: no complete NBO block{hint}")
    blocks = []
    for i in range(n):
        nb = first if i == 0 else NBOProps(out_path, block_index=i, spin="both", content=text)
        full = full_blocks == "all" or i in full_blocks
        nb.get_NPA()
        nb.get_cycle_info()
        if full:
            nb.get_NAO()
            nb.get_NBO_summary()
            nb.get_orbital_details()
            nb.get_NHO_directionality()
            nb.get_second_order_perturbation()
        res = dict(nb.results)
        res["source_file"] = out_path.name
        for k in TABLE_KEYS:
            res[k] = pack_table(res.get(k))
        blocks.append({
            "index": i,
            "state_guess": next((s for s, b in DEFAULT_NBO_BLOCKS.items() if b == i), None),
            "closed_shell": nb.is_closed_shell,
            "spin_stored": None if nb.is_closed_shell else "both",
            "level": "full" if full else "npa",
            "results": res,
        })
    return blocks


def _vibrations(conf_dir, text, mode_decimals, n_atoms):
    freqs, modes, source = op.frequencies_and_modes(conf_dir, text)
    base = {"modes": [], "source": source, "displacement_decimals": mode_decimals, "normalization": "unit_norm",
            "n_dropped_zero_modes": 0, "n_imaginary": 0,
            "note": "mass-weighted Cartesian displacements (M^-1/2 * eigenvector, mode k = column k of Hessian.MODES) "
                    "scaled to unit length as in ORCA's NORMAL MODES table; IR intensities from the IR SPECTRUM "
                    "table, null where ORCA printed none"}
    if freqs is None:
        base["note"] = ("no frequency data: neither frequency.property.json nor VIBRATIONAL FREQUENCIES / "
                        "NORMAL MODES in calcParam.out")
        return base
    ir = op.ir_intensities(text) if text else {}
    keep = op.drop_zero_modes(freqs, modes)
    out_modes = []
    for k in keep:
        vec = modes[k]
        if len(vec) != 3 * n_atoms:
            raise CondenseError(f"mode {k}: {len(vec)} displacement components for {n_atoms} atoms")
        disp = [{"dx": round(vec[3 * a], mode_decimals), "dy": round(vec[3 * a + 1], mode_decimals),
                 "dz": round(vec[3 * a + 2], mode_decimals)} for a in range(n_atoms)]
        out_modes.append({"mode_index": k, "frequency_cm-1": freqs[k],
                          "ir_intensity_km_per_mol": ir.get(k), "displacements": disp})
    base.update({"modes": out_modes, "n_dropped_zero_modes": len(freqs) - len(keep),
                 "n_imaginary": sum(1 for k in keep if freqs[k] < 0)})
    return base


def build_condensed(conf_dir, mo_states=("singlepoint",), nbo_full_blocks=(0,), mode_decimals=6):
    """The condensed dict for one conformer folder (see the module docstring for the schema).

    Required inputs: ``optimization.xyz``, ``singlepoint.property.json`` and at least one complete
    NBO block in ``calcParam.out``. Everything else is optional and recorded under ``producer.sources``;
    whether the ORCA job actually finished is recorded under ``job`` (see :func:`orca_parse.job_status`).
    ``mo_states``: states whose MO energies / occupations are stored (the GBW dumps are large).
    ``nbo_full_blocks``: NBO block indices stored with every table (others keep NPA and cycle info only).
    """
    conf_dir = Path(conf_dir)
    if not conf_dir.is_dir():
        raise CondenseError(f"{conf_dir}: not a directory")
    xyz_path = conf_dir / "optimization.xyz"
    if not xyz_path.is_file():
        raise CondenseError(f"{conf_dir}: optimization.xyz missing")
    out_path = conf_dir / "calcParam.out"
    if not out_path.is_file():
        raise CondenseError(f"{conf_dir}: calcParam.out missing")
    symbols, coords = op.read_xyz(xyz_path)
    n_atoms = len(symbols)
    text = out_path.read_text(errors="replace")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # the parsers report missing sections with print()
        states = {s: op.property_state(conf_dir, s) for s in STATES}
        if states["singlepoint"] is None:
            raise CondenseError(f"{conf_dir}: singlepoint.property.json missing")
        for s in mo_states:
            if states.get(s) is not None:
                states[s]["mo"] = op.mo_list(conf_dir, s)
        blocks = _nbo_blocks(out_path, text, nbo_full_blocks)
        vib = _vibrations(conf_dir, text, mode_decimals, n_atoms)
    notes = [ln for ln in buf.getvalue().splitlines() if ln.strip()]

    for s, st in states.items():
        if st is None:
            continue
        for scheme, vals in st["charges"].items():
            if vals is not None and len(vals) != n_atoms:
                raise CondenseError(f"{s}: {len(vals)} {scheme} charges for {n_atoms} atoms")
        nmr = st.get("nmr") or {}
        for key in ("siso_ppm", "saniso_ppm"):
            if nmr.get(key) is not None and len(nmr[key]) != n_atoms:
                raise CondenseError(f"{s}: {len(nmr[key])} {key} values for {n_atoms} atoms")
    sp = states["singlepoint"]
    if sp["n_atoms"] not in (None, n_atoms):
        raise CondenseError(f"singlepoint has {sp['n_atoms']} atoms, optimization.xyz has {n_atoms}")

    npa_atoms = blocks[0]["results"]["natural_population_analysis"].get("atoms") or {}
    nbo_charges = []
    for i in range(1, n_atoms + 1):
        props = npa_atoms.get(i, npa_atoms.get(str(i)))
        if props is None:
            raise CondenseError(f"NBO block 0: no NPA charge for atom {i}")
        nbo_charges.append(props["charge"])
    hirshfeld = sp["charges"].get("hirshfeld")
    charges = {
        "nbo": nbo_charges,
        "hirshfeld": hirshfeld,
        "cm5": list(hirshfeld) if hirshfeld is not None else None,
        "note": "nbo: NPA natural charges of NBO block 0 (singlepoint); cm5 is a copy of hirshfeld because the "
                "CM5 correction is not implemented",
    }

    dipole = None
    if sp.get("dipole_au"):
        x, y, z = sp["dipole_au"]
        total = sp.get("dipole_magnitude_au")
        if total is None:
            total = math.sqrt(x * x + y * y + z * z)
        dipole = {"x": x * AU_TO_DEBYE, "y": y * AU_TO_DEBYE, "z": z * AU_TO_DEBYE, "total": total * AU_TO_DEBYE,
                  "units": "debye", "state": "singlepoint", "au_to_debye": AU_TO_DEBYE}

    molecule = {
        "atoms": [{"atom": s, "x": c[0], "y": c[1], "z": c[2]} for s, c in zip(symbols, coords)],
        "units": "angstrom",
        "source": "optimization.xyz",
        "n_atoms": n_atoms,
        "charge": sp.get("charge"),
        "multiplicity": sp.get("multiplicity"),
    }
    if sp.get("mo") and sp["mo"].get("charge") is not None:
        molecule["charge"], molecule["multiplicity"] = sp["mo"]["charge"], sp["mo"]["multiplicity"]

    return {
        "schema_version": SCHEMA_VERSION,
        "producer": _producer(conf_dir, text),
        "molecule": molecule,
        "properties": {
            "dipole": dipole,
            "polarizability": sp.get("polarizability"),
            "energies": {"final_single_point_energy_hartree_orca": sp.get("scf_energy_eh")},
        },
        "vibrations": vib,
        "charges": charges,
        "states": states,
        "nbo": {"source_file": "calcParam.out", "n_blocks": len(blocks), "block_map": DEFAULT_NBO_BLOCKS,
                "blocks": blocks},
        "notes": notes,
        "job": _job_entry(conf_dir, text),
    }


def _job_entry(conf_dir, text):
    js = op.job_status(conf_dir, text)
    return {"status": js["status"], "detail": js["detail"], "checked_utc": _utc(),
            "timings": op.compound_step_timings(text)}


def stored_job_status(cond) -> str:
    """``job.status`` of a parsed condensed file; ``unknown`` for files written before the key existed."""
    return ((cond or {}).get("job") or {}).get("status") or JOB_UNKNOWN


def _read_job_entry(path) -> dict:
    """The ``job`` entry stored in the condensed file at ``path`` (``{"status": "unknown"}`` when unreadable or absent)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            job = dict((json.load(fh) or {}).get("job") or {})
    except Exception:
        job = {}
    job.setdefault("status", JOB_UNKNOWN)
    return job


def _read_job_status(path) -> str:
    """``job.status`` stored in the condensed file at ``path`` (``unknown`` when unreadable or absent)."""
    return _read_job_entry(path)["status"]


def conformer_job_entry(conf_dir, condensed_file: Optional[str] = CONDENSED_FILENAME) -> dict:
    """The ``job`` entry of one conformer folder: status plus, when the file carries them, the step timings.

    From the condensed file when there is one (``status`` is ``unknown`` for a file that predates the key,
    ``timings`` absent for one that predates 2026-10-02), else judged from the raw ``calcParam.out``
    (status and timings), else ``{"status": "missing"}``.
    """
    conf_dir = Path(conf_dir)
    cond = find_condensed(conf_dir, condensed_file)
    if cond is not None:
        return _read_job_entry(cond)
    out = conf_dir / "calcParam.out"
    if out.is_file():
        text = out.read_text(errors="replace")
        return {"status": op.job_status(conf_dir, text)["status"], "timings": op.compound_step_timings(text)}
    return {"status": JOB_MISSING}


def conformer_job_status(conf_dir, condensed_file: Optional[str] = CONDENSED_FILENAME) -> str:
    """Job status of a conformer folder as the pipeline sees it.

    From the condensed file when there is one (``unknown`` if it predates the ``job`` key), else from the
    raw ``calcParam.out`` (``completed`` / ``failed`` / ``running`` / ``not_started``), else ``missing``:
    the folder holds neither, typically a job that had not started when the library was copied.
    """
    cond = find_condensed(conf_dir, condensed_file)
    if cond is not None:
        return _read_job_status(cond)
    if (Path(conf_dir) / "calcParam.out").is_file():
        return op.job_status(conf_dir)["status"]
    return JOB_MISSING


def write_condensed(conf_dir, data, name=CONDENSED_FILENAME, overwrite=False):
    """Write ``data`` atomically to ``<conf_dir>/<name>`` (compact JSON, no NaN) and return the path."""
    target = Path(conf_dir) / name
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, target)
    return target


def check_condensed(path):
    """List of problems with a condensed file (empty when it is consistent)."""
    p = Path(path)
    try:
        head = p.read_bytes()[:_SNIFF_BYTES].decode("utf-8", "replace")
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        return [f"unreadable: {exc}"]
    problems = []
    if '"molecule"' not in head or '"atoms"' not in head:
        problems.append(f"'molecule' / 'atoms' not within the first {_SNIFF_BYTES} bytes (DescriPyTor would skip the file)")
    problems += [f"missing key {k!r}" for k in REQUIRED_TOP_KEYS if k not in data]
    if problems:
        return problems
    if data["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version {data['schema_version']!r} != {SCHEMA_VERSION}")
    atoms = data["molecule"].get("atoms") or []
    n = len(atoms)
    if n == 0:
        problems.append("no atoms")
    if data["molecule"].get("n_atoms") != n:
        problems.append(f"molecule.n_atoms {data['molecule'].get('n_atoms')} != {n}")
    for key in ("nbo", "hirshfeld", "cm5"):
        vals = data["charges"].get(key)
        if vals is not None and len(vals) != n:
            problems.append(f"charges.{key}: {len(vals)} values for {n} atoms")
    for m in data["vibrations"].get("modes") or []:
        if len(m.get("displacements") or []) != n:
            problems.append(f"vibrations: mode {m.get('mode_index')} has {len(m.get('displacements') or [])} displacements")
            break
    for s, st in (data["states"] or {}).items():
        for scheme, vals in ((st or {}).get("charges") or {}).items():
            if vals is not None and len(vals) != n:
                problems.append(f"states.{s}.charges.{scheme}: {len(vals)} values for {n} atoms")
    if (data["states"] or {}).get("singlepoint") is None:
        problems.append("states.singlepoint missing")
    blocks = data["nbo"].get("blocks") or []
    if not blocks:
        problems.append("no NBO block")
    for b in blocks:
        npa_atoms = ((b.get("results") or {}).get("natural_population_analysis") or {}).get("atoms") or {}
        if len(npa_atoms) != n:
            problems.append(f"NBO block {b.get('index')}: NPA table has {len(npa_atoms)} atoms")
    return problems


def condense_folder(conf_dir, name=CONDENSED_FILENAME, overwrite=False, **kwargs):
    """Build and write the condensed file of one conformer folder; returns its path."""
    data = build_condensed(conf_dir, **kwargs)
    return write_condensed(conf_dir, data, name=name, overwrite=overwrite)


def iter_conformer_dirs(root, glob="conf_*"):
    """Conformer folders under ``root``: ``<root>/<molecule>/<glob>``, else flat ``<root>/<glob>``, else root itself."""
    root = Path(root)
    dirs = sorted(p for p in root.glob(f"*/{glob}") if p.is_dir())
    if not dirs:
        dirs = sorted(p for p in root.glob(glob) if p.is_dir())
    if not dirs and fnmatch.fnmatch(root.name, glob) and (root / "optimization.xyz").is_file():
        dirs = [root]
    return dirs


def _work(item):
    """Condense one folder; returns ``(folder, ok | skipped | failed, message, bytes, job status)``.

    An existing file is kept only if it says the job completed. Any other stored status (``running``,
    ``failed``, or ``unknown`` for a file written before the status existed) is rebuilt from the raw files,
    since the job may have finished since; without raw files there is nothing to rebuild from and the file
    is kept as it is.
    """
    conf_dir, name, overwrite, mo_states, full, decimals = item
    target = Path(conf_dir) / name
    previous = None
    if target.exists() and not overwrite:
        previous = _read_job_status(target)
        if previous == JOB_COMPLETED:
            return (str(conf_dir), "skipped", "exists (completed)", target.stat().st_size, previous)
        if not (Path(conf_dir) / "calcParam.out").is_file():
            return (str(conf_dir), "skipped", f"exists ({previous}); no raw files to rebuild from",
                    target.stat().st_size, previous)
    try:
        data = build_condensed(conf_dir, mo_states=mo_states, nbo_full_blocks=full, mode_decimals=decimals)
        out = write_condensed(conf_dir, data, name=name, overwrite=True)
    except Exception as exc:  # one bad folder must not stop the batch
        try:
            job = op.job_status(conf_dir)["status"]
        except Exception:
            job = JOB_UNKNOWN
        return (str(conf_dir), "failed", f"{type(exc).__name__}: {exc} [job {job}]", 0, job)
    size = out.stat().st_size
    job = data["job"]["status"]
    msg = f"{size / 1e6:.2f} MB"
    if job != JOB_COMPLETED:
        msg += f", job {job}: {data['job']['detail']}"
    if previous is not None:
        msg += f" (rebuilt, was {previous})"
    return (str(conf_dir), "ok", msg, size, job)


def condense_tree(dirs: Iterable[Union[str, Path]], name=CONDENSED_FILENAME, overwrite=False, jobs=1,
                  mo_states=("singlepoint",), nbo_full_blocks=(0,), mode_decimals=6, log=print):
    """Condense every folder in ``dirs``.

    Returns ``{"ok", "skipped", "failed", "bytes", "job_status", "results"}``; ``job_status`` counts the
    folders per ORCA job state (``completed`` / ``running`` / ``failed`` / ...), whatever happened to their
    condensed file, so the summary says at a glance whether the library is finished.
    """
    items = [(str(d), name, overwrite, tuple(mo_states), nbo_full_blocks, mode_decimals) for d in dirs]
    if jobs and jobs > 1 and len(items) > 1:
        import multiprocessing

        with multiprocessing.Pool(jobs) as pool:
            results = list(pool.imap_unordered(_work, items))
    else:
        results = [_work(it) for it in items]
    summary = {"ok": 0, "skipped": 0, "failed": 0, "bytes": 0, "job_status": {}, "results": sorted(results)}
    for d, status, msg, size, job in summary["results"]:
        summary[status] += 1
        summary["bytes"] += size
        summary["job_status"][job] = summary["job_status"].get(job, 0) + 1
        if log:
            log(f"{status:7s} {d}  {msg}")
    if log:
        log(summary_lines(summary))
    return summary


def job_status_line(counts) -> str:
    """``job status: completed 290, running 5, failed 5`` (known states first, in a fixed order)."""
    order = list(JOB_STATUSES) + [JOB_UNKNOWN, JOB_MISSING]
    keys = [k for k in order if counts.get(k)] + sorted(k for k in counts if k not in order and counts[k])
    return "job status: " + ", ".join(f"{k} {counts[k]}" for k in keys)


def summary_lines(summary) -> str:
    return (f"condensed {summary['ok']} folder(s), skipped {summary['skipped']}, failed {summary['failed']}, "
            f"total {summary['bytes'] / 1e6:.1f} MB\n" + job_status_line(summary["job_status"]))


# --------------------------------------------------------------------------- consumer side

_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_CACHE_SIZE = 4


def find_condensed(conf_dir, name: Optional[str] = CONDENSED_FILENAME) -> Optional[Path]:
    if not name:
        return None
    p = Path(conf_dir) / name
    return p if p.is_file() else None


def load_condensed(path):
    """Parsed condensed file (small LRU cache keyed by path, mtime and size)."""
    p = Path(path)
    key = str(p.resolve())
    st = p.stat()
    sig = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == sig:
        _CACHE.move_to_end(key)
        return hit[1]
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{p}: schema_version {data.get('schema_version')!r}; this code reads version {SCHEMA_VERSION}")
    _CACHE[key] = (sig, data)
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return data


def property_json_view(cond, basename):
    """Rebuild the ``<basename>.property.json`` structure ``OrcaReader`` reads from a condensed dict.

    Only the keys the reader accesses are present (``Geometries[-1]``: ``DFT_Energy.finalEn``,
    ``<Scheme>_Population_Analysis[0].AtomicCharges`` as column vectors, Hirshfeld ``ATNO``,
    ``Geometry.Coordinates.Cartesians`` in Bohr, ``Chemical_Shift``, ``THERMOCHEMISTRY_Energies``,
    ``Dipole_Moment``). ``{}`` when the state is not in the file.
    """
    st = (cond.get("states") or {}).get(basename)
    if st is None:
        return {}
    atoms = (cond.get("molecule") or {}).get("atoms") or []
    G: dict = {
        "Geometry": {
            "NAtoms": len(atoms),
            "Coordinates": {
                "Cartesians": [[a["atom"], a["x"] / bohr_to_A, a["y"] / bohr_to_A, a["z"] / bohr_to_A] for a in atoms],
                "Type": "Cartesians", "Units": "a.u.",
            },
        },
    }
    if st.get("scf_energy_eh") is not None:
        G["DFT_Energy"] = {"finalEn": st["scf_energy_eh"], "nAlphaEl": st.get("n_alpha"),
                           "nBetaEl": st.get("n_beta"), "nTotalEl": st.get("n_electrons")}
    for scheme, name in op._POPULATION_KEYS.items():
        vals = (st.get("charges") or {}).get(scheme)
        if vals is None:
            continue
        entry = {"AtomicCharges": [[v] for v in vals], "NAtoms": len(vals)}
        if scheme == "hirshfeld" and st.get("atomic_numbers"):
            entry["ATNO"] = [[z] for z in st["atomic_numbers"]]
        G[name] = [entry]
    if st.get("thermochemistry"):
        th = st["thermochemistry"]
        G["THERMOCHEMISTRY_Energies"] = [{"enthalpyH": th.get("enthalpy_eh"), "freeEnergyG": th.get("gibbs_eh")}]
    if st.get("nmr"):
        nmr = st["nmr"]
        G["Chemical_Shift"] = [{"siso": nmr.get("siso_ppm"), "saniso": nmr.get("saniso_ppm"),
                                "NUC": nmr.get("nuc"), "Elems": nmr.get("elems")}]
    if st.get("dipole_au"):
        G["Dipole_Moment"] = [{"dipoleTotal": [[v] for v in st["dipole_au"]],
                               "dipoleMagnitude": st.get("dipole_magnitude_au")}]
    return {"Geometries": [G]}


def gbw_view(cond, basename):
    """Rebuild the ``Molecule.MolecularOrbitals.MOs`` structure ``OrcaReader.get_homolumo`` reads (``{}`` if absent)."""
    st = (cond.get("states") or {}).get(basename) or {}
    mo = st.get("mo")
    if not mo:
        return {}
    mos = [{"OrbitalEnergy": e, "Occupancy": o} for e, o in zip(mo.get("energies_eh") or [], mo.get("occupations") or [])]
    return {"Molecule": {"MolecularOrbitals": {"MOs": mos, "EnergyUnit": mo.get("energy_unit")},
                         "Charge": mo.get("charge"), "Multiplicity": mo.get("multiplicity")}}


def nbo_results(cond, block_index=0, spin="alpha", require_full=True):
    """``NBOProps.results`` of one NBO block, filtered to ``spin`` like ``NBOProps(spin=...)`` would parse it.

    Raises ``ValueError`` when ``require_full`` and the block was stored at level ``npa`` (re-run
    ``condense --nbo-full all``). Mirrors ``NBOProps``: an out-of-range index falls back to block 0.
    """
    if spin not in NBOProps.SPIN_CHOICES:
        raise ValueError(f"spin must be one of {NBOProps.SPIN_CHOICES}, got {spin!r}")
    blocks = (cond.get("nbo") or {}).get("blocks") or []
    if not blocks:
        raise ValueError("condensed file has no NBO block")
    if not (0 <= int(block_index) < len(blocks)):
        print(f"Requested NBO block index {block_index} out of range (found {len(blocks)} blocks). Using block 0.")
        block_index = 0
    blk = blocks[int(block_index)]
    if require_full and blk.get("level") != "full":
        raise ValueError(f"NBO block {block_index} is stored at level {blk.get('level')!r} (NPA only); "
                         f"re-run `python -m desc_utils.condense --nbo-full all --overwrite` for the full tables")
    res = dict(blk["results"])
    for k in TABLE_KEYS:
        res[k] = unpack_table(res.get(k))
    if res.get("spin") is None:  # closed shell: a single spin-summed set
        return res
    wanted = ("alpha", "beta") if spin == "both" else (spin,)
    for k in SPIN_TABLE_KEYS:
        rows = res.get(k)
        if isinstance(rows, list):
            res[k] = [r for r in rows if r.get("spin") in wanted]
    res["spin"] = spin
    return res


def scf_energy(cond, basename="singlepoint"):
    st = (cond.get("states") or {}).get(basename) or {}
    return st.get("scf_energy_eh")


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m desc_utils.condense", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--conf-dir", action="append", help="conformer folder (repeatable)")
    g.add_argument("--root", help="library root: <root>/<molecule>/<conformer> (or flat <root>/<conformer>)")
    p.add_argument("--glob", default="conf_*", help="conformer folder pattern under --root (default conf_*)")
    p.add_argument("--name", default=CONDENSED_FILENAME, help=f"output file name (default {CONDENSED_FILENAME})")
    p.add_argument("--overwrite", action="store_true", help="rewrite existing condensed files")
    p.add_argument("--jobs", type=int, default=1, help="parallel workers (keep small on a login node)")
    p.add_argument("--check", action="store_true", help="validate existing condensed files instead of writing them")
    p.add_argument("--mo-states", default="singlepoint",
                   help="comma list of states whose MO energies / occupations are stored, or 'all' (default singlepoint)")
    p.add_argument("--nbo-full", default="0",
                   help="comma list of NBO block indices stored with every table, or 'all' (default 0; the other "
                        "blocks keep NPA charges and cycle info only)")
    p.add_argument("--mode-decimals", type=int, default=6, help="decimals kept for normal-mode displacements (default 6)")
    p.add_argument("--quiet", action="store_true", help="only print the summary")
    args = p.parse_args(argv)

    dirs = [Path(d) for d in args.conf_dir] if args.conf_dir else iter_conformer_dirs(args.root, args.glob)
    if not dirs:
        print("no conformer folders found", file=sys.stderr)
        return 1
    if args.check:
        bad = 0
        job_counts: dict = {}
        for d in dirs:
            path = Path(d) / args.name
            problems = check_condensed(path) if path.is_file() else ["missing"]
            job = _read_job_status(path) if path.is_file() else JOB_MISSING
            job_counts[job] = job_counts.get(job, 0) + 1
            if problems:
                bad += 1
                print(f"FAIL {path}: {'; '.join(problems)}")
            elif job != JOB_COMPLETED:
                print(f"ok   {path}  (job {job})")
            elif not args.quiet:
                print(f"ok   {path}")
        print(f"checked {len(dirs)} file(s): {len(dirs) - bad} ok, {bad} failed")
        print(job_status_line(job_counts))
        return 1 if bad else 0
    mo_states = STATES if args.mo_states.strip() == "all" else tuple(s.strip() for s in args.mo_states.split(",") if s.strip())
    full = "all" if args.nbo_full.strip() == "all" else tuple(int(x) for x in args.nbo_full.split(",") if x.strip())
    summary = condense_tree(dirs, name=args.name, overwrite=args.overwrite, jobs=args.jobs, mo_states=mo_states,
                            nbo_full_blocks=full, mode_decimals=args.mode_decimals,
                            log=(lambda s: None) if args.quiet else print)
    if args.quiet:
        print(summary_lines(summary))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
