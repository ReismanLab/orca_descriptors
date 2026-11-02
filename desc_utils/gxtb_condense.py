"""Condensed per-conformer file for a g-xTB stage-6 run, in the schema ``desc_utils.condense`` defines.

A g-xTB conformer folder looks like this, one subfolder per step the config asked for::

    conf_N/
        input.xyz              the geometry the stage-4 ensemble produced
        optimization.xyz       the geometry the properties are on (a copy of input.xyz without opt_geom)
        gxtb/
            run.json           the resolved options and the wall time of every step
            optimization/      xtb.out, xtbopt.xyz          (only with opt_geom)
            singlepoint/       xtb.out, charges, wbo
            frequency/         xtb.out, g98.out             (only with freq_calc)
            oxidation/         xtb.out, charges             (only with vertical_redox)
            reduction/         xtb.out, charges             (only with vertical_redox)
        orca_condensed.json    written here

The output file carries the **same name, same schema version and same section layout** as the ORCA one,
so nothing downstream changes: ``OrcaReader``, ``desc_utils.data_extraction`` and DescriPyTor's JSON
reader all read it as they always have, and a g-xTB spreadsheet lines up column-for-column with a DFT one.
The sections g-xTB cannot fill are ``null`` or empty, which every consumer already tolerates
(``desc_utils.aggregate`` drops all-NaN columns, so they simply do not appear in the output table).

What differs from an ORCA file, and why:

``charges.nbo`` / ``charges.cm5``
    both hold the g-xTB Mulliken-type atomic charges. g-xTB has no NPA, Hirshfeld or CM5 analysis, but
    these are the slots the descriptor families read (``charge_NBO_*`` for DescriPyTor, the ``cm5``
    prefix for the redox family), so filling them keeps the column names identical between engines.
    ``charges.note`` records what the numbers really are, as the ORCA path already does for cm5.
``nbo.blocks``
    empty. ``data_extraction.get_all_nbo_properties`` catches the resulting error into an
    ``nbo_extraction_error`` column, so the NBO descriptors come out NaN and are dropped.
``states.nmr``, ``properties.polarizability``
    ``null``: g-xTB computes neither.
``gxtb``
    an extra section with the data g-xTB offers and ORCA does not: Wiberg/Mayer bond orders, Fukui
    indices and shell populations. Every consumer reads with ``.get()``, so an extra key is inert.
"""

from __future__ import annotations

import json
import platform
import socket
from pathlib import Path
from typing import Optional

from desc_utils import gxtb_parse as gp
from desc_utils.condense import AU_TO_DEBYE, GXTB_ENGINE, SCHEMA_VERSION, CondenseError, _git_commit, _utc
from desc_utils.orca_parse import CONDENSED_FILENAME, JOB_COMPLETED, JOB_FAILED, JOB_RUNNING

#: the engine name stored in ``producer.engine``; ``condense.check_condensed`` keys its NBO exemption on it
ENGINE = GXTB_ENGINE
#: subfolder of the conformer directory holding one folder per step
RUN_DIRNAME = "gxtb"
#: the file the runner writes with the resolved options and per-step wall times
RUN_RECORD = "run.json"
#: states of the condensed schema, in the order they are written; ``nmr`` is always null for g-xTB
STATES = ("singlepoint", "oxidation", "reduction", "nmr")
#: every step a g-xTB conformer job can run, in the order ``desc_utils.submit.cost`` expects them
STEPS = ("optimization", "singlepoint", "frequency", "oxidation", "reduction")

CHARGES_NOTE = ("nbo/cm5: g-xTB atomic charges (Mulliken-type q from the shell populations); g-xTB has "
                "no NPA, Hirshfeld or CM5 analysis, and these are the slots the descriptor families read")


def _step_dir(conf_dir: Path, step: str) -> Path:
    return Path(conf_dir) / RUN_DIRNAME / step


def _step_text(conf_dir: Path, step: str) -> Optional[str]:
    """``xtb.out`` of one step, or None when the step did not run (or left nothing behind)."""
    out = _step_dir(conf_dir, step) / "xtb.out"
    return out.read_text(errors="replace") if out.is_file() else None


def read_run_record(conf_dir: Path) -> dict:
    """The runner's ``gxtb/run.json``; an empty dict when it is missing (a hand-run folder)."""
    p = Path(conf_dir) / RUN_DIRNAME / RUN_RECORD
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text()) or {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- one state

def state_from_step(conf_dir: Path, step: str, basename: str) -> Optional[dict]:
    """One ``states`` entry from a step's output, shaped like ``orca_parse.property_state``.

    ``charges.mulliken`` is the honest slot for g-xTB's ``q``; ``loewdin``, ``chelpg`` and ``hirshfeld``
    stay ``None`` because g-xTB does not compute them.
    """
    text = _step_text(conf_dir, step)
    if text is None:
        return None
    gp.check_is_gxtb(text)
    counts = gp.electron_counts(text)
    # the charges file carries full precision; the log's table is the fallback
    charges = gp.charges_file(_step_dir(conf_dir, step) / "charges") or gp.atomic_charges(text)
    return {
        "basename": basename,
        "n_geometries": 1,
        "n_atoms": len(charges) if charges else None,
        "charge": counts["charge"],
        "multiplicity": counts["multiplicity"],
        "n_alpha": counts["n_alpha"],
        "n_beta": counts["n_beta"],
        "n_electrons": counts["n_electrons"],
        "scf_energy_eh": gp.total_energy(text),
        "atomic_numbers": gp.atomic_numbers(text),
        "charges": {"mulliken": charges, "loewdin": None, "chelpg": None, "hirshfeld": None},
        "dipole_au": [d["x"], d["y"], d["z"]] if (d := gp.dipole(text)) else None,
        "dipole_magnitude_au": d["total"] if d else None,
        "polarizability": None,     # --alpha is silently ignored under --gxtb
        "thermochemistry": gp.thermochemistry(text),
        "nmr": None,                # g-xTB has no GIAO shieldings
        "mo": gp.orbitals(text),
    }


# --------------------------------------------------------------------------- the file

def build_condensed(conf_dir: Path | str) -> dict:
    """The condensed dict for one g-xTB conformer folder.

    Raises :class:`CondenseError` when the folder has no usable single point; a missing optional step
    (a frequency run that aborted, redox that was not asked for) is left ``null`` and noted.
    """
    conf_dir = Path(conf_dir)
    if not conf_dir.is_dir():
        raise CondenseError(f"{conf_dir}: not a directory")
    xyz_path = conf_dir / "optimization.xyz"
    if not xyz_path.is_file():
        raise CondenseError(f"{conf_dir}: optimization.xyz missing")
    sp_text = _step_text(conf_dir, "singlepoint")
    if sp_text is None:
        raise CondenseError(f"{conf_dir}: {RUN_DIRNAME}/singlepoint/xtb.out missing")
    gp.check_is_gxtb(sp_text)

    symbols, coords = _read_xyz(xyz_path)
    n_atoms = len(symbols)
    record = read_run_record(conf_dir)
    notes: list[str] = []

    states: dict[str, Optional[dict]] = {
        "singlepoint": state_from_step(conf_dir, "singlepoint", "singlepoint"),
        "oxidation": state_from_step(conf_dir, "oxidation", "oxidation"),
        "reduction": state_from_step(conf_dir, "reduction", "reduction"),
        "nmr": None,
    }
    # a frequency run is a second calculation on the same geometry: its thermochemistry belongs to the
    # single point's state, which is where OrcaReader.get_enthalpy and get_gfe look for it
    freq_text = _step_text(conf_dir, "frequency")
    if freq_text is not None and states["singlepoint"] is not None:
        thermo = gp.thermochemistry(freq_text)
        if thermo:
            states["singlepoint"]["thermochemistry"] = thermo
        if not gp.terminated_normally(freq_text):
            notes.append("frequency: the g-xTB run did not terminate normally; "
                         "frequencies and thermochemistry may be missing")

    for name, st in states.items():
        if st is None:
            continue
        for scheme, vals in st["charges"].items():
            if vals is not None and len(vals) != n_atoms:
                raise CondenseError(f"{name}: {len(vals)} {scheme} charges for {n_atoms} atoms")

    sp = states["singlepoint"]
    charges = sp["charges"]["mulliken"] if sp else None
    vib = gp.frequencies(_step_dir(conf_dir, "frequency") / "g98.out") or {
        "modes": [], "source": None, "n_dropped_zero_modes": 0, "n_imaginary": 0}
    for mode in vib["modes"]:
        if len(mode["displacements"]) != n_atoms:
            raise CondenseError(
                f"frequency: mode {mode['mode_index']} has {len(mode['displacements'])} displacements "
                f"for {n_atoms} atoms")
    if not vib["modes"] and freq_text is not None:
        notes.append("frequency: no g98.out; the vibration descriptors will be empty")

    dip = gp.dipole(sp_text)
    dipole = None
    if dip:
        dipole = {"x": dip["x"] * AU_TO_DEBYE, "y": dip["y"] * AU_TO_DEBYE, "z": dip["z"] * AU_TO_DEBYE,
                  "total": dip["total"] * AU_TO_DEBYE, "units": "debye", "state": "singlepoint",
                  "au_to_debye": AU_TO_DEBYE}

    molecule = {
        "atoms": [{"atom": s, "x": c[0], "y": c[1], "z": c[2]} for s, c in zip(symbols, coords)],
        "units": "angstrom",
        "source": "optimization.xyz",
        "n_atoms": n_atoms,
        "charge": sp.get("charge") if sp else None,
        "multiplicity": sp.get("multiplicity") if sp else None,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "producer": _producer(conf_dir, sp_text, record),
        "molecule": molecule,
        "properties": {
            "dipole": dipole,
            "polarizability": None,
            # the key name is the ORCA one on purpose: it is what DescriPyTor's JSON reader looks for
            "energies": {"final_single_point_energy_hartree_orca": sp.get("scf_energy_eh") if sp else None},
        },
        "vibrations": vib,
        "charges": {"nbo": charges, "hirshfeld": None, "cm5": list(charges) if charges else None,
                    "note": CHARGES_NOTE},
        "states": states,
        "nbo": {"source_file": None, "n_blocks": 0, "block_map": {}, "blocks": []},
        "gxtb": {
            "wiberg": gp.wiberg(_step_dir(conf_dir, "singlepoint") / "wbo"),
            "fukui": gp.fukui(sp_text),
            "shell_populations": gp.shell_populations(sp_text),
            "options": record.get("options"),
        },
        "notes": notes,
        "job": _job_entry(conf_dir, record),
    }


def _producer(conf_dir: Path, sp_text: str, record: dict) -> dict:
    """Kept small on purpose: DescriPyTor only sniffs the first 8192 bytes for ``molecule``/``atoms``."""
    return {
        "tool": "desc_utils.gxtb_condense",
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "created_utc": _utc(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "git_commit": _git_commit(Path(__file__).resolve().parent.parent),
        "xtb_version": gp.version(sp_text) or record.get("xtb_version"),
        "steps": sorted(record.get("steps") or {}) or [s for s in STEPS if _step_text(conf_dir, s) is not None],
        "conformer_dir": str(conf_dir.resolve()),
    }


def _job_entry(conf_dir: Path, record: dict) -> dict:
    """``job.status`` and the per-step timings, in the shape ``desc_utils.submit.cost`` sums."""
    steps = record.get("steps") or {}
    timings = {"total_run_time_s": record.get("wall_time_s"), "nprocs": record.get("nprocs"),
               "steps": {name: {"index": i, "wall_s": (steps.get(name) or {}).get("wall_s"), "modules": {}}
                         for i, name in enumerate(STEPS) if name in steps}}
    status = record.get("status")
    if status not in (JOB_COMPLETED, JOB_FAILED, JOB_RUNNING):
        # a folder condensed by hand: the single point is there, so the calculation ran
        status = JOB_COMPLETED if _step_text(conf_dir, "singlepoint") else JOB_FAILED
    return {"status": status, "detail": record.get("detail"), "checked_utc": _utc(), "timings": timings}


def _read_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    lines = path.read_text().splitlines()
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise CondenseError(f"{path}: not an xyz file ({exc})") from exc
    symbols, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        if len(parts) < 4:
            raise CondenseError(f"{path}: short coordinate line {line!r}")
        symbols.append(parts[0])
        coords.append(tuple(float(v) for v in parts[1:4]))
    if len(symbols) != n:
        raise CondenseError(f"{path}: header says {n} atoms, found {len(symbols)}")
    return symbols, coords


def condense_folder(conf_dir: Path | str, overwrite: bool = True,
                    name: str = CONDENSED_FILENAME) -> dict:
    """Write ``<conf_dir>/<name>``. Returns ``{"ok": True, "at": ...}`` or the error, never raising —
    the runner stores the result in the conformer's record exactly as the ORCA path does."""
    conf_dir = Path(conf_dir)
    out = conf_dir / name
    if out.is_file() and not overwrite:
        return {"ok": True, "at": _utc(), "skipped": True}
    try:
        data = build_condensed(conf_dir)
    except Exception as exc:
        import traceback
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                "at": _utc()}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(out)
    return {"ok": True, "at": _utc()}


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Condense a g-xTB conformer folder (or a whole library).")
    ap.add_argument("--conf-dir", help="one conformer folder")
    ap.add_argument("--root", help="a library: every <root>/*/conf_*/ with a gxtb/ subfolder")
    ap.add_argument("--name", default=CONDENSED_FILENAME, help="output file name")
    ap.add_argument("--overwrite", action="store_true", help="rebuild files that already exist")
    args = ap.parse_args(argv)
    if not args.conf_dir and not args.root:
        ap.error("give --conf-dir or --root")
    dirs = [Path(args.conf_dir)] if args.conf_dir else sorted(
        p.parent for p in Path(args.root).glob("*/conf_*/" + RUN_DIRNAME) if p.is_dir())
    bad = 0
    for d in dirs:
        res = condense_folder(d, overwrite=args.overwrite, name=args.name)
        bad += not res["ok"]
        print(f"{d}: {'ok' if res['ok'] else res['error']}")
    print(f"{len(dirs) - bad}/{len(dirs)} condensed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
