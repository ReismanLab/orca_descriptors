"""Readers for the output of one g-xTB run (``xtb --gxtb``), the stage-6 counterpart of ``orca_parse``.

g-xTB is the Grimme group's general-purpose tight-binding method, shipped as a statically linked build
of ``xtb`` 6.7.1 that runs it when ``--gxtb`` is given (https://github.com/grimme-lab/g-xTB). One run
writes its log plus a few machine-readable side files into its own folder; the functions here turn those
into the pieces ``desc_utils.gxtb_condense`` assembles into a condensed file.

Everything is stdlib only and takes text (or a path to a side file), never a running process: the
condenser runs on the cluster next to the job, and the same functions are what the tests read the
committed fixtures with.

What one ``--gxtb`` run offers, measured against the standard ORCA compound job:

===========================  ==============================================================
total energy, MO energies    yes (``* Orbital Energies and Occupations``, alpha and beta)
atomic charges               yes, Mulliken-type ``q`` (plus shell populations)
dipole                       yes, in atomic units and Debye
Wiberg/Mayer bond orders     yes (``wbo`` file)
Fukui indices                yes, with ``--vfukui``
frequencies, IR intensities  yes, with ``--hess`` (Gaussian-98 format in ``g98.out``)
thermochemistry              yes, with ``--hess``
NBO / NPA, NMR shieldings    no
Hirshfeld, CHELPG, CM5       no
polarizability               no (``--alpha`` is silently ignored under ``--gxtb``)
solvation                    no (ALPB/GBSA/COSMO are not parameterised for g-xTB)
===========================  ==============================================================
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

#: ``xtb`` ends a successful run with this line; anything else is a failure, however it exited.
TERMINATION = "normal termination of xtb"

HARTREE_TO_EV = 27.211386245988

_VERSION = re.compile(r"\*\s*xtb version\s+(\S+)")
_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_ENTHALPY = re.compile(r"TOTAL ENTHALPY\s+(-?\d+\.\d+)\s+Eh")
_FREE_ENERGY = re.compile(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)\s+Eh")
_ZPVE = re.compile(r"zero point energy\s+(-?\d+\.\d+)\s+Eh")
#: the thermo table's first column, ``ate 298.15  VIB   1.02 ...``
_THERMO_TEMP = re.compile(r"^\s*(\d+\.\d+)\s+VIB\s", re.MULTILINE)
_RUN_TIME = re.compile(r"\*\s*wall-time:\s*(\d+)\s*d,\s*(\d+)\s*h,\s*(\d+)\s*min,\s*([\d.]+)\s*sec")
_N_ELECTRONS = re.compile(r"number of electrons\s+(-?[\d.]+E?[-+]?\d*)\s+e")
_N_UNPAIRED = re.compile(r"number of unpaired electrons\s+(-?[\d.]+E?[-+]?\d*)\s+e")
_MOL_CHARGE = re.compile(r"molecular charge\s+(-?[\d.]+E?[-+]?\d*)\s+e")
#: a standard xtb warns about the flag it does not know and carries on with GFN2: that is not a g-xTB result
UNKNOWN_FLAG = re.compile(r"unknown (?:option|argument|flag)[^\n]*gxtb", re.IGNORECASE)

_ORBITAL_HEADER = re.compile(r"^\s*(?:(alpha|beta)\s+)?#\s+Occupation\s+Energy/Eh")
#: ``1        2.0000      -0.7725476     -21.0221 (HOMO)`` — the occupation is blank for a virtual
_ORBITAL_ROW = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)?\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*(?:\(\w+\))?\s*$")
#: ``    ...           ...                  ...                  ...`` — a long table's elided middle
_ORBITAL_ELLIPSIS = re.compile(r"^\s*\.\.\.(\s+\.\.\.)+\s*$")
#: ``1    6 C      -0.357729      1.177870   3.179858``
_CHARGE_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,2})\s+(-?\d+\.\d+)((?:\s+-?\d+\.\d+)*)\s*$")
#: ``1C       0.493    0.392    0.443``
_FUKUI_ROW = re.compile(r"^\s*(\d+)([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_DIPOLE_TOTAL = re.compile(r"^\s*total\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
_DIPOLE_MAGNITUDE = re.compile(r"\|total\|\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+Debye")


class GxtbParseError(Exception):
    """A g-xTB output cannot be read (truncated, or not a g-xTB run at all)."""


# --------------------------------------------------------------------------- run-level facts

def check_is_gxtb(text: str, exe: str = "xtb") -> None:
    """Raise when ``text`` came from a standard xtb that ignored ``--gxtb`` and ran GFN2 instead."""
    if UNKNOWN_FLAG.search(text):
        raise GxtbParseError(
            f"{exe} does not know --gxtb: it is not the g-xTB build of xtb "
            f"(https://github.com/grimme-lab/g-xTB). Put that binary first on PATH or set "
            f"dft.gxtb.executable to its path.")


def terminated_normally(text: str) -> bool:
    return TERMINATION in text


def version(text: str) -> Optional[str]:
    m = _VERSION.search(text)
    return m.group(1) if m else None


def run_time_s(text: str) -> Optional[float]:
    """Wall time of the run, from the first ``* wall-time:`` line (the whole-run total)."""
    m = _RUN_TIME.search(text)
    if not m:
        return None
    d, h, mi, s = m.groups()
    return int(d) * 86400 + int(h) * 3600 + int(mi) * 60 + float(s)


def total_energy(text: str) -> Optional[float]:
    """The last ``TOTAL ENERGY`` in Hartree (an optimisation prints one per step)."""
    found = _ENERGY.findall(text)
    return float(found[-1]) if found else None


def _last_float(pattern: re.Pattern, text: str) -> Optional[float]:
    found = pattern.findall(text)
    return float(found[-1]) if found else None


def thermochemistry(text: str) -> Optional[dict]:
    """``{enthalpy_eh, gibbs_eh, zpe_eh}`` from a ``--hess`` run, ``None`` without one.

    The key names are the ones ``orca_parse.property_state`` uses, so the condensed file reads the
    same whichever engine wrote it.
    """
    enthalpy = _last_float(_ENTHALPY, text)
    gibbs = _last_float(_FREE_ENERGY, text)
    if enthalpy is None and gibbs is None:
        return None
    return {"enthalpy_eh": enthalpy, "gibbs_eh": gibbs, "zpe_eh": _last_float(_ZPVE, text),
            "temperature_K": _last_float(_THERMO_TEMP, text)}


def electron_counts(text: str) -> dict:
    """``{n_electrons, n_unpaired, n_alpha, n_beta, charge, multiplicity}``.

    xtb prints the counts as floats (``1.3999999999900E+01``); they are rounded here.
    """
    def count(pattern):
        m = pattern.search(text)
        return round(float(m.group(1))) if m else None

    n_el, n_unpaired, charge = count(_N_ELECTRONS), count(_N_UNPAIRED), count(_MOL_CHARGE)
    out = {"n_electrons": n_el, "n_unpaired": n_unpaired, "charge": charge,
           "n_alpha": None, "n_beta": None, "multiplicity": None}
    if n_el is not None and n_unpaired is not None:
        out["n_alpha"] = (n_el + n_unpaired) // 2
        out["n_beta"] = (n_el - n_unpaired) // 2
        out["multiplicity"] = n_unpaired + 1
    return out


# --------------------------------------------------------------------------- orbitals

def orbitals(text: str) -> Optional[dict]:
    """``{energies_eh, occupations, indices, energy_unit, spin, truncated}`` of the orbital table.

    An unrestricted run prints an ``alpha`` table and a ``beta`` one; the alpha set is returned as
    ``energies_eh`` (what ``OrcaReader.get_homolumo`` reads, matching the ORCA path's ``spin: alpha``
    default) with the beta set under ``beta``. Virtual orbitals have a blank occupation column and are
    stored as ``0.0``.

    Above ~40 orbitals xtb prints only the two ends of the table and elides the middle with a ``...``
    row, so ``energies_eh`` can have a gap: ``indices`` carries the orbital number each entry really
    had and ``truncated`` says whether any were left out. The frontier is always printed, which is
    what HOMO/LUMO need (``get_homolumo`` takes the highest occupied and the lowest virtual, so the
    missing middle cannot change either).
    """
    tables: dict[str, dict] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = _ORBITAL_HEADER.match(lines[i])
        if not header:
            i += 1
            continue
        spin = header.group(1) or "restricted"
        energies: list[float] = []
        occupations: list[float] = []
        indices: list[int] = []
        truncated = False
        i += 1
        while i < len(lines):
            line = lines[i]
            row = _ORBITAL_ROW.match(line)
            if row:
                indices.append(int(row.group(1)))
                occupations.append(float(row.group(2)) if row.group(2) else 0.0)
                energies.append(float(row.group(3)))
            elif _ORBITAL_ELLIPSIS.match(line):
                truncated = True          # xtb printed only the ends of a long table
            elif "HL-Gap" in line:
                break
            elif line.strip() and not line.strip().startswith("---"):
                break
            i += 1
        if energies:
            tables[spin] = {"energies_eh": energies, "occupations": occupations,
                            "indices": indices, "truncated": truncated}
    if not tables:
        return None
    main = tables.get("alpha") or tables.get("restricted")
    if main is None:
        return None
    out = {**main, "energy_unit": "Eh", "spin": "alpha" if "alpha" in tables else "restricted"}
    if "beta" in tables:
        out["beta"] = tables["beta"]
    return out


# --------------------------------------------------------------------------- per-atom tables

def atomic_charges(text: str) -> Optional[list[float]]:
    """Mulliken-type ``q`` per atom from the ``Atomic charges and shell populations`` table."""
    table = _charge_table(text)
    return [row["charge"] for row in table] if table else None


def shell_populations(text: str) -> Optional[list[dict]]:
    """``[{atom, element, atomic_number, populations: [p(s), p(p), ...]}]``, one entry per atom."""
    table = _charge_table(text)
    if not table:
        return None
    return [{"atom": r["index"], "element": r["element"], "atomic_number": r["atomic_number"],
             "populations": r["populations"]} for r in table]


def atomic_numbers(text: str) -> Optional[list[int]]:
    table = _charge_table(text)
    return [row["atomic_number"] for row in table] if table else None


def elements(text: str) -> Optional[list[str]]:
    table = _charge_table(text)
    return [row["element"] for row in table] if table else None


def _charge_table(text: str) -> Optional[list[dict]]:
    start = text.find("Atomic charges and shell populations")
    if start < 0:
        return None
    rows = []
    for line in text[start:].splitlines()[1:]:
        m = _CHARGE_ROW.match(line)
        if m:
            rows.append({"index": int(m.group(1)), "atomic_number": int(m.group(2)),
                         "element": m.group(3), "charge": float(m.group(4)),
                         "populations": [float(v) for v in m.group(5).split()]})
        elif rows and line.strip().startswith("---"):
            break
    return rows or None


def charges_file(path: Path | str) -> Optional[list[float]]:
    """The ``charges`` side file: one float per atom, one per line. Preferred over the log when present
    (full printed precision, no table to match)."""
    p = Path(path)
    if not p.is_file():
        return None
    values = [float(ln) for ln in p.read_text().split() if ln.strip()]
    return values or None


def fukui(text: str) -> Optional[list[dict]]:
    """``[{atom, element, f_plus, f_minus, f_zero}]`` from a ``--vfukui`` run."""
    start = text.find("Fukui functions:")
    if start < 0:
        return None
    rows = []
    for line in text[start:].splitlines()[2:]:
        m = _FUKUI_ROW.match(line)
        if m:
            rows.append({"atom": int(m.group(1)), "element": m.group(2), "f_plus": float(m.group(3)),
                         "f_minus": float(m.group(4)), "f_zero": float(m.group(5))})
        elif rows:
            break
    return rows or None


def wiberg(path: Path | str) -> Optional[list[dict]]:
    """The ``wbo`` side file: ``i j order`` triples, 1-based, one bond per line."""
    p = Path(path)
    if not p.is_file():
        return None
    bonds = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            bonds.append({"i": int(parts[0]), "j": int(parts[1]), "order": float(parts[2])})
    return bonds or None


# --------------------------------------------------------------------------- dipole

def dipole(text: str) -> Optional[dict]:
    """``{x, y, z, total}`` in atomic units from the ``Atomic dipole moments`` block.

    The block's ``total`` row carries the components and the ``|total|`` line the magnitude in both
    atomic units and Debye; the magnitude is taken from there rather than recomputed.
    """
    start = text.find("Atomic dipole moments")
    if start < 0:
        return None
    block = text[start:]
    comp = _DIPOLE_TOTAL.search(block)
    mag = _DIPOLE_MAGNITUDE.search(block)
    if not comp:
        return None
    x, y, z = (float(v) for v in comp.groups())
    return {"x": x, "y": y, "z": z, "total": float(mag.group(1)) if mag else (x * x + y * y + z * z) ** 0.5}


# --------------------------------------------------------------------------- frequencies

def frequencies(path: Path | str) -> Optional[dict]:
    """Normal modes from ``g98.out``, the Gaussian-98 file ``xtb --hess`` writes.

    Returns ``{"modes": [{mode_index, frequency_cm-1, ir_intensity_km_per_mol, displacements:
    [{dx, dy, dz}]}], "source", "n_imaginary"}`` in the shape ``desc_utils.condense`` uses, so
    DescriPyTor's vibration families read it unchanged. xtb has already dropped the translations and
    rotations, so every mode in the file is a real vibration.
    """
    p = Path(path)
    if not p.is_file():
        return None
    lines = p.read_text(errors="replace").splitlines()
    freqs: list[float] = []
    intensities: list[Optional[float]] = []
    displacements: list[list[dict]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("Frequencies --"):
            block_freqs = [float(v) for v in line.split("--", 1)[1].split()]
            block_inten: list[Optional[float]] = [None] * len(block_freqs)
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("Atom AN"):
                if lines[j].strip().startswith("IR Inten"):
                    block_inten = [float(v) for v in lines[j].split("--", 1)[1].split()]
                j += 1
            block_disp: list[list[dict]] = [[] for _ in block_freqs]
            j += 1  # the "Atom AN  X Y Z ..." header
            while j < len(lines):
                parts = lines[j].split()
                # ``  1   6    -0.00  -0.00  -0.06     0.70 ...``: atom, Z, then 3 floats per mode
                if len(parts) != 2 + 3 * len(block_freqs):
                    break
                for k in range(len(block_freqs)):
                    dx, dy, dz = (float(v) for v in parts[2 + 3 * k:5 + 3 * k])
                    block_disp[k].append({"dx": dx, "dy": dy, "dz": dz})
                j += 1
            freqs += block_freqs
            intensities += block_inten
            displacements += block_disp
            i = j
            continue
        i += 1
    if not freqs:
        return None
    modes = [{"mode_index": n, "frequency_cm-1": f, "ir_intensity_km_per_mol": ir, "displacements": disp}
             for n, (f, ir, disp) in enumerate(zip(freqs, intensities, displacements), start=1)]
    return {"modes": modes, "source": p.name, "n_dropped_zero_modes": 0,
            "n_imaginary": sum(1 for f in freqs if f < 0)}
