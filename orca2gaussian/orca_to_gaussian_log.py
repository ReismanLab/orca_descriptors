#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any

SYMBOL_TO_Z = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Co": 27,
    "Ni": 28,
    "Br": 35,
    "I": 53,
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _flatten_scalar(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _to_float(value: Any, default: float = 0.0) -> float:
    value = _flatten_scalar(value)
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    value = _flatten_scalar(value)
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _parse_cartesian_angstrom_blocks(text: str) -> list[tuple[int, list[tuple[str, float, float, float]]]]:
    """Parse all CARTESIAN COORDINATES (ANGSTROEM) blocks with line offsets."""
    lines = text.splitlines()
    blocks: list[tuple[int, list[tuple[str, float, float, float]]]] = []

    i = 0
    while i < len(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" not in lines[i]:
            i += 1
            continue

        start_idx = i
        i += 1

        while i < len(lines):
            s = lines[i].strip()
            if not s or set(s) == {"-"}:
                i += 1
                continue
            break

        coords: list[tuple[str, float, float, float]] = []
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                break
            if (
                "CARTESIAN COORDINATES" in s
                or "INTERNAL COORDINATES" in s
                or set(s) == {"-"}
            ):
                break

            parts = s.split()
            if len(parts) < 4:
                break

            symbol = parts[0]
            if not symbol or not symbol[0].isalpha():
                break

            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except Exception:
                break

            coords.append((symbol, x, y, z))
            i += 1

        if coords:
            blocks.append((start_idx, coords))

    return blocks


def _parse_orbital_energy_blocks(text: str) -> list[tuple[int, float, float]]:
    """Parse ORBITAL ENERGIES blocks and return (line_index, HOMO, LUMO)."""
    lines = text.splitlines()
    blocks: list[tuple[int, float, float]] = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "ORBITAL ENERGIES":
            i += 1
            continue

        block_start = i
        i += 1

        # Move to table rows after header lines.
        while i < len(lines) and "E(Eh)" not in lines[i]:
            i += 1
        if i >= len(lines):
            break
        i += 1

        occ_energies: list[float] = []
        virt_energies: list[float] = []
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                break
            m = re.match(r"^\d+\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)", s)
            if not m:
                break
            occ = _to_float(m.group(1), 0.0)
            ene = _to_float(m.group(2), 0.0)
            if occ > 1e-8:
                occ_energies.append(ene)
            else:
                virt_energies.append(ene)
            i += 1

        if occ_energies:
            homo = occ_energies[-1]
            lumo = virt_energies[0] if virt_energies else homo + 0.01
            blocks.append((block_start, homo, lumo))

    return blocks


def _parse_final_single_point_energies(text: str) -> list[tuple[int, float]]:
    lines = text.splitlines()
    out: list[tuple[int, float]] = []
    for i, line in enumerate(lines):
        m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d*\.\d+)", line)
        if m:
            out.append((i, _to_float(m.group(1), 0.0)))
    return out


def _parse_frequencies_from_text(text: str) -> list[float]:
    freqs: list[float] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+:\s*([-+]?\d*\.\d+)\s*cm\*\*-1", line)
        if m:
            freqs.append(_to_float(m.group(1), 0.0))
    return freqs


def _parse_labeled_orbital_energies_from_text(text: str) -> dict[str, tuple[float, float]]:
    """Map singlepoint/oxidation/reduction labels to HOMO/LUMO from ORCA text."""
    lines = text.splitlines()
    orbital_blocks = _parse_orbital_energy_blocks(text)
    if not orbital_blocks:
        return {}

    label_positions: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = re.search(r"Storing NBOs:\s*(singlepoint|oxidation|reduction)\.nbo", line)
        if m:
            label_positions[m.group(1)] = i

    out: dict[str, tuple[float, float]] = {}
    for key, pos in label_positions.items():
        prior = [b for b in orbital_blocks if b[0] < pos]
        if prior:
            _, homo, lumo = prior[-1]
            out[key] = (homo, lumo)

    return out


def _extract_optimized_geometry_from_calcparam(text: str) -> tuple[list[tuple[str, float, float, float]], list[int]]:
    """Get converged optimization geometry from FINAL ENERGY EVALUATION section."""
    coords_blocks = _parse_cartesian_angstrom_blocks(text)
    if not coords_blocks:
        return [], []

    lines = text.splitlines()
    marker_idx = -1
    for i, line in enumerate(lines):
        if "FINAL ENERGY EVALUATION AT THE STATIONARY POINT" in line:
            marker_idx = i
            break

    if marker_idx >= 0:
        chosen = next((item for item in coords_blocks if item[0] > marker_idx), coords_blocks[-1])
    else:
        chosen = coords_blocks[-1]

    _, coords = chosen
    atomic_numbers = [SYMBOL_TO_Z.get(symbol, 0) for symbol, _, _, _ in coords]
    return coords, atomic_numbers


def _extract_single_point_from_calcparam(text: str) -> dict[str, Any] | None:
    """Fallback extraction when ORCA JSON files are unavailable."""
    coords_blocks = _parse_cartesian_angstrom_blocks(text)
    if not coords_blocks:
        return None

    lines = text.splitlines()
    marker_idx = -1
    for i, line in enumerate(lines):
        if "FINAL ENERGY EVALUATION AT THE STATIONARY POINT" in line:
            marker_idx = i
            break

    if marker_idx >= 0:
        chosen = next((item for item in coords_blocks if item[0] > marker_idx), coords_blocks[-1])
    else:
        chosen = coords_blocks[-1]

    coord_idx, coords = chosen
    atomic_numbers = [SYMBOL_TO_Z.get(symbol, 0) for symbol, _, _, _ in coords]

    orbital_blocks = _parse_orbital_energy_blocks(text)
    if orbital_blocks:
        after_coord = [b for b in orbital_blocks if b[0] > coord_idx]
        _, homo, lumo = (after_coord[0] if after_coord else orbital_blocks[-1])
    else:
        homo, lumo = -0.2, -0.19

    energy_blocks = _parse_final_single_point_energies(text)
    if energy_blocks:
        after_coord_e = [e for e in energy_blocks if e[0] > coord_idx]
        _, energy = (after_coord_e[0] if after_coord_e else energy_blocks[-1])
    else:
        energy = 0.0

    n_atoms = len(coords)
    return {
        "coords": coords,
        "atomic_numbers": atomic_numbers,
        "mulliken_charges": [0.0] * n_atoms,
        "hirsh_charges": [0.0] * n_atoms,
        "cm5_like": [0.0] * n_atoms,
        "dipole": (0.0, 0.0, 0.0, 0.0),
        "polar": (0.0, 0.0),
        "energy": energy,
        "homo": homo,
        "lumo": lumo,
        "label": "Ground State",
    }


def _extract_single_point_data(conf_dir: Path, prefix: str = "singlepoint") -> dict[str, Any]:
    """Extract data for a specific single point calculation (singlepoint, oxidation, reduction)."""
    sp_prop = _load_json(conf_dir / f"{prefix}.property.json")
    sp_json = _load_json(conf_dir / f"{prefix}.json")
    
    sp_geom = (sp_prop.get("Geometries") or [{}])[0]

    coord_rows = (
        sp_geom.get("Geometry", {})
        .get("Coordinates", {})
        .get("Cartesians", [])
    )

    coords = []
    atomic_numbers = []
    for row in coord_rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        symbol = str(row[0])
        x, y, z = _to_float(row[1]), _to_float(row[2]), _to_float(row[3])
        coords.append((symbol, x, y, z))
        atomic_numbers.append(SYMBOL_TO_Z.get(symbol, 0))

    mulliken = (
        (sp_geom.get("Mulliken_Population_Analysis") or [{}])[0]
        .get("AtomicCharges", [])
    )
    mulliken_charges = [_to_float(item, 0.0) for item in mulliken]

    hirsh = (
        (sp_geom.get("Hirshfeld_Population_Analysis") or [{}])[0]
        .get("AtomicCharges", [])
    )
    hirsh_charges = [_to_float(item, 0.0) for item in hirsh]

    chelpg = (
        (sp_geom.get("CHELPG_Population_Analysis") or [{}])[0]
        .get("AtomicCharges", [])
    )
    cm5_like = [_to_float(item, 0.0) for item in chelpg]

    dipole = (sp_geom.get("Dipole_Moment") or [{}])[0]
    dipole_total = dipole.get("dipoleTotal", [0.0, 0.0, 0.0])
    dx = _to_float(dipole_total[0] if len(dipole_total) > 0 else 0.0)
    dy = _to_float(dipole_total[1] if len(dipole_total) > 1 else 0.0)
    dz = _to_float(dipole_total[2] if len(dipole_total) > 2 else 0.0)
    dmag = _to_float(dipole.get("dipoleMagnitude"), (dx * dx + dy * dy + dz * dz) ** 0.5)

    pol = (sp_geom.get("Polarizability") or [{}])[0]
    isotropic_pol = _to_float(pol.get("isotropicPolar"), 0.0)
    diag = pol.get("diagonalizedTensor", [])
    diag_vals = [_to_float(v, 0.0) for v in diag[:3]]
    anisotropic_pol = (max(diag_vals) - min(diag_vals)) if len(diag_vals) == 3 else 0.0

    energy = _to_float(
        ((sp_geom.get("Energy") or [{}])[0].get("totalEnergy", [0.0])),
        _to_float(sp_geom.get("DFT_Energy", {}).get("finalEn"), 0.0),
    )

    mos = (
        sp_json.get("Molecule", {})
        .get("MolecularOrbitals", {})
        .get("MOs", [])
    )
    occ_energies = []
    virt_energies = []
    for mo in mos:
        occ = _to_float(mo.get("Occupancy"), 0.0)
        ene = _to_float(mo.get("OrbitalEnergy"), 0.0)
        if occ > 1e-8:
            occ_energies.append(ene)
        else:
            virt_energies.append(ene)

    homo = occ_energies[-1] if occ_energies else None
    lumo = virt_energies[0] if virt_energies else None

    return {
        "coords": coords,
        "atomic_numbers": atomic_numbers,
        "mulliken_charges": mulliken_charges,
        "hirsh_charges": hirsh_charges,
        "cm5_like": cm5_like,
        "dipole": (dx, dy, dz, dmag),
        "polar": (isotropic_pol, anisotropic_pol),
        "energy": energy,
        "homo": homo,
        "lumo": lumo,
    }


def _extract_data(conf_dir: Path) -> dict[str, Any]:
    """Extract all data including multiple single point calculations."""
    fr_prop = _load_json(conf_dir / "frequency.property.json")
    nmr_prop = _load_json(conf_dir / "nmr.property.json")
    calcparam_path = conf_dir / "calcParam.out"
    calc_text = ""
    if calcparam_path.exists():
        try:
            calc_text = calcparam_path.read_text(errors="ignore")
        except Exception:
            calc_text = ""

    fr_geom = (fr_prop.get("Geometries") or [{}])[0]

    # Extract frequency and normal modes data
    thermo = (fr_geom.get("THERMOCHEMISTRY_Energies") or [{}])[0]
    freq_raw = thermo.get("FREQ", [])
    freqs = [_to_float(f, 0.0) for f in freq_raw]

    modes = fr_geom.get("Hessian", {}).get("MODES", [])
    mode_vectors = []
    for mode in modes:
        if isinstance(mode, list):
            mode_vectors.append([_to_float(v, 0.0) for v in mode])

    # Extract NMR data
    nmr_geom = (nmr_prop.get("Geometries") or [{}])[0]
    nmr_shift = (nmr_geom.get("Chemical_Shift") or [{}])[0]
    nmr_siso = [_to_float(v, 0.0) for v in (nmr_shift.get("siso") or [])]
    nmr_saniso = [_to_float(v, 0.0) for v in (nmr_shift.get("saniso") or [])]
    nmr_elems = [_to_int(v, 0) for v in (nmr_shift.get("Elems") or [])]
    nmr_nuc = [_to_int(v, 0) for v in (nmr_shift.get("NUC") or [])]

    # Extract NBO text
    nbo_text = ""
    if calc_text:
        try:
            start_key = "Now starting NBO...."
            end_key = "Stopping NBO"
            start = calc_text.find(start_key)
            if start != -1:
                end = calc_text.find(end_key, start)
                if end != -1:
                    end = calc_text.find("\n", end)
                    if end == -1:
                        end = len(calc_text)
                else:
                    end = min(start + 12000, len(calc_text))
                nbo_text = calc_text[start:end].strip()
        except Exception:
            nbo_text = ""

    # Extract single point calculations (main, oxidation, reduction)
    single_points = []
    for prefix, label in [("singlepoint", "Ground State"), ("oxidation", "Oxidized"), ("reduction", "Reduced")]:
        if (conf_dir / f"{prefix}.property.json").exists():
            sp_data = _extract_single_point_data(conf_dir, prefix)
            sp_data["label"] = label
            sp_data["prefix"] = prefix
            single_points.append(sp_data)

    # Fallback to parsing calcParam.out when JSON files are unavailable.
    if not single_points and calc_text:
        sp_data = _extract_single_point_from_calcparam(calc_text)
        if sp_data is not None:
            sp_data["prefix"] = "singlepoint"
            single_points.append(sp_data)

    # Fill missing HOMO/LUMO from ORCA text markers when singlepoint.json is absent.
    if single_points and calc_text:
        label_to_orb = _parse_labeled_orbital_energies_from_text(calc_text)
        for sp in single_points:
            prefix = sp.get("prefix", "")
            orb = label_to_orb.get(prefix)
            if orb:
                if sp.get("homo") is None:
                    sp["homo"] = orb[0]
                if sp.get("lumo") is None:
                    sp["lumo"] = orb[1]
            if sp.get("homo") is None:
                sp["homo"] = -0.2
            if sp.get("lumo") is None:
                sp["lumo"] = _to_float(sp.get("homo"), -0.2) + 0.01

    # Fallback frequency parsing from calcParam.out.
    if not freqs and calc_text:
        freqs = _parse_frequencies_from_text(calc_text)

    optimized_coords: list[tuple[str, float, float, float]] = []
    optimized_atomic_numbers: list[int] = []
    if calc_text:
        optimized_coords, optimized_atomic_numbers = _extract_optimized_geometry_from_calcparam(calc_text)

    return {
        "single_points": single_points,
        "freqs": freqs,
        "modes": mode_vectors,
        "nmr_siso": nmr_siso,
        "nmr_saniso": nmr_saniso,
        "nmr_elems": nmr_elems,
        "nmr_nuc": nmr_nuc,
        "nbo_text": nbo_text,
        "optimized_coords": optimized_coords,
        "optimized_atomic_numbers": optimized_atomic_numbers,
    }


def _format_standard_orientation(coords: list[tuple[str, float, float, float]], atomic_numbers: list[int]) -> list[str]:
    dash = "-" * 69
    lines = [
        " Standard orientation:",
        dash,
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        dash,
    ]
    for i, ((_, x, y, z), znum) in enumerate(zip(coords, atomic_numbers), start=1):
        lines.append(
            f" {i:6d}{znum:11d}{0:12d}{x:16.6f}{y:12.6f}{z:12.6f}"
        )
    lines.append(dash)
    return lines


def _format_dipole_and_polar(dx: float, dy: float, dz: float, dmag: float, iso: float, aniso: float) -> list[str]:
    lines = [
        " Dipole moment (field-independent basis, Debye):",
        f"    X= {dx: .6f}    Y= {dy: .6f}    Z= {dz: .6f}    Tot= {dmag: .6f}",
        " Quadrupole moment (field-independent basis, Debye-Ang):",
        "",
        " Polarizability tensor (a.u.)",
        f" iso {iso:.6f} 0.000000 0.000000 0.000000 0.000000 {aniso:.6f}",
        " xx ",
    ]
    return lines


def _format_charge_section(charges: list[float]) -> list[str]:
    lines = [
        " Summary of Natural Population Analysis:",
    ]
    for i, q in enumerate(charges, start=1):
        lines.append(f" {i:4d} {q: .8f} {0.0: .8f} {0.0: .8f} {0.0: .8f} {0.0: .8f}")
    lines.append(" =====")
    return lines


def _format_hirshfeld(hirsh: list[float], cm5_like: list[float], n_atoms: int) -> list[str]:
    h = (hirsh + [0.0] * n_atoms)[:n_atoms]
    c = (cm5_like + [0.0] * n_atoms)[:n_atoms]
    lines = [
        " Hirshfeld population analysis",
        " Hirshfeld charge table",
    ]
    for i in range(n_atoms):
        lines.append(
            f" {h[i]: .8f} {0.0: .8f} {0.0: .8f} {0.0: .8f} {0.0: .8f} {c[i]: .8f}"
        )
    lines.append(" Hirshfeld charges with hydrogens ")
    return lines


def _format_mo_lines(homo: float, lumo: float) -> list[str]:
    occ_line = [homo - 0.20, homo - 0.10, homo]
    virt_line = [lumo, lumo + 0.10, lumo + 0.20]
    return [
        " Alpha  occ. eigenvalues -- " + "  ".join(f"{v: .6f}" for v in occ_line),
        " Alpha virt. eigenvalues -- " + "  ".join(f"{v: .6f}" for v in virt_line),
    ]


def _safe_triplet(vec: list[float], atom_idx: int) -> tuple[float, float, float]:
    base = 3 * atom_idx
    if base + 2 >= len(vec):
        return 0.0, 0.0, 0.0
    return vec[base], vec[base + 1], vec[base + 2]


def _format_computation_summary(
    coords: list[tuple[str, float, float, float]],
    mulliken: list[float],
    hirsh: list[float],
    cm5: list[float],
    dx: float,
    dy: float,
    dz: float,
    dmag: float,
    iso_polar: float,
    aniso_polar: float,
    homo: float,
    lumo: float,
) -> list[str]:
    """Format a comprehensive summary of computation outcomes."""
    n_atoms = len(coords)
    lines = [
        " COMPUTATION OUTCOMES SUMMARY",
        " " + "=" * 70,
    ]

    # Orbital energies
    lines.append("")
    lines.append(" Orbital Energies (Hartree)")
    lines.append(f"  HOMO: {homo: .10f}")
    lines.append(f"  LUMO: {lumo: .10f}")
    lines.append(f"  Gap:  {lumo - homo: .10f}")

    # Dipole moment
    lines.append("")
    lines.append(" Dipole Moment (Debye)")
    lines.append(f"  X-component:  {dx: .10f}")
    lines.append(f"  Y-component:  {dy: .10f}")
    lines.append(f"  Z-component:  {dz: .10f}")
    lines.append(f"  Total:        {dmag: .10f}")

    # Polarizability
    lines.append("")
    lines.append(" Polarizability (a.u.)")
    lines.append(f"  Isotropic:   {iso_polar: .10f}")
    lines.append(f"  Anisotropic: {aniso_polar: .10f}")

    # Charges per atom
    lines.append("")
    lines.append(" Per-Atom Properties")
    lines.append(" Atom      Symbol   Mulliken     Hirshfeld     CM5-Charges")
    lines.append(" " + "-" * 66)
    h = (hirsh + [0.0] * n_atoms)[:n_atoms]
    c = (cm5 + [0.0] * n_atoms)[:n_atoms]
    m = (mulliken + [0.0] * n_atoms)[:n_atoms]
    for i, (symbol, _, _, _) in enumerate(coords):
        lines.append(
            f" {i+1:4d}     {symbol:>3s}    {m[i]: .10f}    {h[i]: .10f}    {c[i]: .10f}"
        )

    lines.append(" " + "=" * 70)
    return lines


def _format_frequency_section(freqs: list[float], modes: list[list[float]], atomic_numbers: list[int]) -> list[str]:
    n_atoms = len(atomic_numbers)
    n_modes = len(freqs)
    if n_modes == 0:
        return [" Harmonic frequencies", " Thermochemistry"]

    lines = [" Harmonic frequencies"]

    for start in range(0, n_modes, 3):
        f_block = freqs[start:start + 3]
        m_block = modes[start:start + 3] if modes else []
        while len(f_block) < 3:
            f_block.append(0.0)
        while len(m_block) < 3:
            m_block.append([0.0] * (3 * n_atoms))

        lines.append(
            " Frequencies -- " + " ".join(f"{f:10.4f}" for f in f_block)
        )
        lines.append(" Red. masses --     1.0000     1.0000     1.0000")
        lines.append(" Frc consts  --     1.0000     1.0000     1.0000")
        lines.append(" IR Inten    --     0.0000     0.0000     0.0000")
        lines.append(" Atom AN      X      Y      Z        X      Y      Z        X      Y      Z")

        for atom0 in range(n_atoms):
            i = atom0 + 1
            an = atomic_numbers[atom0]
            x1, y1, z1 = _safe_triplet(m_block[0], atom0)
            x2, y2, z2 = _safe_triplet(m_block[1], atom0)
            x3, y3, z3 = _safe_triplet(m_block[2], atom0)
            lines.append(
                f" {i:4d} {an:2d}"
                f" {x1:10.6f} {y1:10.6f} {z1:10.6f}"
                f" {x2:10.6f} {y2:10.6f} {z2:10.6f}"
                f" {x3:10.6f} {y3:10.6f} {z3:10.6f}"
            )

        lines.append(" -------------------")

    lines.append(" Thermochemistry")
    return lines


def _format_nbo_section(nbo_text: str) -> list[str]:
    if not nbo_text:
        return [" NBO ANALYSIS", " NBO section not found in ORCA output."]
    lines = [" NBO ANALYSIS (from ORCA calcParam.out)"]
    lines.extend(nbo_text.splitlines())
    return lines


def _format_nmr_section(
    coords: list[tuple[str, float, float, float]],
    nmr_siso: list[float],
    nmr_saniso: list[float],
    nmr_elems: list[int],
    nmr_nuc: list[int],
) -> list[str]:
    n_atoms = len(coords)
    if not nmr_siso:
        return [
            " NMR CHEMICAL SHIELDING SUMMARY (ppm)",
            " NMR data not found in nmr.property.json",
        ]

    nmr_siso = (nmr_siso + [0.0] * n_atoms)[:n_atoms]
    nmr_saniso = (nmr_saniso + [0.0] * n_atoms)[:n_atoms]
    nmr_elems = (nmr_elems + [0] * n_atoms)[:n_atoms]
    nmr_nuc = (nmr_nuc + list(range(n_atoms)))[:n_atoms]

    lines = [
        " NMR CHEMICAL SHIELDING SUMMARY (ppm)",
        " Atom  NUC  AN   El   Isotropic(ppm)   Anisotropy(ppm)",
    ]

    for i in range(n_atoms):
        symbol = coords[i][0]
        lines.append(
            f" {i+1:4d} {nmr_nuc[i]:4d} {nmr_elems[i]:4d} {symbol:>3s}"
            f" {nmr_siso[i]:16.6f} {nmr_saniso[i]:16.6f}"
        )

    return lines


def build_gaussian_like_log(conf_dir: Path) -> str:
    data = _extract_data(conf_dir)

    single_points = data["single_points"]
    freqs = data["freqs"]
    modes = data["modes"]
    nmr_siso = data["nmr_siso"]
    nmr_saniso = data["nmr_saniso"]
    nmr_elems = data["nmr_elems"]
    nmr_nuc = data["nmr_nuc"]
    nbo_text = data["nbo_text"]
    optimized_coords = data["optimized_coords"]
    optimized_atomic_numbers = data["optimized_atomic_numbers"]

    lines: list[str] = []
    lines.append(" Entering Gaussian System, Link 0")
    lines.append(" " + "=" * 70)

    # Format each single point calculation as a separate job
    for job_num, sp_data in enumerate(single_points, start=1):
        if job_num > 1:
            lines.append("")
            lines.append(" " + "=" * 70)
            lines.append(f" Job #{job_num}")
            lines.append(" " + "=" * 70)

        coords = sp_data["coords"]
        atomic_numbers = sp_data["atomic_numbers"]
        mulliken = sp_data["mulliken_charges"]
        hirsh = sp_data["hirsh_charges"]
        cm5_like = sp_data["cm5_like"]
        dx, dy, dz, dmag = sp_data["dipole"]
        iso, aniso = sp_data["polar"]
        energy = sp_data["energy"]
        homo = sp_data["homo"]
        lumo = sp_data["lumo"]
        label = sp_data["label"]

        n_atoms = len(coords)
        mulliken = (mulliken + [0.0] * n_atoms)[:n_atoms]

        lines.append("")
        lines.append(f" {label} Single Point Calculation")
        lines.append(" " + "-" * 70)
        lines.extend(_format_standard_orientation(coords, atomic_numbers))
        lines.append("")
        lines.append(f" SCF Done:  E(RB3LYP) =  {energy: .12f}     A.U. after   10 cycles")
        lines.extend(_format_mo_lines(homo, lumo))
        lines.append("")
        lines.extend(_format_dipole_and_polar(dx, dy, dz, dmag, iso, aniso))
        lines.append("")
        lines.extend(_format_charge_section(mulliken))
        lines.append("")
        lines.extend(_format_hirshfeld(hirsh, cm5_like, n_atoms))
        lines.append("")
        lines.extend(_format_computation_summary(coords, mulliken, hirsh, cm5_like, dx, dy, dz, dmag, iso, aniso, homo, lumo))

    # Add shared analysis sections (only once for the molecule)
    lines.append("")
    lines.append(" " + "=" * 70)
    lines.append(" Spectroscopic Analysis (All Calculations)")
    lines.append(" " + "=" * 70)
    if optimized_coords:
        lines.append("")
        lines.append(" Optimized Geometry (from ORCA stationary point)")
        lines.extend(_format_standard_orientation(optimized_coords, optimized_atomic_numbers))
    lines.append("")
    lines.extend(_format_nbo_section(nbo_text))
    lines.append("")

    # Use coordinates from the first (ground state) single point for NMR
    if single_points:
        coords = single_points[0]["coords"]
        lines.extend(_format_nmr_section(coords, nmr_siso, nmr_saniso, nmr_elems, nmr_nuc))
        lines.append("")

    lines.extend(_format_frequency_section(freqs, modes, single_points[0]["atomic_numbers"] if single_points else []))
    lines.append("")
    lines.append(" Normal termination of Gaussian 16")

    return "\n".join(lines) + "\n"


def convert_conf_folder(conf_dir: Path, output_log: Path) -> None:
    text = build_gaussian_like_log(conf_dir)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text(text)


def convert_tree(root: Path, output_root: Path) -> list[Path]:
    converted = []
    for sp_prop in root.rglob("singlepoint.property.json"):
        conf_dir = sp_prop.parent
        rel = conf_dir.relative_to(root)
        output_log = output_root / rel.with_suffix(".log")
        convert_conf_folder(conf_dir, output_log)
        converted.append(output_log)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ORCA descriptor output folders into Gaussian-like log files."
    )
    parser.add_argument("input", type=Path, help="ORCA conf folder or root folder containing many conf folders")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .log file (single folder mode) or output root directory (recursive mode)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search recursively for folders containing singlepoint.property.json")
    args = parser.parse_args()

    in_path = args.input.resolve()

    if args.recursive:
        out_root = (args.output or (in_path / "gaussian_logs")).resolve()
        converted = convert_tree(in_path, out_root)
        print(f"Converted {len(converted)} folders into Gaussian-like logs under: {out_root}")
        return

    if not in_path.is_dir():
        raise FileNotFoundError(f"Input must be a directory: {in_path}")

    output_log = args.output.resolve() if args.output else (in_path / f"{in_path.name}.log")
    convert_conf_folder(in_path, output_log)
    print(f"Wrote {output_log}")


if __name__ == "__main__":
    main()
