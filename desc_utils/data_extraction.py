import math
import pandas as pd
import numpy as np
import re
from pathlib import Path
from desc_utils import orca_descriptors as od
from desc_utils.orca_parse import CONDENSED_FILENAME, nbo_atom_numbers
from tqdm import tqdm

def _resolve_calc_path(raw_path, repo_root):
    raw_path = Path(raw_path)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.exists():
        return raw_path.resolve()
    return (repo_root / raw_path).resolve()


def conf_key(path_value, repo_root):
    """Canonical conformer key: repository-relative POSIX path without a trailing slash.

    ``/abs/repo/DATA/lib/X/conf_1/`` -> ``DATA/lib/X/conf_1``. Paths outside the repository
    are returned absolute. This is the join key shared by every extraction engine.
    """
    s = str(path_value).replace("\\", "/").rstrip("/")
    p = Path(s)
    if not p.is_absolute():
        p = (Path(repo_root) / p)
    try:
        return p.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _find_nbo_out_file(calc_path, basename="singlepoint", strict_basename=False):
    calc_path = Path(calc_path)
    preferred = calc_path / f"{basename}.out"
    if preferred.exists():
        return preferred

    if strict_basename:
        raise FileNotFoundError(f"Expected {preferred.name} in {calc_path}, but it was not found")

    # Fallbacks for folders where NBO output was written with a different basename.
    out_files = sorted(calc_path.glob("*.out"))
    if not out_files:
        raise FileNotFoundError(f"No .out file found in {calc_path}")

    matching = [p for p in out_files if "nbo" in p.name.lower()]
    if matching:
        return matching[0]

    return out_files[0]


def _flatten_nested_results(value, prefix="", flat_results=None):
    if flat_results is None:
        flat_results = {}

    if isinstance(value, dict):
        for key, subvalue in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_nested_results(subvalue, next_prefix, flat_results)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}_{idx}" if prefix else str(idx)
            _flatten_nested_results(item, next_prefix, flat_results)
        if not value and prefix:
            flat_results[prefix] = np.nan
    else:
        # Store scalar leaves directly as descriptor columns.
        flat_results[prefix] = value

    return flat_results


def _get_dict_value(data, key):
    if key in data:
        return data[key]
    key_str = str(key)
    if key_str in data:
        return data[key_str]
    try:
        key_int = int(key)
    except Exception:
        return None
    return data.get(key_int)


def _extract_atom_numbers_from_text(text):
    # None for a label NBO could not print ("C***", atom > 999), so the row never counts as all-selected atoms
    return nbo_atom_numbers(text)


def _replace_atom_numbers_with_labels(text, atom_index_to_label):
    def repl(match):
        atom_symbol = match.group(1)
        atom_num = int(match.group(2))
        return atom_index_to_label.get(atom_num, f"{atom_symbol}{atom_num}")

    return re.sub(r"([A-Z][a-z]?)\s*(\d+)", repl, str(text))


def _sanitize_descriptor_token(value):
    token = str(value).strip().replace(" ", "")
    token = token.replace("(", "").replace(")", "")
    token = token.replace("/", "-")
    token = token.replace("*", "star")
    token = token.replace(".", "p")
    return token


def _collect_selected_nbo_results(nbo_results, atom_index_map):
    selected_indices = set(atom_index_map.values())
    atom_index_to_label = {v: k for k, v in atom_index_map.items()}
    flat = {}

    npa = nbo_results.get("natural_population_analysis", {})
    npa_atoms = npa.get("atoms", {})
    for label, atom_idx in atom_index_map.items():
        atom_props = _get_dict_value(npa_atoms, atom_idx)
        if atom_props is None:
            continue
        for prop in ["charge", "core", "valence", "rydberg", "total"]:
            flat[f"npa_{label}_{prop}"] = atom_props.get(prop)

    npa_totals = npa.get("totals", {})
    for prop, value in npa_totals.items():
        flat[f"npa_totals_{prop}"] = value

    nao_name_counters = {}
    for entry in nbo_results.get("natural_atomic_orbitals", []):
        atom_idx = entry.get("atom_num")
        if atom_idx not in selected_indices:
            continue
        label = atom_index_to_label[atom_idx]

        orbital_type = _sanitize_descriptor_token(entry.get("orbital_type", "unk"))
        orbital = _sanitize_descriptor_token(entry.get("orbital", "unk"))
        nao_name = f"{orbital_type}-{orbital}"

        # Some molecules can contain repeated NAO labels for the same atom.
        key = (label, nao_name)
        nao_name_counters[key] = nao_name_counters.get(key, 0) + 1
        suffix = "" if nao_name_counters[key] == 1 else f"_{nao_name_counters[key]}"

        flat[f"nao_{label}_{nao_name}{suffix}_occupancy"] = entry.get("occupancy")

    # Compute HOMO nbo_num from ALL natural_bond_orbitals (global ORCA index, shared with bond_orbital_details)
    _homo_nbo_num = None
    for _e in nbo_results.get("natural_bond_orbitals", []):
        if _e.get("occupancy", 0.0) >= 0.1:
            _n = _e.get("nbo_num", 0)
            if _n is None:      # NBO number printed as "****" and not recoverable
                continue
            if _homo_nbo_num is None or _n > _homo_nbo_num:
                _homo_nbo_num = _n

    def _homo_rank_token(nbo_num, homo_num):
        if homo_num is None:
            return str(nbo_num)
        rank = nbo_num - homo_num
        if rank == 0:
            return "homo"
        elif rank < 0:
            return f"hm{abs(rank)}"
        else:
            return f"lp{rank}"

    nbo_rank_counters = {}
    for entry in nbo_results.get("natural_bond_orbitals", []):
        atoms_list = entry.get("atoms", [])
        if not atoms_list or not all(atom in selected_indices for atom in atoms_list):
            continue
        if entry.get("nbo_num", 0) is None:
            continue
        atom_labels = "_".join(atom_index_to_label[atom] for atom in atoms_list)
        rank_token = _homo_rank_token(entry.get("nbo_num", 0), _homo_nbo_num)
        dup_key = (atom_labels, rank_token)
        nbo_rank_counters[dup_key] = nbo_rank_counters.get(dup_key, 0) + 1
        dup_suffix = "" if nbo_rank_counters[dup_key] == 1 else f"_{nbo_rank_counters[dup_key]}"
        base = f"nbo_{rank_token}{dup_suffix}_{atom_labels}"
        flat[f"{base}_type"] = entry.get("nbo_type")
        flat[f"{base}_bond_index"] = entry.get("bond_index")
        flat[f"{base}_occupancy"] = entry.get("occupancy")
        flat[f"{base}_energy"] = entry.get("energy")

    detail_rank_counters = {}
    for entry in nbo_results.get("bond_orbital_details", []):
        atoms_list = entry.get("atoms", [])
        if not atoms_list or not all(atom in selected_indices for atom in atoms_list):
            continue
        if entry.get("bond_num", 0) is None:
            continue
        atom_labels = "_".join(atom_index_to_label[atom] for atom in atoms_list)
        rank_token = _homo_rank_token(entry.get("bond_num", 0), _homo_nbo_num)
        dup_key = (atom_labels, rank_token)
        detail_rank_counters[dup_key] = detail_rank_counters.get(dup_key, 0) + 1
        dup_suffix = "" if detail_rank_counters[dup_key] == 1 else f"_{detail_rank_counters[dup_key]}"
        base = f"bond_orbital_details_{rank_token}{dup_suffix}_{atom_labels}"
        flat[f"{base}_bond_type"] = entry.get("bond_type")
        flat[f"{base}_bond_index"] = entry.get("bond_index")
        flat[f"{base}_occupancy"] = entry.get("occupancy")

        contrib_counter = 0
        for contrib in entry.get("contributions", []):
            contrib_atom = contrib.get("atom_num")
            if contrib_atom not in selected_indices or contrib.get("one_centre"):
                continue
            contrib_counter += 1
            contrib_label = atom_index_to_label[contrib_atom]
            cbase = f"{base}_contrib_{contrib_counter}_{contrib_label}"
            flat[f"{cbase}_percentage"] = contrib.get("percentage")
            flat[f"{cbase}_coefficient"] = contrib.get("coefficient")
            flat[f"{cbase}_s_percent"] = contrib.get("s_percent")
            flat[f"{cbase}_p_power"] = contrib.get("p_power")
            flat[f"{cbase}_p_percent"] = contrib.get("p_percent")

    # Fallback lookup for legacy outputs whose second-order table has no NBO numbers:
    # (nbo_type, bond_index, atoms) -> nbo_num
    _nbo_key_to_num = {}
    for _e in nbo_results.get("natural_bond_orbitals", []):
        _key = (_e.get("nbo_type"), _e.get("bond_index"), tuple(_e.get("atoms", [])))
        _nbo_key_to_num.setdefault(_key, _e.get("nbo_num", 0))

    dir_rank_counters = {}
    for entry in nbo_results.get("nho_directionality", []):
        atoms_list = entry.get("atoms", [])
        if not atoms_list or not all(atom in selected_indices for atom in atoms_list):
            continue
        if entry.get("nbo_num", 0) is None:
            continue
        atom_labels = "_".join(atom_index_to_label[atom] for atom in atoms_list)
        rank_token = _homo_rank_token(entry.get("nbo_num", 0), _homo_nbo_num)
        dup_key = (atom_labels, rank_token)
        dir_rank_counters[dup_key] = dir_rank_counters.get(dup_key, 0) + 1
        dup_suffix = "" if dir_rank_counters[dup_key] == 1 else f"_{dir_rank_counters[dup_key]}"
        base = f"nho_directionality_{rank_token}{dup_suffix}_{atom_labels}"
        flat[f"{base}_type"] = entry.get("nbo_type")
        flat[f"{base}_bond_index"] = entry.get("bond_index")
        flat[f"{base}_line_theta"] = entry.get("line_theta")
        flat[f"{base}_line_phi"] = entry.get("line_phi")
        flat[f"{base}_hybrid_theta"] = entry.get("hybrid_theta")
        flat[f"{base}_hybrid_phi"] = entry.get("hybrid_phi")
        flat[f"{base}_deviation"] = entry.get("deviation")

    sop_rank_counters = {}
    for entry in nbo_results.get("second_order_perturbation", []):
        donor_atoms = _extract_atom_numbers_from_text(entry.get("donor_atoms", ""))
        acceptor_atoms = _extract_atom_numbers_from_text(entry.get("acceptor_atoms", ""))
        interaction_atoms = donor_atoms + acceptor_atoms

        if not interaction_atoms or not all(atom in selected_indices for atom in interaction_atoms):
            continue

        donor_label = "_".join(atom_index_to_label[atom] for atom in donor_atoms)
        acceptor_label = "_".join(atom_index_to_label[atom] for atom in acceptor_atoms)
        donor_nbo_num = entry.get("donor_nbo_num")
        if donor_nbo_num is None:
            donor_nbo_num = _nbo_key_to_num.get((entry.get("donor_type"), entry.get("donor_index"), tuple(donor_atoms)), 0)
        acceptor_nbo_num = entry.get("acceptor_nbo_num")
        if acceptor_nbo_num is None:
            acceptor_nbo_num = _nbo_key_to_num.get((entry.get("acceptor_type"), entry.get("acceptor_index"), tuple(acceptor_atoms)), 0)
        donor_rank = _homo_rank_token(donor_nbo_num, _homo_nbo_num)
        acceptor_rank = _homo_rank_token(acceptor_nbo_num, _homo_nbo_num)
        dup_key = (donor_label, donor_rank, acceptor_label, acceptor_rank)
        sop_rank_counters[dup_key] = sop_rank_counters.get(dup_key, 0) + 1
        dup_suffix = "" if sop_rank_counters[dup_key] == 1 else f"_{sop_rank_counters[dup_key]}"
        base = f"second_order_{donor_rank}_{donor_label}_to_{acceptor_rank}_{acceptor_label}{dup_suffix}"
        flat[f"{base}_donor_type"] = entry.get("donor_type")
        flat[f"{base}_donor_index"] = entry.get("donor_index")
        flat[f"{base}_acceptor_type"] = entry.get("acceptor_type")
        flat[f"{base}_acceptor_index"] = entry.get("acceptor_index")
        flat[f"{base}_e2_energy"] = entry.get("e2_energy")
        flat[f"{base}_energy_diff"] = entry.get("energy_diff")
        flat[f"{base}_fock_element"] = entry.get("fock_element")

    for key, value in nbo_results.get("analysis_info", {}).items():
        flat[f"analysis_info_{key}"] = value

    return flat


def _quadrant_label(quadrant) -> str:
    """The column label of one quadrant/octant entry: its own, else the metal."""
    if isinstance(quadrant, dict):
        atoms = list(quadrant.get("atoms") or [])
        return str(quadrant.get("label") or (atoms[0] if atoms else "?"))
    return "_".join(quadrant)


def _atom_index(row, label):
    """The 1-based index of one label on one conformer's numbering row, or None if it has none.

    None is the ordinary case, not an error: a ring stage is `required: false`, so a ligand
    of another class simply carries no pyridine labels and every term mentioning one has to be
    skipped rather than crashing the extraction of the whole library. `int(float(v))` because
    a numbering column read back from CSV arrives as `5.0`, and sometimes as the string
    `'5.0'`, which `int()` alone refuses.
    """
    if label not in row:
        return None
    value = row[label]
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ring_keys(rings, ring_pairs, ring_probes):
    """One representative column name per declared ring term, for the cache check."""
    keys = []
    for ring in rings or []:
        keys.append(f"Ring_{ring['label']}_planarity_rms")
        if ring.get("homa"):
            keys.append(f"Ring_{ring['label']}_HOMA")
    for a, b in ring_pairs or []:
        keys.append(f"RingPair_{a}_{b}_interplanar_angle")
    for probe in ring_probes or []:
        keys.append(f"RingProbe_{probe['ring']}_{probe['atom']}_height")
    return keys


def _add_ring_descriptors(P, row, rings, ring_pairs, ring_probes):
    """Ring-level geometry (desc_utils.rings) written into ``P.results``.

    A ring whose labels are not all present on this conformer is skipped and NAMED in
    ``Ring_<label>_note`` rather than left silently absent, because "this column is blank"
    and "this ring was never asked for" look identical in a finished table.
    """
    from desc_utils import rings as ring_math

    coords = {}
    for ring in rings or []:
        idx = [_atom_index(row, label) for label in ring["atoms"]]
        if any(i is None for i in idx):
            missing = [l for l, i in zip(ring["atoms"], idx) if i is None]
            P.results[f"Ring_{ring['label']}_note"] = f"not numbered on this conformer: {missing}"
            continue
        X = [P.coordinates[i - 1] for i in idx]
        symbols = [P.elements[i - 1] for i in idx]
        coords[ring["label"]] = X
        try:
            values = ring_math.ring_descriptors(X, symbols, with_homa=bool(ring.get("homa")))
        except ValueError as exc:                     # e.g. HOMA has no parameters for a bond type
            P.results[f"Ring_{ring['label']}_note"] = str(exc)
            values = ring_math.ring_descriptors(X, symbols, with_homa=False)
        for name, value in values.items():
            P.results[f"Ring_{ring['label']}_{name}"] = float(value)

    for a, b in ring_pairs or []:
        if a not in coords or b not in coords:
            continue
        P.results[f"RingPair_{a}_{b}_interplanar_angle"] = ring_math.interplanar_angle(coords[a], coords[b])
        P.results[f"RingPair_{a}_{b}_centroid_distance"] = float(np.linalg.norm(
            np.mean(np.asarray(coords[a], dtype=float), axis=0)
            - np.mean(np.asarray(coords[b], dtype=float), axis=0)))

    for probe in ring_probes or []:
        ring_label, atom_label = probe["ring"], probe["atom"]
        idx = _atom_index(row, atom_label)
        if ring_label not in coords or idx is None:
            continue
        point = P.coordinates[idx - 1]
        P.results[f"RingProbe_{ring_label}_{atom_label}_height"] = ring_math.atom_height(coords[ring_label], point)
        P.results[f"RingProbe_{ring_label}_{atom_label}_centroid_distance"] = ring_math.centroid_distance(
            coords[ring_label], point)


def get_xyz_descriptors(df, atoms, bonds, angles, dihedrals, quadrants=None, overwrite_json=False, vbur_radii=None,
                        progress=True, geometry_file=od.PropertiesXYZ.DEFAULT_GEOMETRY_FILE,
                        rings=None, ring_pairs=None, ring_probes=None, distances=None):
    """Morfeus descriptors of ``geometry_file`` in every conformer folder of ``df`` (``optimization.xyz`` by default).

    Any single-frame xyz works, so a library optimised with UMA (or any ASE calculator) can be described
    from its xyz alone. The results are cached next to the geometry (``PropertiesXYZ.cache_name``).
    """
    results_dict = {}
    repo_root = Path(__file__).resolve().parent.parent
    for row in tqdm(df.itertuples(), total=len(df), disable=not progress):
        P = od.PropertiesXYZ(path=df.loc[row.Index, "path"], overwrite_results=overwrite_json, geometry_file=geometry_file)
        if vbur_radii is not None:
            P.radius_list = [float(r) for r in vbur_radii]
        
        # A cached xyz.json is only trusted when it holds every column the current spec asks for;
        # otherwise the atom map changed since it was written and the geometry is recomputed.
        expected_keys = [f"SASA_{a}" for a in atoms] + [f"Bond_{b[0]}_{b[1]}" for b in bonds] \
            + [f"Angle_{x[0]}_{x[1]}_{x[2]}" for x in angles] + [f"Dihed_{'_'.join(d)}" for d in dihedrals] \
            + [f"%Vbur_{_quadrant_label(q)}_octant_+,+,+" for q in (quadrants or [])] \
            + [f"Dist_{d[0]}_{d[1]}" for d in (distances or [])] \
            + _ring_keys(rings, ring_pairs, ring_probes)
        # The octant key is in the list because it was NOT, and a library that had been
        # extracted once without quadrants then kept every cached xyz.json and quietly
        # produced no octant column at all. Note what this check still cannot see: it
        # compares NAMES, and the hemisphere convention changes which atom `N1` is
        # without changing one column name - so a re-numbered library must be extracted
        # with `use_cached_json: false`, not merely with a wider expected-key list.
        cache_ok = P.results is not None and len(P.results) > 1 and set(expected_keys) <= set(P.results)
        if len(P.results) > 1 and not overwrite_json and not cache_ok:
            print(f"{P.cache_file} in {df.loc[row.Index, 'path']} lacks columns for the current spec; recomputing.")
        if cache_ok and not overwrite_json:
            if P.results is not None:
                P.results["path"] = conf_key(df.loc[row.Index, "path"], repo_root)
                results_dict[row.Index] = P.results
            else:
                print(f"No results found for {df.loc[row.Index, 'path']}.")

        else:
            # Zero atom properties
            P.get_SASA_and_Volume()

            # One atom properties
            numbering_row = df.loc[row.Index]
            atom_map = {atom: _atom_index(numbering_row, atom) for atom in atoms}
            atom_map = {a: i for a, i in atom_map.items() if i is not None}
            P.define_atom_list(atom_map)
            P.get_Vbur()
            P.get_atomic_SASA()
            P.get_pyramidalizations()

            # Two atom properties
            # A term is skipped when any of its labels is absent on this conformer rather than
            # raising: an optional numbering stage (a ring that only a pyox ligand has) leaves
            # another class's rows without those labels, and one such ligand must not take the
            # whole library's extraction down with it.
            bond_map = {
                f"{bond[0]}_{bond[1]}": tuple(_atom_index(numbering_row, l) for l in bond)
                for bond in bonds
            }
            bond_map = {k: v for k, v in bond_map.items() if None not in v}
            P.define_bond_list(bond_map)
            P.get_bond_distances()

            distance_map = {
                "_".join(pair): tuple(_atom_index(numbering_row, l) for l in pair)
                for pair in (distances or [])
            }
            P.define_distance_list({k: v for k, v in distance_map.items() if None not in v})
            P.get_distances()
            P.get_Sterimol_parameters(morfeus=True)

            # Three atom properties
            angle_map = {
                "_".join(angle): tuple(_atom_index(numbering_row, l) for l in angle)
                for angle in angles
            }
            angle_map = {k: v for k, v in angle_map.items() if None not in v}
            P.define_angle_list(angle_map)
            P.get_angles()

            # Four atom properties
            dihedral_map = {
                "_".join(dihedral): tuple(_atom_index(numbering_row, l) for l in dihedral)
                for dihedral in dihedrals
            }
            dihedral_map = {k: v for k, v in dihedral_map.items() if None not in v}
            P.define_dihedral_list(dihedral_map)
            P.get_dihedrals()

            # Quadrant properties
            if quadrants is not None:
                quadrant_map = {}
                for quadrant in quadrants:
                    if isinstance(quadrant, dict):
                        # NOT `atoms`: that is the loop-wide list of labels this function was
                        # called with, and rebinding it here narrowed `atom_map` to the
                        # quadrant frame for every LATER row - silently dropping %Vbur, SASA
                        # and pyramidalization for the other atoms on every conformer but the
                        # first. Never fired, because every ORCA spec ships `quadrants: []`.
                        quadrant_atoms = list(quadrant["atoms"])
                        # The label defaults to the metal rather than to every atom in the
                        # frame, so the column is `%Vbur_Ni_octant_+,+,+` as BUNNY names it
                        # and not `%Vbur_Ni_N1_N2_N2_octant_+,+,+`.
                        label = str(quadrant.get("label") or quadrant_atoms[0])
                        quadrant_map[label] = {
                            "atoms": tuple(int(df.loc[row.Index, a]) for a in quadrant_atoms),
                            "exclude": tuple(int(df.loc[row.Index, a])
                                             for a in (quadrant.get("exclude") or [])) or None,
                            "radius": quadrant.get("radius"),
                        }
                    else:
                        quadrant_map["_".join(quadrant)] = tuple(
                            int(df.loc[row.Index, atom]) for atom in quadrant)
                P.define_quadrant_list(quadrant_map)
                P.get_vbur_quadrants()
                P.get_vbur_octants()

            _add_ring_descriptors(P, numbering_row, rings, ring_pairs, ring_probes)

            results = P.results
            results["path"] = conf_key(df.loc[row.Index, "path"], repo_root)
            results_dict[row.Index] = results

            P.save_results_to_json() # save results to json file in same directory as xyz file for later use if needed
    return results_dict

def _load_nbo_results(calc_path, basename="singlepoint", nbo_block_index=0, spin="alpha",
                      condensed_file=CONDENSED_FILENAME, full=True):
    """``NBOProps.results`` for one NBO block of a conformer folder.

    Read from ``<calc_path>/<condensed_file>`` (see ``desc_utils.condense``) when that file exists,
    otherwise parsed from the ORCA output found by ``_find_nbo_out_file``. ``full=False`` fills only
    the natural population analysis (all the redox family needs).
    """
    from desc_utils.condense import find_condensed, load_condensed, nbo_results

    cond_path = find_condensed(calc_path, condensed_file)
    if cond_path is not None:
        return nbo_results(load_condensed(cond_path), nbo_block_index, spin=spin, require_full=full)
    out_file = _find_nbo_out_file(calc_path, basename=basename, strict_basename=False)
    nbo_reader = od.NBOProps(out_file, block_index=nbo_block_index, spin=spin)
    nbo_reader.get_NPA()
    if full:
        nbo_reader.get_NAO()
        nbo_reader.get_NBO_summary()
        nbo_reader.get_orbital_details()
        nbo_reader.get_NHO_directionality()
        nbo_reader.get_second_order_perturbation()
        nbo_reader.get_cycle_info()
    return nbo_reader.results


def _extract_nbo_charges_for_reader(reader, calc_path, atom_map, basename="singlepoint", nbo_block_index=0,
                                    condensed_file=CONDENSED_FILENAME):
    """Add the NBO natural charges of one NBO block to ``reader.results``.

    Returns ``None`` on success, else a one-line description of what went wrong (the block is missing
    or truncated, typically because the ORCA job died); the Fukui values of that state are then NaN and
    the caller records the reason in ``redox_nbo_error``.
    """
    try:
        results = _load_nbo_results(calc_path, basename, nbo_block_index, condensed_file=condensed_file, full=False)
        npa_atoms = results.get("natural_population_analysis", {}).get("atoms", {})

        for label, atom_idx in atom_map.items():
            atom_props = _get_dict_value(npa_atoms, atom_idx)
            if atom_props is not None:
                charge = atom_props.get("charge")
                reader.results[f"nbo_charge_{label}"] = charge
                # Compatibility alias used elsewhere in the codebase.
                reader.results[f"npa_{label}_charge"] = charge
        return None
    except Exception as exc:
        return f"{basename} (NBO block {nbo_block_index}): {type(exc).__name__}: {str(exc)[:100]}"


def _normalize_nbo_block_map(nbo_block_map):
    """Normalize NBO block map to 0-based indices.

    If values look like 1-based user counting (all positive and none is 0),
    convert by subtracting 1.
    """
    normalized = dict(nbo_block_map or {})
    vals = [v for v in normalized.values() if v is not None]
    if vals and all(int(v) >= 1 for v in vals) and all(int(v) != 0 for v in vals):
        return {k: int(v) - 1 for k, v in normalized.items()}
    return {k: int(v) for k, v in normalized.items()}


def get_redox_descriptors(df, atoms, descriptorPrefixes=["cm5"], nbo_block_map=None, condensed_file=CONDENSED_FILENAME):
    results_dict = {}
    repo_root = Path(__file__).resolve().parent.parent
    if nbo_block_map is None:
        # Default assumes one combined ORCA output with three NBO blocks ordered as:
        # singlepoint (neutral), oxidation (cation), reduction (anion).
        nbo_block_map = {"singlepoint": 0, "oxidation": 1, "reduction": 2}
    else:
        nbo_block_map = _normalize_nbo_block_map(nbo_block_map)

    for row in df.itertuples():
        calc_path = _resolve_calc_path(df.loc[row.Index, "path"], repo_root)

        atom_map = {atom: int(df.loc[row.Index, atom]) for atom in atoms}

        neutral_reader = od.OrcaReader(calc_path, basename="singlepoint", out_file_path=None, nbo_block_index=0, nbo_job_number=None,
                                        condensed_file=condensed_file)
        neutral_reader.set_atoms(atom_map)
        neutral_reader.parse_property_json()
        neutral_reader.get_scfenergy()
        neutral_reader.get_all_charges()
        nbo_errors = []
        nbo_errors.append(_extract_nbo_charges_for_reader(
            neutral_reader,
            calc_path,
            atom_map,
            "singlepoint",
            nbo_block_index=int(nbo_block_map.get("singlepoint", 0)),
            condensed_file=condensed_file,
        ))

        cation_reader = od.OrcaReader(calc_path, basename="oxidation", out_file_path=None, nbo_block_index=0, nbo_job_number=None,
                                        condensed_file=condensed_file)
        cation_reader.set_atoms(atom_map)
        cation_reader.parse_property_json()
        cation_reader.get_scfenergy()
        cation_reader.get_all_charges()
        nbo_errors.append(_extract_nbo_charges_for_reader(
            cation_reader,
            calc_path,
            atom_map,
            "oxidation",
            nbo_block_index=int(nbo_block_map.get("oxidation", 1)),
            condensed_file=condensed_file,
        ))

        anion_reader = od.OrcaReader(calc_path, basename="reduction", out_file_path=None, nbo_block_index=0, nbo_job_number=None,
                                        condensed_file=condensed_file)
        anion_reader.set_atoms(atom_map)
        anion_reader.parse_property_json()
        anion_reader.get_scfenergy()
        anion_reader.get_all_charges()
        nbo_errors.append(_extract_nbo_charges_for_reader(
            anion_reader,
            calc_path,
            atom_map,
            "reduction",
            nbo_block_index=int(nbo_block_map.get("reduction", 2)),
            condensed_file=condensed_file,
        ))

        P = od.RedoxSet(neutralReader=neutral_reader, cationReader=cation_reader, anionReader=anion_reader, descriptorPrefixes=descriptorPrefixes)

        results = P.aggregate()
        results['path'] = conf_key(calc_path, repo_root)
        errors = [e for e in nbo_errors if e]
        if errors:
            results['redox_nbo_error'] = "; ".join(errors)
        results_dict[row.Index] = results
    return results_dict

def get_all_charges(df, atoms, basename="singlepoint", condensed_file=CONDENSED_FILENAME):
    results_dict = {}
    repo_root = Path(__file__).resolve().parent.parent
    for row in df.itertuples():
        calc_path = _resolve_calc_path(df.loc[row.Index, "path"], repo_root)

        atom_map = {atom: int(df.loc[row.Index, atom]) for atom in atoms}

        reader = od.OrcaReader(calc_path, basename=basename, out_file_path=None, nbo_block_index=0, nbo_job_number=None,
                               condensed_file=condensed_file)
        reader.set_atoms(atom_map)
        reader.parse_property_json()
        reader.aggregate_results()
        results = reader.results
        results['path'] = conf_key(calc_path, repo_root)
        results_dict[row.Index] = results
    return results_dict

def _conformer_coords(calc_path, condensed_file=CONDENSED_FILENAME):
    """Cartesian coordinates (angstrom, file order) of the geometry the NBO job ran on: the
    condensed file's ``molecule.atoms`` when present, else ``optimization.xyz``."""
    from desc_utils.condense import find_condensed, load_condensed
    cond = find_condensed(calc_path, condensed_file) if condensed_file else None
    if cond is not None:
        atoms = (load_condensed(cond).get("molecule") or {}).get("atoms") or []
        if atoms:
            return np.array([[float(a["x"]), float(a["y"]), float(a["z"])] for a in atoms])
    xyz = Path(calc_path) / "optimization.xyz"
    lines = xyz.read_text(encoding="utf-8").splitlines()
    return np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + int(lines[0])]])


def _nlmo_columns(calc_path, atom_map, nlmo_basename, frame, coords):
    """NLMO columns from ``<calc_path>/<nlmo_basename>.out`` (an NBO rerun with ``$NBO NLMO $END``).
    The NLMOs are keyed through that output's own NBO section, so they need not share NBO
    numbers with the singlepoint; its energies and E2 (zero under NoIter) are not used."""
    from desc_utils import nbo_keys
    out = Path(calc_path) / f"{nlmo_basename}.out"
    if not out.is_file():
        return {"nbo_nlmo_error": f"no {out.name}"}
    reader = od.NBOProps(out, block_index=0, spin="both")
    reader.get_NAO()
    reader.get_NBO_summary()
    reader.get_orbital_details()
    nlmo = reader.get_NLMO()
    if not nlmo:
        return {"nbo_nlmo_error": f"{out.name} has no NLMO section"}
    return nbo_keys.nlmo_descriptors(nlmo, reader.results, atom_map, frame, coords)


def _chemical_nbo_row(calc_path, atom_map, basename, nbo_block_index, spin, condensed_file, frame,
                      nlmo_basename=None):
    """One conformer's NBO descriptors under ``naming: chemical`` (desc_utils.nbo_keys): the NPA and
    NAO columns as under rank naming, then the chemically keyed NBO, hybrid and E2 columns, and the
    NLMO columns when ``nlmo_basename`` is set."""
    from desc_utils import nbo_keys
    base = _collect_selected_nbo_results(
        _load_nbo_results(calc_path, basename, nbo_block_index, spin=spin,
                          condensed_file=condensed_file, full=True), atom_map)
    flat = {k: v for k, v in base.items() if k.startswith(("npa_", "nao_"))}
    results = _load_nbo_results(calc_path, basename, nbo_block_index, spin="both",
                                condensed_file=condensed_file, full=True)
    if not nbo_keys.has_hybrids(results):
        # A condensed file written before one-centre hybrids were parsed: read the output itself
        # when it is there, else say what to do.
        try:
            results = _load_nbo_results(calc_path, basename, nbo_block_index, spin="both",
                                        condensed_file=None, full=True)
        except Exception:
            results = None
        if results is None or not nbo_keys.has_hybrids(results):
            raise ValueError("the NBO data has no lone-pair hybrids (a condensed file written before "
                             "they were parsed and no ORCA output beside it): re-run "
                             "`python -m desc_utils.condense --root <library_dir> --overwrite` on this library")
    coords = _conformer_coords(calc_path, condensed_file) if frame else None
    flat.update(nbo_keys.chemical_descriptors(results, atom_map, frame, coords))
    if nlmo_basename:
        try:
            flat.update(_nlmo_columns(calc_path, atom_map, nlmo_basename, frame, coords))
        except Exception as exc:
            flat["nbo_nlmo_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    return flat


def get_all_nbo_properties(df, atoms, basename="singlepoint", nbo_block_index=0, spin="alpha",
                           condensed_file=CONDENSED_FILENAME, consolidate_ranks=True,
                           naming="rank", frame=None, nlmo_basename=None):
    """Extract all properties available from ``od.NBOProps`` into a flat dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least a ``path`` column with calculation folders
        and atom index columns named by ``atoms``.
    atoms : list[str]
        Atom labels to keep (e.g. ``["H1", "H2", "C1", "C2", "N1", "N2", "Ni"]``).
        Output columns are keyed using these labels rather than numeric atom indices.
    basename : str, optional
        Preferred output filename stem, by default ``singlepoint`` (expects
        ``<basename>.out`` inside each calculation folder).
    nbo_block_index : int, optional
        Which NBO block of the output to read (0-based), by default the first one.
    spin : {"alpha", "beta", "both"}, optional
        Spin set returned for open-shell NBO blocks, by default ``"alpha"``. See
        ``od.NBOProps`` for the rationale; closed-shell blocks ignore it.
    condensed_file : str | None, optional
        Name of the condensed per-conformer JSON (``desc_utils.condense``) read instead of the
        ORCA output when it exists in the folder; ``None`` always parses the output.
    consolidate_ranks : bool, optional
        Apply :func:`consolidate_nbo_rank_columns` to the result (default). The parallel extraction
        passes ``False`` and consolidates once over the concatenated table: the decision to merge
        ``_lpN_`` / ``_hmN_`` columns depends on every molecule of the library.
    naming : {"rank", "chemical"}, optional
        ``rank`` (default) keeps the historical rank-named columns; ``chemical`` keys every NBO by
        type, sigma/pi class and atoms (:mod:`desc_utils.nbo_keys`).
    frame : list[str] | None, optional
        ``[centre, x_atom, y_atom]`` labels for the ligand-frame d character (chemical only).
    nlmo_basename : str | None, optional
        Stem of an NBO rerun with NLMOs in each conformer folder (chemical only): adds the
        ``_nlmo_parent`` / ``_nlmo_on_<label>`` columns of :func:`desc_utils.nbo_keys.nlmo_descriptors`.

    Returns
    -------
    pandas.DataFrame
        One row per input row/index, with NBO properties limited to selected atoms
        and column names keyed by atom labels.
    """
    nbo_results = {}
    repo_root = Path(__file__).resolve().parent.parent

    for row in df.itertuples():
        calc_path = _resolve_calc_path(df.loc[row.Index, "path"], repo_root)
        atom_map = {atom: int(df.loc[row.Index, atom]) for atom in atoms}

        row_results = {
            "path": conf_key(calc_path, repo_root)
        }

        try:
            if naming == "chemical":
                row_results.update(_chemical_nbo_row(calc_path, atom_map, basename, nbo_block_index,
                                                     spin, condensed_file, frame, nlmo_basename))
            else:
                # Every descriptor family implemented by NBOProps (from the condensed file when present).
                results = _load_nbo_results(calc_path, basename, nbo_block_index, spin=spin,
                                            condensed_file=condensed_file, full=True)

                filtered = _collect_selected_nbo_results(results, atom_map)
                row_results.update(filtered)
        except Exception as exc:
            row_results["nbo_extraction_error"] = str(exc)

        nbo_results[row.Index] = row_results

    df_out = pd.DataFrame.from_dict(nbo_results, orient="index")
    return consolidate_nbo_rank_columns(df_out) if (consolidate_ranks and naming == "rank") else df_out


def consolidate_nbo_rank_columns(df_out):
    """Merge ``_lpN_`` / ``_hmN_`` columns whose rank index varies across molecules into one rank-agnostic column.

    Library-wide operation: a group is merged only when no row of ``df_out`` holds more than one
    value among its members, so it must see the whole table, not a slice of it.
    """
    # --- Consolidate lp/hm columns whose rank index varies across molecules ---
    # e.g. nbo_lp3_H1_energy, nbo_lp4_H1_energy, nbo_lp5_H1_energy
    #   -> nbo_lp_H1_energy  (rank-agnostic merged column)
    #
    # Only columns with exactly ONE rank token are candidates (second_order columns
    # carry two tokens — donor + acceptor — and are skipped automatically).
    # A group is merged only when no molecule has more than one non-NaN value among
    # the members, confirming they represent the same orbital at varying positions.
    import re as _re
    _lp_num_re = _re.compile(r'_lp(\d+)')
    _hm_num_re = _re.compile(r'_hm(\d+)')

    _rank_groups = {}
    for col in df_out.columns:
        if len(_lp_num_re.findall(col)) + len(_hm_num_re.findall(col)) == 1:
            base = _lp_num_re.sub('_lp', col)
            base = _hm_num_re.sub('_hm', base)
            if base != col:
                _rank_groups.setdefault(base, []).append(col)

    _new_cols = {}
    _cols_to_drop = []
    for base, members in _rank_groups.items():
        # Include single-member groups: even if only one lpX/hmX variant exists
        # across the library, rename it to the numberless base name.
        # For multi-member groups, only merge when no molecule has >1 non-NaN
        # (i.e. members represent the same orbital at varying HOMO-relative rank,
        # not genuinely distinct orbitals).
        non_nan_per_row = df_out[members].notna().sum(axis=1)
        if non_nan_per_row.max() <= 1:
            # Safe: each molecule has at most one non-NaN -> coalesce
            _new_cols[base] = df_out[members].bfill(axis=1).iloc[:, 0]
            _cols_to_drop.extend(members)

    if _new_cols:
        df_out = df_out.drop(columns=_cols_to_drop)
        df_out = pd.concat([df_out, pd.DataFrame(_new_cols, index=df_out.index)], axis=1)

    return df_out

def remove_atom_labeled_columns(df, atom_labels):
    """Remove columns whose names contain atom-label tokens.

    This is useful before aggregate descriptor generation (e.g. *_min/*_max/*_lowE/*_boltz)
    when atom-specific descriptors should be excluded.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    atom_labels : list[str]
        Atom tokens such as ["H1", "H2", "C1", "C2", "N1", "N2", "Ni"].

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        Filtered dataframe and sorted list of removed column names.
    """
    labels = [str(a).strip() for a in atom_labels if str(a).strip()]
    if not labels:
        return df.copy(), []

    # Match whole atom tokens with non-alphanumeric boundaries (or string boundaries).
    atom_pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(map(re.escape, labels)) + r")(?![A-Za-z0-9])")

    removed = [col for col in df.columns if atom_pattern.search(str(col))]
    filtered = df.drop(columns=removed, errors="ignore")
    return filtered, sorted(removed)


def filter_numeric_descriptors(
    df,
    corr_threshold=0.95,
    ratio_threshold=0.01,
    orbital_filter=None,
    ni_orbital_filter=None,
    cor_orbital_filter=None,
    nmr_filter=None,
    bond_analysis_filter=None,
    atom_value_filter=None,
    atom_labels=None,
    remove_low_occupancy_nbo=False,
    occupancy_boltz_threshold=0.001,
    corr_pruning_mode="component_ranked",
    metal_label="Ni",
):
    """Filter noisy numeric descriptors using variance/ratio/correlation rules.

    Rules applied in order:
    1. Keep only numeric columns.
     2. Optionally remove NBO orbital families whose ``occupancy_boltz`` is below
         a threshold for all compounds.
     3. Optionally remove Rydberg descriptors (except Ni when requested).
     4. Optionally remove Ni 4f Rydberg descriptors.
     5. Optionally remove NMR aniso descriptors.
     6. Optionally remove phi/theta bond-analysis descriptors.
     7. Optionally remove atom-index/atom-value columns (e.g. H1/H2/C1...).
     8. Remove columns with zero standard deviation.
     9. Remove columns where std/abs(mean) < ``ratio_threshold``.
    10. Remove highly correlated descriptors (|r| > ``corr_threshold``).
         Default mode uses connected components with deterministic ranking,
         which is less sensitive to pre-filters than pairwise greedy dropping.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing descriptors.
    corr_threshold : float, optional
        Correlation threshold for pair pruning, by default 0.95.
    ratio_threshold : float, optional
        Minimum allowed std/abs(mean), by default 0.01 (1%).
    orbital_filter : str | None, optional
        Orbital-level filter option. Supported values:
        - ``"remove rydbergs"``: remove columns whose names contain ``ryd``
          except those tied to Ni (contains ``_Ni_`` / ``_ni_``).
        ni_orbital_filter : str | None, optional
                Ni-specific orbital filter option. Supported values:
                - ``"remove 4f rydbergs"``: remove Ni descriptors containing both
                    ``ryd`` and ``4f`` (e.g. ``nao_Ni_Ryd-4f_occupancy``).
        nmr_filter : str | None, optional
                NMR descriptor filter option. Supported values:
                - ``"remove aniso"``: remove anisotropic shielding terms
                    (e.g. ``saniso_*``, ``*aniso*``), while keeping isotropic terms.
        bond_analysis_filter : str | None, optional
                Bond-analysis filter option. Supported values:
                - ``"remove phi theta"``: remove descriptors containing
                    ``phi`` or ``theta`` in their column names.
        atom_value_filter : str | None, optional
            Atom-value filter option. Supported values:
            - ``"remove atom values"``: remove columns that represent atom
              numbering/index values for labels provided in ``atom_labels``
              (e.g. ``H1``, ``H1_boltz``, ``H1_min``, ...).
        atom_labels : list[str] | None, optional
            Atom labels to use with ``atom_value_filter``. If ``None``, defaults
            to ``["H1", "H2", "C1", "C2", "N1", "N2", "Ni"]``.
        remove_low_occupancy_nbo : bool, optional
            If True, remove all NBO descriptor columns for orbital families where
            ``*_occupancy_boltz`` is below ``occupancy_boltz_threshold`` for all compounds.
        occupancy_boltz_threshold : float, optional
            Threshold used with ``remove_low_occupancy_nbo``, by default 0.001.
        metal_label : str | None, optional
        Atom label of the metal centre whose Rydberg descriptors are kept by
        ``orbital_filter`` and targeted by ``ni_orbital_filter``. Matched as the
        ``_<label>_`` token (case-insensitive). ``None`` disables both metal rules,
        so ``"remove rydbergs"`` removes every Rydberg column.
    corr_pruning_mode : str, optional
                Correlation pruning strategy. Supported values:
                - ``"component_ranked"`` (default): for each connected component in the
                    correlation graph, keep one best-ranked descriptor.
                - ``"greedy_pairwise"``: legacy pairwise greedy pruning.
                - ``"bridge_star_centers"``: remove articulation-point bridge descriptors,
                    then keep one center descriptor per star-like component; use pairwise
                    pruning for non-star residual components.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Filtered numeric dataframe and a report dictionary with dropped columns.
    """

    df_num = df.select_dtypes(include=[np.number]).copy()

    removed_low_occ_nbo_orbitals = []
    removed_low_occ_nbo_columns = []
    if remove_low_occupancy_nbo:
        occ_suffix = "_occupancy_boltz"
        occ_cols = [c for c in df_num.columns if c.endswith(occ_suffix)]

        low_orbital_bases = []
        for occ_col in occ_cols:
            series = pd.to_numeric(df_num[occ_col], errors="coerce")
            max_occ = series.max(skipna=True)
            if pd.isna(max_occ) or max_occ < occupancy_boltz_threshold:
                low_orbital_bases.append(occ_col[: -len(occ_suffix)])

        if low_orbital_bases:
            removed_low_occ_nbo_orbitals = sorted(low_orbital_bases)
            cols_to_remove = []
            for col in df_num.columns:
                if any(col.startswith(base + "_") for base in low_orbital_bases):
                    cols_to_remove.append(col)
            removed_low_occ_nbo_columns = sorted(cols_to_remove)
            if removed_low_occ_nbo_columns:
                df_num = df_num.drop(columns=removed_low_occ_nbo_columns, errors="ignore")

    metal_token = f"_{str(metal_label).strip().lower()}_" if metal_label else None

    removed_rydberg = []
    if isinstance(orbital_filter, str) and orbital_filter.strip().lower() == "remove rydbergs":
        cols_to_remove = []
        for col in df_num.columns:
            c = col.lower()
            # Drop Rydberg-related descriptors, but keep the metal-centre ones.
            if "ryd" in c and (metal_token is None or metal_token not in c):
                cols_to_remove.append(col)
        removed_rydberg = sorted(cols_to_remove)
        if removed_rydberg:
            df_num = df_num.drop(columns=removed_rydberg, errors="ignore")

    removed_ni_4f_rydberg = []
    if isinstance(ni_orbital_filter, str) and ni_orbital_filter.strip().lower() == "remove 4f rydbergs":
        cols_to_remove = []
        for col in df_num.columns:
            c = col.lower()
            # Remove only metal-centre 4f Rydberg descriptors.
            if metal_token is not None and metal_token in c and "ryd" in c and "4f" in c:
                cols_to_remove.append(col)
        removed_ni_4f_rydberg = sorted(cols_to_remove)
        if removed_ni_4f_rydberg:
            df_num = df_num.drop(columns=removed_ni_4f_rydberg, errors="ignore")

    removed_cor = []
    if isinstance(cor_orbital_filter, str) and cor_orbital_filter.strip().lower() == "remove cor":
        cols_to_remove = []
        for col in df_num.columns:
            c = col.lower()
            # Remove core (Cor) orbital descriptors.
            if "_cor-" in c or "_cor_" in c:
                cols_to_remove.append(col)
        removed_cor = sorted(cols_to_remove)
        if removed_cor:
            df_num = df_num.drop(columns=removed_cor, errors="ignore")

    removed_nmr_aniso = []
    if isinstance(nmr_filter, str) and nmr_filter.strip().lower() == "remove aniso":
        cols_to_remove = []
        for col in df_num.columns:
            c = col.lower()
            # Remove anisotropic NMR shielding descriptors (saniso / aniso terms).
            if "saniso" in c or "aniso" in c:
                cols_to_remove.append(col)
        removed_nmr_aniso = sorted(cols_to_remove)
        if removed_nmr_aniso:
            df_num = df_num.drop(columns=removed_nmr_aniso, errors="ignore")

    removed_phi_theta = []
    if isinstance(bond_analysis_filter, str) and bond_analysis_filter.strip().lower() == "remove phi theta":
        cols_to_remove = []
        for col in df_num.columns:
            c = col.lower()
            if "phi" in c or "theta" in c:
                cols_to_remove.append(col)
        removed_phi_theta = sorted(cols_to_remove)
        if removed_phi_theta:
            df_num = df_num.drop(columns=removed_phi_theta, errors="ignore")

    removed_atom_value_columns = []
    if isinstance(atom_value_filter, str) and atom_value_filter.strip().lower() == "remove atom values":
        labels = atom_labels if atom_labels is not None else ["H1", "H2", "C1", "C2", "N1", "N2", "Ni"]
        label_set = {str(x).strip() for x in labels if str(x).strip()}

        cols_to_remove = []
        for col in df_num.columns:
            col_str = str(col)
            for label in label_set:
                # Remove pure atom-index columns and their aggregate variants.
                if col_str == label or col_str.startswith(f"{label}_"):
                    cols_to_remove.append(col)
                    break

        removed_atom_value_columns = sorted(set(cols_to_remove))
        if removed_atom_value_columns:
            df_num = df_num.drop(columns=removed_atom_value_columns, errors="ignore")

    std_vals = df_num.std(axis=0, skipna=True)
    keep_nonzero_std = std_vals[std_vals > 0].index
    removed_zero_std = sorted(set(df_num.columns) - set(keep_nonzero_std))
    df_num = df_num.loc[:, keep_nonzero_std]

    std_vals = df_num.std(axis=0, skipna=True)
    abs_mean_vals = df_num.mean(axis=0, skipna=True).abs()
    ratio_std_mean = std_vals / abs_mean_vals.replace(0, np.nan)
    keep_ratio = ratio_std_mean[(ratio_std_mean >= ratio_threshold) | ratio_std_mean.isna()].index
    removed_low_ratio = sorted(set(df_num.columns) - set(keep_ratio))
    df_num = df_num.loc[:, keep_ratio]

    def suffix_priority(col_name):
        c = col_name.lower()
        if c.endswith("_boltz") or c.endswith("_boltzmann"):
            return 0
        if c.endswith("_lowe"):
            return 1
        if c.endswith("_min"):
            return 2
        if c.endswith("_max"):
            return 3
        return 4

    corr_abs = df_num.corr().abs()
    cols = corr_abs.columns.tolist()
    to_drop_corr = set()

    neighbors_map = {
        c: set(corr_abs.index[corr_abs.loc[c] > corr_threshold].tolist()) - {c}
        for c in cols
    }

    def _rank_key(col_name):
        std_val = std_vals.get(col_name, np.nan)
        std_sort = -std_val if pd.notna(std_val) else np.inf
        missing_frac = df_num[col_name].isna().mean()
        return (suffix_priority(col_name), missing_frac, std_sort, col_name)

    def _components_from_neighbors(neigh):
        visited_local = set()
        comps = []
        for node in neigh:
            if node in visited_local:
                continue
            stack_local = [node]
            visited_local.add(node)
            comp = []
            while stack_local:
                u = stack_local.pop()
                comp.append(u)
                for v in neigh.get(u, set()):
                    if v not in visited_local:
                        visited_local.add(v)
                        stack_local.append(v)
            comps.append(comp)
        return comps

    def _articulation_points(neigh):
        # Tarjan articulation-point detection on an undirected graph.
        disc = {}
        low = {}
        parent = {}
        ap = set()
        time = [0]

        def dfs(u):
            disc[u] = time[0]
            low[u] = time[0]
            time[0] += 1
            child_count = 0

            for v in neigh.get(u, set()):
                if v not in disc:
                    parent[v] = u
                    child_count += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent.get(u) is None and child_count > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        for node in neigh:
            if node not in disc:
                parent[node] = None
                dfs(node)

        return ap

    def _pairwise_prune(component, existing_drop):
        comp_drop = set()
        comp_cols = list(component)
        for i in range(len(comp_cols)):
            c1 = comp_cols[i]
            if c1 in comp_drop or c1 in existing_drop:
                continue
            for j in range(i + 1, len(comp_cols)):
                c2 = comp_cols[j]
                if c2 in comp_drop or c2 in existing_drop:
                    continue
                if corr_abs.loc[c1, c2] <= corr_threshold:
                    continue

                p1 = suffix_priority(c1)
                p2 = suffix_priority(c2)
                if p1 < p2:
                    drop_col = c2
                elif p2 < p1:
                    drop_col = c1
                else:
                    s1 = std_vals.get(c1, 0)
                    s2 = std_vals.get(c2, 0)
                    if s1 > s2:
                        drop_col = c2
                    elif s2 > s1:
                        drop_col = c1
                    else:
                        drop_col = c2 if c1 <= c2 else c1

                comp_drop.add(drop_col)
        return comp_drop

    if corr_pruning_mode == "component_ranked":
        visited = set()
        for root in cols:
            if root in visited:
                continue

            stack = [root]
            component = []
            visited.add(root)

            while stack:
                node = stack.pop()
                component.append(node)
                neighbors = corr_abs.index[corr_abs.loc[node] > corr_threshold].tolist()
                for nb in neighbors:
                    if nb == node or nb in visited:
                        continue
                    visited.add(nb)
                    stack.append(nb)

            if len(component) <= 1:
                continue

            keep_col = min(component, key=_rank_key)
            for c in component:
                if c != keep_col:
                    to_drop_corr.add(c)
    elif corr_pruning_mode == "bridge_star_centers":
        # Step 1: iteratively remove bridge descriptors (articulation points)
        # to split correlation components.
        work_neighbors = {k: set(v) for k, v in neighbors_map.items()}
        while True:
            aps = _articulation_points(work_neighbors)
            if not aps:
                break

            # Drop the least preferred articulation points first.
            aps_sorted = sorted(list(aps), key=_rank_key, reverse=True)
            to_remove_now = set(aps_sorted)
            to_drop_corr.update(to_remove_now)

            for rm in to_remove_now:
                for nb in list(work_neighbors.get(rm, set())):
                    work_neighbors[nb].discard(rm)
                work_neighbors.pop(rm, None)

            if not work_neighbors:
                break

        # Step 2: in each resulting component, keep one star center when possible;
        # otherwise apply local pairwise pruning.
        residual_components = _components_from_neighbors(work_neighbors)
        for comp in residual_components:
            if len(comp) <= 1:
                continue

            comp_set = set(comp)
            centers = [c for c in comp if work_neighbors.get(c, set()) >= (comp_set - {c})]
            if centers:
                keep_center = min(centers, key=_rank_key)
                for c in comp:
                    if c != keep_center:
                        to_drop_corr.add(c)
            else:
                to_drop_corr.update(_pairwise_prune(comp, to_drop_corr))
    elif corr_pruning_mode == "greedy_pairwise":
        to_drop_corr.update(_pairwise_prune(cols, to_drop_corr))
    else:
        raise ValueError(
            "Unsupported corr_pruning_mode. Use 'component_ranked', 'bridge_star_centers', or 'greedy_pairwise'."
        )

    df_filtered = df_num.drop(columns=sorted(to_drop_corr), errors="ignore")

    report = {
        "initial_numeric_count": len(df.select_dtypes(include=[np.number]).columns),
        "removed_zero_std": removed_zero_std,
        "removed_low_ratio": removed_low_ratio,
        "removed_low_occ_nbo_orbitals": removed_low_occ_nbo_orbitals,
        "removed_low_occ_nbo_columns": removed_low_occ_nbo_columns,
        "removed_rydberg": removed_rydberg,
        "removed_ni_4f_rydberg": removed_ni_4f_rydberg,
        "removed_cor": removed_cor,
        "removed_nmr_aniso": removed_nmr_aniso,
        "removed_phi_theta": removed_phi_theta,
        "removed_atom_value_columns": removed_atom_value_columns,
        "removed_high_corr": sorted(to_drop_corr),
        "final_numeric_count": df_filtered.shape[1],
        "corr_threshold": corr_threshold,
        "ratio_threshold": ratio_threshold,
        "orbital_filter": orbital_filter,
        "ni_orbital_filter": ni_orbital_filter,
        "cor_orbital_filter": cor_orbital_filter,
        "nmr_filter": nmr_filter,
        "bond_analysis_filter": bond_analysis_filter,
        "atom_value_filter": atom_value_filter,
        "atom_labels": atom_labels,
        "remove_low_occupancy_nbo": remove_low_occupancy_nbo,
        "occupancy_boltz_threshold": occupancy_boltz_threshold,
        "corr_pruning_mode": corr_pruning_mode,
    }

    return df_filtered, report
