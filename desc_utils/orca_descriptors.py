import pandas as pd
from pathlib import Path
import re
import numpy as np
import os
import json

from morfeus import Sterimol, BuriedVolume, Pyramidalization, SASA, BiteAngle

import dbstep.Dbstep as db

import logging
logging.basicConfig(level=logging.WARNING) 

h_to_eV = 27.2114
bohr_to_A = 0.529177

COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Fe": 1.24,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Pd": 1.39,
    "Pt": 1.36,
}


class OrcaReader:
    """ORCA results reader with unified atom index labeling.

    Atom handling:
    - atom_index_map: Dict[int, str] mapping 1-based atom index -> label string
    - set_atoms(list[int]) assigns numeric string labels automatically
    - set_atoms(dict[int,str]) assigns explicit labels
    - set_atom_labels(list[str]|dict[int,str]) updates labels

    Result keys follow: f"{prefix}_{label}".
    """
    def __init__(self, directory, basename="singlepoint", out_file_path=None, nbo_block_index=0, nbo_job_number=None):
        self.directory = Path(directory)
        self.basename = basename
        self.outfile = Path(out_file_path) if out_file_path else self.directory / f"{self.basename}.out"
        self.nbo_block_index = nbo_block_index
        self.nbo_job_number = nbo_job_number

        # Core data stores
        self.results = {}
        self.propertyjson = {}
        self.gbwjson = {}

        # Unified atom index mapping (1-based indices)
        self.atom_index_map = {}
        self.atom_labels_map = {}
        self.atom_list = []

        # NBO parsing caches (kept for future extension)
        self.nbo_sections = []
        self.nbo_section = None
        self._outfile_content = None
        self._nbo_blocks_meta = []
        self._job_headers = []
        self.nboresults = {}

    # ---------------- Atom management ----------------
    def set_atoms(self, atoms):
        """Define atom indices with optional labels.

        Supported inputs:
        - `list[int]` -> labels default to the numeric atom indices
        - `dict[label, index]` -> preferred labeled form
        - `dict[index, label]` -> legacy form
        """
        if isinstance(atoms, dict):
            if all(isinstance(v, (int, np.integer)) or str(v).isdigit() for v in atoms.values()):
                self.atom_index_map = {int(v): str(k).strip() for k, v in atoms.items()}
            else:
                self.atom_index_map = {
                    int(k): (str(v).strip() if v is not None else str(k)) for k, v in atoms.items()
                }
        elif isinstance(atoms, (list, tuple)):
            self.atom_index_map = {int(k): str(int(k)) for k in atoms}
        else:
            raise TypeError("atoms must be list[int] or a dict mapping labels and indices")

        self.atom_labels_map = dict(self.atom_index_map)
        self.atom_list = list(self.atom_index_map.keys())

    def set_atom_labels(self, labels):
        """Update labels (dict[int,str] or list[str] matching current order)."""
        if not self.atom_index_map:
            print("Set atoms before labels.")
            return
        if isinstance(labels, dict):
            for k, v in labels.items():
                k = int(k)
                if k in self.atom_index_map and isinstance(v, str):
                    self.atom_index_map[k] = v.strip()
        elif isinstance(labels, (list, tuple)):
            if len(labels) != len(self.atom_index_map):
                print("Label list length must match number of atoms.")
                return
            for key, new_label in zip(self.atom_index_map.keys(), labels):
                if isinstance(new_label, str):
                    self.atom_index_map[key] = new_label.strip()
        else:
            raise TypeError("labels must be dict or list[str]")

        self.atom_labels_map = dict(self.atom_index_map)
        self.atom_list = list(self.atom_index_map.keys())

    @property
    def atom_indices(self):
        return list(self.atom_index_map.keys())

    def _atom_key(self, prefix, atom_index):
        label = self.atom_index_map.get(atom_index, str(atom_index))
        return f"{prefix}_{label}"
    
    # -------------------- configuration helpers --------------------
    def set_outfile(self, out_file_path):
        changed = Path(out_file_path) != self.outfile
        self.outfile = Path(out_file_path)
        if changed:
            self.nbo_sections = []
            self.nbo_section = None
            self._outfile_content = None
            self._nbo_blocks_meta = []
            self._job_headers = []
            self._clear_nbo_data()

    def set_nbo_block_index(self, index:int):
        prev_index = self.nbo_block_index
        self.nbo_block_index = index
        if self.nbo_sections:
            if 0 <= index < len(self.nbo_sections):
                if index != prev_index:
                    self.nbo_section = self.nbo_sections[index]
                    self._clear_nbo_data()
            else:
                print(f"Requested NBO block index {index} out of range (found {len(self.nbo_sections)} blocks).")

    def set_nbo_job_number(self, job_number:int|None):
        prev_job = self.nbo_job_number
        self.nbo_job_number = job_number
        if prev_job != job_number:
            self.nbo_section = None
            self._clear_nbo_data()

    # -------------------- internal clear helper --------------------
    def _clear_nbo_data(self):
        """Clear previously parsed NBO results and any NBO charge entries in self.results."""
        nbo_keys = [k for k in self.results.keys() if k.startswith("nbo_charge_")]
        for k in nbo_keys:
            del self.results[k]
        self.nboresults = {}

    # -------------------- atom label helpers --------------------
    def _atom_key(self, prefix, atom_index):
        """Return result key using custom label if present else index."""
        label = self.atom_index_map.get(atom_index)
        if label:
            return f"{prefix}_{label}"
        return f"{prefix}_{atom_index}"

    # -------------------- configuration helpers --------------------
    def set_outfile(self, out_file_path):
        changed = Path(out_file_path) != self.outfile
        self.outfile = Path(out_file_path)
        if changed:
            self.nbo_sections = []
            self.nbo_section = None
            self._outfile_content = None
            self._nbo_blocks_meta = []
            self._job_headers = []
            self._clear_nbo_data()

    def set_nbo_block_index(self, index:int):
        prev_index = self.nbo_block_index
        self.nbo_block_index = index
        if self.nbo_sections:
            if 0 <= index < len(self.nbo_sections):
                if index != prev_index:
                    self.nbo_section = self.nbo_sections[index]
                    self._clear_nbo_data()
            else:
                print(f"Requested NBO block index {index} out of range (found {len(self.nbo_sections)} blocks).")

    def set_nbo_job_number(self, job_number:int|None):
        prev_job = self.nbo_job_number
        self.nbo_job_number = job_number
        if prev_job != job_number:
            self.nbo_section = None
            self._clear_nbo_data()

    # -------------------- generic safe traversal helpers --------------------
    @staticmethod
    def _nested_get(root, path):
        cur = root
        for p in path:
            try:
                cur = cur[p]
            except (KeyError, IndexError, TypeError):
                return None
        return cur

    def _last_geometry(self):
        geoms = self._nested_get(self.propertyjson, ["Geometries"])
        if not geoms or not isinstance(geoms, list):
            return None
        if not geoms:
            return None
        return geoms[-1]

    # -------------------- JSON parsing --------------------
    def parse_property_json(self):
        file_name = f'{self.basename}.property.json'
        if file_name in os.listdir(self.directory):
            try:
                self.propertyjson = json.load(open(self.directory / file_name))
            except Exception as e:
                print(f"Failed to load {file_name}: {e}")
                self.propertyjson = {}
        else:
            print(f"No {self.basename}.property.json file found in the specified path. Results dictionary will be empty.")

    def parse_gbw_json(self):
        file_name = f'{self.basename}.json'
        if file_name in os.listdir(self.directory):
            try:
                self.gbwjson = json.load(open(self.directory / file_name))
            except Exception as e:
                print(f"Failed to load {file_name}: {e}")
                self.gbwjson = {}
        else:
            print(f"No {self.basename}.json file found in the specified path. Results dictionary will be empty.")
    # ---------------- Charge extraction ----------------
    def _charge_table(self, key):
        geom = self._last_geometry()
        return self._nested_get(geom, [key, 0, "AtomicCharges"]) if geom else None

    def _extract_charges(self, table_key, prefix, preserve_case=False, local_atom_index_map=None):
        """Generic extractor supporting alternate atom index maps.

        local_atom_index_map: optional dict[int,str]; if None uses self.atom_index_map.
        """
        if local_atom_index_map is None:
            local_atom_index_map = self.atom_index_map
        if not local_atom_index_map:
            print("No atoms defined for charge extraction.")
            return
        table = self._charge_table(table_key)
        if table is None:
            print(f"{table_key} charges not found; skipping.")
            return
        name_prefix = prefix if preserve_case else prefix.lower()
        for idx, label in local_atom_index_map.items():
            value = None
            entry = self._nested_get(table, [idx - 1])
            if entry and isinstance(entry, list) and entry:
                value = entry[0]
            elif entry:
                value = entry
            self.results[f"{name_prefix}_{label}"] = value

    def get_mulliken_charges(self, local_atom_index_map=None):
        self._extract_charges("Mulliken_Population_Analysis", "mulliken", local_atom_index_map=local_atom_index_map)

    def get_loewdin_charges(self, local_atom_index_map=None):
        self._extract_charges("Loewdin_Population_Analysis", "loewdin", local_atom_index_map=local_atom_index_map)

    def get_chelpg_charges(self, local_atom_index_map=None):
        self._extract_charges("CHELPG_Population_Analysis", "chelpG", preserve_case=True, local_atom_index_map=local_atom_index_map)

    def get_hirshfeld_charges(self, local_atom_index_map=None):
        self._extract_charges("Hirshfeld_Population_Analysis", "hirshfeld", local_atom_index_map=local_atom_index_map)

    def get_cm5_charges(self, extended=True, atomradius_setting="hokru", local_atom_index_map=None):
        if local_atom_index_map is None:
            local_atom_index_map = self.atom_index_map
        if not local_atom_index_map:
            print("No atoms defined for CM5.")
            return
        geom = self._last_geometry()
        if geom is None:
            print("No geometry for CM5.")
            return
        hir_raw = self._nested_get(geom, ["Hirshfeld_Population_Analysis", 0, "AtomicCharges"])
        atnos = self._nested_get(geom, ["Hirshfeld_Population_Analysis", 0, "ATNO"])
        cart = self._nested_get(geom, ["Geometry", "Coordinates", "Cartesians"])
        if hir_raw is None or atnos is None or cart is None:
            print("Missing CM5 inputs.")
            return
        try:
            hirshfeld_charges = np.array([x[0] for x in hir_raw])
            z = np.array([x[0] for x in atnos])
            xyz = np.array([[x, y, zv] for _, x, y, zv in cart]) * bohr_to_A
        except Exception as e:
            print(f"Failed CM5 prep: {e}")
            return
        atomradius = np.zeros(119)
        atomradius[:] = 0.7
        alpha = 2.474
        def _tij(i, j):
            return 0.0
        for idx, label in local_atom_index_map.items():
            i = idx - 1
            if i >= len(hirshfeld_charges):
                self.results[f"cm5_{label}"] = None
                continue
            s = 0.0
            for j in range(len(hirshfeld_charges)):
                if i == j:
                    continue
                try:
                    rij = np.linalg.norm(xyz[i] - xyz[j])
                except Exception:
                    continue
                bij = np.exp(-alpha * (rij - atomradius[z[i]] - atomradius[z[j]]))
                tij = _tij(z[i], z[j])
                s += tij * bij
            self.results[f"cm5_{label}"] = hirshfeld_charges[i] + s

    # ---------------- Chemical shift tensors ----------------
    def get_saniso(self, local_atom_index_map=None):
        if local_atom_index_map is None:
            local_atom_index_map = self.atom_index_map
        if not local_atom_index_map:
            print("Atom list is not defined. Please define it.")
            return
        geom = self._last_geometry()
        table = self._nested_get(geom, ["Chemical_Shift", 0, "saniso"]) if geom else None
        if table is None:
            print("saniso data not found; skipping.")
            return
        for at_idx, label in local_atom_index_map.items():
            value = None
            entry = self._nested_get(table, [at_idx - 1])
            if entry and isinstance(entry, list) and entry:
                value = entry[0]
            elif entry:
                value = entry
            self.results[f"saniso_{label}"] = value

    def get_siso(self, local_atom_index_map=None):
        if local_atom_index_map is None:
            local_atom_index_map = self.atom_index_map
        if not local_atom_index_map:
            print("Atom list is not defined. Please define it.")
            return
        geom = self._last_geometry()
        table = self._nested_get(geom, ["Chemical_Shift", 0, "siso"]) if geom else None
        if table is None:
            print("siso data not found; skipping.")
            return
        for at_idx, label in local_atom_index_map.items():
            value = None
            entry = self._nested_get(table, [at_idx - 1])
            if entry and isinstance(entry, list) and entry:
                value = entry[0]
            elif entry:
                value = entry
            self.results[f"siso_{label}"] = value

    def get_all_charges(self, local_atom_index_map=None):
        self.get_mulliken_charges(local_atom_index_map)
        self.get_loewdin_charges(local_atom_index_map)
        self.get_chelpg_charges(local_atom_index_map)
        self.get_hirshfeld_charges(local_atom_index_map)
        self.get_cm5_charges(local_atom_index_map=local_atom_index_map)


    def get_scfenergy(self):
        geom = self._last_geometry()
        value = self._nested_get(geom, ["DFT_Energy", "finalEn"]) if geom else None
        if value is None:
            print("SCF energy not found.")
        self.results["SCF_energy"] = value

    def get_homolumo(self):
        mos = self._nested_get(self.gbwjson, ["Molecule", "MolecularOrbitals", "MOs"])
        if mos is None:
            print("MO data not found; skipping HOMO/LUMO.")
            self.results["HOMO_energy"] = None
            self.results["LUMO_energy"] = None
            return
        mo_energies = []
        mo_occupancy = []
        for mo in mos:
            try:
                mo_energies.append(mo.get("OrbitalEnergy"))
                mo_occupancy.append(mo.get("Occupancy"))
            except Exception:
                continue
        mo_energies = np.array([e for e in mo_energies if e is not None])
        mo_occupancy = np.array([o for o in mo_occupancy if o is not None])
        if mo_energies.size == 0 or mo_occupancy.size == 0:
            print("Incomplete MO data; skipping HOMO/LUMO.")
            self.results["HOMO_energy"] = None
            self.results["LUMO_energy"] = None
            return
        try:
            homo_mask = (mo_occupancy == 1.0) | (mo_occupancy == 2.0)
            HOMO_energy = np.max(mo_energies[homo_mask]) * h_to_eV if np.any(homo_mask) else None
            lumo_mask = (mo_occupancy == 0.0)
            LUMO_energy = np.min(mo_energies[lumo_mask]) * h_to_eV if np.any(lumo_mask) else None
        except Exception:
            HOMO_energy = None
            LUMO_energy = None
        self.results["HOMO_energy"] = HOMO_energy
        self.results["LUMO_energy"] = LUMO_energy

    def get_enthalpy(self):
        geom = self._last_geometry()
        value = self._nested_get(geom, ["THERMOCHEMISTRY_Energies", 0, "enthalpyH"]) if geom else None
        self.results["Enthalpy"] = value

    def get_gfe(self):
        geom = self._last_geometry()
        value = self._nested_get(geom, ["THERMOCHEMISTRY_Energies", 0, "freeEnergyG"]) if geom else None
        self.results["GibbsFreeEnergy"] = value


    def _ensure_outfile_content(self):
        if self._outfile_content is None:
            try:
                self._outfile_content = self.outfile.read_text()
            except Exception as e:
                print(f"Failed to read output file '{self.outfile}': {e}")
                self._outfile_content = ""

    def _scan_job_headers(self):
        self._ensure_outfile_content()
        text = self._outfile_content
        self._job_headers = []
        if not text:
            return
        job_re = re.compile(r"(?m)^[ \t]*COMPOUND[ \t]+JOB[ \t]+(?:\{[ \t]*(\d+)[ \t]*\}|(\d+))[ \t]*$")
        for m in job_re.finditer(text):
            num = m.group(1) or m.group(2)
            try:
                num_i = int(num)
            except Exception:
                continue
            self._job_headers.append((num_i, m.start()))
        self._job_headers.sort(key=lambda x: x[1])

    def write_results_to_file(self, output_path=None):
        if output_path is None:
            output_path = self.directory / f"{self.basename}_descriptors.json"
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=4)
        except Exception as e:
            print(f"Failed writing results file: {e}")

    def aggregate_results(self):
        # Electronic properties are read from the current basename (typically singlepoint).
        self.parse_property_json()
        self.parse_gbw_json()
        self.get_all_charges()
        self.get_scfenergy()
        self.get_homolumo()
        self.get_enthalpy()
        self.get_gfe()

        # NMR shielding is often written to nmr.property.json, not singlepoint.property.json.
        original_basename = self.basename
        if original_basename != "nmr":
            self.basename = "nmr"
            self.parse_property_json()
        self.get_saniso()
        self.get_siso()
        self.basename = original_basename

class NBOProps:
    """Properties from out file from Orca NBO calculations"""

    def __init__(self, path):
        self.path = Path(path)
        self.content = self.path.read_text()

        self.nbo_start = self.content.find("Now starting NBO")
        if self.nbo_start != -1:
            # Find the end of NBO section (marked by "returned from NBO program")
            self.nbo_end = self.content.find("*** returned from  NBO  program ***", self.nbo_start)
            if self.nbo_end != -1:
                self.nbo_section = self.content[self.nbo_start:self.nbo_end + len("*** returned from  NBO  program ***")]
                #print(f"Found NBO section: {len(self.nbo_section)} characters")
            else:
                print("Could not find end of NBO section")
                self.nbo_section = None
        else:
            print("Could not find 'Now starting NBO' in the file")
            self.nbo_section = None

        self.results = {
            'source_file': str(self.path),
            'analysis_type': 'NBO 7.0',
            'natural_population_analysis': {
                'atoms': {},
                'totals': {}
            },
            'natural_atomic_orbitals': {},
            'natural_bond_orbitals': {},
            'bond_orbital_details': {},
            'nho_directionality': {},
            'second_order_perturbation': {},
            'analysis_info': {}
        }

    def get_NPA(self):

        # Extract Natural Population Analysis
        # Find the NPA subsection first
        npa_start = self.nbo_section.find("Summary of Natural Population Analysis:")
        npa_end = self.nbo_section.find("Natural Electron Configuration", npa_start)
        if npa_end == -1:
            npa_end = self.nbo_section.find("\n\n NATURAL BOND ORBITAL ANALYSIS:", npa_start)

        if npa_start != -1 and npa_end != -1:
            npa_section = self.nbo_section[npa_start:npa_end]
            
            # Initialize arrays
            atom_properties = []
            total_properties = {}
            
            # Extract atom charges and populations
            # Look for the table between dashed lines with pattern: "    H  1    0.00000      0.00000     0.99914    0.00086     1.00000"
            # The line has: Atom Symbol, Atom Number, Charge, Core, Valence, Rydberg, Total
            atom_lines = re.findall(r'\n\s+([A-Z][a-z]?)\s+(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', npa_section)
            
            for atom_data in atom_lines:
                atom_symbol = atom_data[0]
                atom_num = int(atom_data[1])
                
                # Extend array if needed to include this atom number
                # This indexes atoms starting from 1
                while len(atom_properties) <= atom_num:
                    atom_properties.append(None)
                
                atom_properties[atom_num] = {
                    'atom': atom_symbol,
                    'charge': float(atom_data[2]),
                    'core': float(atom_data[3]),
                    'valence': float(atom_data[4]),
                    'rydberg': float(atom_data[5]),
                    'total': float(atom_data[6])
                }

            self.results['natural_population_analysis']['atoms'] = {i: prop for i, prop in enumerate(atom_properties) if prop is not None}
            
            # Extract total properties (line starting with "* Total *")
            total_match = re.search(r'\*\s*Total\s*\*\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', npa_section)
            if total_match:
                total_properties = {
                    'charge': float(total_match.group(1)),
                    'core': float(total_match.group(2)),
                    'valence': float(total_match.group(3)),
                    'rydberg': float(total_match.group(4)),
                    'total': float(total_match.group(5))
                }

                self.results['natural_population_analysis']['totals'] = total_properties
        else:
            print("Could not find Natural Population Analysis section")

    def get_NAO(self):
        # Extract Natural Atomic Orbital (NAO) occupancies
        # Find the NAO subsection
        nao_start = self.nbo_section.find("NATURAL POPULATIONS:  Natural atomic orbital occupancies")
        nao_end = self.nbo_section.find("Summary of Natural Population Analysis:", nao_start)

        nao_properties = []
        if nao_start != -1 and nao_end != -1:
            nao_subsection = self.nbo_section[nao_start:nao_end]
            
            # Look for the table header and then parse lines
            # Pattern: "   1    H  1  s      Val( 1s)     0.99914       0.02931"
            nao_lines = re.findall(r'\n\s+(\d+)\s+([A-Z][a-z]?)\s+(\d+)\s+(\S+)\s+(\S+\([^)]+\))\s+([\d.]+)\s+([-\d.]+)', nao_subsection)
            
            for line_data in nao_lines:
                orbital_type = line_data[4].split('(')[0]  # Extract orbital type before '('
                orbital = line_data[4].split('(')[1].strip(')').strip()  # Extract orbital details inside '()'
                nao_properties.append({
                    'orbital_num': int(line_data[0]),
                    'atom': line_data[1],
                    'atom_num': int(line_data[2]),
                    'angular': line_data[3],
                    'orbital_type': orbital_type,
                    'orbital': orbital,
                    'occupancy': float(line_data[5]),
                    'energy': float(line_data[6])
                })

            self.results['natural_atomic_orbitals'] = nao_properties
            
        else:
            print("Could not find NAO section")

    def get_NBO_summary(self):

        # Extract Natural Bond Orbital (NBO) summary
        # Find the NBO summary subsection
        nbo_summary_start   = self.nbo_section.find("NATURAL BOND ORBITALS (Summary):")
        nbo_summary_end     = self.nbo_section.find("$CHOOSE", nbo_summary_start)
        if nbo_summary_end  == -1:
            nbo_summary_end = self.nbo_section.find("Checkpoint orbital", nbo_summary_start)

        nbo_properties = []

        if nbo_summary_start != -1 and nbo_summary_end != -1:
            nbo_summary_section = self.nbo_section[nbo_summary_start:nbo_summary_end]
            
            # Parse NBO summary lines
            # Pattern: "    1. BD ( 1) H  1- H  2       2.00000    -0.43520"
            # Use explicit atom pattern to avoid cutting off at last atom number
            nbo_lines = re.findall(r'\n\s+(\d+)\.\s+(BD\*?|LP|CR|RY)\s*\(\s*(\d+)\)\s+([A-Z][a-z]?\s+\d+(?:\s*-\s*[A-Z][a-z]?\s+\d+)*)\s+([\d.]+)\s+([-\d.]+)', nbo_summary_section)
            
            for line_data in nbo_lines:
                atoms_string = line_data[3].strip()
                
                # Parse atom numbers from atoms_string (e.g., "H  1- H  2" -> [1, 2] or "H  1" -> [1])
                atom_numbers = []
                atom_num_matches = re.findall(r'([A-Z][a-z]?)\s+(\d+)', atoms_string)
                for atom_match in atom_num_matches:
                    atom_numbers.append(int(atom_match[1]))
                
                nbo_properties.append({
                    'nbo_num': int(line_data[0]),
                    'nbo_type': line_data[1],
                    'bond_index': int(line_data[2]),
                    'atoms': atom_numbers,
                    'occupancy': float(line_data[4]),
                    'energy': float(line_data[5])
                })

            self.results['natural_bond_orbitals'] = nbo_properties
            
        else:
            print("Could not find NBO summary section")

        
    def get_orbital_details(self):
        # Extract detailed bond orbital properties with hybridization
        # Find the detailed bond orbital subsection
        bond_detail_start = self.nbo_section.find("(Occupancy)   Bond orbital / Coefficients / Hybrids")
        lewis_start = self.nbo_section.find("---------------- Lewis", bond_detail_start)
        non_lewis_start = self.nbo_section.find("---------------- non-Lewis", lewis_start)

        bond_orbital_details = []

        if bond_detail_start != -1 and lewis_start != -1:
            # Determine the end of the section we want to parse (Lewis section only, or both)
            # For now, let's get both Lewis and non-Lewis
            if non_lewis_start != -1:
                # Get up to NHO DIRECTIONALITY or SECOND ORDER
                end_markers = ["NHO DIRECTIONALITY", "SECOND ORDER", "\n\n NATURAL BOND ORBITALS (Summary)"]
                detail_end = -1
                for marker in end_markers:
                    pos = self.nbo_section.find(marker, non_lewis_start)
                    if pos != -1:
                        detail_end = pos
                        break
            else:
                detail_end = -1
                end_markers = ["NHO DIRECTIONALITY", "SECOND ORDER", "\n\n NATURAL BOND ORBITALS (Summary)"]
                for marker in end_markers:
                    pos = self.nbo_section.find(marker, lewis_start)
                    if pos != -1:
                        detail_end = pos
                        break
            
            if detail_end != -1:
                bond_detail_section = self.nbo_section[lewis_start:detail_end]
                
                # Split into individual bond entries
                # Each bond starts with a pattern like "   1. (2.00000) BD ( 1) H  1- H  2"
                bond_entries = re.finditer(r'\n\s+(\d+)\.\s+\(([\d.]+)\)\s+(BD\*?|LP|CR|RY)\s*\(\s*(\d+)\)\s+(.*?)(?=\n\s+\d+\.\s+\(|\Z)', bond_detail_section, re.DOTALL)
                
                for bond_entry in bond_entries:
                    bond_num = int(bond_entry.group(1))
                    occupancy = float(bond_entry.group(2))
                    bond_type = bond_entry.group(3)
                    bond_index = int(bond_entry.group(4))
                    atoms_and_contrib = bond_entry.group(5)
                    
                    # Extract atom information from first line
                    atoms_line_match = re.match(r'(.*?)(?=\n|\()', atoms_and_contrib)
                    atoms_line = atoms_line_match.group(1).strip() if atoms_line_match else ""
                    
                    # Parse atom numbers from atoms_line (e.g., "H  1- H  2" -> [1, 2])
                    atom_numbers = []
                    atom_num_matches = re.findall(r'([A-Z][a-z]?)\s+(\d+)', atoms_line)
                    for atom_match in atom_num_matches:
                        atom_numbers.append(int(atom_match[1]))
                    
                    bond_info = {
                        'bond_num': bond_num,
                        'occupancy': occupancy,
                        'bond_type': bond_type,
                        'bond_index': bond_index,
                        'atoms': atom_numbers,
                        'contributions': []
                    }
                    
                    # Look for atomic contributions
                    # Pattern: "               ( 50.00%)   0.7071* H  1 s( 99.91%)p 0.00(  0.09%)"
                    contrib_pattern = r'\(\s*([\d.]+)%\)\s+([-\d.]+)\*\s+([A-Z][a-z]?)\s+(\d+)\s+s\(\s*([\d.]+)%\)p\s*([\d.]+)\(\s*([\d.]+)%\)'
                    contributions = re.finditer(contrib_pattern, atoms_and_contrib)
                    
                    for contrib in contributions:
                        bond_info['contributions'].append({
                            'percentage': float(contrib.group(1)),
                            'coefficient': float(contrib.group(2)),
                            'atom': contrib.group(3),
                            'atom_num': int(contrib.group(4)),
                            's_percent': float(contrib.group(5)),
                            'p_power': float(contrib.group(6)),
                            'p_percent': float(contrib.group(7))
                        })
                    
                    bond_orbital_details.append(bond_info)
                self.results['bond_orbital_details'] = bond_orbital_details
            else:
                print("Could not determine end of bond detail section")
        else:
            print("Could not find bond orbital detail section")

    def get_NHO_directionality(self):
        # Extract NHO Directionality and Bond Bending
        # Find the directionality subsection
        dir_start = self.nbo_section.find("NHO DIRECTIONALITY AND BOND BENDING")
        dir_end = self.nbo_section.find("\n SECOND ORDER PERTURBATION", dir_start)
        if dir_end == -1:
            dir_end = self.nbo_section.find("\n\n NATURAL BOND ORBITALS", dir_start)

        nbo_directionality = []

        if dir_start != -1 and dir_end != -1:
            dir_section = self.nbo_section[dir_start:dir_end]
            
            # Check if there are entries exceeding thresholds
            if "None exceeding thresholds" in dir_section:
                print("NHO Directionality: No entries exceeding thresholds (all bonds are linear)")
            else:
                # Parse directionality entries if they exist
                # Match atoms as: one or more occurrences of (AtomSymbol  Number) with optional dash/space separator
                # "  32. BD ( 1) N  4- C  5   78.9   53.4    79.7   50.0   3.5   102.2  234.7   1.6"
                dir_lines = re.findall(r'\n\s+(\d+)\.\s+(BD|LP|CR)\s*\(\s*(\d+)\)\s+([A-Z][a-z]?\s+\d+(?:\s*-\s*[A-Z][a-z]?\s+\d+)*)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', dir_section)
                
                for line_data in dir_lines:
                    atoms_string = line_data[3].strip()
                    
                    # Parse atom numbers from atoms_string (e.g., "N  4- C  5" -> [4, 5] or "Ni  2" -> [2])
                    atom_numbers = []
                    atom_num_matches = re.findall(r'([A-Z][a-z]?)\s+(\d+)', atoms_string)
                    for atom_match in atom_num_matches:
                        atom_numbers.append(int(atom_match[1]))
                    
                    nbo_directionality.append({
                        'nbo_num': int(line_data[0]),
                        'nbo_type': line_data[1],
                        'bond_index': int(line_data[2]),
                        'atoms': atom_numbers,  # Now a list of atom numbers
                        'line_theta': float(line_data[4]) if line_data[4] != '--' else None,
                        'line_phi': float(line_data[5]) if line_data[5] != '--' else None,
                        'hybrid_theta': float(line_data[6]) if line_data[6] != '--' else None,
                        'hybrid_phi': float(line_data[7]) if line_data[7] != '--' else None,
                        'deviation': float(line_data[8]) if line_data[8] != '--' else None
                    })
                
            self.results['nho_directionality'] = nbo_directionality
        else:
            print("Could not find NHO Directionality section")

    def get_second_order_perturbation(self):
        # Extract Second-Order Perturbation Theory Analysis
        # Find the second-order perturbation subsection
        second_start = self.nbo_section.find("SECOND ORDER PERTURBATION THEORY ANALYSIS")
        second_end = self.nbo_section.find("\n\n NATURAL BOND ORBITALS (Summary):", second_start)
        if second_end == -1:
            second_end = self.nbo_section.find("\n $CHOOSE", second_start)

        second_order_perturbation = []

        if second_start != -1 and second_end != -1:
            second_section = self.nbo_section[second_start:second_end]
            
            # Check if there are any interactions above threshold
            if "None above threshold" in second_section or "within unit" not in second_section:
                print("Second-Order Perturbation: No interactions above threshold")
            else:
                # Parse donor-acceptor interactions
                # Pattern: "   1. BD ( 1) H  1- H  2   ->  BD*( 1) H  3- H  4      12.34     0.56     0.078"
                # Format: donor -> acceptor E(2) E(j)-E(i) F(i,j)
                interaction_pattern = r'\n\s+(BD\*?|LP|CR|RY)\s*\(\s*(\d+)\)\s+(.*?)\s+->\s+(BD\*?|LP\*?|CR\*?|RY)\s*\(\s*(\d+)\)\s+(.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
                interactions = re.finditer(interaction_pattern, second_section)
                
                for inter in interactions:
                    second_order_perturbation.append({
                        'donor_type': inter.group(1),
                        'donor_index': int(inter.group(2)),
                        'donor_atoms': inter.group(3).strip(),
                        'acceptor_type': inter.group(4),
                        'acceptor_index': int(inter.group(5)),
                        'acceptor_atoms': inter.group(6).strip(),
                        'e2_energy': float(inter.group(7)),  # E(2) stabilization energy in kcal/mol
                        'energy_diff': float(inter.group(8)),  # E(j) - E(i) in a.u.
                        'fock_element': float(inter.group(9))  # F(i,j) in a.u.
                    })

            self.results['second_order_perturbation'] = second_order_perturbation
                
        else:
            print("Could not find Second-Order Perturbation section")

    def get_cycle_info(self):
        # Extract NBO Analysis Cycle Information
        # Find the analysis cycle subsection
        cycle_start = self.nbo_section.find("NATURAL BOND ORBITAL ANALYSIS:")
        cycle_end = self.nbo_section.find("Structure accepted:", cycle_start)
        if cycle_end == -1:
            cycle_end = self.nbo_section.find("\n\n", cycle_start + 100)

        nbo_analysis_info = {}

        if cycle_start != -1 and cycle_end != -1:
            cycle_section = self.nbo_section[cycle_start:cycle_end + 200]
            
            # Parse the cycle line
            # Pattern: "    1     2     1.90     2.00000   0.00000      0   1   0   0     0      0"
            cycle_match = re.search(r'\n\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', cycle_section)
            
            if cycle_match:
                nbo_analysis_info = {
                    'cycle': int(cycle_match.group(1)),
                    'center': int(cycle_match.group(2)),
                    'threshold': float(cycle_match.group(3)),
                    'lewis_occupancy': float(cycle_match.group(4)),
                    'non_lewis_occupancy': float(cycle_match.group(5)),
                    'num_core': int(cycle_match.group(6)),
                    'num_bonds': int(cycle_match.group(7)),
                    'num_non_bonding_cores': int(cycle_match.group(8)),
                    'num_lone_pairs': int(cycle_match.group(9)),
                    'low_occ_lewis': int(cycle_match.group(10)),
                    'high_occ_non_lewis': int(cycle_match.group(11))
                }
                
                self.results['analysis_info'] = nbo_analysis_info
        else:
            print("Could not find NBO analysis section")

    def print_summary(self):
        # Summary of extracted data using self.results
        print("=" * 70)
        print("SUMMARY OF EXTRACTED NBO PROPERTIES")
        print("=" * 70)

        npa_atoms = self.results.get('natural_population_analysis', {}).get('atoms', {})
        npa_totals = self.results.get('natural_population_analysis', {}).get('totals', {})
        nao_props = self.results.get('natural_atomic_orbitals', [])
        nbo_props = self.results.get('natural_bond_orbitals', [])
        bond_details = self.results.get('bond_orbital_details', [])
        nho_dir = self.results.get('nho_directionality', [])
        sop = self.results.get('second_order_perturbation', [])
        analysis_info = self.results.get('analysis_info', {})

        # 1. Natural Population Analysis
        print(f"\n1. Natural Population Analysis:")
        print(f"   - Atoms: {len(npa_atoms)} atoms")
        print(f"   - Total charge: {npa_totals.get('charge', 'N/A')}")
        print(f"   - Total electrons: {npa_totals.get('total', 'N/A')}")

        # 2. Natural Atomic Orbitals (NAO)
        print(f"\n2. Natural Atomic Orbitals (NAO):")
        print(f"   - Total NAOs: {len(nao_props)}")

        # 3. Natural Bond Orbitals (NBO) Summary
        print(f"\n3. Natural Bond Orbitals (NBO) Summary:")
        print(f"   - Total NBOs: {len(nbo_props)}")
        if nbo_props:
            lewis_count = len([n for n in nbo_props if not str(n.get('nbo_type', '')).endswith('*')])
            non_lewis_count = len([n for n in nbo_props if str(n.get('nbo_type', '')).endswith('*')])
            print(f"   - Lewis orbitals: {lewis_count}")
            print(f"   - Non-Lewis orbitals: {non_lewis_count}")

        # 4. Detailed Bond Orbital Properties
        print(f"\n4. Detailed Bond Orbital Properties:")
        print(f"   - Bonds with hybridization details: {len(bond_details)}")

        # 5. NHO Directionality
        print(f"\n5. NHO Directionality:")
        print(f"   - Bonds with angular deviation: {len(nho_dir)}")

        # 6. Second-Order Perturbation
        print(f"\n6. Second-Order Perturbation:")
        print(f"   - Donor-acceptor interactions: {len(sop)}")

        # 7. NBO Analysis Info
        print(f"\n7. NBO Analysis Info:")
        if analysis_info:
            print(f"   - Analysis converged in cycle {analysis_info.get('cycle', 'N/A')}")
            print(f"   - Number of bonds identified: {analysis_info.get('num_bonds', 'N/A')}")

        print("\n" + "=" * 70)
        print("All data structures are ready for further analysis!")
        print("=" * 70)

    def write_results_to_file(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)

class PropertiesJOB:
    """Properties from JSON file from Orca NMR calculations"""

    def __init__(self, path):
        """this class reads an orca json file from a given path and extracts properties
        Args:
            path (Path): Path to the orca property.json file
        The class is initialized by reading the json file
        """
        
        self.path = path

        #find the *compound_detailed.txt file in the path
        compound_files = [f for f in os.listdir(path) if f.endswith('_compound_detailed.txt')]
        if compound_files:
            self.compound_file = compound_files[0]
            with open(os.path.join(path, self.compound_file), 'r') as f:
                self.compound_text = f.read()
        else:
            self.compound_file = None
            self.compound_text = None
            print("No compound_detailed.txt file found in the specified path.")

        self.results = {}

    def get_opt_theory(self):
        """Extract optimization method from compound_detailed.txt file"""
        if self.compound_text is None:
            print("Compound_detailed.txt file is not defined. Please ensure it exists in the specified path.")
            return

        # Find OPT_METHOD section and extract the value
        if "OPT_METHOD" in self.compound_text:
            # Split at OPT_METHOD and search for Value in the rest
            after_opt_method = self.compound_text.split("OPT_METHOD", 1)[1]
            if "Value" in after_opt_method:
                value_line = after_opt_method.split("Value", 1)[1].split('\n', 1)[0]
                opt_method = value_line.split(':', 1)[1].strip()
                self.results["opt_method"] = opt_method.lower()

    def get_svp_theory(self):
        """Extract svp theory from compound_detailed.txt file"""
        if self.compound_text is None:
            print("Compound_detailed.txt file is not defined. Please ensure it exists in the specified path.")
            return

        # Find SVP_METHOD section and extract the value
        if "SVP_METHOD" in self.compound_text:
            # Split at SVP_METHOD and search for Value in the rest
            after_svp_theory = self.compound_text.split("SVP_METHOD", 1)[1]
            if "Value" in after_svp_theory:
                value_line = after_svp_theory.split("Value", 1)[1].split('\n', 1)[0]
                svp_theory = value_line.split(':', 1)[1].strip()
                self.results["svp_theory"] = svp_theory.lower()

class RedoxSet:
    def __init__(self, neutralReader=None, cationReader=None, anionReader=None, descriptorPrefixes=None):
        self.neutralReader = neutralReader
        self.cationReader = cationReader
        self.anionReader = anionReader
        self.descriptorPrefixes = descriptorPrefixes or ["mulliken", "loewdin", "nbo_charge", "cm5", "chelpg", "hirshfeld"]
        self.results = {}
        # Adopt atom list and labels from neutral reader if available
        self.atom_labels_map = {}
        self.atom_list = None
        if self.neutralReader and getattr(self.neutralReader, 'atom_list', None):
            self.atom_list = list(self.neutralReader.atom_list)
            self.atom_labels_map = dict(getattr(self.neutralReader, 'atom_labels_map', {}))

    def _atom_key(self, prefix, atom_index):
        label = self.atom_labels_map.get(atom_index)
        if label:
            return f"{prefix}_{label}"
        return f"{prefix}_{atom_index}"

    def _pull_charge(self, reader, prefix, atom_index):
        if reader is None:
            return None
        # Attempt labeled key first
        labeled = self._atom_key(prefix, atom_index)
        if labeled in reader.results:
            return reader.results.get(labeled)
        # Fallback to numeric key
        return reader.results.get(f"{prefix}_{atom_index}")

    def get_redox_properties(self):
        self.results['Vertical_Ionization_Potential'] = self.neutralReader.results.get('SCF_energy') - self.cationReader.results.get('SCF_energy') if (self.neutralReader and self.cationReader and self.neutralReader.results.get('SCF_energy') is not None and self.cationReader.results.get('SCF_energy') is not None) else None
        self.results['Vertical_Electron_Affinity'] = self.anionReader.results.get('SCF_energy') - self.neutralReader.results.get('SCF_energy') if (self.neutralReader and self.anionReader and self.anionReader.results.get('SCF_energy') is not None and self.neutralReader.results.get('SCF_energy') is not None) else None
        ip = self.results['Vertical_Ionization_Potential']
        ea = self.results['Vertical_Electron_Affinity']
        if ip is not None and ea is not None:
            hardness = (ip - ea) / 2
            electronegativity = (ip + ea) / 2
            self.results['Hardness'] = hardness
            self.results['Electronegativity'] = electronegativity
            self.results['Electrophilicity'] = (electronegativity ** 2) / (2 * hardness) if hardness else None
        else:
            self.results['Hardness'] = None
            self.results['Electronegativity'] = None
            self.results['Electrophilicity'] = None

    def get_fukui_functions(self):
        if not self.atom_list:
            print("Atom list not available in RedoxSet.")
            return
        for prefix in self.descriptorPrefixes:
            for idx in self.atom_list:
                qN = self._pull_charge(self.neutralReader, prefix, idx)
                qC = self._pull_charge(self.cationReader, prefix, idx)
                qA = self._pull_charge(self.anionReader, prefix, idx)
                f_plus = (qN - qA) if (qN is not None and qA is not None) else None
                f_minus = (qC - qN) if (qC is not None and qN is not None) else None
                f_zero = (0.5 * (f_plus + f_minus)) if (f_plus is not None and f_minus is not None) else None
                self.results[self._atom_key(f"fplus_{prefix}", idx)] = f_plus
                self.results[self._atom_key(f"fminus_{prefix}", idx)] = f_minus
                self.results[self._atom_key(f"fzero_{prefix}", idx)] = f_zero

    def aggregate(self):
        self.get_redox_properties()
        self.get_fukui_functions()
        return self.results

class PropertiesXYZ:
    """Properties from XYZ file using Morfeus"""

    def __init__(self, path, overwrite_results=False):
        """this class reads an xyz file from a given path and extracts properties using Morfeus
        Args:
            path (str): path to the folder containing optimization.xyz
            overwrite_results (bool): whether to overwrite existing results in xyz.json or to load them if they exist. Default is False (load existing results if available).
        The class is initialized by reading the xyz file and storing the number of atoms, elements, and coordinates
        and reading a json file with results from DFT calculations.
        """
        if not path.endswith('/'):
            path = path + '/'
        self.path = path
        with open(path + 'optimization.xyz', 'r') as file:
            lines = file.readlines()
        self.n_atoms = int(lines[0].strip())
        self.elements = [line.split()[0] for line in lines[2:] if line.strip()]
        self.coordinates = [[float(x) for x in line.split()[1:4]] for line in lines[2:] if line.strip()]

        if 'xyz.json' in os.listdir(path) and not overwrite_results:
            print(f"Loading existing results from {path + 'xyz.json'}.")
            self.results = json.load(open(path + 'xyz.json'))
        else:
            print("No xyz.json file found in the specified path or overwrite_results is True. Results dictionary will be empty.")
            self.results = {}

        # add path to results for reference
        self.results['path'] = self.path

        # initialize lists for atoms, bonds, angles, dihedrals
        self.atom_list = None
        self.bond_list = None
        self.angle_list = None
        self.dihedral_list = None
        self.atom_connectivity = {}
        self.atom_neighbor_counts = {}
        self.atom_labels_map = {}
        self.atom_index_to_label = {}
        self.bond_labels_map = {}
        self.angle_labels_map = {}
        self.dihedral_labels_map = {}
        self.quadrant_labels_map = {}

        # default radius list for Vbur calculation
        self.radius_list = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        # default scan for dbstep Sterimol calculation
        self.sterimol_scan = None #  format required: str(), start:end:step. Example: "3:5:0.5"

        # default parameters for quadrant calculation
        self.quadrant_list = None
        self.quadrant_radius = 6.5
        self.quadrant_excluded_atoms = None

    def reset_results(self):
        """Reset the results dictionary to an empty dictionary -- and remove the xyz.json file if it exists."""
        self.results = {}
        json_path = self.path + 'xyz.json'
        if os.path.exists(json_path):
            os.remove(json_path)

    def _default_atom_label(self, at_idx):
        """Fallback label for an atom when no custom label is provided."""
        at_idx = int(at_idx)
        return f"{self.elements[at_idx - 1]}{at_idx}"

    def _get_atom_label(self, at_idx):
        """Return the custom atom label if available, else the default element/index label."""
        return self.atom_index_to_label.get(int(at_idx), self._default_atom_label(at_idx))

    def _normalize_atom_map(self, atom_list):
        """Accept either a dict[label, atom_index] or a legacy list of atom indices."""
        if isinstance(atom_list, dict):
            return {str(label): int(at_idx) for label, at_idx in atom_list.items()}
        if isinstance(atom_list, (list, tuple)):
            return {self._default_atom_label(at_idx): int(at_idx) for at_idx in atom_list}
        raise TypeError("atom_list must be a dict[label, atom_index] or a list/tuple of atom indices")

    def _normalize_group_map(self, items, expected_len, item_name):
        """Accept either a dict[label, index_sequence] or a legacy list of index sequences."""
        if isinstance(items, dict):
            iterable = items.items()
        elif isinstance(items, (list, tuple)):
            iterable = [
                ("_".join(self._get_atom_label(i) for i in entry), tuple(int(i) for i in entry))
                for entry in items
            ]
        else:
            raise TypeError(
                f"{item_name} must be a dict[label, indices] or a list/tuple of index sequences"
            )

        normalized = {}
        for label, indices in iterable:
            indices = tuple(int(i) for i in indices)
            if len(indices) != expected_len:
                raise ValueError(f"Each {item_name} entry must contain exactly {expected_len} atom indices.")
            normalized[str(label)] = indices
        return normalized

    def _normalize_quadrant_map(self, quadrant_list):
        """Accept either a dict[label, index_sequence] or a legacy list of index sequences for quadrant/octant analysis."""
        if isinstance(quadrant_list, dict):
            iterable = quadrant_list.items()
        elif isinstance(quadrant_list, (list, tuple)):
            iterable = [
                ("_".join(self._get_atom_label(i) for i in atom_group), tuple(int(i) for i in atom_group))
                for atom_group in quadrant_list
            ]
        else:
            raise TypeError(
                "quadrant_list must be a dict[label, indices] or a list/tuple of index sequences"
            )

        normalized = {}
        for label, indices in iterable:
            indices = tuple(int(i) for i in indices)
            if len(indices) < 4:
                raise ValueError("Each quadrant entry must contain at least 4 atom indices.")
            normalized[str(label)] = indices
        return normalized

    def _get_covalent_radius(self, element):
        """Return an approximate covalent radius in Å for connectivity estimation."""
        return COVALENT_RADII.get(str(element).capitalize(), 0.77)

    def compute_connectivity(self, scale_factor=1.25):
        """Estimate bonded neighbors from the XYZ geometry using covalent radii."""
        coords = np.array(self.coordinates, dtype=float)
        connectivity = {idx: [] for idx in range(1, self.n_atoms + 1)}

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                distance = np.linalg.norm(coords[i] - coords[j])
                cutoff = scale_factor * (
                    self._get_covalent_radius(self.elements[i]) + self._get_covalent_radius(self.elements[j])
                )
                if 0.1 < distance <= cutoff:
                    connectivity[i + 1].append(j + 1)
                    connectivity[j + 1].append(i + 1)

        self.atom_connectivity = connectivity
        self.atom_neighbor_counts = {idx: len(neighbors) for idx, neighbors in connectivity.items()}
        return connectivity

    def define_atom_list(self, atom_list):
        """Define atoms via dict[label, atom_index] or a legacy list of atom indices."""
        self.atom_labels_map = self._normalize_atom_map(atom_list)
        self.atom_list = list(self.atom_labels_map.values())
        self.atom_index_to_label = {idx: label for label, idx in self.atom_labels_map.items()}
        self.compute_connectivity()

    def define_bond_list(self, bond_list):
        """Define bonds via dict[label, (atom1, atom2)] or a legacy list of atom-index pairs."""
        self.bond_labels_map = self._normalize_group_map(bond_list, expected_len=2, item_name="bond")
        self.bond_list = list(self.bond_labels_map.values())

    def define_angle_list(self, angle_list):
        """Define angles via dict[label, (atom1, atom2, atom3)] or a legacy list of triplets."""
        self.angle_labels_map = self._normalize_group_map(angle_list, expected_len=3, item_name="angle")
        self.angle_list = list(self.angle_labels_map.values())
    
    def define_dihedral_list(self, dihedral_list):
        """Define dihedrals via dict[label, (atom1, atom2, atom3, atom4)] or a legacy list of quartets."""
        self.dihedral_labels_map = self._normalize_group_map(dihedral_list, expected_len=4, item_name="dihedral")
        self.dihedral_list = list(self.dihedral_labels_map.values())
        
    def define_quadrant_list(self, quadrant_list):
        """Define quadrant/octant analysis entries via dict[label, indices] or a legacy list of index sequences."""
        self.quadrant_labels_map = self._normalize_quadrant_map(quadrant_list)
        self.quadrant_list = list(self.quadrant_labels_map.values())

    ## 0 atom properties ##
    #-- SASA -- & -- Molecular Volume --#
    def get_SASA_and_Volume(self):
        """Calculate solvent accessible surface area (SASA) using Morfeus
        Results are stored in the results dictionary with key 'SASA'.
        """
        sasa = SASA(self.elements, self.coordinates) # calls Morfeus
        self.results['SASA'] = sasa.area
        self.results['SASA_Volume'] = sasa.volume

    ## 1 atom properties ##
    #-- Vbur calculation --#
    def get_Vbur(self):
        """
        Calculate buried volume for all atoms in atom_list at radius from 2.0 to 5.0 Å with a step of 0.5 Å.
        Result keys use the provided atom labels when available.
        """
        if self.atom_list is None:
            print("Atom list is not defined. Please define it using define_atom_list().")
            return

        atom_items = self.atom_labels_map.items() if self.atom_labels_map else [
            (self._default_atom_label(at_idx), at_idx) for at_idx in self.atom_list
        ]
        for atom_label, at_idx in atom_items:
            for radius in self.radius_list:
                self.get_Vbur_at_radii(at_idx, radius, atom_label=atom_label)
          
    def get_Vbur_at_radii(self, at_idx, radius=3.5, atom_label=None):
        """Calculate buried volume for a specific atom at a specific radius using Morfeus."""
        vbur = BuriedVolume(self.elements, self.coordinates, int(at_idx), include_hs=True, radius=radius)
        atom_label = atom_label or self._get_atom_label(at_idx)
        self.results[f'%Vbur_atom_{atom_label}_radius_{radius}'] = vbur.fraction_buried_volume * 100

    #-- Pyramidalization --#
    def get_pyramidalizations(self):
        """Calculate pyramidalization for all labeled atoms in `define_atom_list()`."""
        if self.atom_list is None:
            print("Atom list is not defined. Please define it using define_atom_list().")
            return

        atom_items = self.atom_labels_map.items() if self.atom_labels_map else [
            (self._default_atom_label(at_idx), at_idx) for at_idx in self.atom_list
        ]
        for atom_label, at_idx in atom_items:
            self.get_pyramidalization(at_idx, atom_label=atom_label)

    def get_pyramidalization(self, at_idx, atom_label=None):
        """Calculate pyramidalization for a single atom when it has exactly three neighbors."""
        if not self.atom_connectivity:
            self.compute_connectivity()

        atom_label = atom_label or self._get_atom_label(at_idx)
        neighbors = self.atom_connectivity.get(at_idx, [])
        neighbor_count = len(neighbors)
        if neighbor_count != 3:
            logging.info(
                "Skipping pyramidalization for %s: expected 3 neighbors, found %s (%s)",
                atom_label,
                neighbor_count,
                neighbors,
            )
            return

        pyramidalization = Pyramidalization(
            self.coordinates,
            at_idx,
            elements=self.elements,
            method="connectivity",
        )
        self.results[f'Pyr_{atom_label}'] = pyramidalization.P
        self.results[f'Pyr_angle_{atom_label}'] = pyramidalization.P_angle

    #-- Atomic SASA --#
    def get_atomic_SASA(self):
        """Calculate atomic solvent accessible surface area (SASA) for all labeled atoms using Morfeus."""
        if self.atom_list is None:
            print("Atom list is not defined. Please define it using define_atom_list().")
            return
        
        sasa = SASA(self.elements, self.coordinates)
        atomic_areas = sasa.atom_areas

        atom_items = self.atom_labels_map.items() if self.atom_labels_map else [
            (self._default_atom_label(at_idx), at_idx) for at_idx in self.atom_list
        ]
        for atom_label, at_idx in atom_items:
            self.results[f'SASA_{atom_label}'] = atomic_areas[at_idx]
    

    ## 2 atom properties ##
    #-- Bond Distances --#
    def get_bond_distances(self):
        """Calculate bond distances for all labeled entries in `define_bond_list()`."""
        if self.bond_list is None:
            print("Bond list is not defined. Please define it using define_bond_list().")
            return

        bond_items = self.bond_labels_map.items() if self.bond_labels_map else [
            ("_".join(self._get_atom_label(i) for i in bond_atoms), bond_atoms) for bond_atoms in self.bond_list
        ]
        for bond_label, bond_atoms in bond_items:
            self.get_bond_distance(bond_atoms, bond_label=bond_label)

    def get_bond_distance(self, bond_list, bond_label=None):
        """Calculate bond distance for a specific set of two atoms using Morfeus."""
        bond = Sterimol(self.elements, self.coordinates, bond_list[0], bond_list[1])
        bond_label = bond_label or '_'.join(self._get_atom_label(i) for i in bond_list)
        self.results[f'Bond_{bond_label}'] = bond.bond_length

    def get_Sterimol_parameters(self, morfeus=True):
        """Calculate Sterimol parameters for all labeled entries in `define_bond_list()`."""
        if self.bond_list is None:
            print("Bond list is not defined. Please define it using define_bond_list().")
            return

        bond_items = self.bond_labels_map.items() if self.bond_labels_map else [
            ("_".join(self._get_atom_label(i) for i in bond_atoms), bond_atoms) for bond_atoms in self.bond_list
        ]
        for bond_label, bond_atoms in bond_items:
            self.get_Sterimol_parameters_for_bond(bond_atoms, morfeus=morfeus, bond_label=bond_label)

    def get_Sterimol_parameters_for_bond(self, bond_list, morfeus=True, bond_label=None):
        """Calculate Sterimol parameters for a specific bond using dbstep (fallback) or Morfeus."""
        key_prefix = 'Sterimol_' + (bond_label or '_'.join(self._get_atom_label(i) for i in bond_list))

        if morfeus:
            bond = Sterimol(self.elements, self.coordinates, bond_list[0], bond_list[1])
            self.results[key_prefix + '_Bmin'] = bond.B_1_value
            self.results[key_prefix + '_Bmax'] = bond.B_5_value
            self.results[key_prefix + '_L'] = float(bond.L_value)

        elif self.sterimol_scan is not None:
            mol = db.dbstep(
                self.path + 'optimization.xyz',
                atom1=bond_list[0],
                atom2=bond_list[1],
                commandline=True,
                verbose=False,
                sterimol=True,
                scan=self.sterimol_scan,
            )

            start, end, step = self.sterimol_scan.split(':')
            values = []
            v = float(start)
            while v < float(end):
                values.append(v)
                v += float(step)

            for idx, scan_value in enumerate(values):
                scan_key_prefix = key_prefix + f'_{scan_value}Å'
                self.results[scan_key_prefix + '_Bmin'] = mol.sterimol[scan_value].Bmin[idx]
                self.results[scan_key_prefix + '_Bmax'] = mol.sterimol[scan_value].Bmax[idx]

            self.results[key_prefix + '_L'] = float(mol.L)

        else:
            mol = db.dbstep(
                self.path + 'optimization.xyz',
                atom1=bond_list[0],
                atom2=bond_list[1],
                commandline=True,
                verbose=False,
                sterimol=True,
            )
            self.results[key_prefix + '_Bmin'] = mol.Bmin
            self.results[key_prefix + '_Bmax'] = mol.Bmax
            self.results[key_prefix + '_L'] = float(mol.L)

    ## 3 atom properties ##
    #-- Angles --#
    def get_angles(self):
        """Calculate angles for all labeled entries in `define_angle_list()`."""
        if self.angle_list is None:
            print("Angle list is not defined. Please define it using define_angle_list().")
            return

        angle_items = self.angle_labels_map.items() if self.angle_labels_map else [
            ("_".join(self._get_atom_label(i) for i in angle_atoms), angle_atoms) for angle_atoms in self.angle_list
        ]
        for angle_label, angle_atoms in angle_items:
            self.get_angle(angle_atoms, angle_label=angle_label)

    def get_bite_angle(self, angle_list: list, angle_label=None):
        """Calculate bite angle for a specific set of three atoms using Morfeus."""
        angle = BiteAngle(self.coordinates, angle_list[0], angle_list[1], angle_list[2])
        angle_label = angle_label or '_'.join(self._get_atom_label(i) for i in angle_list)
        self.results[f'BiteAngle_{angle_label}'] = angle.angle
    
    def get_angle(self, angle_list: list, angle_label=None):
        """Calculate angle for a specific set of three atoms using 3D coordinates."""
        u = np.array(self.coordinates[angle_list[0]-1]) - np.array(self.coordinates[angle_list[1]-1])
        v = np.array(self.coordinates[angle_list[2]-1]) - np.array(self.coordinates[angle_list[1]-1])
        
        cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        angle_label = angle_label or '_'.join(self._get_atom_label(i) for i in angle_list)
        self.results[f'Angle_{angle_label}'] = float(angle_deg)

    ## 4 atom properties ##
    #-- Dihedrals --#
    def get_dihedrals(self):
        """Calculate dihedral angles for all labeled entries in `define_dihedral_list()`."""
        if self.dihedral_list is None:
            print("Dihedral list is not defined. Please define it using define_dihedral_list().")
            return

        dihedral_items = self.dihedral_labels_map.items() if self.dihedral_labels_map else [
            ("_".join(self._get_atom_label(i) for i in dihedral_atoms), dihedral_atoms)
            for dihedral_atoms in self.dihedral_list
        ]
        for dihedral_label, dihedral_atoms in dihedral_items:
            self.get_dihedral(dihedral_atoms, dihedral_label=dihedral_label)

    def get_dihedral(self, dihedral_list, dihedral_label=None):
        """Calculate dihedral angle for a specific set of four atoms using Morfeus."""

        b1 = np.array(self.coordinates[dihedral_list[1]-1]) - np.array(self.coordinates[dihedral_list[0]-1])
        b2 = np.array(self.coordinates[dihedral_list[2]-1]) - np.array(self.coordinates[dihedral_list[1]-1])
        b3 = np.array(self.coordinates[dihedral_list[3]-1]) - np.array(self.coordinates[dihedral_list[2]-1])
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        x = np.dot(n1, n2)
        y = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))
        
        dihedral = np.degrees(np.arctan2(y, x))
        dihedral_label = dihedral_label or '_'.join(self._get_atom_label(i) for i in dihedral_list)
        self.results[f'Dihed_{dihedral_label}'] = float(dihedral)

    
    ## 5 atom properties ##
    #-- quadrants --#
    def get_vbur_quadrants(self):
        """Calculate buried volume in quadrants for all labeled entries in `define_quadrant_list()`."""
        if self.quadrant_list is None:
            print("Quadrant list is not defined. Please define it using define_quadrant_list().")
            return

        quadrant_items = self.quadrant_labels_map.items() if self.quadrant_labels_map else [
            ("_".join(self._get_atom_label(i) for i in atom_list), atom_list) for atom_list in self.quadrant_list
        ]
        for quadrant_label, atom_list in quadrant_items:
            self.get_vbur_quadrants_only(atom_list, quadrant_label=quadrant_label)

    def get_vbur_quadrants_only(self, atom_list, quadrant_label=None):
        """Use Morfeus to calculate buried volume in quadrants for one labeled entry."""
        metal_center = atom_list[0]
        z_axis_atoms = [atom_list[1], atom_list[2]]
        xz_plane_atoms = atom_list[3:]
        quadrant_label = quadrant_label or '_'.join(self._get_atom_label(i) for i in atom_list)
        
        vbur = BuriedVolume(
            self.elements,
            self.coordinates,
            metal_center,
            radius=self.quadrant_radius,
            include_hs=True,
            excluded_atoms=self.quadrant_excluded_atoms,
            z_axis_atoms=z_axis_atoms,
            xz_plane_atoms=xz_plane_atoms,
        )
                
        vbur.octant_analysis()
                
        quadrants = vbur.quadrants
        bv_quadrants = quadrants['percent_buried_volume']

        self.results[f'%Vbur_{quadrant_label}_quadrant_+,+'] = bv_quadrants[1]
        self.results[f'%Vbur_{quadrant_label}_quadrant_–,+'] = bv_quadrants[2]
        self.results[f'%Vbur_{quadrant_label}_quadrant_–,–'] = bv_quadrants[3]
        self.results[f'%Vbur_{quadrant_label}_quadrant_+,–'] = bv_quadrants[4]

    #-- octants --#
    def get_vbur_octants(self):
        """Calculate buried volume in octants for all labeled entries in `define_quadrant_list()`."""
        if self.quadrant_list is None:
            print("Quadrant list is not defined. Please define it using define_quadrant_list().")
            return

        quadrant_items = self.quadrant_labels_map.items() if self.quadrant_labels_map else [
            ("_".join(self._get_atom_label(i) for i in atom_list), atom_list) for atom_list in self.quadrant_list
        ]
        for quadrant_label, atom_list in quadrant_items:
            self.get_vbur_octants_only(atom_list, quadrant_label=quadrant_label)

    def get_vbur_octants_only(self, atom_list, quadrant_label=None): 
        metal_center = atom_list[0]
        z_axis_atoms = [atom_list[1], atom_list[2]]
        xz_plane_atoms = atom_list[3:]
        quadrant_label = quadrant_label or '_'.join(self._get_atom_label(i) for i in atom_list)
        
        vbur = BuriedVolume(
            self.elements,
            self.coordinates,
            metal_center,
            radius=self.quadrant_radius,
            include_hs=True,
            excluded_atoms=self.quadrant_excluded_atoms,
            z_axis_atoms=z_axis_atoms,
            xz_plane_atoms=xz_plane_atoms,
        )
                
        vbur.octant_analysis()

        octants = vbur.octants

        bv_octants = octants['percent_buried_volume']
        self.results[f'%Vbur_{quadrant_label}_octant_+,+,+'] = bv_octants[0]
        self.results[f'%Vbur_{quadrant_label}_octant_–,+,+'] = bv_octants[1]
        self.results[f'%Vbur_{quadrant_label}_octant_–,–,+'] = bv_octants[2]
        self.results[f'%Vbur_{quadrant_label}_octant_+,–,+'] = bv_octants[3]
        self.results[f'%Vbur_{quadrant_label}_octant_+,–,–'] = bv_octants[4]
        self.results[f'%Vbur_{quadrant_label}_octant_–,–,–'] = bv_octants[5]
        self.results[f'%Vbur_{quadrant_label}_octant_–,+,–'] = bv_octants[6]
        self.results[f'%Vbur_{quadrant_label}_octant_+,+,–'] = bv_octants[7]
    


    ## Save results to json ##
    def save_results_to_json(self, filename='xyz.json'):
        """Save the results dictionary to a JSON file.
        Args:
            filename (str): name of the JSON file to save the results to (default is 'xyz.json')
        """
        sorted_results = dict(sorted(self.results.items()))
        self.results = sorted_results
        with open(self.path + filename, 'w') as json_file:
            json.dump(self.results, json_file, indent=4)