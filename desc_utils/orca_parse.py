"""Pure parsers for ORCA 6 outputs: property JSON, GBW JSON, NBO 7 text blocks, frequencies.

Standard library only (no numpy / pandas / morfeus), so this module also runs on a cluster login
node with a bare ``python`` module. ``desc_utils.condense`` builds the condensed per-conformer
file from these parsers; ``desc_utils.orca_descriptors`` re-exports the classes so the descriptor
pipeline keeps calling ``od.OrcaReader`` / ``od.NBOProps`` / ``od.RedoxSet``.

* ``OrcaReader``  ``<basename>.property.json`` / ``<basename>.json``: charges, SCF energy, HOMO/LUMO,
  NMR shieldings, thermochemistry. Reads ``orca_condensed.json`` instead when the folder has one.
* ``NBOProps``    one NBO block of ``calcParam.out``: NPA, NAO occupancies, NBO summary, hybrids,
  NHO directionality, second-order perturbation, cycle info.
* ``RedoxSet``    IP / EA, hardness, electronegativity, electrophilicity, Fukui functions.
* module functions (``read_xyz``, ``property_state``, ``mo_list``, ``frequencies_and_modes``,
  ``ir_intensities``, ``drop_zero_modes``, ``orca_version``, ``nbo_program``) used by the condenser.
* ``job_status``  completed / failed / running / not_started of one conformer folder, from the
  termination lines of ``calcParam.out`` and the SLURM ``error_<jobid>.txt``; recorded in the
  condensed file and shared with ``scripts-for-orca-computations/check_jobs.py``.
"""
from __future__ import annotations

import json
import math
import numbers
import os
import re
from pathlib import Path
from typing import Optional

h_to_eV = 27.2114
bohr_to_A = 0.529177

#: Default name of the condensed per-conformer file written by ``desc_utils.condense``.
CONDENSED_FILENAME = "orca_condensed.json"

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

ELEMENTS = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
    'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
)
SYMBOL_TO_Z = {symbol: z for z, symbol in enumerate(ELEMENTS, start=1)}


class OrcaReader:
    """ORCA results reader with unified atom index labeling.

    Atom handling:
    - atom_index_map: Dict[int, str] mapping 1-based atom index -> label string
    - set_atoms(list[int]) assigns numeric string labels automatically
    - set_atoms(dict[int,str]) assigns explicit labels
    - set_atom_labels(list[str]|dict[int,str]) updates labels

    Result keys follow: f"{prefix}_{label}".
    """
    def __init__(self, directory, basename="singlepoint", out_file_path=None, nbo_block_index=0, nbo_job_number=None,
                 condensed_file=CONDENSED_FILENAME):
        self.directory = Path(directory)
        # Name of the condensed per-conformer JSON (``desc_utils.condense``). When that file exists in
        # ``directory`` it is read instead of the raw ORCA files; ``None`` forces the raw files.
        self.condensed_file = condensed_file
        self.source = "raw"
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
            if all(isinstance(v, numbers.Integral) or str(v).isdigit() for v in atoms.values()):
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

    def set_nbo_block_index(self, index: int):
        prev_index = self.nbo_block_index
        self.nbo_block_index = index
        if self.nbo_sections:
            if 0 <= index < len(self.nbo_sections):
                if index != prev_index:
                    self.nbo_section = self.nbo_sections[index]
                    self._clear_nbo_data()
            else:
                print(f"Requested NBO block index {index} out of range (found {len(self.nbo_sections)} blocks).")

    def set_nbo_job_number(self, job_number: Optional[int] = None):
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
    def _condensed(self):
        """Parsed ``orca_condensed.json`` of this folder (see ``desc_utils.condense``), or None."""
        if not self.condensed_file:
            return None
        path = self.directory / self.condensed_file
        if not path.is_file():
            return None
        from desc_utils.condense import load_condensed  # local import: condense imports this module

        return load_condensed(path)

    def parse_property_json(self):
        cond = self._condensed()
        if cond is not None:
            from desc_utils.condense import property_json_view

            self.propertyjson = property_json_view(cond, self.basename)
            self.source = "condensed"
            if not self.propertyjson:
                print(f"No {self.basename} state in {self.condensed_file}. Results dictionary will be empty.")
            return
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
        cond = self._condensed()
        if cond is not None:
            from desc_utils.condense import gbw_view

            self.gbwjson = gbw_view(cond, self.basename)
            self.source = "condensed"
            return
        print(f"Parsing GBW JSON for {self.basename} properties...")
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
            hirshfeld_charges = [float(x[0]) for x in hir_raw]
            z = [int(x[0]) for x in atnos]
            xyz = [[float(x) * bohr_to_A, float(y) * bohr_to_A, float(zv) * bohr_to_A] for _, x, y, zv in cart]
        except Exception as e:
            print(f"Failed CM5 prep: {e}")
            return
        atomradius = [0.7] * 119
        alpha = 2.474

        # The CM5 pairwise correction is not implemented: _tij is 0 for every pair, so every
        # cm5_<label> equals hirshfeld_<label> exactly: the CM5 correction is not implemented.
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
                    rij = math.dist(xyz[i], xyz[j])
                    bij = math.exp(-alpha * (rij - atomradius[z[i]] - atomradius[z[j]]))
                except Exception:
                    continue
                s += _tij(z[i], z[j]) * bij
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
        """HOMO / LUMO energies (eV) from the orbital energies and occupations of the GBW JSON.

        HOMO = highest orbital with occupation 1 or 2, LUMO = lowest with occupation 0.
        """
        mos = self._nested_get(self.gbwjson, ["Molecule", "MolecularOrbitals", "MOs"])
        if mos is None:
            print("MO data not found; skipping HOMO/LUMO.")
            self.results["HOMO_energy"] = None
            self.results["LUMO_energy"] = None
            return
        pairs = []
        for mo in mos:
            try:
                energy, occupancy = mo.get("OrbitalEnergy"), mo.get("Occupancy")
            except Exception:
                continue
            if energy is not None and occupancy is not None:
                pairs.append((float(energy), float(occupancy)))
        if not pairs:
            print("Incomplete MO data; skipping HOMO/LUMO.")
            self.results["HOMO_energy"] = None
            self.results["LUMO_energy"] = None
            return
        occupied = [e for e, o in pairs if o in (1.0, 2.0)]
        virtual = [e for e, o in pairs if o == 0.0]
        self.results["HOMO_energy"] = max(occupied) * h_to_eV if occupied else None
        self.results["LUMO_energy"] = min(virtual) * h_to_eV if virtual else None

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
    """Properties from out file from Orca NBO calculations.

    Parameters
    ----------
    path : str or Path
        ORCA output holding one or more ``Now starting NBO`` ... ``returned from NBO`` blocks.
    block_index : int
        0-based index of the NBO block to read when the file has several (compound jobs).
    content : str, optional
        Text of ``path`` if the caller already read it.
    spin : {"alpha", "beta", "both"}
        Which spin set the orbital parsers return for an **open-shell** (UHF/UKS) block.

        NBO 7 prints two complete NBO analyses for an open-shell block -- bond orbital
        details, NHO directionality, second-order table and summary -- under
        ``*** Alpha spin orbitals ***`` and ``*** Beta spin orbitals ***`` banners, and the
        closing ``$CHOOSE`` / ``Checkpoint orbital`` lines only follow the beta set.
        ``get_NBO_summary``, ``get_orbital_details``, ``get_NHO_directionality`` and
        ``get_second_order_perturbation`` therefore work on the slice of the block that
        belongs to the requested spin, so each ``nbo_num`` appears once (the column
        flattener in ``desc_utils.data_extraction`` relies on that). Rows from an
        open-shell block carry a ``'spin'`` key (``'alpha'`` / ``'beta'``); rows from a
        closed-shell block do not.

        * ``"alpha"`` (default): the alpha set only. This is what the detail, NHO and
          second-order parsers have always returned for open-shell blocks, so existing
          descriptor columns keep their meaning.
        * ``"beta"``: the beta set only.
        * ``"both"``: alpha rows followed by beta rows. Every ``nbo_num`` then occurs twice
          and the flattener disambiguates the duplicates with ``_2`` column suffixes.

        Closed-shell blocks have a single set and ignore this option. The natural
        population analysis (``get_NPA`` / ``get_NAO``) always reads the spin-summed
        tables that precede the spin banners, whatever ``spin`` is.
    """

    SPIN_CHOICES = ("alpha", "beta", "both")
    _SPIN_BANNER_RE = re.compile(r"\*{5,}\s+(Alpha|Beta)\s+spin orbitals\s+\*{5,}")

    def __init__(self, path, block_index=0, spin="alpha", content=None):
        self.path = Path(path)
        # ``content`` lets a caller that parses several blocks of one file read it once.
        self.content = self.path.read_text() if content is None else content
        self.nbo_block_index = int(block_index) if block_index is not None else 0
        if spin not in self.SPIN_CHOICES:
            raise ValueError(f"spin must be one of {self.SPIN_CHOICES}, got {spin!r}")
        self.spin = spin

        nbo_start_marker = "Now starting NBO"
        nbo_end_marker = "*** returned from  NBO  program ***"
        self.nbo_blocks = []

        for m in re.finditer(re.escape(nbo_start_marker), self.content):
            start = m.start()
            end = self.content.find(nbo_end_marker, start)
            if end != -1:
                self.nbo_blocks.append(self.content[start:end + len(nbo_end_marker)])

        if not self.nbo_blocks:
            print("Could not find any complete NBO section in the file")
            self.nbo_section = None
            self.spin_sections = {}
        else:
            if not (0 <= self.nbo_block_index < len(self.nbo_blocks)):
                print(
                    f"Requested NBO block index {self.nbo_block_index} out of range "
                    f"(found {len(self.nbo_blocks)} blocks). Using block 0."
                )
                self.nbo_block_index = 0
            self.nbo_section = self.nbo_blocks[self.nbo_block_index]
            self.spin_sections = self._split_spin_sections(self.nbo_section)

        self.results = {
            'source_file': str(self.path),
            'analysis_type': 'NBO 7.0',
            # None for a closed-shell block, otherwise the `spin` option in force.
            'spin': None if self.is_closed_shell else self.spin,
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

    # ------------------------------------------------------------------ spin handling
    @property
    def is_closed_shell(self):
        """True when the selected block has a single (spin-summed) NBO set."""
        return None in self.spin_sections or not self.spin_sections

    @classmethod
    def _split_spin_sections(cls, section):
        """Map spin label -> text of that spin's NBO analysis.

        Closed-shell blocks give ``{None: section}``. Open-shell blocks give
        ``{"alpha": ..., "beta": ...}``: the alpha slice runs from the alpha banner to the
        beta banner, the beta slice from the beta banner to the end of the block.
        """
        banners = [(m.group(1).lower(), m.start()) for m in cls._SPIN_BANNER_RE.finditer(section)]
        if not banners:
            return {None: section}
        out = {}
        for i, (label, start) in enumerate(banners):
            end = banners[i + 1][1] if i + 1 < len(banners) else len(section)
            out[label] = section[start:end]
        return out

    def _orbital_sections(self):
        """``(spin_label, text)`` pairs the orbital parsers read, honouring ``self.spin``."""
        if None in self.spin_sections:
            return [(None, self.spin_sections[None])]
        wanted = ("alpha", "beta") if self.spin == "both" else (self.spin,)
        missing = [s for s in wanted if s not in self.spin_sections]
        if missing:
            print(f"NBO block has no {', '.join(missing)} spin set")
        return [(s, self.spin_sections[s]) for s in wanted if s in self.spin_sections]

    @staticmethod
    def _tag_spin(rows, spin_label):
        if spin_label is not None:
            for row in rows:
                row['spin'] = spin_label
        return rows

    @staticmethod
    def _find_end(text, start, markers):
        """Offset of the earliest of ``markers`` at or after ``start`` (``len(text)`` if none)."""
        positions = [pos for pos in (text.find(m, start) for m in markers) if pos != -1]
        return min(positions) if positions else len(text)

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

    # Anything that can follow one spin set's summary table. In an open-shell block the
    # beta analysis (starting with its NPA / "(Occupancy)" header) follows the alpha
    # summary directly and "$CHOOSE" only appears after the beta summary, so the slice
    # must also stop at the next NBO header.
    # Orbital types printed by NBO 3-7 in the summary / detail / directionality / second-order
    # tables: 2-centre bonds (BD, BD*), lone pairs (LP, LP*), cores (CR), Rydbergs (RY, RY*),
    # lone vacancies (LV; empty metal-centred orbitals such as "87. LV ( 1)Ni  2") and
    # 3-centre orbitals (3C, 3Cn, 3C*). Kept identical to ``_SOP_TYPE`` below on purpose.
    # Note: a two-letter element symbol abuts the closing parenthesis ("LP ( 1)Ni  2"), so the
    # patterns below must allow zero whitespace between ")" and the atom list.
    _NBO_TYPE = r"(BD\*?|LP\*?|CR|RY\*?|LV|3C\*?|3Cn)"
    _NBO_ATOMS = r"([A-Z][a-z]?\s+\d+(?:\s*-\s*[A-Z][a-z]?\s+\d+)*)"

    _SUMMARY_END_MARKERS = (
        "$CHOOSE", "Checkpoint orbital", "NATURAL BOND ORBITALS (Summary):",
        "(Occupancy)   Bond orbital / Coefficients / Hybrids", "spin orbitals",
    )
    # Occupancy is 0..2 (one leading digit) and energy is a signed decimal. Requiring both
    # keeps NHO-directionality rows such as "45. BD ( 1) H  1-Ni  2   97.1   53.4 ..." out.
    # "   87. LV ( 1)Ni  2             0.26397    -0.07374" (no space before "Ni")
    _SUMMARY_ROW_RE = re.compile(
        r'\n\s+(\d+)\.\s+' + _NBO_TYPE + r'\s*\(\s*(\d+)\)\s*' + _NBO_ATOMS
        + r'\s+(\d\.\d+)\s+(-?\d+\.\d+)'
    )

    def get_NBO_summary(self):
        """Parse "NATURAL BOND ORBITALS (Summary):" into ``results['natural_bond_orbitals']``.

        One dict per NBO row: ``nbo_num``, ``nbo_type``, ``bond_index``, ``atoms``,
        ``occupancy``, ``energy`` (plus ``spin`` for open-shell blocks, see the class
        docstring for which spin set(s) are returned).
        """
        header = "NATURAL BOND ORBITALS (Summary):"
        nbo_properties = []
        found = False
        for spin_label, section in self._orbital_sections():
            start = section.find(header)
            if start == -1:
                continue
            found = True
            end = self._find_end(section, start + len(header), self._SUMMARY_END_MARKERS)
            nbo_properties.extend(self._tag_spin(self._parse_summary_rows(section[start:end]), spin_label))

        if found:
            self.results['natural_bond_orbitals'] = nbo_properties
        else:
            print("Could not find NBO summary section")

    @classmethod
    def _parse_summary_rows(cls, summary_section):
        # Pattern: "    1. BD ( 1) H  1- H  2       2.00000    -0.43520"
        rows = []
        for line_data in cls._SUMMARY_ROW_RE.findall(summary_section):
            # Parse atom numbers from atoms_string (e.g., "H  1- H  2" -> [1, 2] or "H  1" -> [1])
            atom_numbers = [int(num) for _, num in re.findall(r'([A-Z][a-z]?)\s+(\d+)', line_data[3].strip())]
            rows.append({
                'nbo_num': int(line_data[0]),
                'nbo_type': line_data[1],
                'bond_index': int(line_data[2]),
                'atoms': atom_numbers,
                'occupancy': float(line_data[4]),
                'energy': float(line_data[5])
            })
        return rows

    _DETAIL_END_MARKERS = (
        "NHO DIRECTIONALITY", "SECOND ORDER", "\n\n NATURAL BOND ORBITALS (Summary)",
        "$CHOOSE", "Checkpoint orbital", "spin orbitals",
    )

    def get_orbital_details(self):
        """Parse the "(Occupancy)   Bond orbital / Coefficients / Hybrids" table
        (Lewis and non-Lewis) into ``results['bond_orbital_details']``.
        """
        bond_orbital_details = []
        found = False
        for spin_label, section in self._orbital_sections():
            rows = self._parse_orbital_details(section)
            if rows is None:
                continue
            found = True
            bond_orbital_details.extend(self._tag_spin(rows, spin_label))

        if found:
            self.results['bond_orbital_details'] = bond_orbital_details
        else:
            print("Could not find bond orbital detail section")

    @classmethod
    def _parse_orbital_details(cls, section):
        bond_detail_start = section.find("(Occupancy)   Bond orbital / Coefficients / Hybrids")
        lewis_start = section.find("---------------- Lewis", bond_detail_start)
        if bond_detail_start == -1 or lewis_start == -1:
            return None
        detail_end = cls._find_end(section, lewis_start, cls._DETAIL_END_MARKERS)
        bond_detail_section = section[lewis_start:detail_end]

        bond_orbital_details = []
        # Split into individual bond entries
        # Each bond starts with a pattern like "   1. (2.00000) BD ( 1) H  1- H  2"
        # or "  87. (0.26397) LV ( 1)Ni  2            s( 99.10%)p 0.00(  0.31%)d 0.01(  0.59%)"
        bond_entries = re.finditer(
            r'\n\s+(\d+)\.\s+\(([\d.]+)\)\s+' + cls._NBO_TYPE + r'\s*\(\s*(\d+)\)\s*(.*?)(?=\n\s+\d+\.\s+\(|\Z)',
            bond_detail_section, re.DOTALL)

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
            #      or: "               ( 39.78%)   0.6307*Ni  2 s( 48.78%)p 0.01(  0.41%)d 1.04( 50.81%)"
            # (two-letter elements abut the "*", so no whitespace may be required there)
            contrib_pattern = r'\(\s*([\d.]+)%\)\s+([-\d.]+)\*\s*([A-Z][a-z]?)\s+(\d+)\s+s\(\s*([\d.]+)%\)p\s*([\d.]+)\(\s*([\d.]+)%\)'
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
        return bond_orbital_details

    _NHO_END_MARKERS = (
        "\n SECOND ORDER PERTURBATION", "\n\n NATURAL BOND ORBITALS",
        "$CHOOSE", "Checkpoint orbital", "spin orbitals",
    )

    def get_NHO_directionality(self):
        """Parse "NHO DIRECTIONALITY AND BOND BENDING" into ``results['nho_directionality']``."""
        nbo_directionality = []
        found = False
        for spin_label, section in self._orbital_sections():
            dir_start = section.find("NHO DIRECTIONALITY AND BOND BENDING")
            if dir_start == -1:
                continue
            found = True
            dir_end = self._find_end(section, dir_start, self._NHO_END_MARKERS)
            dir_section = section[dir_start:dir_end]

            # Check if there are entries exceeding thresholds
            if "None exceeding thresholds" in dir_section:
                print("NHO Directionality: No entries exceeding thresholds (all bonds are linear)")
                continue
            nbo_directionality.extend(self._tag_spin(self._parse_NHO_rows(dir_section), spin_label))

        if found:
            self.results['nho_directionality'] = nbo_directionality
        else:
            print("Could not find NHO Directionality section")

    @classmethod
    def _parse_NHO_rows(cls, dir_section):
        # Match atoms as: one or more occurrences of (AtomSymbol  Number) with optional dash/space separator
        # "  32. BD ( 1) N  4- C  5   78.9   53.4    79.7   50.0   3.5   102.2  234.7   1.6"
        dir_lines = re.findall(
            r'\n\s+(\d+)\.\s+' + cls._NBO_TYPE + r'\s*\(\s*(\d+)\)\s*' + cls._NBO_ATOMS
            + r'\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)',
            dir_section)

        rows = []
        for line_data in dir_lines:
            atoms_string = line_data[3].strip()

            # Parse atom numbers from atoms_string (e.g., "N  4- C  5" -> [4, 5] or "Ni  2" -> [2])
            atom_numbers = []
            atom_num_matches = re.findall(r'([A-Z][a-z]?)\s+(\d+)', atoms_string)
            for atom_match in atom_num_matches:
                atom_numbers.append(int(atom_match[1]))

            rows.append({
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
        return rows

    # One donor/acceptor row of the second-order perturbation table. Two layouts are
    # accepted:
    #   NBO 7 (ORCA 6.x):  "  46. BD ( 1) H  1-Ni  2      88. BD*( 1)Ni  2- H  3     29.34    0.38   0.095"
    #   legacy / NBO 3-6:  "BD ( 1) H  1- H  2   ->  BD*( 1) H  3- H  4      12.34     0.56     0.078"
    # i.e. an optional NBO number before each orbital, and an optional "->" between them.
    _SOP_TYPE = _NBO_TYPE
    _SOP_ATOMS = r"([A-Z][a-z]?\s*\d+(?:\s*-\s*[A-Z][a-z]?\s*\d+)*)"
    _SOP_NUM = r"(-?\d+\.\d+)"
    _SOP_LINE_RE = re.compile(
        r"^\s*(?:(\d+)\.\s*)?" + _SOP_TYPE + r"\s*\(\s*(\d+)\)\s*" + _SOP_ATOMS
        + r"\s+(?:->\s+)?(?:(\d+)\.\s*)?" + _SOP_TYPE + r"\s*\(\s*(\d+)\)\s*" + _SOP_ATOMS
        + r"\s+" + _SOP_NUM + r"\s+" + _SOP_NUM + r"\s+" + _SOP_NUM + r"\s*$"
    )
    _SOP_UNIT_RE = re.compile(r"^\s*(?:within unit\s+(\d+)|from unit\s+(\d+)\s+to unit\s+(\d+))\s*$")
    _SOP_END_MARKERS = (
        "\n NATURAL BOND ORBITALS (Summary)", "\n $CHOOSE", "Checkpoint orbital",
        "(Occupancy)   Bond orbital / Coefficients / Hybrids", "spin orbitals",
    )

    def get_second_order_perturbation(self):
        """Parse "SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS".

        Fills ``self.results['second_order_perturbation']`` with one dict per donor ->
        acceptor row: ``donor_nbo_num``/``acceptor_nbo_num`` (NBO index in the summary
        table, ``None`` for legacy outputs without it), ``*_type``, ``*_index`` (the
        number in parentheses), ``*_atoms`` (e.g. ``"H  1-Ni  2"``), ``*_unit``
        (molecular unit, ``None`` if not printed), ``e2_energy`` (kcal/mol),
        ``energy_diff`` (E(j)-E(i), a.u.) and ``fock_element`` (F(i,j), a.u.).
        Open-shell rows also carry ``spin`` (see the class docstring).
        """
        second_order_perturbation = []
        found = False
        for spin_label, section in self._orbital_sections():
            second_start = section.find("SECOND ORDER PERTURBATION THEORY ANALYSIS")
            if second_start == -1:
                continue
            found = True
            second_end = self._find_end(section, second_start, self._SOP_END_MARKERS)
            second_section = section[second_start:second_end]
            rows = self._parse_second_order_rows(second_section)
            if not rows and "None above threshold" not in second_section:
                print("Second-Order Perturbation: section found but no donor/acceptor rows could be parsed")
            second_order_perturbation.extend(self._tag_spin(rows, spin_label))

        if not found:
            print("Could not find Second-Order Perturbation section")
            return
        self.results['second_order_perturbation'] = second_order_perturbation

    @classmethod
    def _parse_second_order_rows(cls, second_section):
        rows = []
        donor_unit = acceptor_unit = None
        for line in second_section.splitlines():
            unit_match = cls._SOP_UNIT_RE.match(line)
            if unit_match:
                if unit_match.group(1) is not None:
                    donor_unit = acceptor_unit = int(unit_match.group(1))
                else:
                    donor_unit, acceptor_unit = int(unit_match.group(2)), int(unit_match.group(3))
                continue

            m = cls._SOP_LINE_RE.match(line)
            if not m:
                continue
            g = m.groups()
            rows.append({
                'donor_nbo_num': int(g[0]) if g[0] is not None else None,
                'donor_type': g[1],
                'donor_index': int(g[2]),
                'donor_atoms': g[3].strip(),
                'donor_unit': donor_unit,
                'acceptor_nbo_num': int(g[4]) if g[4] is not None else None,
                'acceptor_type': g[5],
                'acceptor_index': int(g[6]),
                'acceptor_atoms': g[7].strip(),
                'acceptor_unit': acceptor_unit,
                'e2_energy': float(g[8]),      # E(2) stabilization energy in kcal/mol
                'energy_diff': float(g[9]),    # E(j) - E(i) in a.u.
                'fock_element': float(g[10]),  # F(i,j) in a.u.
            })
        return rows

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


class RedoxSet:
    def __init__(self, neutralReader=None, cationReader=None, anionReader=None, descriptorPrefixes=["cm5"]):
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
        label = self.atom_labels_map.get(atom_index, atom_index)
        pref_raw = str(prefix)
        pref_norm = pref_raw.strip().lower()

        candidate_keys = [f"{pref_raw}_{label}", f"{pref_raw}_{atom_index}"]

        # Normalize common spelling/casing variants.
        if pref_norm == "nbo_charges":
            candidate_keys.extend([f"nbo_charge_{label}", f"nbo_charge_{atom_index}"])
        if pref_norm == "chelpg":
            candidate_keys.extend([f"chelpG_{label}", f"chelpG_{atom_index}"])

        # NBO natural charges are often stored as npa_<atom>_charge.
        if pref_norm in {"nbo_charge", "nbo_charges", "npa"}:
            candidate_keys.extend([
                f"npa_{label}_charge",
                f"npa_{atom_index}_charge",
                f"nbo_charge_{label}",
                f"nbo_charge_{atom_index}",
            ])

        seen = set()
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            if key in reader.results:
                return reader.results.get(key)

        return None

    def get_redox_properties(self):
        # IP = E(cation) - E(neutral), EA = E(neutral) - E(anion): positive for bound electrons (Hartree).
        self.results['Vertical_Ionization_Potential'] = self.cationReader.results.get('SCF_energy') - self.neutralReader.results.get('SCF_energy') if (self.neutralReader and self.cationReader and self.neutralReader.results.get('SCF_energy') is not None and self.cationReader.results.get('SCF_energy') is not None) else None
        self.results['Vertical_Electron_Affinity'] = self.neutralReader.results.get('SCF_energy') - self.anionReader.results.get('SCF_energy') if (self.neutralReader and self.anionReader and self.anionReader.results.get('SCF_energy') is not None and self.neutralReader.results.get('SCF_energy') is not None) else None
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


# --------------------------------------------------------------------------- file-level helpers
# Used by ``desc_utils.condense`` on the cluster. Every function returns plain lists / dicts
# that ``json.dump`` can write, and none of them imports anything outside the standard library.

def _nested(root, path):
    cur = root
    for p in path:
        try:
            cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def _flat(values):
    """ORCA column vectors ``[[v], [v], ...]`` or plain ``[v, v, ...]`` -> ``[v, v, ...]``; None stays None."""
    if values is None:
        return None
    return [v[0] if isinstance(v, list) and len(v) == 1 else v for v in values]


def read_xyz(path):
    """``(symbols, coordinates)`` of an xyz file, coordinates in Å. Only the first frame is read."""
    with open(path, "r") as fh:
        lines = fh.readlines()
    n_atoms = int(lines[0].strip())
    rows = [ln.split() for ln in lines[2:2 + n_atoms] if ln.strip()]
    symbols = [r[0] for r in rows]
    coords = [[float(x) for x in r[1:4]] for r in rows]
    if len(symbols) != n_atoms:
        raise ValueError(f"{path}: header announces {n_atoms} atoms, found {len(symbols)}")
    return symbols, coords


def load_property_json(conf_dir, basename):
    """Parsed ``<conf_dir>/<basename>.property.json`` or None when the file does not exist."""
    p = Path(conf_dir) / f"{basename}.property.json"
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


_POPULATION_KEYS = {
    "mulliken": "Mulliken_Population_Analysis",
    "loewdin": "Loewdin_Population_Analysis",
    "chelpg": "CHELPG_Population_Analysis",
    "hirshfeld": "Hirshfeld_Population_Analysis",
}


def polar_anisotropy(eigenvalues):
    """sqrt(1/2 * sum_(i<j) (a_i - a_j)^2) of the three polarizability eigenvalues, or None."""
    if not eigenvalues or len(eigenvalues) != 3 or any(e is None for e in eigenvalues):
        return None
    a, b, c = eigenvalues
    return math.sqrt(0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2))


def property_state(conf_dir, basename):
    """Condensed content of ``<basename>.property.json`` (last geometry), or None if absent.

    Keys: ``scf_energy_eh``, electron counts, ``charge`` / ``multiplicity`` (derived from the
    atomic numbers and electron counts), ``atomic_numbers``, ``charges`` (flat per-atom lists per
    scheme, None when ORCA did not print that scheme), ``dipole_au``, ``dipole_magnitude_au``,
    ``polarizability``, ``thermochemistry``, ``nmr`` and a ``mo`` slot filled by the caller.
    The geometry is not copied: every state of a compound job shares ``optimization.xyz``.
    """
    data = load_property_json(conf_dir, basename)
    if data is None:
        return None
    geoms = data.get("Geometries") or []
    G = geoms[-1] if geoms else {}
    dft = G.get("DFT_Energy") or {}
    cart = _nested(G, ["Geometry", "Coordinates", "Cartesians"]) or []
    symbols = [row[0] for row in cart]
    atno = _flat(_nested(G, ["Hirshfeld_Population_Analysis", 0, "ATNO"]))
    if atno is None and symbols:
        atno = [SYMBOL_TO_Z.get(s) for s in symbols]
    n_alpha, n_beta, n_total = dft.get("nAlphaEl"), dft.get("nBetaEl"), dft.get("nTotalEl")
    charge = None
    if atno and all(z is not None for z in atno) and n_total is not None:
        charge = int(round(sum(atno) - n_total))
    multiplicity = int(n_alpha - n_beta + 1) if n_alpha is not None and n_beta is not None else None
    dip = _nested(G, ["Dipole_Moment", 0]) or {}
    pol = _nested(G, ["Polarizability", 0]) or {}
    thermo = _nested(G, ["THERMOCHEMISTRY_Energies", 0])
    shifts = _nested(G, ["Chemical_Shift", 0])
    state = {
        "basename": basename,
        "n_geometries": len(geoms),
        "n_atoms": _nested(G, ["Geometry", "NAtoms"]) or len(symbols),
        "charge": charge,
        "multiplicity": multiplicity,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "n_electrons": n_total,
        "scf_energy_eh": dft.get("finalEn"),
        "atomic_numbers": atno,
        "charges": {k: _flat(_nested(G, [name, 0, "AtomicCharges"])) for k, name in _POPULATION_KEYS.items()},
        "dipole_au": _flat(dip.get("dipoleTotal")) if dip else None,
        "dipole_magnitude_au": dip.get("dipoleMagnitude") if dip else None,
        "polarizability": None,
        "thermochemistry": None,
        "nmr": None,
        "mo": None,
    }
    if pol:
        eig = _flat(pol.get("diagonalizedTensor"))
        state["polarizability"] = {"iso": pol.get("isotropicPolar"), "aniso": polar_anisotropy(eig),
                                   "eigenvalues": eig, "units": "au"}
    if thermo:
        state["thermochemistry"] = {"enthalpy_eh": thermo.get("enthalpyH"), "gibbs_eh": thermo.get("freeEnergyG"),
                                    "zpe_eh": thermo.get("zpe"), "temperature_K": thermo.get("temperature")}
    if shifts:
        state["nmr"] = {"nuc": shifts.get("NUC"), "elems": shifts.get("Elems"),
                        "siso_ppm": _flat(shifts.get("siso")), "saniso_ppm": _flat(shifts.get("saniso"))}
    return state


def mo_list(conf_dir, basename):
    """Orbital energies (Hartree) and occupations from the GBW dump ``<basename>.json``, or None.

    The dump is tens of MB because of the MO coefficients; only the two per-orbital scalars that
    ``OrcaReader.get_homolumo`` needs are kept.
    """
    p = Path(conf_dir) / f"{basename}.json"
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    mol = data.get("Molecule") or {}
    mos = _nested(mol, ["MolecularOrbitals", "MOs"])
    if mos is None:
        return None
    return {
        "energies_eh": [mo.get("OrbitalEnergy") for mo in mos],
        "occupations": [mo.get("Occupancy") for mo in mos],
        "energy_unit": _nested(mol, ["MolecularOrbitals", "EnergyUnit"]),
        "charge": mol.get("Charge"),
        "multiplicity": mol.get("Multiplicity"),
        "hf_type": mol.get("HFTyp"),
    }


_FREQ_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1")
_IR_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(\S+)\s+(-?\d+\.\d+)")


def parse_text_frequencies(text):
    """Frequencies (cm^-1, index = ORCA mode number) of the last ``VIBRATIONAL FREQUENCIES`` table."""
    start = text.rfind("VIBRATIONAL FREQUENCIES")
    if start == -1:
        return None
    end = text.find("NORMAL MODES", start)
    found = {}
    for line in text[start:end if end != -1 else len(text)].splitlines():
        m = _FREQ_ROW.match(line)
        if m:
            found[int(m.group(1))] = float(m.group(2))
    if not found:
        return None
    return [found.get(k) for k in range(max(found) + 1)]


def parse_text_normal_modes(text):
    """``modes[k]`` = the 3N Cartesian displacements of mode k from the last ``NORMAL MODES`` table.

    ORCA prints the 3N x 3N matrix in chunks of six columns: a header line of mode indices, then
    one row per Cartesian coordinate (``3*atom + xyz``). Mode k is column k, as in
    ``Hessian.MODES`` of ``frequency.property.json``.
    """
    start = text.rfind("NORMAL MODES")
    if start == -1:
        return None
    end = text.find("IR SPECTRUM", start)
    columns = {}
    current = None
    for line in text[start:end if end != -1 else len(text)].splitlines():
        parts = line.split()
        if not parts:
            continue
        if all(p.isdigit() for p in parts):
            current = [int(p) for p in parts]
            for k in current:
                columns.setdefault(k, {})
            continue
        if current is None or not parts[0].isdigit() or len(parts) != len(current) + 1:
            continue
        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        i = int(parts[0])
        for k, v in zip(current, values):
            columns[k][i] = v
    if not columns:
        return None
    n_coord = max(len(c) for c in columns.values())
    modes = []
    for k in sorted(columns):
        col = columns[k]
        if len(col) != n_coord:
            raise ValueError(f"NORMAL MODES: mode {k} has {len(col)} of {n_coord} coordinates")
        modes.append([col[i] for i in range(n_coord)])
    return modes


def _unit(vec):
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0.0 else list(vec)


def frequencies_and_modes(conf_dir, calcparam_text=None):
    """``(frequencies_cm1, modes, source)``; ``modes[k]`` holds the 3N displacements of mode k.

    ``frequency.property.json`` is preferred (full precision; ``Hessian.MODES`` is column-major,
    mode k = column k, and its columns are the un-normalised mass-weighted vectors, so each one is
    scaled to unit length here). Without it the ``VIBRATIONAL FREQUENCIES`` / ``NORMAL MODES``
    tables of ``calcParam.out`` are parsed; ORCA prints those already normalised (six decimals),
    so both sources agree to ~1e-6. ``(None, None, None)`` when neither source is available.
    """
    data = load_property_json(conf_dir, "frequency")
    if data is not None:
        geoms = data.get("Geometries") or []
        G = geoms[-1] if geoms else {}
        freqs = _flat(_nested(G, ["THERMOCHEMISTRY_Energies", 0, "FREQ"]))
        M = _nested(G, ["Hessian", "MODES"])
        if freqs and M:
            modes = [_unit([M[i][k] for i in range(len(M))]) for k in range(len(M[0]))]
            if len(freqs) == len(modes):
                return [float(f) for f in freqs], modes, "frequency.property.json"
    if calcparam_text is None:
        out = Path(conf_dir) / "calcParam.out"
        if out.is_file():
            calcparam_text = out.read_text(errors="replace")
    if calcparam_text:
        freqs = parse_text_frequencies(calcparam_text)
        modes = parse_text_normal_modes(calcparam_text)
        if freqs and modes and len(freqs) == len(modes):
            return freqs, modes, "calcParam.out"
    return None, None, None


def ir_intensities(text):
    """``{mode_index: intensity_km_per_mol}`` from the last ``IR SPECTRUM`` table ({} if absent)."""
    start = text.rfind("IR SPECTRUM")
    if start == -1:
        return {}
    out = {}
    for line in text[start:].splitlines()[1:]:
        m = _IR_ROW.match(line)
        if m:
            out[int(m.group(1))] = float(m.group(4))
            continue
        if out and line.strip().startswith("-"):
            break
    return out


def drop_zero_modes(freqs, modes):
    """Indices of the modes to keep: all but the translations/rotations (zero frequency, zero vector)."""
    keep = []
    for k, (f, vec) in enumerate(zip(freqs, modes)):
        if f == 0.0 and all(v == 0.0 for v in vec):
            continue
        keep.append(k)
    return keep


def orca_version(text):
    m = re.search(r"Program Version\s+(\S+)", text)
    return m.group(1) if m else None


def nbo_program(text):
    m = re.search(r"\*{3,}\s+NBO\s+([\d.]+)\s+\*{3,}", text)
    return f"NBO {m.group(1)}" if m else None


# --------------------------------------------------------------------------- job status

#: the four states of an ORCA job as seen from its folder
JOB_COMPLETED, JOB_FAILED, JOB_RUNNING, JOB_NOT_STARTED = "completed", "failed", "running", "not_started"
JOB_STATUSES = (JOB_COMPLETED, JOB_FAILED, JOB_RUNNING, JOB_NOT_STARTED)
ORCA_NORMAL = "ORCA TERMINATED NORMALLY"
ORCA_ERROR_RE = re.compile(r"ORCA finished by error termination.*|aborting the run", re.IGNORECASE)
#: messages SLURM / Open MPI write to the job's error_<jobid>.txt when the job dies before ORCA can say so
SLURM_ERRORS = {
    "DUE TO TIME LIMIT": "killed by SLURM: time limit",
    "oom-kill": "killed by SLURM: out of memory",
    "Out Of Memory": "killed by SLURM: out of memory",
    "CANCELLED AT": "cancelled (scancel or node failure)",
    "not enough slots": "Open MPI: not enough slots for the requested processes",
}
_ERROR_FILE_RE = re.compile(r"error_(\d+)\.txt")


def latest_slurm_error_file(conf_dir):
    """``error_<jobid>.txt`` of the most recent submission, or None (earlier ones stay after a resubmit)."""
    files = []
    for p in Path(conf_dir).glob("error_*.txt"):
        m = _ERROR_FILE_RE.fullmatch(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return max(files)[1] if files else None


def slurm_kill_reason(conf_dir):
    """Why SLURM killed the last job of this folder, or None when its error file says nothing of the kind."""
    err = latest_slurm_error_file(conf_dir)
    if err is None:
        return None
    text = err.read_text(errors="replace")
    for marker, reason in SLURM_ERRORS.items():
        if marker in text:
            return reason
    return None


def job_status(conf_dir, text=None, output_name="calcParam.out"):
    """State of the ORCA job of one conformer folder, judged from its files alone.

    Returns ``{"status", "detail", "n_normal_terminations", "failed_after_success"}`` with ``status`` one of
    ``completed`` (the last termination line is ``ORCA TERMINATED NORMALLY``), ``failed`` (an ORCA error
    termination, or a SLURM kill recorded in ``error_<jobid>.txt``), ``running`` (``calcParam.out`` exists
    but has no termination line yet: the job is still writing it, or it was killed without a message) and
    ``not_started`` (no ``calcParam.out``). ``calcParam.out`` is appended to on every submission, so a file
    can hold several runs: the verdict follows the last one, ``n_normal_terminations`` and
    ``failed_after_success`` tell when there were several. ``text`` is the content of ``calcParam.out``
    when the caller already read it.
    """
    conf_dir = Path(conf_dir)
    out = conf_dir / output_name
    result = {"status": JOB_NOT_STARTED, "detail": f"no {output_name}", "n_normal_terminations": 0,
              "failed_after_success": False}
    if text is None:
        if not out.is_file():
            return result
        text = out.read_text(errors="replace")

    last_normal = last_error = -1
    n_normal = 0
    orca_error = None
    for i, line in enumerate(text.splitlines()):
        if ORCA_NORMAL in line:
            last_normal, n_normal = i, n_normal + 1
        else:
            m = ORCA_ERROR_RE.search(line)
            if m:
                if orca_error is None or last_error < last_normal:  # keep the first, most specific, line of a run
                    orca_error = m.group(0).strip()
                last_error = i
    result["n_normal_terminations"] = n_normal

    if last_normal > last_error:
        detail = "" if n_normal == 1 else f"{n_normal} normal terminations in {output_name} (submitted several times?)"
        result.update(status=JOB_COMPLETED, detail=detail)
        return result
    kill = slurm_kill_reason(conf_dir)
    if last_error >= 0:
        result.update(status=JOB_FAILED, detail=kill or orca_error, failed_after_success=last_normal >= 0)
        return result
    if kill:
        result.update(status=JOB_FAILED, detail=kill)
        return result
    result.update(status=JOB_RUNNING,
                  detail=f"no termination line in {output_name}: still running, or killed without a message")
    return result

# --------------------------------------------------------------------------- run times
COMPOUND_JOB_RE = re.compile(r"^\s*COMPOUND JOB\s+(MainJOB|\d+)\s*$", re.MULTILINE)
_BASE_RE = re.compile(r'%base\s+"([^"]+)"')
_NPROCS_RE = re.compile(r"\bnprocs\s+(\d+)", re.IGNORECASE)
MODULE_TIMINGS_HEADER = "Timings for individual modules:"
_MODULE_LINE_RE = re.compile(r"^\s*(?P<name>[A-Za-z][^.]*?)\s*\.{3,}\s*(?P<sec>\d+(?:\.\d+)?)\s*sec\b")
_TOTAL_RUN_TIME_RE = re.compile(r"TOTAL RUN TIME:\s*(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec")


def run_time_seconds(text) -> Optional[float]:
    """ORCA's ``TOTAL RUN TIME`` line (the last one in ``text``) in seconds; None when there is none."""
    matches = _TOTAL_RUN_TIME_RE.findall(text)
    if not matches:
        return None
    d, h, m, sec, ms = (int(x) for x in matches[-1])
    return round(((d * 24 + h) * 60 + m) * 60 + sec + ms / 1000, 3)


def _module_timings(segment) -> Optional[dict]:
    """The ``Timings for individual modules`` block of one ORCA run: ``{"wall_s", "modules"}``, None if absent."""
    start = segment.rfind(MODULE_TIMINGS_HEADER)
    if start < 0:
        return None
    wall, modules = None, {}
    for line in segment[start + len(MODULE_TIMINGS_HEADER):].splitlines():
        if not line.strip():
            if modules or wall is not None:
                break
            continue
        m = _MODULE_LINE_RE.match(line)
        if not m:
            break
        name, sec = m.group("name").strip(), float(m.group("sec"))
        if name == "Sum of individual times":
            wall = sec
        else:
            modules[name] = sec
    return {"wall_s": wall, "modules": modules}


def compound_step_timings(text) -> dict:
    """How long each step of a ``calcParam.out`` took, from ORCA's own timing blocks.

    Returns ``{"total_run_time_s", "nprocs", "steps": {<%base name>: {"index", "wall_s", "modules"}}}``.
    Each ``COMPOUND JOB N`` block ends with a ``Timings for individual modules`` table whose
    ``Sum of individual times`` is the step's wall time (``wall_s``; None for a step that did not get
    there) and whose other lines (``SCF iterations``, ``SCF Gradient evaluation``, ...) are kept under
    ``modules``. Steps are keyed by the ``%base`` name echoed in the step's input (``optimization``,
    ``frequency``, ``singlepoint``, ``nmr``, ``oxidation``, ``reduction`` in the library template), a
    repeated name gets ``_2``, ``_3``...; a step without a ``%base`` is ``step_N``. ``nprocs`` is the
    ``%pal`` setting echoed in the main input, so ``total_run_time_s x nprocs`` is the job's core-seconds.
    ``calcParam.out`` is appended to on resubmission: only the last run (from its last ``COMPOUND JOB
    MainJOB`` header) is read. A plain, non-compound output gives one step named ``main``.
    """
    last_main = text.rfind("COMPOUND JOB MainJOB")
    if last_main >= 0:
        text = text[max(0, text.rfind("\n", 0, last_main)):]
    out = {"total_run_time_s": run_time_seconds(text), "nprocs": None, "steps": {}}
    m = _NPROCS_RE.search(text)
    if m:
        out["nprocs"] = int(m.group(1))
    headers = list(COMPOUND_JOB_RE.finditer(text))
    if not headers:
        block = _module_timings(text)
        if block is not None:
            out["steps"]["main"] = {"index": 0, **block}
        return out
    for i, h in enumerate(headers):
        if h.group(1) == "MainJOB":
            continue
        segment = text[h.end(): headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        index = int(h.group(1))
        base_m = _BASE_RE.search(segment)
        name = base_m.group(1) if base_m else f"step_{index}"
        key, n = name, 1
        while key in out["steps"]:
            n += 1
            key = f"{name}_{n}"
        out["steps"][key] = {"index": index, **(_module_timings(segment) or {"wall_s": None, "modules": {}})}
    return out
