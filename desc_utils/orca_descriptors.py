from pathlib import Path
import re
import numpy as np
import os
import json

from morfeus import Sterimol, BuriedVolume, Pyramidalization, SASA, BiteAngle

import dbstep.Dbstep as db

import logging
logging.basicConfig(level=logging.WARNING) 

# The pure parsers live in ``desc_utils.orca_parse`` (standard library only, usable on the cluster)
# and are re-exported here so ``od.OrcaReader`` / ``od.NBOProps`` / ``od.RedoxSet`` keep working.
from desc_utils.orca_parse import (  # noqa: F401,E402
    COVALENT_RADII, CONDENSED_FILENAME, NBOProps, OrcaReader, RedoxSet, bohr_to_A, h_to_eV,
)


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

class PropertiesXYZ:
    """Properties from XYZ file using Morfeus"""

    DEFAULT_GEOMETRY_FILE = 'optimization.xyz'

    @staticmethod
    def cache_name(geometry_file=DEFAULT_GEOMETRY_FILE):
        """Name of the morfeus cache next to the geometry: ``xyz.json`` for ``optimization.xyz`` (legacy),
        ``xyz_<stem>.json`` for any other geometry file, so that two geometries of the same conformer folder
        (the ORCA one and e.g. ``optimized_uma-s-1p1.xyz``) never share a cache."""
        if geometry_file == PropertiesXYZ.DEFAULT_GEOMETRY_FILE:
            return 'xyz.json'
        return f"xyz_{os.path.splitext(os.path.basename(geometry_file))[0]}.json"

    def __init__(self, path, overwrite_results=False, geometry_file=DEFAULT_GEOMETRY_FILE):
        """this class reads an xyz file from a given path and extracts properties using Morfeus
        Args:
            path (str): path to the folder containing the geometry file
            overwrite_results (bool): whether to overwrite existing results in the cache file or to load them if they exist. Default is False (load existing results if available).
            geometry_file (str): name of the xyz file in ``path`` (default ``optimization.xyz``, the ORCA geometry;
                any single-frame xyz works, extra columns after x y z such as the forces of an extended xyz are ignored).
        The class is initialized by reading the xyz file and storing the number of atoms, elements, and coordinates
        and reading a json file with results from DFT calculations.
        """
        if not path.endswith('/'):
            path = path + '/'
        self.path = path
        self.geometry_file = geometry_file
        self.xyz_path = path + geometry_file
        self.cache_file = self.cache_name(geometry_file)
        with open(self.xyz_path, 'r') as file:
            lines = file.readlines()
        self.n_atoms = int(lines[0].strip())
        body = [line for line in lines[2:2 + self.n_atoms] if line.strip()]
        self.elements = [line.split()[0] for line in body]
        self.coordinates = [[float(x) for x in line.split()[1:4]] for line in body]
        if len(self.elements) != self.n_atoms:
            raise ValueError(f"{self.xyz_path}: header announces {self.n_atoms} atoms, {len(self.elements)} read")

        if self.cache_file in os.listdir(path) and not overwrite_results:
            print(f"Loading existing results from {path + self.cache_file}.")
            self.results = json.load(open(path + self.cache_file))
        else:
            print(f"No {self.cache_file} file found in the specified path or overwrite_results is True. Results dictionary will be empty.")
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
        """Reset the results dictionary to an empty dictionary -- and remove the cache file if it exists."""
        self.results = {}
        json_path = self.path + self.cache_file
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
                self.xyz_path,
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
                self.xyz_path,
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
    def save_results_to_json(self, filename=None):
        """Save the results dictionary to a JSON file.
        Args:
            filename (str): name of the JSON file to save the results to (default: the cache file of this
                geometry, ``xyz.json`` for ``optimization.xyz``, see :meth:`cache_name`)
        """
        filename = filename or self.cache_file
        sorted_results = dict(sorted(self.results.items()))
        self.results = sorted_results
        with open(self.path + filename, 'w') as json_file:
            json.dump(self.results, json_file, indent=4)