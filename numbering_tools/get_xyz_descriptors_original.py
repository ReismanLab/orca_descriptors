import pandas as pd
import numpy as np
import os
import json

from morfeus import Sterimol, BuriedVolume, Pyramidalization, SASA, BiteAngle

import dbstep.Dbstep as db

import logging

logging.basicConfig(level=logging.WARNING) 


class PropertiesXYZ:
    """Properties from XYZ file using Morfeus"""

    def __init__(self, path):
        """this class reads an xyz file from a given path and extracts properties using Morfeus
        Args:
            path (str): path to the folder containing optimization.xyz
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

        if 'xyz.json' in os.listdir(path):
            self.results = json.load(open(path + 'xyz.json'))
        else:
            print("No xyz.json file found in the specified path. Results dictionary will be empty.")
            self.results = {}

        # initialize lists for atoms, bonds, angles, dihedrals
        self.atom_list = None
        self.bond_list = None
        self.angle_list = None
        self.dihedral_list = None

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



    def define_atom_list(self, atom_list):
        self.atom_list = atom_list

    def define_bond_list(self, bond_list):
        self.bond_list = bond_list

    def define_angle_list(self, angle_list):
        self.angle_list = angle_list
    
    def define_dihedral_list(self, dihedral_list):
        self.dihedral_list = dihedral_list

    def define_dihedral_list(self, dihedral_list):
        self.dihedral_list = dihedral_list
        
    def define_quadrant_list(self, quadrant_list):
        self.quadrant_list = quadrant_list

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
        Calculate buried volume for all atoms in atom_list at radius from 2.0 to 5.0 Å with a step of 0.5 Å
        The calculation is done using Morfeus.
        The atom_list must be defined before calling this function.
        The radius list is [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] by default but can be changed by modifying the radius parameter: self.radius_list
        Results are stored in the results dictionary with keys formatted as 'Vbur_atom_{at_idx}_radius_{radius}'.
        """
        if self.atom_list is None:
            print("Atom list is not defined. Please define it using define_atom_list().")
            return
        
        for at_idx in self.atom_list:
            for radius in self.radius_list:
                self.get_Vbur_at_radii(at_idx, radius)
          
    def get_Vbur_at_radii(self, at_idx, radius=3.5):
        """Calculate buried volume for a specific atom at a specific radius using Morfeus
        Args:
            at_idx (int): index of the atom (1-based)
            radius (float): radius in Å for the buried volume calculation
        """
        vbur = BuriedVolume(self.elements, self.coordinates, int(at_idx), include_hs=True, radius=radius) # calls Morfeus
        self.results['%Vbur_atom_' + str(self.elements[at_idx-1]) + str(at_idx) + '_radius_' + str(radius)] = vbur.fraction_buried_volume * 100

    #-- Pyramidalization --#
    def get_pyramidalizations(self):
        """Calculate dihedral angles for all sets of four atoms in dihedral_list
        The dihedral_list must be defined before calling this function.
        Results are stored in the results dictionary with keys formatted as 'Dihedral_{atom1_symbol}{atom1}_{atom2_symbol}{atom2}_{atom3_symbol}{atom3}_{atom4_symbol}{atom4}'.
        """
        if self.atom_list is None:
            print("Dihedral list is not defined. Please define it using define_dihedral_list().")
            return
        
        for atoms in self.atom_list:
            self.get_pyramidalization(atoms)

    def get_pyramidalization(self, at_idx):
        """Calculate dihedral angle for a specific set of four atoms using Morfeus
        Args:
            dihedral_list (list): list of four atom indexes (1-based)
        """
        pyramidalization = Pyramidalization(self.coordinates, at_idx, elements=self.elements, method="connectivity") # calls Morfeus AVC: edited ot call elements
        key1 = 'Pyr_' + str(self.elements[at_idx-1]) + str(at_idx) 
        key2 = 'Pyr_angle_' + str(self.elements[at_idx-1]) + str(at_idx) #AVC: altered to not treat at_idx as list
        self.results[key1] = pyramidalization.P
        self.results[key2] = pyramidalization.P_angle

    #-- Atomic SASA --#
    def get_atomic_SASA(self):
        """Calculate atomic solvent accessible surface area (SASA) for all atoms in atom_list using Morfeus
        The atom_list must be defined before calling this function.
        Results are stored in the results dictionary with keys formatted as 'Atomic_SASA_{atom_symbol}{atom_index}'.
        """
        if self.atom_list is None:
            print("Atom list is not defined. Please define it using define_atom_list().")
            return
        
        sasa = SASA(self.elements, self.coordinates) # calls Morfeus
        atomic_areas = sasa.atom_areas

        for at_idx in self.atom_list:
            key = 'SASA_' + str(self.elements[at_idx-1]) + str(at_idx)
            self.results[key] = atomic_areas[at_idx] #AVC: changed to convert back to correct indexing 
    

    ## 2 atom properties ##
    #-- Bond Distances --#
    def get_bond_distances(self):
        """Calculate bond distances for all sets of two atoms in bond_list
        The bond_list must be defined before calling this function.
        Results are stored in the results dictionary with keys formatted as 'Bond_{atom1_symbol}{atom1}_{atom2_symbol}{atom2}'.
        """
        if self.bond_list is None:
            print("Bond list is not defined. Please define it using define_bond_list().")
            return
        
        for bond_atoms in self.bond_list:
            self.get_bond_distance(bond_atoms)

    def get_bond_distance(self, bond_list):
        """Calculate bond distance for a specific set of two atoms using Morfeus
        Args:
            bond_list (list): list of two atom indexes (1-based)
        """
        bond = Sterimol(self.elements, self.coordinates, bond_list[0], bond_list[1]) # calls Morfeus
        key = 'Bond_' + '_'.join([str(self.elements[i-1]) + str(i) for i in bond_list])
        self.results[key] = bond.bond_length

    def get_Sterimol_parameters(self):
        """Calculate Sterimol parameters for all sets of two atoms in bond_list
        The bond_list must be defined before calling this function.
        Results are stored in the results dictionary with keys formatted as 'Sterimol_{atom1_symbol}{atom1}_{atom2_symbol}{atom2}_L', 'Sterimol_{atom1_symbol}{atom1}_{atom2_symbol}{atom2}_B1', and 'Sterimol_{atom1_symbol}{atom1}_{atom2_symbol}{atom2}_B5'.
        """
        if self.bond_list is None:
            print("Bond list is not defined. Please define it using define_bond_list().")
            return
        
        for bond_atoms in self.bond_list:
            self.get_Sterimol_parameters_for_bond(bond_atoms)

    def get_Sterimol_parameters_for_bond(self, bond_list):
        """Calculate Sterimol parameters for a specific set of two atoms using dbstep (fallback to Morfeus).
        Args:
            bond_list (list): list of two atom indexes (1-based)
        """

        # case where sterimol parameters are computed along a scan
        if self.sterimol_scan is not None:
            mol = db.dbstep(self.path + 'optimization.xyz', atom1=(bond_list[0]), atom2=(bond_list[1]),
                        commandline=True, verbose=False, sterimol=True,
                        scan=self.sterimol_scan)
            
            start, end, step = self.sterimol_scan.split(':')
            values = []
            v = float(start)
            while v < end:
                values.append(v)
                v += float(step)

            for _, scan_value in enumerate(values):
                key_prefix = 'Sterimol_' + '_'.join([str(self.elements[i-1]) + str(i) for i in bond_list]) + f'_{scan_value}Å'
                self.results[key_prefix + '_Bmin'] = mol.sterimol[scan_value].Bmin[_]
                self.results[key_prefix + '_Bmax'] = mol.sterimol[scan_value].Bmax[_]


        # default case without scan
        else:
            mol = db.dbstep(self.path + 'optimization.xyz', atom1=bond_list[0], atom2=bond_list[1], 
                        commandline=True, verbose=False, sterimol=True)
        
            self.results['Sterimol_' + '_'.join([str(self.elements[i-1]) + str(i) for i in bond_list]) + '_Bmin'] = mol.Bmin
            self.results['Sterimol_' + '_'.join([str(self.elements[i-1]) + str(i) for i in bond_list]) + '_Bmax'] = mol.Bmax
        
        self.results['Sterimol_' + '_'.join([str(self.elements[i-1]) + str(i) for i in bond_list]) + '_L']  = float(mol.L)

    ## 3 atom properties ##
    #-- Angles --#
    def get_angles(self):
        """Calculate angles for all sets of three atoms in angle_list
        The angle_list must be defined before calling this function.
        Results are stored in the results dictionary with keys formatted as 'Angle_{atom1_symbol}{atom1}_{atom2_symbol}{atom2}_{atom3_symbol}{atom3}'.
        """
        if self.angle_list is None:
            print("Angle list is not defined. Please define it using define_angle_list().")
            return
        
        for angle_atoms in self.angle_list:
            self.get_angle(angle_atoms)

    def get_bite_angle(self, angle_list: list):
        """Calculate angle for a specific set of three atoms using Morfeus
        Args:
            angle_list (list): list of three atom indexes (1-based)
        """
        angle = BiteAngle(self.coordinates, angle_list[0], angle_list[1], angle_list[2]) # calls Morfeus
        key = 'BiteAngle_' + '_'.join([str(self.elements[i-1]) + str(i) for i in angle_list])
        self.results[key] = angle.angle
    
    def get_angle(self, angle_list: list):
        """Calculate angle for a specific set of three atoms using 3D coordinates
        Args:
            angle_list (list): list of three atom indexes (1-based)
        """
        # define the two vectors (1->0 and 1->2) 0, 1, 2 are the indexes in angle_list
        u = np.array(self.coordinates[angle_list[0]-1]) - np.array(self.coordinates[angle_list[1]-1])
        v = np.array(self.coordinates[angle_list[2]-1]) - np.array(self.coordinates[angle_list[1]-1])
        
        # calculate the angle in radians 
        cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        angle_rad = np.arccos(cos_theta) # 
        # convert to degrees
        angle_deg = np.degrees(angle_rad)
        key = 'Angle_' + '_'.join([str(self.elements[i-1]) + str(i) for i in angle_list])
        #print(f"Calculated angle for {key}: {angle_deg} degrees, type: {type(angle_deg)}")
        self.results[key] = float(angle_deg)

    ## 4 atom properties ##
    #-- Dihedrals --#
    def get_dihedrals(self):
        """Calculate dihedral angles for all sets of four atoms in dihedral_list
        The dihedral_list must be defined before calling this function.
        Results are stored in the results dictionary with keys formatted as 'Dihedral_{atom1_symbol}{atom1}_{atom2_symbol}{atom2}_{atom3_symbol}{atom3}_{atom4_symbol}{atom4}'.
        """
        if self.dihedral_list is None:
            print("Dihedral list is not defined. Please define it using define_dihedral_list().")
            return
        
        for dihedral_atoms in self.dihedral_list:
            self.get_dihedral(dihedral_atoms)

    def get_dihedral(self, dihedral_list):
        """Calculate dihedral angle for a specific set of four atoms using Morfeus
        Args:
            dihedral_list (list): list of four atom indexes (1-based)
        """

        b1 = np.array(self.coordinates[dihedral_list[1]-1]) - np.array(self.coordinates[dihedral_list[0]-1])
        b2 = np.array(self.coordinates[dihedral_list[2]-1]) - np.array(self.coordinates[dihedral_list[1]-1])
        b3 = np.array(self.coordinates[dihedral_list[3]-1]) - np.array(self.coordinates[dihedral_list[2]-1])
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # atan2(y, x) form recommended in molecular mechanics
        x = np.dot(n1, n2)
        y = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))
        
        dihedral = np.degrees(np.arctan2(y, x))
        dihedral_key = 'Dihed_' + '_'.join([str(self.elements[i-1]) + str(i) for i in dihedral_list])  
        self.results[dihedral_key] = float(dihedral)  

    
    ## 5 atom properties ##
    #-- quadrants --#
    def get_vbur_quadrants(self):
        """Calculate buried volume in quadrants for a specific atom at a specific radius using Morfeus
        Args:
            at_idx (int): index of the atom (1-based)
            radius (float): radius in Å for the buried volume calculation
        """
        if self.quadrant_list is None:
            print("Quadrant list is not defined. Please define it using define_quadrant_list().")
            return

        for atom_list in self.quadrant_list:
            self.get_vbur_quadrants_only(atom_list)

    def get_vbur_quadrants_only(self, atom_list): #uses morfeus to calculate vbur at a single radius for an atom (a1) in df
        # check documentation of morfeus buried volume for explanation of parameters: https://digital-chemistry-laboratory.github.io/morfeus/buried_volume.html
        metal_center   = atom_list[0]
        z_axis_atoms   = [atom_list[1], atom_list[2]]
        xz_plane_atoms = atom_list[3:]
        
        vbur = BuriedVolume(self.elements, self.coordinates, metal_center,
                            radius=self.quadrant_radius, 
                            include_hs=True, 
                            excluded_atoms=self.quadrant_excluded_atoms, 
                            z_axis_atoms=z_axis_atoms, 
                            xz_plane_atoms=xz_plane_atoms) #calls morfeus
                
        vbur.octant_analysis()
                
        quadrants = vbur.quadrants
        bv_quadrants = quadrants['percent_buried_volume']

        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_quadrant_+,+'] = bv_quadrants[1]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_quadrant_–,+'] = bv_quadrants[2]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_quadrant_–,–'] = bv_quadrants[3]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_quadrant_+,–'] = bv_quadrants[4]                  

    #-- octants --#
    def get_vbur_octants(self):
        """Calculate buried volume in octants for a specific atom at a specific radius using Morfeus
        """
        if self.quadrant_list is None:
            print("Quadrant list is not defined. Please define it using define_quadrant_list().")
            return

        for atom_list in self.quadrant_list:
            self.get_vbur_octants_only(atom_list)

    def get_vbur_octants_only(self, atom_list): 
        # check documentation of morfeus buried volume for explanation of parameters: https://digital-chemistry-laboratory.github.io/morfeus/buried_volume.html
        metal_center   = atom_list[0]
        z_axis_atoms   = [atom_list[1], atom_list[2]]
        xz_plane_atoms = atom_list[3:]
        
        vbur = BuriedVolume(self.elements, self.coordinates, metal_center,
                            radius=self.quadrant_radius, 
                            include_hs=True, 
                            excluded_atoms=self.quadrant_excluded_atoms, 
                            z_axis_atoms=z_axis_atoms, 
                            xz_plane_atoms=xz_plane_atoms) #calls morfeus
                
        vbur.octant_analysis()

        octants = vbur.octants

        bv_octants = octants['percent_buried_volume']
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_+,+,+'] = bv_octants[0]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_–,+,+'] = bv_octants[1]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_–,–,+'] = bv_octants[2]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_+,–,+'] = bv_octants[3]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_+,–,–'] = bv_octants[4]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_–,–,–'] = bv_octants[5]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_–,+,–'] = bv_octants[6]
        self.results['%Vbur_' + str(self.elements[metal_center-1]) + str(metal_center) + '_octant_+,+,–'] = bv_octants[7]
    


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

    