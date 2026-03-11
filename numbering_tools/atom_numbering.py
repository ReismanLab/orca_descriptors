import os
import math

import numpy as np
import pandas as pd

from openbabel import pybel #JULES: openbabel is not currently installed in the yml environment
pybel.ob.obErrorLog.StopLogging()
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolAlign, GetMolFrags
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
from IPython.display import Image, display
import orca_descriptors as od

def get_sdf_from_xyz(path):
    for filename in os.listdir(path):
        if filename.endswith(".xyz"):
            xyz_path = os.path.join(path, filename)
            base_name = os.path.splitext(filename)[0]
            sdf_path = os.path.join(path, f"{base_name}.sdf")
            for mol in pybel.readfile("xyz", xyz_path):
                mol.write("sdf", sdf_path, overwrite=True)

def get_xyz_ids(path):
    with open(f"{path}/xyz_ids.txt", "w") as a:
        for filename in os.listdir(path):
            if filename.endswith('.xyz'):
                f = os.path.join(path, filename)
                a.write(str(f) + os.linesep)

def get_ligand_core_type_xyz(path):
    mols_5 = {} #5-membered metallacycles
    mols_6 = {} #6-membered metallacycles
    results = []
    with open(f"{path}/xyz_ids.txt", 'r') as file: #get list of xyz files
        xyz_names = [line.strip() for line in file]

    for xyz in xyz_names:
        remove_path = xyz.split(path)[1]
        base_name = remove_path.split('.')[0] 
        sdf_name = base_name + '.sdf'
        sdf_path = os.path.join(path, sdf_name)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False) #get mol object from sdf file
        original_mol = next((m for m in suppl if m is not None), None)
        if original_mol is None:
            print(f"sdf error for {sdf_name}, cannot automatically get atom numbering")
            continue

        Ni_atom = [atom for atom in original_mol.GetAtoms() if atom.GetAtomicNum() == 28] #find Ni atom
        Ni_atom_idx = Ni_atom[0].GetIdx() #get Ni atom index
        Ni_ringsize_5 = original_mol.GetAtomWithIdx(Ni_atom_idx).IsInRingSize(5) #check if Ni is in 5-membered ring
        Ni_ringsize_6 = original_mol.GetAtomWithIdx(Ni_atom_idx).IsInRingSize(6) #check if Ni is in 6-membered ring

        if Ni_ringsize_5 and not Ni_ringsize_6:
            mols_5[base_name] = original_mol
            results.append({'id': xyz, 'ligand_type': 'core_5'})
        elif not Ni_ringsize_5 and Ni_ringsize_6:
            mols_6[base_name] = original_mol
            results.append({'id': xyz,  'ligand_type': 'core_6'})
    return mols_5, mols_6, pd.DataFrame(results)

def get_initial_atom_numbers(mol_dict5, mol_dict6, path):
    substructure_5 = ['[H][Ni]1([H])[N]C=C[N]1', '[H][Ni]1([H])[N]CC[N]1', '[H][Ni]1([H])[#7]~[#6]~[#6]~[#7]1' ]
    
    results5 = []
    results6 = []
    for key, value in mol_dict5.items():
        ligID = key
        mol = value

        for substructure in substructure_5:
            substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
            if len(substructure_match) > 0:
                atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
                results5.append({'xyz_name': ligID, 'ligand_type': 'core_5', 'H1': atoms_idx_in_substructure_rdkit[0], 'Ni': atoms_idx_in_substructure_rdkit[1], 'H2': atoms_idx_in_substructure_rdkit[2], 'N1': atoms_idx_in_substructure_rdkit[3], 'C1': atoms_idx_in_substructure_rdkit[4], 'C2': atoms_idx_in_substructure_rdkit[5], 'N2': atoms_idx_in_substructure_rdkit[6]})
                break
        if ligID not in [entry['xyz_name'] for entry in results5]: #rdkit is weird sometimes just generating the mol again fixes the issue
            sdf_name = key + '.sdf'
            print (sdf_name)
            sdf_path = os.path.join(path, sdf_name)
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None) 
            for substructure in substructure_5:
                print (substructure)
                substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
                print (substructure_match)
                if len(substructure_match) > 0:
                    atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
                    results5.append({'xyz_name': ligID, 'ligand_type': 'core_5', 'H1': atoms_idx_in_substructure_rdkit[0], 'Ni': atoms_idx_in_substructure_rdkit[1], 'H2': atoms_idx_in_substructure_rdkit[2], 'N1': atoms_idx_in_substructure_rdkit[3], 'C1': atoms_idx_in_substructure_rdkit[4],'C3': atoms_idx_in_substructure_rdkit[5], 'C2': atoms_idx_in_substructure_rdkit[6], 'N2': atoms_idx_in_substructure_rdkit[7]}) #JULES: why is there C3 here?
                    break

    substructure_6 = ['[H][Ni]1([H])[N]CCC[N]1', '[H][Ni]1([H])[N]CC=C[N]1', '[H][Ni]1([H])[#7]~[#6]~[#6]~[#6]~[#7]1']
    for key, value in mol_dict6.items():
        ligID = key
        mol = value

        for substructure in substructure_6:
            substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
            if len(substructure_match) > 0:
                atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist]) #JULES: I don't undertand how the list ordering works here
                results6.append({'xyz_name': ligID, 'ligand_type': 'core_6', 'H1': atoms_idx_in_substructure_rdkit[0], 'Ni': atoms_idx_in_substructure_rdkit[1], 'H2': atoms_idx_in_substructure_rdkit[2], 'N1': atoms_idx_in_substructure_rdkit[3], 'C1': atoms_idx_in_substructure_rdkit[4],'C3': atoms_idx_in_substructure_rdkit[5], 'C2': atoms_idx_in_substructure_rdkit[6], 'N2': atoms_idx_in_substructure_rdkit[7]})
                break
        if ligID not in [entry['xyz_name'] for entry in results6]: #rdkit is weird sometimes just generating the mol again fixes the issue
            sdf_name = key + '.sdf'
            print (sdf_name)
            sdf_path = os.path.join(path, sdf_name)
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None) 
            for substructure in substructure_6:
                print (substructure)
                substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
                print (substructure_match)
                if len(substructure_match) > 0:
                    atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
                    results6.append({'xyz_name': ligID, 'ligand_type': 'core_6', 'H1': atoms_idx_in_substructure_rdkit[0], 'Ni': atoms_idx_in_substructure_rdkit[1], 'H2': atoms_idx_in_substructure_rdkit[2], 'N1': atoms_idx_in_substructure_rdkit[3], 'C1': atoms_idx_in_substructure_rdkit[4],'C3': atoms_idx_in_substructure_rdkit[5], 'C2': atoms_idx_in_substructure_rdkit[6], 'N2': atoms_idx_in_substructure_rdkit[7]})
                    break
    if len(results6) > 0 and len(results5) > 0:
        results6 = (pd.DataFrame(results6)).drop(columns=['C3'])
        results5 = pd.DataFrame(results5)
        results = pd.DataFrame(pd.concat([results5, results6], ignore_index=True))
    elif len(results6) > 0 and len(results5) == 0:
        results = pd.DataFrame(results6).drop(columns=['C3'])
    elif len(results6) == 0 and len(results5) > 0:
        results = pd.DataFrame(results5)

    return pd.DataFrame(results)

def make_ensemble_atom_numbering_consistent (df):
    df['lig_name'] = df['xyz_name'].str.split('_').str[0]
    ligands = df.groupby('lig_name')
    for prefix, ligand in ligands:
        unique_rows = ligand.drop(columns=['xyz_name', 'lig_name']).drop_duplicates()
        if len(unique_rows) > 1:
            if (len(unique_rows) == 2 and #if there are only 2 unique sets of values
                unique_rows.iloc[0]['Ni'] == unique_rows.iloc[1]['Ni'] and #and the Ni values are the same
                ((unique_rows.iloc[0]['N1'] == unique_rows.iloc[1]['N2']) and
                    (unique_rows.iloc[0]['N2'] == unique_rows.iloc[1]['N1']))): # and the N1/N2 values are just swapped
                chosen_values = unique_rows.iloc[0]
                for col in unique_rows.columns:
                    df.loc[df['lig_name'] == prefix, col] = chosen_values[col]
                continue
            if len(unique_rows) > 2: #this indicates that there is a more complex issue and potentially different substructures
                print(f"\nFlagged Prefix: {prefix}")
                print("Detected differing sets of values:")

                for i, unique_row in enumerate(unique_rows.iterrows(), start=1):
                    print(f"Option {i}:")
                    print(unique_row[1].to_dict())

                print(f"{len(unique_rows) + 1}: Leave as is")

                choice = int(input(f"\nEnter the option number (1-{len(unique_rows) + 1}) you want to apply for prefix '{prefix}': ").strip())

                if choice <= len(unique_rows):
                    chosen_values = unique_rows.iloc[choice - 1]
                    for col in unique_rows.columns:
                        df.loc[df['lig_name'] == prefix, col] = chosen_values[col]
                else:
                    print(f"Leaving the rows for prefix '{prefix}' as they are.")
    df.sort_values(by=['lig_name'], inplace=True)
    df = df.drop(columns=['lig_name'])
    return df

def verify_atom_numbering(df, path):
    df_columns = list(df.columns)
    for i,row in df.iterrows():
        xyz_name_base = row['xyz_name']
        sdf_name = xyz_name_base + '.sdf'
        sdf_path = os.path.join(path,sdf_name)
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except:
            print (f'error reading sdf for {xyz_name_base}')
            continue

        N1_index = row['N1']  
        N1_atomic_num = mol.GetAtomWithIdx(N1_index).GetAtomicNum()

        N2_index = row['N2'] 
        N2_atomic_num = mol.GetAtomWithIdx(N2_index).GetAtomicNum()

        C1_index = row['C1'] 
        C1_atomic_num = mol.GetAtomWithIdx(C1_index).GetAtomicNum()
        
        C2_index = row['C2'] 
        C2_atomic_num = mol.GetAtomWithIdx(C2_index).GetAtomicNum()

        if C2_atomic_num != 6 or C1_atomic_num != 6 or N2_atomic_num !=7 or N1_atomic_num !=7:
            print (f'error in atom numbering for {xyz_name_base}')

        if 'F1' in df_columns and 'F2' in df_columns:
            F1_index = row['F1'] 
            F2_index = row['F2'] 

            F1_atomic_num = mol.GetAtomWithIdx(F1_index).GetAtomicNum()
            F2_atomic_num = mol.GetAtomWithIdx(F2_index).GetAtomicNum()

            if F1_atomic_num !=9 or F2_atomic_num !=9:
                print (f'error in atom numbering for {xyz_name_base}')
        if 'H1' in df_columns and 'H2' in df_columns:
            H1_index = row['H1'] 
            H2_index = row['H2'] 

            H1_atomic_num = mol.GetAtomWithIdx(H1_index).GetAtomicNum()
            H2_atomic_num = mol.GetAtomWithIdx(H2_index).GetAtomicNum()

            if H1_atomic_num !=1 or H2_atomic_num !=1:
                print (f'error in atom numbering for {xyz_name_base}')

def find_pyox_ligands_and_renumber(df, path):
    # first deal with the 5-membered cores
    df5 = df[df['ligand_type'] == 'core_5'].copy()
    pyox_ligands = []
    biox_ligands = []
    biim_ligands = []
    bpy_ligands = []
    phen_ligands = []
    other_ligands = []
    for i,row in df5.iterrows():
        xyz_name_base = row['xyz_name']
        N1_index = row['N1'] 
        N2_index = row['N2'] 
        C1_index = row['C1'] 
        C2_index = row['C2'] 
        
        sdf_name = xyz_name_base + '.sdf'
        sdf_path = os.path.join(path,sdf_name)

        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except:
            print (f'bad input file for {sdf_path}')
        ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
        ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

        ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
        ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)

        if ringsize_N1_is_6 is False and ringsize_N2_is_6 is False and ringsize_N2_is_5 is True and ringsize_N1_is_5 is True: 
            # if there are no 6 membered rings and core is 5 membered, it is biim/biox
            for neighbor in mol.GetAtomWithIdx(C1_index).GetNeighbors(): #if carbon in mettalacycle is bonded to oxygen, it should be biox
                if neighbor.GetAtomicNum() == 8: 
                    biox_ligands.append(xyz_name_base)
                    other_ligands.append(xyz_name_base)
                    break
            else: #if not bonded to oxygen, it should be biim
                biim_ligands.append(xyz_name_base)
                other_ligands.append(xyz_name_base) 
        elif ringsize_N1_is_6 is True and ringsize_N2_is_6 is True and ringsize_N2_is_5 is True and ringsize_N1_is_5 is True: 
            # if N1 and N2 are each in a 6 membered ring and also a 5 membered core, it is a bpy or phen
            ring_info = mol.GetRingInfo()
            if ring_info.NumAtomRings(C1_index) > 2: # if carbon in mettalacycle is in more than 2 rings, it should be phen
                phen_ligands.append(xyz_name_base)
                other_ligands.append(xyz_name_base)
            else: # if not, it should be bpy
                bpy_ligands.append(xyz_name_base)
                other_ligands.append(xyz_name_base)
        else:
            # should be a pyox ligand
            pyox_ligands.append(xyz_name_base)
    
    #JULES: not sure if this is the best (or even a good way) to edit the dataframe, but it allowed for minimal change to the rest of the code
    if len(other_ligands) > 0:
        other_lig_df = df5.loc[df5['xyz_name'].isin(other_ligands)].copy()
        other_lig_df.loc[other_lig_df['xyz_name'].isin(biox_ligands), 'ligand_class'] = 'biox'
        other_lig_df.loc[other_lig_df['xyz_name'].isin(biim_ligands), 'ligand_class'] = 'biim'
        other_lig_df.loc[other_lig_df['xyz_name'].isin(bpy_ligands), 'ligand_class'] = 'bpy'
        other_lig_df.loc[other_lig_df['xyz_name'].isin(phen_ligands), 'ligand_class'] = 'phen'
    if len(pyox_ligands) > 0:
        pyox_lig_df = df5.loc[df5['xyz_name'].isin(pyox_ligands)].copy()
        pyox_lig_df.loc[:, 'ligand_class'] = 'pyox'
        for i,row in pyox_lig_df.iterrows():
                xyz_name_base = row['xyz_name']
                N1_index = row['N1'] 
                N2_index = row['N2'] 
                C1_index = row['C1']              
                C2_index = row['C2'] 

                sdf_name = xyz_name_base + '.sdf'
                sdf_path = os.path.join(path,sdf_name)

                try:
                    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
                    mol = next((m for m in suppl if m is not None), None)
                except:
                    print (f'bad input file for {sdf_path}')
                
                ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
                ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

                ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
                ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)
                
                if ringsize_N1_is_6 is True and ringsize_N2_is_5 is True and ringsize_N2_is_6 is False: # is N1 is already pyridine, df is correct as is 
                    # write back corrected values (+1 because dataframe is 1-based) AVC: don't think this is needed anymore since we are in 0-based indexing now
                    #pyox_lig_df.loc[i, 'N1'], pyox_lig_df.loc[i, 'N2']  = N1_index + 1, N2_index + 1
                    #pyox_lig_df.loc[i, 'C1'], pyox_lig_df.loc[i, 'C2'] = C1_index + 1, C2_index + 1
                    break

                elif ringsize_N2_is_6 is True and ringsize_N1_is_5 is True and ringsize_N1_is_6 is False: #Need to swap N1 and N2 indexes
                    pyox_lig_df.loc[i, 'N1'], pyox_lig_df.loc[i, 'N2'] = N2_index, N1_index 
                    pyox_lig_df.loc[i, 'C1'], pyox_lig_df.loc[i, 'C2'] = C2_index, C1_index
    
    if len(other_ligands) > 0 and len(pyox_ligands) > 0:
        df5_pyox_corrected = pd.concat([pyox_lig_df, other_lig_df], ignore_index=True)
    elif len(other_ligands) == 0 and len(pyox_ligands) > 0:
        df5_pyox_corrected = pyox_lig_df
    elif len(other_ligands) > 0 and len(pyox_ligands) == 0:
        df5_pyox_corrected = other_lig_df
    if len(other_ligands) == 0 and len(pyox_ligands) == 0:
        df5_pyox_corrected = pd.DataFrame()

    # now deal with the 6-membered cores
    df6 = df[df['ligand_type'] == 'core_6'].copy()
    pyr6_ligands = []
    box_ligands = []
    for i,row in df6.iterrows():
        xyz_name_base = row['xyz_name']
        N1_index = row['N1']  
        N2_index = row['N2'] 
        C1_index = row['C1'] 
        C2_index = row['C2'] 
        
        sdf_name = xyz_name_base + '.sdf'
        sdf_path = os.path.join(path,sdf_name)

        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except:
            print (f'bad input file for {sdf_path}')

        ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
        ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

        ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
        ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)

        if ringsize_N2_is_6 is True and ringsize_N2_is_5 is True and ringsize_N1_is_6 is True and ringsize_N1_is_5 is True:
            # if N1 and N2 are each in a 5 membered ring and also a 6 membered core, it is a box ligand 
            box_ligands.append(xyz_name_base)
        else:
            # should be a pyr6 ligand
            pyr6_ligands.append(xyz_name_base)
    if len(box_ligands) > 0:
        box_lig_df = df6.loc[df6['xyz_name'].isin(box_ligands)].copy()
        box_lig_df.loc[:, 'ligand_class'] = 'box'
    if len(pyr6_ligands) > 0:
        pyr6_lig_df = df6.loc[df6['xyz_name'].isin(pyr6_ligands)].copy()
        pyr6_lig_df.loc[:, 'ligand_class'] = 'pyr6'
        for i,row in pyr6_lig_df.iterrows():
                xyz_name_base = row['xyz_name']
                N1_index = row['N1'] 
                N2_index = row['N2'] 
                C1_index = row['C1']                 
                C2_index = row['C2'] 

                sdf_name = xyz_name_base + '.sdf'
                sdf_path = os.path.join(path,sdf_name)

                try:
                    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
                    mol = next((m for m in suppl if m is not None), None)
                except:
                    print (f'bad input file for {sdf_path}')
                
                ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
                ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

                ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
                ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)
                
                if ringsize_N1_is_6 is True and ringsize_N1_is_5 is True and ringsize_N2_is_6 is True and ringsize_N2_is_5 is False: # is N1 is already pyridine, df is correct as is
                    # then N1 is the oxazoline and N2 is the pyridine so we need to swap them 
                    pyr6_lig_df.loc[i, 'N1'], pyr6_lig_df.loc[i, 'N2']  = N2_index + 1, N1_index + 1
                    pyr6_lig_df.loc[i, 'C1'], pyr6_lig_df.loc[i, 'C2'] = C2_index + 1, C1_index + 1

                elif ringsize_N1_is_6 is True and ringsize_N1_is_5 is False and ringsize_N2_is_6 is True and ringsize_N2_is_5 is True:
                    # then N2 is the oxazoline and N1 is the pyridine, which is correct
                    pyr6_lig_df.loc[i, 'N1'], pyr6_lig_df.loc[i, 'N2'] = N1_index + 1, N2_index + 1
                    pyr6_lig_df.loc[i, 'C1'], pyr6_lig_df.loc[i, 'C2'] = C1_index + 1, C2_index + 1
    if len(box_ligands) > 0 and len(pyr6_ligands) > 0:
        df6_pyox_corrected = pd.concat([box_lig_df, pyr6_lig_df], ignore_index=True)
    elif len(box_ligands) == 0 and len(pyr6_ligands) > 0:
        df6_pyox_corrected = pyr6_lig_df
    elif len(box_ligands) > 0 and len(pyr6_ligands) == 0:
        df6_pyox_corrected = box_lig_df
    if len(box_ligands) == 0 and len(pyr6_ligands) == 0:
        df6_pyox_corrected = pd.DataFrame()
    
    df_pyox_corrected = pd.concat([df5_pyox_corrected, df6_pyox_corrected], ignore_index=True)
    
    return df_pyox_corrected

def get_smiles(df, path):
    df.insert(2, 'smiles', '') # add empty column for smiles
    df.insert(1, 'path', '') # add empty column for path
    df.insert(3, 'inchi', '') # add empty column for inchikey
    for i, row in df.iterrows():
        xyz_name_base = row['xyz_name']
        sdf_name = xyz_name_base + '.sdf'
        xyz_name = xyz_name_base + '.xyz'
        sdf_path = os.path.join(path, sdf_name)
        xyz_path = os.path.join(path, xyz_name)
        df.loc[i, 'path'] = path
        Ni_index = row['Ni']
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except:
            print (f'bad input file for {sdf_path}')
        rw_mol = Chem.rdchem.RWMol(mol)  # Make a writable copy of the molecule
        rw_mol = Chem.rdchem.RWMol(mol)  # Make a writable copy of the molecule
        Chem.rdchem.RWMol.RemoveAtom(rw_mol, Ni_index)
        frags = GetMolFrags(rw_mol, asMols=True)
        for frag in frags:
            if len(frag.GetAtoms()) > 1:
                lig = Chem.MolToSmiles(frag)
                df.loc[i, 'smiles'] = lig
                inchi = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(lig))
                df.loc[i, 'inchi'] = inchi
                break
    return df

def get_scf_energy(df):
    df.insert(4, 'SCF_energy', '') # add empty column for scf energy
    for i, row in df.iterrows():
        path = row['path']
        out_file_path = os.path.join(path, 'calcParam.out')
        singlepoint = od.OrcaReader(directory       = f"{path}",   #<- this is the folder with the job outputs
                                    basename        = "singlepoint",        #<- this is the type of job outputs the parameters will be extracted from (to extract descriptors for the json summary file)
                                    out_file_path   = f"{out_file_path}") #<- this is the complete ORCA output file path
        singlepoint.parse_property_json()               #<- loads the json file associated with the basename 
        singlepoint.get_scfenergy()                     #<- extract scf energy
        df.at[i, 'SCF_energy'] = singlepoint.results['SCF_energy']                      
        
    return df



def compile_final(path, index=1, subdirectory=True):
    if not path[-1] == '/':
        path = path + '/'
        print (f'added slash to end of path, new path is {path}')
    if subdirectory == False:
        get_sdf_from_xyz(path)
        get_xyz_ids(path)
        mols_5, mols_6, ligand_core_class = get_ligand_core_type_xyz(path)
        ligand_initial_atom_numbering = get_initial_atom_numbers(mols_5, mols_6, path)
        ligand_consistent_atom_numbering = make_ensemble_atom_numbering_consistent(ligand_initial_atom_numbering)
        verify_atom_numbering(ligand_consistent_atom_numbering, path)
        df_pyox_corrected = find_pyox_ligands_and_renumber(ligand_consistent_atom_numbering, path)
        df_with_smiles = get_smiles(df_pyox_corrected, path)
        df_with_energy = get_scf_energy(df_with_smiles)
        if index == 1: #change zero indexing to 1-indexing if desired
            df_with_smiles['H1'] = df_with_smiles['H1'] + 1
            df_with_smiles['Ni'] = df_with_smiles['Ni'] + 1
            df_with_smiles['H2'] = df_with_smiles['H2'] + 1
            df_with_smiles['N1'] = df_with_smiles['N1'] + 1
            df_with_smiles['C1'] = df_with_smiles['C1'] + 1
            df_with_smiles['C2'] = df_with_smiles['C2'] + 1
            df_with_smiles['N2'] = df_with_smiles['N2'] + 1
        df_final = df_with_energy
    else:
        df_final = pd.DataFrame()
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    sub_path = os.path.join(path, entry.name)
                    if not sub_path[-1] == '/':
                        sub_path = sub_path + '/'
                    get_sdf_from_xyz(sub_path)
                    get_xyz_ids(sub_path)
                    mols_5, mols_6, ligand_core_class = get_ligand_core_type_xyz(sub_path)
                    ligand_initial_atom_numbering = get_initial_atom_numbers(mols_5, mols_6, sub_path)
                    ligand_consistent_atom_numbering = make_ensemble_atom_numbering_consistent(ligand_initial_atom_numbering)
                    verify_atom_numbering(ligand_consistent_atom_numbering, sub_path)
                    df_pyox_corrected = find_pyox_ligands_and_renumber(ligand_consistent_atom_numbering, sub_path)
                    df_with_smiles = get_smiles(df_pyox_corrected, sub_path)
                    df_with_energy = get_scf_energy(df_with_smiles)
                    if index == 1: #change zero indexing to 1-indexing if desired
                        df_with_energy['H1'] = df_with_energy['H1'] + 1
                        df_with_energy['Ni'] = df_with_energy['Ni'] + 1
                        df_with_energy['H2'] = df_with_energy['H2'] + 1
                        df_with_energy['N1'] = df_with_energy['N1'] + 1
                        df_with_energy['C1'] = df_with_energy['C1'] + 1
                        df_with_energy['C2'] = df_with_energy['C2'] + 1
                        df_with_energy['N2'] = df_with_energy['N2'] + 1
                    df_final = pd.concat([df_final, df_with_energy], ignore_index=True)
                
    return df_final


    

    

