from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdChemReactions
import argparse
import os

parser = argparse.ArgumentParser(description="Generate 3D coordinates from SMILES and save to XYZ format.")
parser.add_argument("--lig", type=str, help="SMILES string of the molecule")
parser.add_argument("--out", type=str, default="output.xyz", help="Output XYZ file name")
parser.add_argument("--optimize", type=bool, default=False, help="Optimize the 3D structure using xtb")
args = parser.parse_args()

ligand   = args.lig
output   = args.out
optimize = args.optimize


# hardcoded templates for the ligand
temp_dict = {'pyox': "C1(c2ncccc2)=NCCO1",   # Example pyrazole derivative
             'box': "C1(CC2=NCCO2)=NCCO1",  # Example box structure
             'biox': "C1(C2=NCCO2)=NCCO1",  # Example biox structure
             'bilm': "C1(C2=NCCN2)=NCCN1",  # Example bilayer structure
             'bipy': "c1(c2ncccc2)ncccc1"}  # Example bipyridine structure

temp_rxn = {'pyox': rdChemReactions.ReactionFromSmarts('[C:0]1([c:1]2[n:2][c:3][c:4][c:5][c:6]2)=[N:7][C:8][C:9][O:10]1.[Ni:11]>>[C:0]1([c:1]2[n:2](->[Ni:11]3)[c:3][c:4][c:5][c:6]2)=[N:7]->3[C:8][C:9][O:10]1'),
            'box' :  rdChemReactions.ReactionFromSmarts('[C:0]1([C:1][C:2]2=[N:3][C:4][C:5][O:6]2)=[N:7][C:8][C:9][O:10]1.[Ni:11]>>[C:0]1([C:1][C:2]2=[N:3](->[Ni:11]3)[C:4][C:5][O:6]2)=[N:7]->3[C:8][C:9][O:10]1'),
            'biox': rdChemReactions.ReactionFromSmarts('[C:0]1([C:2]2=[N:3][C:4][C:5][O:6]2)=[N:7][C:8][C:9][O:10]1.[Ni:11]>>[C:0]1([C:2]2=[N:3](->[Ni:11]3)[C:4][C:5][O:6]2)=[N:7]->3[C:8][C:9][O:10]1'),
            'bilm': rdChemReactions.ReactionFromSmarts('[C:0]1([C:2]2=[N:3][C:4][C:5][N:6]2)=[N:7][C:8][C:9][N:10]1.[Ni:11]>>[C:0]1([C:2]2=[N:3](->[Ni:11]3)[C:4][C:5][N:6]2)=[N:7]->3[C:8][C:9][N:10]1'),
            'bipy': rdChemReactions.ReactionFromSmarts('[c:1]1([c:2]2[n:3][c:4][c:5][c:6][c:13]2)[n:7][c:8][c:9][c:10][c:11]1.[Ni:12]>>[c:1]1([c:2]2[n:3](->[Ni:12]3)[c:4][c:5][c:6][c:13]2)[n:7]->3[c:8][c:9][c:10][c:11]1')
            }



## identify the ligand nitrogens
# 1. assert that the ligand is one of the known structures
mol = Chem.MolFromSmiles(ligand)
if not mol:
    raise ValueError("Invalid SMILES string provided.")

found_temp = False
for temp_name, temp_smiles in temp_dict.items():
    temp_mol = Chem.MolFromSmiles(temp_smiles)
    if mol.HasSubstructMatch(temp_mol):
        print(f"Found matching substructure: {temp_name}")
        found_temp = temp_name
        break

if found_temp == False:
    raise ValueError("Ligand does not match any known structure templates.")

# make the NiH2 complex
ni  = Chem.MolFromSmiles('[Ni](<-[Be])<-[Be]') # using Be as a placeholder for H2
rxn = temp_rxn[found_temp]
reacts = (mol, ni)
products = rxn.RunReactants(reacts)
if len(products) == 0:
    raise ValueError("No products generated from the reaction. Check the SMILES or reaction template.")

m = products[0][0]
mol = Chem.MolFromSmiles(Chem.MolToSmiles(m))
mol = Chem.AddHs(mol)
for at in mol.GetAtoms():
    if at.GetSymbol() == 'Be':
        at.SetAtomicNum(1)  # Change Be back to H
        at.SetFormalCharge(-1)  # Ensure H has formal charge 0
    if at.GetSymbol() == 'Ni':
        at.SetFormalCharge(2)  # Ensure Ni has formal charge 0
AllChem.EmbedMolecule(mol)

#inchi_key = Chem.rdinchi.MolToInchiKey(mol)


#if inchi_key is None:
inchi_key = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(ligand))

print(f"InChIKey: {inchi_key}")
if inchi_key not in os.listdir("."):
    os.mkdir(inchi_key)
    os.chdir(inchi_key)

else:
    exit(f"Directory {inchi_key} already exists, skipping generation.")


conf = mol.GetConformer()

print(f"Generated molecule: {Chem.MolToSmiles(m)}")
# write to XYZ file a first xyz file
with open("mol.xyz", "w") as f:
    f.write(f"{mol.GetNumAtoms()}\n")
    f.write(f"SMILES: {ligand}\n")
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        if atom.GetSymbol() == 'Be':
            f.write(f"H {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
        else:
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

if optimize:
    os.system("xtb mol.xyz --opt --gfn 2")
    os.system(f"mv xtbopt.xyz {output}")
    os.system("rm mol.xyz")
    os.system("rm xtbopt.log wbo xtbrestart xtbtopo.mol .xtboptok charges")

os.chdir('..')



