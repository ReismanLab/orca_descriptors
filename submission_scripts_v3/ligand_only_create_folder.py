from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdChemReactions
import argparse
import os
import sys
import metadata_utils

parser = argparse.ArgumentParser(description="Generate 3D coordinates from SMILES and save to XYZ format.")
parser.add_argument("--lig", type=str, help="SMILES string of the molecule")
parser.add_argument("--out", type=str, default="output.xyz", help="Output XYZ file name")
parser.add_argument("--optimize", type=bool, default=False, help="Optimize the 3D structure using xtb")
# previously hardcoded as `--gfn 2` in the os.system() call below; now chosen
# via workflow_inputs.yaml's conformer_gen.xtb_gfn_level
parser.add_argument("--xtb-gfn-level", type=int, default=2, help="xtb --gfn <N> level for the pre-optimization")
args = parser.parse_args()

ligand   = args.lig
output   = args.out
optimize = args.optimize

## identify the ligand nitrogens
# 1. assert that the ligand is one of the known structures
mol = Chem.MolFromSmiles(ligand)
if not mol:
    raise ValueError("Invalid SMILES string provided.")

mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
inchi_key = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(ligand))

print(f"InChIKey: {inchi_key}")
if inchi_key not in os.listdir("."):
    os.mkdir(inchi_key)
    os.chdir(inchi_key)

else:
    # An existing directory means this molecule was already built -- the rest
    # of the pipeline (crest.py, uma_opt_confs.py, launch_orca_all.py) each
    # resume from what is already there, so this is a skip, not a failure.
    # Exit 0 so launch_workflow.sh's `set -e` doesn't abort the re-run.
    print(f"Directory {inchi_key} already exists, skipping generation.")
    sys.exit(0)


conf = mol.GetConformer()

# write to XYZ file a first xyz file
with open("mol.xyz", "w") as f:
    f.write(f"{mol.GetNumAtoms()}\n")
    f.write(f"SMILES: {ligand}\n")
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

if optimize:
    os.system(f"xtb mol.xyz --opt --gfn {args.xtb_gfn_level}")
    os.system(f"mv xtbopt.xyz {output}")
    os.system("rm mol.xyz")
    os.system("rm xtbopt.log wbo xtbrestart xtbtopo.mol .xtboptok charges")

metadata_utils.update_metadata(".", {
    "smiles": ligand,
    "inchikey": inchi_key,
    "formal_charge": Chem.GetFormalCharge(mol),
    "conformer_gen": {
        "embed_method": "RDKit ETKDG (AllChem.EmbedMolecule)",
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "xtb_preopt": optimize,
        "xtb_preopt_method": f"GFN{args.xtb_gfn_level}-xTB" if optimize else None,
    },
})

os.chdir('..')
