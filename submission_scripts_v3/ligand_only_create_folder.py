from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdChemReactions
import argparse
import os
import sys
import time
import metadata_utils


def str2bool(value):
    """argparse converter for the true/false text launch_workflow.sh passes
    (read out of workflow_inputs.yaml by get_input.py). `type=bool` would
    call bool("false"), which is True: every non-empty string is truthy."""
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n", ""):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


parser = argparse.ArgumentParser(description="Generate 3D coordinates from SMILES and save to XYZ format.")
parser.add_argument("--lig", type=str, help="SMILES string of the molecule")
parser.add_argument("--out", type=str, default="output.xyz", help="Output XYZ file name")
parser.add_argument("--optimize", type=str2bool, default=False,
                    help="Optimize the 3D structure using xtb (true/false)")
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

mol_noH = mol
mol = Chem.AddHs(mol)
inchi_key = Chem.rdinchi.MolToInchiKey(mol_noH)

print(f"InChIKey: {inchi_key}")
registry_entry = {
    "smiles": ligand,
    "canonical_smiles": Chem.MolToSmiles(mol_noH),
    "inchi": Chem.MolToInchi(mol_noH),
    "formal_charge": Chem.GetFormalCharge(mol_noH),
    "rdkit_version": Chem.rdBase.rdkitVersion,
}

if inchi_key not in os.listdir("."):
    AllChem.EmbedMolecule(mol)
    # The molecule as it went into mol.xyz: explicit hydrogens and the embedded
    # 3D coordinates. Chem.MolFromMolBlock(block, removeHs=False) gives it back.
    registry_entry.update({
        "mol_block": Chem.MolToMolBlock(mol),
        "has_3d_coordinates": mol.GetNumConformers() > 0,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    metadata_utils.register_molecule(inchi_key, registry_entry)
    os.mkdir(inchi_key)
    os.chdir(inchi_key)

else:
    # An existing directory means this molecule was already built -- the rest
    # of the pipeline (crest.py, uma_opt_confs.py, launch_orca_all.py) each
    # resume from what is already there, so this is a skip, not a failure.
    # Exit 0 so launch_workflow.sh's `set -e` doesn't abort the re-run.
    # Still register: a different SMILES spelling of the same molecule is
    # recorded as an alias. Batches built before the registry existed get an
    # entry without the embedded coordinates (the molecule, not the geometry,
    # is what is known).
    if "mol_block" not in (metadata_utils.load_registry().get(inchi_key) or {}):
        registry_entry.update({"mol_block": Chem.MolToMolBlock(mol), "has_3d_coordinates": False})
    metadata_utils.register_molecule(inchi_key, registry_entry)
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
else:
    # crest.py reads <output> (output.xyz) whether or not xtb ran, so the raw
    # RDKit geometry has to end up under that name too.
    os.rename("mol.xyz", output)

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
