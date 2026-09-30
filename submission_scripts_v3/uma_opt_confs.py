from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from ase.optimize import BFGS
from ase.io import read, write
from rdkit import Chem
import os
import subprocess
import argparse
import metadata_utils

parser = argparse.ArgumentParser(description="UMA-optimize CREST conformers")
parser.add_argument("--lig", type=str, required=True, help="SMILES string of the molecule")
# --- previously hardcoded below; now chosen via workflow_inputs.yaml's
#     `uma:` block and passed in here. ---
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--fmax", type=float, required=True,
                     help="BFGS force-convergence threshold, eV/Angstrom")
parser.add_argument("--task-name", type=str, required=True)
args = parser.parse_args()

# The optimised geometry is named after the model that produced it, so two
# libraries run with different UMA models can be told apart on disk, and so
# the UMA-only descriptor spec (data.geometry_file: optimized_<model>.xyz)
# matches without editing. Runs before 2026-09-30 wrote a fixed name
# whatever the model; those files are renamed on resume (check_inputs.py
# has already refused, or the operator confirmed, any change of model since).
uma_xyz = f"optimized_{args.model_name}.xyz"
LEGACY_UMA_XYZ = "optimized_uma-s-1.xyz"

ligand    = args.lig
inchi_key = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(ligand))

os.chdir(inchi_key)

# provide the name of the xyz file with the conformers
conformers_file = "crest_clustered.xyz"

# segment the file in unique xyzs files
subprocess.run(
                ["obabel", conformers_file, "-O", "conf_.xyz", "-m"],
                check=True
                )

conformers = [f for f in os.listdir('.') if f.startswith("conf_") and f.endswith(".xyz")]

### UMA optimization ###
model_name   = args.model_name
predict_unit = get_predict_unit(model_name=model_name, device="cpu")
calc = FAIRChemCalculator(predict_unit=predict_unit, task_name=args.task_name)

for f in conformers:
   folder = f.split('.')[0]
   if folder in os.listdir('.'):
       legacy = os.path.join(folder, LEGACY_UMA_XYZ)
       if uma_xyz != LEGACY_UMA_XYZ and os.path.exists(legacy) and not os.path.exists(os.path.join(folder, uma_xyz)):
           print(f"{legacy}: renaming to {uma_xyz} (fixed name of runs before 2026-09-30)")
           os.rename(legacy, os.path.join(folder, uma_xyz))
       if uma_xyz in os.listdir(folder):
           continue
   else:
       os.mkdir(folder)

   # read the conformer
   atoms = read(f)

   # Initialize the calculator with the loaded predict unit
   atoms.calc = calc

   # Run geometry optimization using BFGS
   dyn = BFGS(atoms, trajectory=f"{folder}/optimization_s1.traj")
   dyn.run(fmax=args.fmax)

   # Save the optimized structure
   write(f"{folder}/{uma_xyz}", atoms)

conf_energies = {}
for f in conformers:
    folder = f.split('.')[0]
    energy_file = os.path.join(folder, uma_xyz)
    energy = None
    if os.path.exists(energy_file):
        with open(energy_file) as ef:
            ef.readline()
            comment = ef.readline()
        for tok in comment.split():
            if tok.startswith("energy="):
                try:
                    energy = float(tok.split("=", 1)[1])
                except ValueError:
                    energy = None
    conf_energies[folder] = energy

metadata_utils.update_metadata(".", {
    "uma": {
        "model_name": model_name,
        "task_name": args.task_name,
        "optimizer": "ASE BFGS",
        "fmax": args.fmax,
        "geometry_file": uma_xyz,
        "conformer_energies_eV": conf_energies,
    },
})

os.chdir('..')
