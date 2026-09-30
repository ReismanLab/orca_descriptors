import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from rdkit import Chem
import numpy as np
import argparse
import metadata_utils
import workflow_inputs as wi
import calc_param_template as cpt


def get_orca_module():
    for mod in os.environ.get("LOADEDMODULES", "").split(":"):
        if mod.startswith("orca"):
            return mod
    return None


def sbatch_resource_overrides(cfg):
    """--time / --mem-per-cpu / --ntasks-per-node flags for the per-conformer
    ORCA job, read from workflow_inputs.yaml's slurm.orca_job block. These
    override slurm-calcParam.sh's own #SBATCH defaults (sbatch CLI flags win
    over the script's embedded ones) -- #SBATCH lines are parsed before the
    script runs, so there's no way to make them read this file directly."""
    orca_job = wi.get(cfg, "slurm.orca_job", default={})
    flags = []
    if "time" in orca_job:
        flags.append(f"--time={orca_job['time']}")
    if "mem_per_cpu" in orca_job:
        flags.append(f"--mem-per-cpu={orca_job['mem_per_cpu']}")
    if "ntasks_per_node" in orca_job:
        flags.append(f"--ntasks-per-node={orca_job['ntasks_per_node']}")
    return flags


parser = argparse.ArgumentParser(description="get smiles for inchikey")
parser.add_argument("--lig", type=str, help="SMILES string of the molecule")
parser.add_argument("--ewin", type=str, help="energy cutoff for the uma optimized conformers (in kcal/mol)",  default=8)
# previously the relative path to a static calcParam.inp; now the ORCA
# Compound TEMPLATE, rendered per ligand from workflow_inputs.yaml's
# dft.level_of_theory block (see calc_param_template.py)
parser.add_argument("--calc-param-template", type=str, default="calcParam.inp.tmpl",
                     help="Path to the ORCA Compound template, resolved from the directory "
                          "this script is invoked from (the batch directory)")
args = parser.parse_args()
ligand    = args.lig
inchi_key = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(ligand))

cfg = wi.load_inputs()
sbatch_overrides = sbatch_resource_overrides(cfg)
# Resolve the template path against the batch directory (our CWD right now)
# before chdir-ing into the ligand's own directory below.
template_path = os.path.abspath(args.calc_param_template)

os.chdir(inchi_key)

ewin = float(args.ewin)

# Render calcParam.inp fresh for this ligand. Every conformer gets an
# identical copy (level_of_theory doesn't vary per conformer), but rendering
# once here means calc_param_md5 below fingerprints the values that were
# ACTUALLY used, not a template file that was never directly fed to ORCA.
rendered_calc_param, level_of_theory = cpt.render(cfg, template_path=template_path)
with open("calcParam.inp", "w") as f:
    f.write(rendered_calc_param)

conformers = [f for f in os.listdir('.') if f.startswith("conf_") and os.path.isdir(f)]

print(conformers)

# Bail out before recording anything. Writing the dft block first (as this
# script used to) left behind a metadata.json that advertised a full level of
# theory for a molecule whose UMA stage had died and produced no conformers
# at all -- the summary then had nothing to flag.
if not conformers:
    sys.exit("ERROR: no conf_* directories found -- the UMA stage produced no "
             "conformers to run DFT on. Check that uma_opt_confs.py completed.")

current_md5 = hashlib.md5(rendered_calc_param.encode()).hexdigest()
dft_record = {
    "engine": "ORCA",
    "orca_module": get_orca_module(),
    "calc_param_template": template_path,
    "calc_param_md5": current_md5,
    "level_of_theory": level_of_theory,
}

# On a resume this block is normally identical to what's already recorded and
# update_metadata leaves the file alone. If it isn't -- workflow_inputs.yaml's
# dft.level_of_theory (or calcParam.inp.tmpl itself) changed since this
# molecule was computed -- don't quietly re-stamp existing conformer results
# with settings they weren't computed under; keep the old hash alongside the
# new one so the summary can flag the molecule. This now catches edits to
# either the template's structure or the YAML values, since both are baked
# into the rendered content this hash is taken from.
existing_dft = metadata_utils.load_metadata(".").get("dft") or {}
previous_md5 = existing_dft.get("calc_param_md5")
if previous_md5 and previous_md5 != current_md5 and existing_dft.get("conformers"):
    print(f"WARNING: the rendered DFT settings for {inchi_key} have changed since "
          f"it was last computed ({previous_md5} -> {current_md5}). Existing "
          f"conformer results were NOT produced with the current settings.")
    dft_record["previous_calc_param_md5"] = previous_md5
    dft_record["settings_changed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

metadata_utils.update_metadata(".", {"dft": dft_record})


# get the conformer energy from UMA to only select conformers withn 8kcal/mol of he min energy conformer 
energies = {}
for f in conformers:
   energy = subprocess.run(f"grep 'energy' {f}/optimized_uma-s-1.xyz | grep -o 'energy=[^ ]*' | cut -d= -f2 | tail -1", capture_output=True, text=True, check=True, shell=True).stdout
   try:
       energies[f] = float(energy)
   except ValueError:
       print(f"could not read energy of {f}/optimized_uma-s-1.xyz, should check that folder")
       energies[f] = np.nan

min_e = min(energies.values())

for f, e in energies.items():
   print(f, e)
   energies[f] = 23.061*(e - min_e)

print(energies)

# launch an ORCA job for each of the conformers
count = 0
for f in conformers:
   if energies[f] != energies[f]:
       print(f"Conformer {f} energy is not defined, likely uma did not converge -- skipping")
       metadata_utils.update_metadata(".", {
           "dft": {"conformers": {f: {"status": "skipped_uma_nonconvergent"}}},
       })
       continue
   elif energies[f] > ewin:
       print(f"Conformer {f} energy: {energies[f]} -- skipping")
       metadata_utils.update_metadata(".", {
           "dft": {"conformers": {f: {
               "status": "skipped_energy_filter",
               "energy_above_min_kcal_mol": energies[f],
               "ewin_kcal_mol": ewin,
           }}},
       })
       continue
   os.chdir(f)
   if 'slurm-calcParam.sh' not in os.listdir('.'):
       # note: the xyz input file to the orca input has to be named input.xyz
       # ../calcParam.inp is the rendered file written once above, at the
       # ligand level -- every conformer gets an identical copy.
       subprocess.run(["cp", "../calcParam.inp", "calcParam.inp"], check=True)
       subprocess.run(["cp", "../../slurm-calcParam.sh", "slurm-calcParam.sh"], check=True)
       # step needed to convert the uma output to a readable xyz file
       subprocess.run("cut -c 1-57 optimized_uma-s-1.xyz > input.xyz", shell=True, check=True)
       sbatch_cmd = ["sbatch"] + sbatch_overrides + ["slurm-calcParam.sh"]
       sbatch_result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
       print(sbatch_result.stdout)
       job_id_match = re.search(r'Submitted batch job (\d+)', sbatch_result.stdout)
       os.chdir('..')
       metadata_utils.update_metadata(".", {
           "dft": {"conformers": {f: {
               "status": "submitted",
               "slurm_job_id": job_id_match.group(1) if job_id_match else None,
               "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "energy_above_min_kcal_mol": energies[f],
               "ewin_kcal_mol": ewin,
           }}},
       })
   else:
       print(f"already a slurm.sh in {inchi_key}/{f} check manually to resubmit orca job")
       os.chdir('..')
   count += 1
