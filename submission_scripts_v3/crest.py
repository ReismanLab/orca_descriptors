import os
import re
import subprocess
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdChemReactions
import argparse
import metadata_utils
sys.path.append('/resnick/groups/ReismanLab/JCS/scripts/')
import gaussian_scripts as gs

parser = argparse.ArgumentParser(description="Run CREST conformer search for a ligand")
parser.add_argument("--lig", type=str, required=True, help="SMILES string of the molecule")
# --- previously hardcoded in the gs.gaussian_scripts(...) call below; now
#     chosen via workflow_inputs.yaml's `crest:` block and passed in here. ---
parser.add_argument("--charge", type=int, required=True)
parser.add_argument("--spin-multiplicity", type=int, required=True)
parser.add_argument("--ewin", type=float, required=True,
                     help="CREST's own conformer energy window, kcal/mol")
parser.add_argument("--n-conf", type=int, required=True)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--norotmd", action="store_true")
parser.add_argument("--noreftopo", action="store_true")
parser.add_argument("--cores", type=int, required=True)
args = parser.parse_args()

ligand   = args.lig
inchi_key = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(ligand))
os.chdir(inchi_key)

job = gs.gaussian_scripts(
    charge=args.charge,
    spin_multiplicity=args.spin_multiplicity,
    ewin=args.ewin,
    quick='--quick' if args.quick else '',
    norotmd='--norotmd' if args.norotmd else '',
    n_conf=args.n_conf,
    func='B3LYP', bs_light='6-31G(d)', bs_heavy='LANL2DZ',  # unused by the CREST/xtb path; left as
                                                             # this class's defaults, not pipeline choices
    spe=False,
    node=args.cores,
    nnlibrary=True,
    noreftopo=args.noreftopo,
)

ran_crest = 'crest_clustered.xyz' not in os.listdir('.')
if ran_crest:
    job.CREST("output.xyz")
else:
    print("crest_clustered.xyz already computed")
    print("")

def get_crest_version():
    """`crest --version` prints a ~30 line banner (ASCII art, citations,
    warranty). Pull just the number out of its
    `Version 2.12,   Thu 19. Mai 16:32:32 CEST 2022` line so metadata.json
    records "2.12" rather than the whole block."""
    try:
        proc = subprocess.run(["crest", "--version"], capture_output=True, text=True)
    except Exception:
        return None
    banner = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r'^\s*Version\s+([^\s,]+)', banner, re.MULTILINE)
    if m:
        return m.group(1)
    # unrecognised banner format -- keep a single identifying line, never the whole blob
    for line in banner.splitlines():
        if "ersion" in line:
            return line.strip()
    return None


# Only record the crest block when CREST actually ran, or when there is no
# block yet to back-fill. On a resume the existing block describes the run
# that produced crest_clustered.xyz; re-deriving it here would re-stamp those
# conformers with today's crest version and today's settings, which is a
# claim about a run that didn't happen.
if ran_crest or not metadata_utils.load_metadata(".").get("crest"):
    metadata_utils.update_metadata(".", {
        "crest": {
            "method": "GFN2-xTB",
            "crest_version": get_crest_version(),
            "charge": job.charge,
            "spin_multiplicity": job.spin_multiplicity,
            "ewin_kcal_mol": job.ewin,
            "n_conf": job.n_conf,
            "quick": bool(job.quick),
            "norotmd": bool(job.norotmd),
            "noreftopo": job.noreftopo,
        },
    })
else:
    print("crest metadata already recorded for these conformers; leaving it unchanged")

os.chdir('..')

#job.OPTIMIZE()
#job.SUBMIT()
