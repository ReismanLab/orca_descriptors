import os
import subprocess

conformers_file = "conformer_file.xyz"                         # provide the name of the xyz file with the conformers

subprocess.run( ["module", "load", "openbabel"], check=True)    # load openbabel on the cluster
subprocess.run(["obabel", conformers_file, "-O", "conf_.xyz", "-m"], check=True )           # segment the file in unique xyzs files
conformers = [f for f in os.listdir('.') if f.startswith("conf_") and f.endswith(".xyz")]   # create a list with all the single xyzs files

print(f"There are {len(conformers)} conformers to compute")

proceed = input("Do you want to proceed with the calculations? (yes/no): ").strip().lower() # call user to avoid launching orca calculation on too many conformers
if proceed not in ['yes', 'y']:
    print("Calculations cancelled.")
    exit()

for f in conformers: # iterate over conformers files to launch an ORCA job for each one of them
   folder = f.split('.')[0]
   if folder in os.listdir('.'):        # if block to avoid resubmitting a job that is already going
       print(f"{folder} is already there skipping")
       continue
   else:
      os.mkdir(folder)                                          # create a new folder for the conformer
      os.chdir(folder)                                          # move to that folder
      print(f"launching orca for {folder}")
      subprocess.run(["cp", f"../{f}", "."], check=True)                # copy the conformer coordinates to the folder
      subprocess.run(["cp", "../calcParam.inp", "."], check=True)       # copy the ORCA input to the folder  
      subprocess.run(["cp", "../slurm-calcParam.sh", "."], check=True)  # copy the slurm.sh input to the folder 
      subprocess.run(["sed", "-i", f"s/inp.xyz/{f}/g", "calcParam.inp"], check=True) # modify the ORCA input to call the correct xyzfile
      subprocess.run(["sbatch", "slurm-calcParam.sh", "."], check=True) # submit the job to the cluster
      os.chdir('..')                                            # move back to parent folder
