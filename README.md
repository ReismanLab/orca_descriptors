# orca_descriptors

Parsing ORCA and Gaussian outputs for MLR descriptors and perform subsequent modeling.

## References

This repository necessitates the installation of two other repositories:

1. [DescriPyTor](https://github.com/edenspec2/DescriPyTor): for the generation of molecular descriptors and their visualization.

2. [ROBERT](https://github.com/jvalegre/robert): for the generation and validation of MLR models.

This repository is designed after the strategy developped by the Sigman Group for descriptor selection that can be found on their [Get_Properties](https://github.com/SigmanGroup/Get_Properties) repository. Of course we use [morfeus-ml](https://github.com/digital-chemistry-laboratory/morfeus) for extraction of steric parameters.

## Installation

We provide a single environemment to run all the notebooks that can be installed using the following commands 

```
git clone https://github.com/ReismanLab/orca_descriptors
conda create -n desc_mlr python=3.12
conda activate desc_mlr
pip install tqdm bokeh openbabel-wheel rdkit morfeus-ml dbstep pathlib ipykernel openpyxl robert igraph networkx plotly ipywidgets dash pyarrow
conda install -y -c conda-forge glib gtk3 pango mscorefonts
```


# The repository is organized as follow:

1. [DATA](DATA): examples of calculation outputs
2. [desc_utils](desc_utils): contains (a) script for the automated numbering of the substructures, you might want to tune that depending on the molecules you are studying... (b) descriptors extraction for ORCA outputs.
3. [mlr_utils](mlr_utils): in-house scripts for multi-variate regression feature selection.
4. [scripts-for-orca-computations](scripts-for-orca-computations): in-house scripts to launch the ORCA jobs & conformer searches on the cluster.
5. [orca2gaussian](orca2gaussian): in-house scripts to convert ORCA jobs to a pseudo log Gaussian that can be read by DescriPyTor utils.

## Pipeline

1. If needed ORCA 2 Gaussian conversion.
2. Atom numbering + DescriPyTor descriptor extraction.
3. Extraction of conformer averaged descriptors (_min, _max, _low_E and _Boltz). 

## Descriptors included

- Numbering atoms: -> make it a python file

- Orca descriptor extraction (single output): Elementary block that takes a set of orca outputs for one calculation and generates a json file with extensive set of descriptors.
- Orca calculation

- Notebook:
   input: specify a folder with conformers computed
   - get the smiles
   - check that the orca file converged
   - get the atoms numbers for atomic descriptors
   - extract descriptors from each conformer
   - ensemble to get the overall descripotrs (_min, _max, _low_E, _Boltz)

## Installation:

conda create --name get_desc python==3.12

conda activate get_desc

pip install pandas rdkit morfeus-ml dbstep ipykernel periodictable


1. orca outputs:
   - .out --> NBO
   - optimization.xyz --> XYZ
   - .json(s):
   -     job2.json; --> SPE
   -     job3.json; --> NBO
   -     job4.json; --> NMR

2. orca.out file:
   task: extract NBO parameters:
    - atomic descriptors
    - bond descriptors

3. job1.json:
  task: get the xyz in a format that can be plugged into the Sigman property extratction 


List of properties:
- [x] SCF Energy
- [x] Enthalpy (need to check after rerunning to see if json parses right)
- [x] Level of Theory
- Polarizability    
-  [x] SASA
-  [x] Volume
- [x] HOMO energy
- [x] LUMO energy
- [x] frontier orbitals

  Atomic properties (1 Atom):
  
- NMR
  - [x] aniso
  - [x] iso

- Charges
  - [x] Mulliken
  - [x] Loewdin
  - [x] NPA
  - [x] Hirshfeld
  - [x] CM5 (NEEDS TO BE Checked)
  - [x] ChelpG ESP
- atomic dipole
- visible volume
-  [x] accessible surface
  
  Bond properties (2 Atoms):
  
- IR frequencies
- [x] NBO bonds
-  [x] Sterimol L, B1, B5
-  [x] Bond distances
- Bond occupancy?
- Antibonding occupancy?
- SASA metal? different for metals than other atoms?

   Angles (3 Atoms):
  
-  [x] angles
- cone angles

   4 Atoms:
  
-  [x] pyramidalization
-  [x] dihedral angles

   5 Atoms:

-  [x] quadrants
-  [x] octants
