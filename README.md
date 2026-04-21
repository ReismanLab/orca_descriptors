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

### Orca descriptors:

```X``` : atom label

```Y``` : integer for the radius by default from 2.0 to 5.0 with a 0.5 increment.

**Fragment specific geometric descriptors:**

- %Vbur:                    ```%Vbur_atom_X_radius_Y``` :white_check_mark:
- atomic SASA:              ```SASA_X``` :white_check_mark:
- Pyramidalization:         ```Pyr_X``` :white_check_mark:
- Pyramidalization angle:   ```Pyr_angle_X``` :white_check_mark:
- Sterimol_B5:              ```Sterimol_X_X_Bmax``` :white_check_mark: / do not match DescripyTor B5:```dist_X-X``` :x:
- Sterimol_B1:              ```Sterimol_X_X_Bmin``` :white_check_mark: / do not match DescripyTor B1:```dist_X-X``` :x:
- Sterimol_L:               ```Sterimol_X_X_L``` :white_check_mark:
- Bond length:              ```Bond_X_X``` :white_check_mark: / matches DescripyTor Bond length:```dist_X-X``` :white_check_mark:
- Angles:                   ```Angle_X_X_X``` :white_check_mark: / matches DescripyTor Angles:```angle_X-X-X``` :white_check_mark:
- Dihed:                    ```Dihed_X_X_X_X``` :white_check_mark:
- Quadrants:
- Octants:

**Molecular geometric descriptors:**

- SASA: ```SASA``` :white_check_mark:
- SASA_Volume: ```SASA_Volume``` :white_check_mark:

**Charges**

- Mulliken: ```mulliken_X``` :white_check_mark:
- Loewdin: ```loewdin_X``` :white_check_mark:
- chelpG: ```chelpG_X``` :white_check_mark:
- Hirshfeld: ```hirshfeld_X``` :white_check_mark:
- cm5: ```cm5_X``` :white_check_mark:
- NBO: ```npa_X_charge``` :white_check_mark: / matches Descripytor NBO charge: ```charge_NBO_X``` :white_check_mark:

**Orbital Analysis**

- NBO Core/Valence/Rydberg: ```npa_X_cor/valence/rydberg``` :white_check_mark:
- NBO "Natural Occupancies": ```nao_X_Cor/Val/Ryd-ns/p/d/f_occupancy``` :white_check_mark:
- Frontier orbital descriptors

**Fukui coefficients (using Z charges)**
Z can be mulliken, loewdin, chelpG, hirshfeld, cm5 or NBO.
- fplus_Z_X: ```fplus_Z_X``` :white_check_mark:
- fminus_Z_X: ```fminus_Z_X``` :white_check_mark:
- fzero_Z_X: ```fzero_Z_X``` :white_check_mark:

**NMR isotropic and anistropic**

- isotropic shift: ```siso_X``` :white_check_mark:
- anisotropic shift: ```saniso_X``` :white_check_mark:

**Molecular descriptors:**

- HOMO energy: ```HOMO_energy``` :white_check_mark:
- LUMO energy: ```LUMO_energy``` :white_check_mark:
- Enthalpy: ```Enthlapy``` :white_check_mark:
- GibbsFreeEnergy: ```GibbsFreeEnergy``` :white_check_mark:
- Vertical_Ionization_Potential: ```Vertical_Ionization_Potential``` :white_check_mark:
- Vertical_Electron_Affinity: ```Vertical_Electron_Affinity``` :white_check_mark:
- Hardness: ```Hardness``` :white_check_mark:
- Electronegativity: ```Electronegativity``` :white_check_mark:
- Electrophilicity: ```Electrophilicity``` :white_check_mark:

### DescriPytor:

**Fragment specific geometric descriptors:**

- Sterimol_B1: ```B1_X-X``` :white_check_mark: / does not match Orca descriptors Sterimol_B1 ```Sterimol_X_X_Bmin```
- Sterimol_B5: ```B5_X-X``` :white_check_mark: / does not match Orca descriptors Sterimol_B5 ```Sterimol_X_X_Bmax```
- Sterimol_L: ```L_X-X``` :white_check_mark: / almost matches Orca descriptors Sterimol_L ```Sterimol_X_X_L```
- Sterimol_B5_loc: ```loc_C5_X-X``` :white_check_mark:
- Sterimol_B1-B5_angle: ```B1-B5_angle_X-X``` :white_check_mark:
- Bond length:```dist_X-X``` :white_check_mark: / matches Orca descriptors Bond length:              ```Bond_X_X``` :white_check_mark:
- Angles:```angle_X-X-X``` :white_check_mark: / matches Orca descriptors Angles:                   ```Angle_X_X_X``` :white_check_mark:

**Charges**

- NBO charge: ```charge_NBO_X``` :white_check_mark: / matches Orca descriptors NBO: ```npa_X_charge``` :white_check_mark: 

**Vibrations**

- Vibration frequency:```vib_freq_X-X``` :white_check_mark:
- Vibration amplitude:```vib_amp_X-X``` :white_check_mark:

**Dipoles**

- Dipole norm:```dipole``` :white_check_mark:
- Dipole x projection (after basis transformation):```dipole_x_X-X-X``` :white_check_mark:
- Dipole y projection (after basis transformation):```dipole_y_X-X-X``` :white_check_mark:
- Dipole z projection (after basis transformation):```dipole_z_X-X-X``` :white_check_mark:


## Description of the ORCA jobs


## Remaining list of descriptors to be checked

List of properties remaining:
- [x] SCF Energy
- [x] Enthalpy (need to check after rerunning to see if json parses right)
- [x] Level of Theory
- [x] Polarizability    

- Charges
- [x] CM5 (NEEDS TO BE Checked)
- [x] atomic dipole
- [x] visible volume
  
  Bond properties (2 Atoms):
  
- [x] Bond occupancy?
- [x] Antibonding occupancy?

   5 Atoms:

-  [x] quadrants
-  [x] octants
