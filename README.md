# orca_descriptors
Parsing ORCA outputs for MLR descriptors

## Philosophy:
Elementary block that takes a set of orca outputs for one calculation and generates a json file with extensive set of descriptors.

## Installation:

conda create --name get_desc python==3.12

conda activate get_desc

pip install pandas rdkit morfeus-ml dbstep ipykernel


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
- Level of Theory (parse input files)
- Polarizability    
-  [x] SASA
-  [x] Volume
- [x] HOMO energy
- [x] LUMO energy
- [ ] frontier orbitals (need to adjust for UHF calc.)

  Atomic properties (1 Atom):
  
- NMR
  - [x] aniso
  - [x] iso
- NBO, atomic
- Charges
  - [x] Mulliken
  - [x] Loewdin
  - [ ] NPA
  - [ ] Hirshfeld (needs Hirshfeld keyword)
  - [ ] CM5 (must be calculated)
  - [ ] ChelpG ESP (needs CHELPG ketword or orca_chelpg program)
- atomic dipole
- visible volume
-  [x] accessible surface
  
  Bond properties (2 Atoms):
  
- IR frequencies
- NBO bonds
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
