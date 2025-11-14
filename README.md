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
   
