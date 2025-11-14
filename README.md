# orca_descriptors
Parsing ORCA outputs for MLR descriptors


1. orca outputs:
   - .out --> NBO
   - .json(s):
   -     job1.json; --> XYZ
   -     job2.json; --> SPE
   -     job3.json; --> NBO
   -     job4.json; --> NMR

2. orca.out file:
   task: extract NBO parameters:
    - atomic descriptors
    - bond descriptors

3. job1.json:
  task: get the xyz in a format that can be plugged into the Sigman property extratction 
   
