# B-CEG-MCD

Benchmarking **C**onformer **E**nsemble **G**eneration and refinement, and their impact on
**M**olecular **D**escriptors used to rationalise reactivity.

This first stage produces, for every input molecule, conformer ensembles from several search
methods, each post-optimised at a common level of theory (UMA by default), in 3 replicates, with
a strict hyper-parameter log. Descriptor extraction and analysis come next.
The roadmap and open decisions are in [PLAN.md](PLAN.md).

## Requirements (project specification)

These requirements define the project; the code implements them.

1. **Input**: a list of SMILES in a text file, submitted together with the pre-processing to apply
   (`NiF2`, `NiH2`, `none`, or something else).
2. **Pre-processing (transformation)**: an optional script that turns each SMILES into the structure to
   compute. Example: a set of N,N ligands and a script that builds the L-NiF2 (or L-NiH2) complex in which
   both ligand nitrogens bind the nickel.
3. **Conformer ensemble search**, chosen by the user: CREST, GOAT, molassembler, RDKit ETKDG or LoQI.
   CREST can be run with the inline single-point or inline geometry-optimisation options
   ([composite workflows](https://crest-lab.github.io/crest-docs/page/examples/composite.html)).
   The default level of theory for these inline steps is **UMA**.
4. **Post-search optimisation**, specified by the user; **uma-s** by default.
5. **Replicates**: every workflow is run **3 times** to evaluate variability, each from its own start
   geometry (a new random embedding per replicate), pre-optimised with GFN2-xTB. The N-Ni-N bite angle
   varies from ligand to ligand, but should be relatively consistent for a single ligand. The Ni-N distance
   must stay within a limit that guarantees coordination (≤ 2.5 Å).
   Identical starts are accepted: for rigid ligands (e.g. bpy) the 3 replicates may start from the same
   geometry, and any variability comes from the searches themselves (CREST's metadynamics, GOAT, Molassembler). Looser pre-optimisation is not used to create
   diversity. Each `run_<i>.json` records the start-geometry RMSDs, the angles at Ni and τ4.
6. **Spin state**: defaults to 1, is easy for the user to change, and is carried through every step of the jobs.
7. **Folder organisation**:
   - depth 1: the transformation applied to the SMILES (e.g. `NiF2`, `NiH2`);
   - depth 2: the InChIKey obtained after canonicalisation of the **input** SMILES;
   - depth 3: the conformer ensemble method.

   The depth-3 folder holds `raw_1.xyz`, `raw_2.xyz`, `raw_3.xyz` (ensemble search) and the sister files
   `raw_X_<lot>.xyz` (post-search optimised at level of theory `<lot>`).
8. **Hyper-parameter log**: each conformer-ensemble folder keeps a log of every hyper-parameter used
   to generate its xyz files. Before a computation is redone, the log is checked. If the parameters match
   exactly, the computation is skipped. Otherwise the method must be given another name.
9. **Cluster**: jobs are submitted to an HPC cluster (SLURM + GPUs). For each input, the conformer search
   and the post-processing run one after the other automatically.
10. **Data**: results go to a `DATA/` folder that is part of the repository layout but never committed.
11. **Next stage**: descriptor extraction to rationalise reactivity. It is not part of the current code.

## Workflow

1. **Inputs**: a text file with one SMILES per line and an optional name (see `inputs/`). Each SMILES
   is canonicalised with RDKit, and the InChIKey of the canonical **input** names its folder.
2. **Transformation** (pre-processing): `none`, `NiF2`, `NiH2` (N,N ligand → L-NiX2, `transforms/ni_x2.py`),
   or your own script (see [transforms/README.md](transforms/README.md)). The spin multiplicity (default 1)
   is set per transformation or with `--multiplicity`, and is passed to every step.
3. **Ensemble search**, one of:
   | searcher | program | notes |
   |---|---|---|
   | `crest` | CREST ≥ 3.1 (TOML input) | iMTD-GC; optional inline `refine: sp` / `refine: opt` ([composite](https://crest-lab.github.io/crest-docs/page/examples/composite.html)), default refinement level **UMA** (`mlip`, `uma-s-1p2`, `omol`) |
   | `goat` | ORCA GOAT | `! GOAT XTB` by default |
   | `molassembler` | SCINE Molassembler | distance geometry from the molecular graph; suited to TM complexes |
   | `rdkit_etkdg` | RDKit | ETKDGv3, optional MMFF/UFF |
   | `loqi` | [LoQI](https://github.com/isayevlab/LoQI) | diffusion model, organics only (skipped as `not_applicable` otherwise) |
4. **Post-search optimisation**: every raw ensemble is optimised with UMA-S (`uma-s-1p2`, fairchem + ASE) by default.
5. Every workflow runs **3 replicates** (`seed = seed_offset + replicate`). CREST, GOAT and molassembler
   start from one geometry: each replicate gets a new ETKDG embedding, pre-optimised with GFN2-xTB, and the
   geometry is accepted only if metal-ligand bonds are ≤ 2.5 Å and no bond is broken or collapsed
   (otherwise it is re-embedded with a new seed).

## Results layout

```
DATA/                                 git-ignored (only DATA/.gitkeep is tracked)
  molecules.csv                       InChIKey <-> input SMILES index
  _jobs/<timestamp>/                  frozen config, task lists, sbatch scripts, SLURM logs
  <transformation>/                   e.g. none, NiF2, NiH2, NiF2_m3 (--multiplicity 3)
    <InChIKey of input SMILES>/
      <method>/                       name given in the config, e.g. crest_gfn2+uma-opt
        raw_1.xyz raw_2.xyz raw_3.xyz                      search output
        raw_1_<lot>.xyz raw_2_<lot>.xyz raw_3_<lot>.xyz    post-optimised, e.g. raw_1_uma-s-1p2.xyz
        params.json                   hyper-parameter log of the search
        params_<lot>.json             hyper-parameter log of the post-optimisation
        run_<i>.json, run_<i>_<lot>.json   per-replicate record (status, seed, versions, timings, host, SLURM ids)
        logs/                         start geometries and program inputs/outputs (CREST TOML, ORCA input, stdout)
```

xyz comment lines start with the energy in Hartree at the level that produced the file. All files in a
method folder share the atom order of `AddHs(MolFromSmiles(<transformed smiles>))`.

Results are keyed by molecule, not by submission: every `bceg submit` (today's ligands, tomorrow's additions,
a re-run after a crash) writes into the same `<transformation>/<InChIKey>/<method>/` tree and only submits
what is not finished there yet. `_jobs/<timestamp>/` is bookkeeping for one submission: the config snapshot
its jobs read, their task lists, sbatch scripts and SLURM logs. Nothing in the results tree refers to it, so
old `_jobs` folders can be deleted once their jobs have finished.

### Hyper-parameter log (skip or rename)

`params.json` stores every parameter that determines the ensemble (all searcher parameters with defaults
filled in, the transformation name and the SHA-256 of its script, charge, multiplicity, and the seeding rule).
Resource settings such as threads, device and executable paths are recorded but not compared. Before
anything runs:

- no log → it is written and the task runs;
- identical log → finished replicates are **skipped**, and missing or failed ones are (re)run;
- different log → the task is **refused** with a field-by-field diff. Give the method a new name in the config.
  Results made with different parameters are never mixed or overwritten.

`params_<lot>.json` does the same for the post-optimisation and is tied to the search parameters' hash.

## Usage

**How to submit jobs on the cluster: see [SUBMITTING.md](SUBMITTING.md).** In short:

```bash
conda env create -f environment.yml && conda activate bceg
huggingface-cli login                                                  # UMA weights are gated
bceg submit configs/example.yaml --smiles inputs/example_NN_ligands.txt -t NiF2 --dry-run
bceg submit configs/example.yaml --smiles inputs/example_NN_ligands.txt -t NiF2
bceg status configs/example.yaml --smiles inputs/example_NN_ligands.txt -t NiF2
bceg run    configs/example.yaml --smiles inputs/example_NN_ligands.txt -t none -m rdkit_etkdg   # locally
```

Each cluster job runs the conformer search of one replicate and then its UMA post-optimisation.

## Status

- Implemented and unit-tested: inputs and canonicalisation, NiF2/NiH2 transformation, folder layout,
  hyper-parameter logs, start-geometry checks, ETKDG searcher, task expansion, SLURM script generation, `submit`.
- Run for real on the development machine (conda env `crest`: CREST 3.0.2 + xtb 6.7.1):
  - GFN2-xTB pre-optimisation of NiF2/NiH2 × {bpy, tBu-pyox} × 3 replicates, all accepted on the first
    attempt (Ni-N 1.87-1.97 Å, Ni-F 1.73 Å, Ni-H 1.41 Å);
  - a full `bceg run` with `crest_gfn2` on ethanol, which gave the documented layout, and a rerun that skipped everything.
  - **But the local CREST 3.0.2 build is unreliable**: for 1-hexanol it optimizes 0 of 1106 snapshots
    (floating-point exceptions) with 1 or 4 threads, and writes no ensemble. The pipeline marks such
    replicates `failed`. Use the cluster build for real runs.
- **Not yet run**: CREST 3.1 with inline UMA (`mlip`), GOAT, molassembler, LoQI and the UMA post-optimisation.
  Things to verify on the first cluster run are listed in [PLAN.md](PLAN.md).

To run the tests locally, use any env with rdkit, pandas and pyyaml. Put the `crest` env's `bin/` on
`PATH` to also run the real-xtb tests:

```bash
PATH=<conda>/envs/crest/bin:$PATH PYTHONPATH=src python -m pytest -q
```

## Repository

```
README.md             this file (requirements, workflow, layout)
SUBMITTING.md         how to submit jobs on the cluster
PLAN.md               roadmap, decisions, open items
configs/              configurations (methods, transformations, post-optimisation, cluster)
inputs/               SMILES lists
transforms/           pre-processing scripts (ni_x2.py -> NiF2 / NiH2)
scripts_4_structure/  original stand-alone NiF2 / NiH2 scripts (reference; ni_x2.py replaces them)
src/bceg/             the package
tests/                pytest suite
DATA/                 results (git-ignored)
```

## Development

```bash
pip install -e ".[dev]" && pytest
```

Add a searcher by subclassing `bceg.search.base.Searcher` (declare `defaults` and `resource_keys`) and
registering it in `bceg/search/__init__.py`. Refiners work the same way in `bceg/refine/`.
