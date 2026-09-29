# B-CEG-MCD

Benchmarking **C**onformer **E**nsemble **G**eneration and refinement, and their impact on
**M**olecular **D**escriptors used to rationalise reactivity.

This first stage produces, for every input molecule, conformer ensembles from several search
methods, each post-optimised at a common level of theory (UMA by default), in 3 replicates, with
a strict hyper-parameter log. Descriptor extraction and analysis come next.

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

## Repository

```
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
