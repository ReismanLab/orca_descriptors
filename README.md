# B-CEG-MCD

Benchmarking **C**onformer **E**nsemble **G**eneration and refinement, and their impact on
**M**olecular **D**escriptors used to rationalise reactivity.

This first stage produces, for every input molecule, conformer ensembles from several search
methods, each post-optimised at a common level of theory (UMA by default), in 3 replicates, with
a strict hyper-parameter log. Descriptor extraction and analysis come next.

## Workflow

1. **Inputs**: a list of SMILES (`data/inputs/*.csv`, column `smiles`, optional `name`).
   Each SMILES is canonicalised with RDKit, and the InChIKey of the canonical **input** names its folder.
2. **Transformation** (optional): a user script that turns each input into the structure to sample,
   e.g. N,N ligand → LNiF2 (see [transforms/README.md](transforms/README.md)). `none` = no transformation.
3. **Ensemble search**, one of:
   | searcher | program | notes |
   |---|---|---|
   | `crest` | CREST ≥ 3.1 (TOML input) | iMTD-GC; optional inline `refine: sp` / `refine: opt` ([composite](https://crest-lab.github.io/crest-docs/page/examples/composite.html)), default refinement level **UMA** (`mlip`, `uma-s-1p2`, `omol`) |
   | `goat` | ORCA GOAT | `! GOAT XTB` by default |
   | `molassembler` | SCINE Molassembler | distance geometry from the molecular graph; suited to TM complexes |
   | `rdkit_etkdg` | RDKit | ETKDGv3, optional MMFF/UFF |
   | `loqi` | [LoQI](https://github.com/isayevlab/LoQI) | diffusion model, organics only (skipped as `not_applicable` otherwise) |
4. **Post-search optimisation**: every raw ensemble is optimised with UMA-S (`uma-s-1p2`, fairchem + ASE) by default.
5. Every workflow runs **3 replicates** (`seed = seed_offset + replicate`; searchers that start
   from one geometry get a new ETKDG start geometry per replicate).

## Results layout

```
results/
  molecules.csv                       InChIKey <-> input SMILES index
  <transformation>/                   e.g. none, NiF2, NiH2
    <InChIKey of input SMILES>/
      <method>/                       name given in the config, e.g. crest_gfn2+uma-opt
        raw_1.xyz raw_2.xyz raw_3.xyz                      search output
        raw_1_<lot>.xyz raw_2_<lot>.xyz raw_3_<lot>.xyz    post-optimised, e.g. raw_1_uma-s-1p2.xyz
        params.json                   hyper-parameter log of the search
        params_<lot>.json             hyper-parameter log of the post-optimisation
        run_<i>.json, run_<i>_<lot>.json   per-replicate record (status, seed, versions, timings, host, SLURM ids)
        logs/                         program inputs/outputs (CREST TOML, ORCA input, stdout)
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

```bash
conda env create -f environment.yml && conda activate bceg
huggingface-cli login                 # UMA weights are gated
bceg plan configs/example.yaml        # check logs, list pending tasks, write SLURM arrays
bash results/_jobs/<timestamp>/submit.sh
bceg status configs/example.yaml      # done / failed / missing per method
bceg run configs/example.yaml         # alternative: everything locally, sequentially
```

`bceg plan` writes one job array per search method and one per post-optimisation level. The
post-optimisation arrays depend (`afterany`) on the search arrays. Partitions, GPUs, `setup` lines
(modules, conda) and `tasks_per_job` are set in the `slurm:` block of the config, and methods may override them.

## Development

```bash
pip install -e ".[dev]" && pytest
```

Add a searcher by subclassing `bceg.search.base.Searcher` (declare `defaults` and `resource_keys`) and
registering it in `bceg/search/__init__.py`. Refiners work the same way in `bceg/refine/`.
