# Submitting jobs on the cluster

One command takes a list of SMILES and a pre-processing choice and submits everything. Each
cluster job runs the **conformer search** and then the **post-optimisation** of one replicate.

```bash
bceg submit configs/example.yaml --smiles my_ligands.txt --transformation NiF2
```

## 1. One-time setup

```bash
git clone https://github.com/ReismanLab/B-CEG-MCD.git && cd B-CEG-MCD
conda env create -f environment.yml
conda activate bceg
huggingface-cli login          # UMA weights are gated on HuggingFace
pytest                         # quick sanity check
```

External programs (see the comments in `environment.yml`):

| needed for | program | how |
|---|---|---|
| CREST methods, all start geometries | `crest` ≥ 3.1, `xtb` | conda env |
| CREST with inline UMA (`refine: sp/opt`) | `fmlip-relay-server` | `pip install '<crest-src>/subprojects/fmlip_relay[uma]'` |
| GOAT | ORCA | cluster module; absolute path in `orca_cmd` |
| LoQI | `loqi` | separate env; path in `executable` |

Then edit the `slurm:` block of your config (copy `configs/example.yaml`, e.g. to `configs/cluster.yaml`):

```yaml
slurm:
  mode: chained              # each job: search, then post-optimisation of that replicate
  setup:                     # run at the start of every job
    - source ~/miniconda3/etc/profile.d/conda.sh
    - conda activate bceg
    - module load orca       # if you use GOAT
  tasks_per_job: 1
  array_throttle: 50         # max jobs running at once per array
  cpu: {partition: <cpu partition>, time: "24:00:00", cpus_per_task: 8, mem: 16G, account: <account>}
  gpu: {partition: <gpu partition>, time: "24:00:00", cpus_per_task: 8, mem: 32G, gres: "gpu:1", account: <account>}
```

If a step needs a GPU (UMA post-optimisation on `cuda`, or CREST with inline UMA), the job goes to the
`gpu` profile; otherwise it goes to `cpu`. A method can override the profile, e.g.
`crest_gfn2: {searcher: crest, params: {...}, slurm: {time: "48:00:00"}}`.

## 2. Input SMILES

A text file with one SMILES per line and an optional name after it. `#` starts a comment:

```
# my_ligands.txt
c1ccc(-c2ccccn2)nc1                     bpy
CC(C)(C)[C@@H]1COC(c2ccccn2)=N1         tBu-pyox
```

A CSV with a `smiles` column (and optionally `name`) also works. Each SMILES is canonicalised,
and the InChIKey of the input SMILES names its folder. Duplicates are dropped.

## 3. Pre-processing (transformation) and spin

`--transformation` / `-t` picks one or more transformations defined in the config:

| name | what | depth-1 folder |
|---|---|---|
| `none` | input molecule as is | `DATA/none/` |
| `NiF2` | N,N ligand → L-NiF2 (`transforms/ni_x2.py`) | `DATA/NiF2/` |
| `NiH2` | N,N ligand → L-NiH2 | `DATA/NiH2/` |

Ligands that match none of the script's scaffolds (pyox, box, biox, bilm, bipy) are skipped for that
transformation. To add another transformation, write a script (see [transforms/README.md](transforms/README.md))
and add it to the config.

**Spin multiplicity** defaults to 1. It is used by every step: the xtb pre-optimisation (`--uhf`), CREST
and GOAT (`uhf` / multiplicity) and UMA (`spin`). It is also stored in `params.json` and in every
`run_*.json`. To change it:

- permanently, in the config: `NiF2: {script: ..., params: {x: F}, multiplicity: 3}`;
- for one submission: `--multiplicity 3`. If this differs from the configured value, results go to
  `DATA/NiF2_m3/`, so singlet and triplet results never mix.

## 4. Submit

```bash
# see what would run (writes nothing)
bceg submit configs/cluster.yaml --smiles my_ligands.txt -t NiF2 -t NiH2 --dry-run

# submit everything
bceg submit configs/cluster.yaml --smiles my_ligands.txt -t NiF2 -t NiH2

# only some methods, triplet
bceg submit configs/cluster.yaml --smiles my_ligands.txt -t NiF2 -m crest_gfn2,molassembler --multiplicity 3
```

`submit` does the following:
1. Checks every `params.json` that already exists. If a method's hyper-parameters changed, nothing is
   submitted and you get a diff; rename the method in the config.
2. Writes `DATA/_jobs/<timestamp>/` containing:
   - `config.yaml`: the frozen configuration the jobs read, with your selections applied;
   - `chain_<method>.jsonl`: the list of tasks;
   - `chain_<method>.sbatch`: one job array per method;
   - `submit.sh` and `slurm_logs/`.
3. Calls `sbatch` for each array. Use `--no-sbatch` to only write the scripts and run `bash submit.sh` yourself.

Only unfinished work is submitted. Finished replicates are skipped, and failed ones are submitted again.
Re-running the same command after a crash or timeout therefore picks up where it stopped.

## 5. Follow up

```bash
squeue -u $USER
bceg status configs/cluster.yaml --smiles my_ligands.txt -t NiF2    # done / failed / missing per method and step
less DATA/_jobs/<timestamp>/slurm_logs/chain_crest_gfn2_<jobid>_<index>.out
```

If a replicate failed, its `run_<i>.json` holds the error and traceback, and `logs/` holds the program
inputs and outputs. Fix the cause, then run the same `bceg submit` again.

## What each job does

For one (transformation, molecule, method, replicate):

1. Build the structure: canonical SMILES → transformation → charge and multiplicity.
2. Start geometry (CREST, GOAT, molassembler):
   1. ETKDG embedding seeded by the replicate, so the 3 replicates start from different geometries
      (the N-Ni-N angle varies freely).
   2. GFN2-xTB pre-optimisation.
   3. Check that every metal-ligand bond is ≤ `max_metal_ligand_distance` (2.5 Å), no other bond is
      broken and no atoms clash (e.g. H-H). A rejected geometry is re-embedded with a new seed, up to
      `start_max_attempts` (10) times.

   The accepted geometry is saved as `logs/raw_<i>_start_geometry.xyz`, and the seeds and rejections are
   in `run_<i>.json`.
3. Conformer search → `raw_<i>.xyz`.
4. UMA-S post-optimisation of every conformer → `raw_<i>_uma-s-1p2.xyz`.

## Split mode

With `slurm: {mode: split}`, searches and post-optimisations are separate arrays. The post-optimisation
arrays start (`afterany`) once all search arrays have finished. This avoids holding a GPU during long
CPU-only searches.
