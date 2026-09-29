# Submitting jobs on the cluster

One command takes a list of SMILES and a pre-processing choice and submits everything. Each
cluster job runs the **conformer search** and then the **post-optimisation** of one replicate.

```bash
bceg submit configs/example.yaml --smiles my_ligands.txt --transformation NiF2
```

## 1. One-time setup

### Use the shared environment

The group maintains one environment at `/groups/ReismanLab/miniconda3/envs/bceg`. **Do not run `conda env create`**:
the environment is read-only for everyone but its maintainer, and you do not need to build or update it.

```bash
source /groups/ReismanLab/miniconda3/etc/profile.d/conda.sh
conda activate bceg
bceg --help
huggingface-cli login     # once per user: the UMA weights are gated and the token is per-account
```

The plain name works because the env lives inside the shared conda you just sourced. If you use a conda
of your own instead, point it at the shared envs directory once:

```bash
conda config --append envs_dirs /groups/ReismanLab/miniconda3/envs
```

The `bceg` package is part of that environment, installed from the shared clone at
`/groups/ReismanLab/B-CEG-MCD`. Cloning the repository yourself is optional and only gives you
`configs/` and `inputs/` to copy from — **editing `src/` in a clone of your own changes
nothing about what runs**. To work on the code, build your own environment (see
[Development](README.md#development)).

External programs, none of which the shared environment can provide for you:

| needed for | program | how |
|---|---|---|
| CREST methods, all start geometries | `crest`, `xtb` | in the shared env (conda-forge `crest`, currently 3.0.2) |
| CREST with inline UMA (`+uma` methods, `refine: sp/opt`) | `crest` ≥ 3.1 and `fmlip-relay-server` | **unreleased**, so these methods are commented out in `configs/example.yaml`. 3.1 exists only on the open PR [crest-lab/crest#483](https://github.com/crest-lab/crest/pull/483) (branch `pprcht/crest@experimental`). To use it: build that branch with `-Dfmlip-relay=enabled`, `pip install '<crest-src>/subprojects/fmlip_relay[uma]'` into the env, then set `params: {executable: /abs/path/to/crest-3.1}` on those methods |
| GOAT | ORCA | cluster module; absolute path in `orca_cmd` |
| LoQI | `loqi` | separate env; path in `executable` |

### Your config

Copy a config into a directory you own and edit it:

```bash
cp /groups/ReismanLab/B-CEG-MCD/configs/example.yaml ~/cluster.yaml
```

Paths inside the config are relative **to the config file**, so a copy keeps working as long as the
files it points at travel with it -- put your SMILES list next to it and `inputs: my_ligands.txt` is
enough. The `NiF2` / `NiH2` `script:` entries need nothing at all: they name a script shipped inside
the package, so they resolve from any directory. (`--smiles` is the one exception to the rule: it is
resolved against your current directory, not the config.)

Two things must change. `results_dir:` should be an absolute path you own, and the `slurm: setup:`
lines must activate the shared environment, because a compute node starts from a bare shell:

```yaml
results_dir: /groups/ReismanLab/DATA     # absolute; group-writable and setgid if users share results
slurm:
  mode: chained              # each job: search, then post-optimisation of that replicate
  setup:                     # run at the start of every job
    - source /groups/ReismanLab/miniconda3/etc/profile.d/conda.sh
    - conda activate bceg
    - module load orca/6.1.0   # if you use GOAT; the version is mandatory, and GOAT needs ORCA >= 6.0
  tasks_per_job: 1
  array_throttle: 50         # max jobs running at once per array
  cpu: {partition: <cpu partition>, time: "24:00:00", cpus_per_task: 8, mem: 16G, account: <account>}
  gpu: {partition: <gpu partition>, time: "24:00:00", cpus_per_task: 8, mem: 32G, gres: "gpu:1", account: <account>}
```

The UMA post-optimisation runs on the CPU by default, so the `gpu:` profile above is optional: define it
only if you set `post_optimization.params.device: cuda` (much faster) or use CREST with inline UMA. A step
that needs a GPU goes to the `gpu` profile; everything else goes to `cpu`. In `chained` mode a GPU step
makes its whole job a GPU job, CREST search included -- use `mode: split` to avoid holding a GPU through
the search.

Thread counts (`threads` for CREST and ETKDG, `nprocs` for GOAT, the xtb pre-optimisation) default to the
job's `cpus_per_task` (`SLURM_CPUS_PER_TASK`). In chained mode,
`post_optimization.slurm` and the method's `slurm` overrides both apply, the method's taking precedence. A method can override the profile, e.g.
`crest_gfn2: {searcher: crest, params: {...}, slurm: {time: "48:00:00"}}`.

### Creating or updating the shared environment (maintainer only)

One person owns the environment. A conda env that several people can `pip install` into drifts and
breaks silently, and concurrent `conda env create` runs corrupt the shared package cache in ways whose
error messages point nowhere near the cause.

Clone the repo and build the env in group space, on a filesystem the compute nodes mount (a home
directory is usually `700`, so an env under `~` is unreachable for the others even when the env itself
is readable):

```bash
export SHARED=/groups/ReismanLab
git clone git@github.com:ReismanLab/B-CEG-MCD.git $SHARED/B-CEG-MCD
source $SHARED/miniconda3/etc/profile.d/conda.sh
conda env create -f $SHARED/B-CEG-MCD/environment.yml
```

`name: bceg` in `environment.yml` puts it at `$SHARED/miniconda3/envs/bceg`, which is already in that
conda's default `envs_dirs` — that is why users get `conda activate bceg` without configuring anything.

`environment.yml` installs the package with `pip install -e .`, so the env points at the clone it was
created from: use the shared clone, not a personal one, and `git pull` there to update every user at
once. To decouple the two instead, reinstall non-editably after creating the env, and repeat that line
whenever you publish an update:

```bash
conda run -n bceg pip install --no-deps --force-reinstall $SHARED/B-CEG-MCD
```

Users need `r` on the files and `x` on every directory above them, and nothing more — a group-writable
env is a writable `bin/python` and `bin/crest` that everyone executes on the compute nodes:

```bash
env=$SHARED/miniconda3/envs/bceg
chgrp -R hpc_ReismanLab $env $SHARED/B-CEG-MCD
chmod -R g+rX,o-rwx $env $SHARED/B-CEG-MCD
find $env $SHARED/B-CEG-MCD -type d -exec chmod g+s {} +
namei -l $env/bin/bceg     # every component must be traversable
```

Scope these to the `bceg` env, not to all of `$SHARED/miniconda3/envs` — other members keep their own
environments there and you do not own those files.

If a second person must also install into the shared conda prefix, the thing that actually bites is file
*creation mode*: with the usual `umask 027` their files land `640`, and the next person cannot write
them — which is what `NotWritableError` on a path under `pkgs/` means. A cache file is safe to delete
(`find $SHARED/miniconda3/pkgs/cache -type f -delete`) and conda re-downloads it. To stop it recurring,
put a **default** ACL on the directories you own; it is inherited by files anyone creates there
afterwards and overrides their umask:

```bash
find $SHARED/miniconda3 -user $USER -type d -print0 | xargs -0 setfacl -d -m g:hpc_ReismanLab:rwX
find $SHARED/miniconda3 -user $USER ! -type l -print0 | xargs -0 setfacl -m g:hpc_ReismanLab:rwX
```

Default ACLs are not retroactive, so files that already exist keep their mode; `find $SHARED/miniconda3
! -user $USER` lists them. Only their owner or root can change those.

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
| `NiF2` | N,N ligand → L-NiF2 (`ni_x2.py`, shipped with bceg) | `DATA/NiF2/` |
| `NiH2` | N,N ligand → L-NiH2 | `DATA/NiH2/` |

Ligands that match none of the script's scaffolds (pyox, box, biox, bilm, bipy) are skipped for that
transformation. To add another transformation, write a script (see
[src/bceg/transforms_builtin/README.md](src/bceg/transforms_builtin/README.md))
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

`-t` and `-m` **select from** what the config defines; they cannot add anything, and naming something
the config does not have is an error. Keep the config exhaustive and filter here rather than commenting
blocks out of the YAML. `--multiplicity` is the one flag that overrides rather than filters, and it
sends results to `<transformation>_m<N>/`.

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

Each `run_<i>.json` also records what the step cost — `wall_time_s`, `cpus`, `cpu_hours`, and `gpus` /
`gpu_hours` with the GPU model on a GPU node — so a method's price can be totalled from the results tree
long after the jobs have left SLURM's accounting database (see [README](README.md#compute-usage)).

If a replicate failed, its `run_<i>.json` holds the error and traceback. `logs/raw_<i>_failed/` holds the
files of its working directory, including every xtb pre-optimisation attempt (`preopt_<n>/`). Fix the
cause, then run the same `bceg submit` again.

To try things before the cluster, `bceg run` executes the same tasks locally, e.g. with the `crest` conda env
(CREST 3.0.2 + xtb) on `PATH`: `bceg run configs/example.yaml --smiles in.txt -t none -m crest_gfn2`.
CREST 3.0.2 has no `mlip` level, so the `+uma` methods cannot run either locally or on the cluster; they
are commented out in `configs/example.yaml` until CREST 3.1 is released (see the table in §1).

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

   The accepted geometry is saved as `logs/raw_<i>_start_geometry.xyz`. `run_<i>.json` records the seeds,
   the rejections, the RMSD of the pre-optimised start to the raw embedding and to the other replicates'
   starts, and the angles and τ4 at the metal (0 = square planar, 1 = tetrahedral).
3. Conformer search → `raw_<i>.xyz`.
4. UMA-S post-optimisation of every conformer → `raw_<i>_uma-s-1p2.xyz`.

## Split mode

With `slurm: {mode: split}`, searches and post-optimisations are separate arrays. The post-optimisation
arrays start (`afterany`) once all search arrays have finished. This avoids holding a GPU during long
CPU-only searches.
