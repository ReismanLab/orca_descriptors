# Running a library: config, `orcadesc submit`, `status`, records

One library is one **library directory** `$LIB` and one config file with one block per stage. One
command runs, for every SMILES of the input list, the six stages: `transformation`, `pre_refinement`,
`conformer_search` and `post_optimization` (the conformer ensemble, run by `bceg`), `pruning` (which
conformers go on) and `dft` (one ORCA compound job per conformer). Everything you choose is in that one
YAML; everything the run discovers is in per-molecule records, and the records decide what is left to
do. The code is never copied: jobs run from the shared clone, whose git commit is recorded.

Names used below: `<LIB>` is the library name (one campaign of ORCA jobs, e.g. `LNiH2_library_002`), `$LIB`
the full path of its folder on the cluster, laid out as `<LIB>/<InChIKey>/conf_N/`.

## 1. Write the config

Copy the example next to your SMILES list and edit it:

```bash
mkdir -p ~/libraries/<LIB> && cd ~/libraries/<LIB>
cp /groups/ReismanLab/orca_descriptors/configs/orca_library_example.yaml <LIB>.yaml    # ensembles + ORCA, one file
nano ligands.txt                  # one SMILES per line, optional name after it; # starts a comment
nano <LIB>.yaml                   # table below; relative paths are relative to the file
```

Each stage document lists every option the stage covers, every parameter with its default and
whether it is hashed, and an example block per option.

| Block | Stage | Content |
|---|---|---|
| `library_dir`, `smiles`, `ensembles_dir`, `run_bceg` | | `$LIB` (absolute), the input list, where the ensembles go (default `$LIB/ensembles`), and `run_bceg: false` when they exist already |
| `transformation` | 1, [stages/1_transformation.md](stages/1_transformation.md) | the SMILES as is (`none`) or a script that turns it into the structure computed (`ni_x2.py` with `params: {x: H}` gives L-NiH2; your own script works too); `name` is the depth-1 folder of the ensembles; `charge` and `multiplicity` apply to every later stage |
| `pre_refinement` | 2, [stages/2_pre_refinement.md](stages/2_pre_refinement.md) | the pre-optimisation of the start geometry handed to CREST, GOAT or molassembler: `xtb` (default, `gfn`, `level`), `uma` (model, `fmax`, ...) or `none` (required for `rdkit_etkdg` and `loqi`); the embedding checks (`max_metal_ligand_distance`, `max_attempts`) |
| `conformer_search` | 3, [stages/3_conformer_search.md](stages/3_conformer_search.md) | `name` (the method folder; **the name is the parameter set**: every parameter is logged with its defaults in the folder's `params.json`, and changing one under an existing name is refused, you rename it), `searcher` (`crest`, `goat`, `molassembler`, `rdkit_etkdg`, `loqi`), its `params`, `replicates` (default 1: how many searches per molecule, each from its own seed; the ORCA stage pools them), and `slurm` overrides for the step |
| `post_optimization` | 4, [stages/4_post_optimization.md](stages/4_post_optimization.md) | every conformer re-optimised with `uma` (`model`, `fmax`, `max_steps`, `device`, `workers`), `gxtb` (g-xTB, `level`, `executable`) or `aimnet2` (`model`, `fmax`, `device`, `workers`), or `null` to send the search output to ORCA as it is |
| `pruning` | 5, [stages/5_pruning.md](stages/5_pruning.md) | `energy_window_kcal_mol`: `null` (default) sends every conformer to ORCA, a number keeps those within the window of the ensemble minimum on the post-optimisation energies |
| `orca`, `dft` | 6, [stages/6_dft.md](stages/6_dft.md) | `orca: off` stops after the ensembles (the same command later with `on` refines them); `dft.level_of_theory`: which compound steps run (`opt_geom`, `freq_calc`, `nmr_calc`, `vertical_redox`), method, basis and solvent of each (`opt_method`, `opt_solvent`, `svp_method`, `property_solvent`, `nmr_method`, `nbo_run`), `extra_basis` (`{Ni: def2-tzvp}`). No charge or multiplicity: both come from the ensemble |
| `slurm` | | `setup` lines run at the start of every job (activate `gp_env`, `module load orca`, the NBO variables, the weight caches of [cluster_setup.md](cluster_setup.md)); `ensembles` (bceg's jobs: `mode`, `tasks_per_job`, `cpu` and `gpu` profiles); `prepare` and `orca` profiles (`time`, `mem_per_cpu`, `ntasks`, `nodes`, `partition`, ...; `ntasks` becomes `%pal nprocs`, `mem_per_cpu` gives `%maxcore` at 75 %); `array_throttle` |
| `condense` | | write `conf_N/orca_condensed.json` when each ORCA job ends ([condense.md](condense.md)) |

The six stage blocks define the results: they are written to `$LIB/params.json` by the first `submit`
(with the searcher's and refiner's defaults filled in), and a later submit with other values is refused
with a diff. A different choice in any of them is a different `library_dir`. `orca`, `run_bceg`, the
paths, `slurm` and `condense` are recorded but may change freely. The shipped example carries the level
of theory of the LNiH2 library. The compound job's structure is
[desc_utils/submit/calcParam.inp.tmpl](../desc_utils/submit/calcParam.inp.tmpl); give `dft.calc_param_template`
your own copy only to change which steps exist.

## 2. Submit

```bash
conda activate gp_env
orcadesc submit <LIB>.yaml --dry-run     # what would run: molecules per state, bceg tasks pending; writes nothing
orcadesc submit <LIB>.yaml               # or: python -m desc_utils.submit submit <LIB>.yaml
```

`submit` prints the pipeline on one line, writes `$LIB/params.json` (first time), freezes the resolved
config, the template and the git commit under `$LIB/_jobs/<timestamp>/`, runs bceg's submission for the
input list (stages 1 to 4; its job arrays, scripts and logs land under `_jobs/<timestamp>/bceg/`), then
queues a `prepare` array, one element per molecule, `afterany` the bceg jobs. Each prepare element reads
the molecule's ensemble `raw_<i>_<lot>.xyz` (every replicate, pooled in order), checks its `params.json` hash,
prunes it (stage 5), splits it into `$LIB/<InChIKey>/conf_N/input.xyz`, renders `calcParam.inp` with the ensemble's
charge and multiplicity, writes the record, and submits that molecule's ORCA array, one element per
conformer (stage 6). Each ORCA element records its outcome and condenses.

Running the same command again is how a library grows or recovers: molecules recorded `complete` or
`in_progress` are skipped, the others are prepared again, and inside a molecule a `completed` conformer is
never resubmitted while a `failed` or `timeout` one is. `--force` also takes the molecules marked
`in_progress` (a job killed before it wrote its end leaves them there). An ensemble regenerated by bceg
with other parameters is refused for a library built on the old one (`ensemble_changed`).

## 3. Check the jobs

```bash
orcadesc status <LIB>.yaml                                                                       # per molecule: one status per stage, conformer counts
orcadesc cost <LIB>.yaml [--csv cost.csv]                                                        # per molecule: core-hours and GPU-hours per stage and per ORCA step, from the records
squeue -u $USER                                                                                  # what SLURM really holds
python3 /groups/ReismanLab/orca_descriptors/scripts-for-orca-computations/check_jobs.py $LIB      # per conformer, from squeue and the ORCA outputs
python3 -m json.tool $LIB/<InChIKey>/metadata.json                                               # the full record of one molecule
less $LIB/_jobs/<timestamp>/slurm_logs/orca_prepare_<jobid>_<index>.out                          # a prepare log; orca_<InChIKey>_... for ORCA
```

`status` prints one row per molecule with one column per stage (`transf`, `preopt`, `search`,
`post_opt`, `pruning`, `dft`), then `n_conf` (ensemble size), `n_dft` (conformers kept / total),
`dft_done` (completed / kept) and the molecule `state`, followed by per-stage counts and a `needs a
look` list.

| Stage value | Meaning |
|---|---|
| `queued` | the job that runs this stage is submitted (bceg's for 2-4, the prepare or ORCA array for 5-6) and has not reached it |
| `running` | the stage is running now (for 2 and 3, from the `phase` the bceg task records; for 6, any conformer `running`) |
| `done` | finished; for `dft`, every kept conformer `completed` |
| `failed` | the stage failed (error in `reason`); for `dft`, a conformer `failed`, `timeout` or never submitted; for `pruning`, the ensemble changed under the library |
| `n/a` | the pipeline has no such step for this molecule (no pre-refinement for `rdkit_etkdg`, no post-optimisation, a transformation that does not apply: every later stage is `n/a`) |
| `-` | nothing recorded and nothing queued |
| `off` | `pruning` and `dft` with `orca: off` |

| Molecule state | Meaning |
|---|---|
| `absent` | nothing recorded yet |
| `in_progress` | prepare queued, or ORCA jobs `submitted` or `running` |
| `incomplete` | ensemble missing, failed or not applicable, or a conformer `failed`, `timeout`, or never submitted |
| `complete` | every conformer `completed` or pruned |

A conformer is `completed` only if ORCA exited 0 *and* printed `ORCA TERMINATED NORMALLY`; a walltime
kill is recorded as `timeout`. States come from the records, not from `squeue`: `check_jobs.py` shows
whether anything is really queued (`QUEUED`, `RUNNING`, `OK`, `FAILED` with the reason, `CHECK`,
`NOT STARTED`). What to do about each failure is in [troubleshooting.md](troubleshooting.md).

## 4. What is recorded, and how to reproduce a library

`$LIB/params.json` holds the six stage blocks as run; `_jobs/<timestamp>/` the frozen config, template,
bceg config, task lists and scripts of each submission; `molecule_registry.json` maps every InChIKey folder
to the SMILES as submitted, its line in the list, the canonical SMILES and the bceg method folder with its
hyper-parameter hashes (an InChIKey is a hash and cannot be turned back into a molecule). Per molecule,
`metadata.json` records when its ensemble was queued (`ensemble_stage`), the ensemble (bceg directory,
method, level, replicate, `params.json` hashes, program versions; bceg's own `run_1[_<lot>].json` in the
method folder has the phase, host, wall time and errors), the clone and git commit, the DFT set-up
(charge, multiplicity, `pruning`, template, MD5 of the rendered input, `nprocs`, `maxcore`, the
`level_of_theory` values and `compound_steps`: method, solvent, extra basis, charge and multiplicity of
every compound step as it ran) and one record per conformer (energy above the minimum, SLURM ids, times,
exit code, termination banner, run time, condense result). To reproduce a library: copy its YAML file, and
the frozen template if you customised it, next to a new SMILES list.

## Other ways to produce the ORCA folders

The previous workflow, [submission_scripts_v3](../submission_scripts_v3) (RDKit, CREST, UMA and ORCA from
scripts copied into a batch directory), is kept for the batches still running under it; its records are
not read by `orcadesc`.

Whatever the procedure, it must leave, somewhere on the cluster, one folder per conformer holding
the output of an ORCA 6.1 compound job run with `calcParam.inp`:

```
$LIB/<InChIKey>/conf_N/
```

The steps of that job and the files the descriptors read from it are in [condense.md](condense.md).
