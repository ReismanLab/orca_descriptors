# orca_descriptors

Compile a per-molecule descriptor table from ORCA calculations of conformer ensembles.

The chemistry is not hard-coded. A **spec** (one YAML file per molecule family) declares which atoms
are labelled, how they are found in every conformer, which descriptors are computed on which labels,
and how the conformers are aggregated. The same code then serves any molecule family. The output is the
complete descriptor library; a noise filter for modelling sets exists but is off by default.
Two engines compute the descriptors: the in-house ORCA readers (`desc_utils`) and
[DescriPyTor](https://github.com/Milo-group/DescriPyTor) (git submodule).

The calculations themselves are produced here too: `bceg/` generates the conformer ensembles
(transformation, conformer search, UMA post-optimisation, hyper-parameter logs; imported from
[ReismanLab/B-CEG-MCD](https://github.com/ReismanLab/B-CEG-MCD)) and `desc_utils.submit` runs one ORCA
compound job per conformer and records everything needed to reproduce a library.

## Contents

- [Step by step](#step-by-step)
  - [1. Laptop: install the code and the environment](#1-laptop-install-the-code-and-the-environment)
  - [2. Cluster: log in and activate the environment](#2-cluster-log-in-and-activate-the-environment)
  - [3. Cluster: submit the jobs and check them](#3-cluster-submit-the-jobs-and-check-them)
  - [4. Cluster: condense the ORCA outputs](#4-cluster-condense-the-orca-outputs)
  - [5. Laptop: copy the condensed files, extract and compile the descriptors](#5-laptop-copy-the-condensed-files-extract-and-compile-the-descriptors)
- [Details](#details)
  - [Re-run after changing the spec or adding conformers](#re-run-after-changing-the-spec-or-adding-conformers)
  - [What you get](#what-you-get)
  - [If something goes wrong](#if-something-goes-wrong)
- [Repository layout](#repository-layout)
- [Pipeline in detail: from ORCA files to the final descriptors](#pipeline-in-detail-from-orca-files-to-the-final-descriptors)
  - [0. ORCA calculations](#0-orca-calculations)
  - [1. Condensed file and transfer](#1-condensed-file-and-transfer)
  - [2. Spec](#2-spec)
  - [3. Atom numbering](#3-atom-numbering)
  - [4. Extraction](#4-extraction)
  - [5. Aggregation and filtering](#5-aggregation-and-filtering)
  - [Adding a new molecule family](#adding-a-new-molecule-family)
  - [Tests](#tests)
- [Descriptors provided](#descriptors-provided)
  - [ORCA descriptors](#orca-descriptors)
  - [DescriPyTor descriptors](#descripytor-descriptors)
- [References](#references)

## Step by step

Five steps in order: 1 and 5 on your laptop, 2 to 4 on the cluster. The commands use these names:

| Name | Meaning |
|---|---|
| `<LIB>` | the library: one campaign of ORCA jobs, e.g. `LNiH2_library_002` |
| `$LIB` | full path of the library folder on the cluster, laid out as `<LIB>/<InChIKey>/conf_N/` |
| `<user>@<cluster>` | your account and the login node |

### 1. Laptop: install the code and the environment

Once, about ten minutes:

```bash
git clone https://github.com/ReismanLab/orca_descriptors
cd orca_descriptors
git submodule update --init          # DescriPyTor, pinned commit
conda env create -f gp_env.yml       # environment "gp_env"
conda activate gp_env
pip install -e . --no-deps           # this clone as a package: `orcadesc` and `bceg` on PATH, imports work from anywhere
pytest tests -m "not slow"           # expected: all passed but 3 bceg tests that need xtb on PATH (see Tests)
```

Never `pip install` DescriPyTor into `gp_env`: it is imported from the `DescriPyTor/` folder, and its
`numpy<2` pin would downgrade the environment.

### 2. Cluster: log in and activate the environment

The repository is cloned in the group folder and the shared environment `gp_env` exists (built by its
maintainer from [gp_env_cluster.yml](gp_env_cluster.yml): the laptop environment plus xtb, CREST, UMA,
molassembler, and this clone installed editable). Nothing to install:

```bash
ssh <user>@<cluster>
source /groups/ReismanLab/miniconda3/etc/profile.d/conda.sh
conda activate gp_env
orcadesc --help                      # sanity check: the ORCA-stage command of this clone
```

The UMA weights are gated and the compute nodes cannot download them, so they are fetched once, on the
login node, into a cache shared by the group and read offline by the jobs (the two `export` lines of
`slurm.setup` in the example config):

```bash
export FAIRCHEM_CACHE_DIR=/groups/ReismanLab/hf_cache/hub  # the shared cache fairchem reads (only model files; your token stays in ~/.cache/huggingface)
hf auth login                                              # once per user: the weights are gated
orcadesc fetch-weights --model uma-s-1p2                   # once per model; sets SSL_CERT_FILE to the certifi bundle (conda's OpenSSL does not
                                                           # find the system CA store, hence CERTIFICATE_VERIFY_FAILED otherwise)
```

Set `FAIRCHEM_CACHE_DIR` before fetching: fairchem reads that variable, not `HF_HUB_CACHE`, and would otherwise
download into a private `~/.cache/fairchem`. If weights ever have to be moved from one cache to another, move
the whole cache: the model directory `models--facebook--UMA/` and the root-level `blobs/` store next to it,
because recent Hugging Face clients keep the data in that store and the model directory only holds symlinks.

Step 4 alone (condensing) needs only the Python of `module load python`. For the maintainer: `conda env
create -f gp_env_cluster.yml` from the root of `/groups/ReismanLab/orca_descriptors` (its `-e .` line installs
that clone, so `git pull` there updates everyone). LoQI is part of that environment.

### 3. Cluster: submit the jobs and check them

One library is one **library directory** `$LIB` and one config file. One command runs, for every SMILES of
the input list, the conformer ensemble (`bceg`: transformation, conformer search, UMA post-optimisation)
and then one ORCA compound job per conformer within the energy window. Everything you choose is in that
one YAML; everything the run discovers is in per-molecule records, and the records decide what is left
to do. The code is never copied: jobs run from the shared clone, whose git commit is recorded.

**3.1 Write the config.** Copy the example next to your SMILES list and edit it:

```bash
mkdir -p ~/libraries/<LIB> && cd ~/libraries/<LIB>
cp /groups/ReismanLab/orca_descriptors/configs/orca_library_example.yaml <LIB>.yaml    # ensembles + ORCA, one file
nano ligands.txt                  # one SMILES per line, optional name after it; # starts a comment
nano <LIB>.yaml                   # table below; relative paths are relative to the file
```

| Block | Content |
|---|---|
| `library_dir` | `$LIB`, absolute |
| `source` | `smiles` (the input list), `transformation` and `method` to refine, `lot` (`null` = the post-optimisation level), `run_bceg` (`false`: the ensembles exist, do not call bceg), `results_dir` (default `$LIB/ensembles`), and `config`: the bceg settings, see the next row |
| `source.config` | the conformer-ensemble settings ([docs/bceg_README.md](docs/bceg_README.md), [docs/bceg_SUBMITTING.md](docs/bceg_SUBMITTING.md)): `replicates` (1), `transformations` (`none`, `NiH2`, `NiF2`, or your script), `methods` (name, searcher and parameters: `crest_gfn2`, `goat_xtb`, `molassembler`, `rdkit_etkdg`, `loqi`), `post_optimization` (UMA model, `fmax`), and bceg's `slurm` block (partitions, resources; its `setup` defaults to the library's). The method name is the parameter set: every parameter is logged in the method folder's `params.json` with the defaults filled in, and changing one for an existing method is refused, you rename the method. A path to a separate bceg YAML is accepted here instead of the block |
| `orca` | `on`, or `off` to stop after the ensembles; the same command later with `on` refines them |
| `dft` | `ewin_kcal_mol` (conformers within this window of the ensemble minimum get an ORCA job); `level_of_theory`: which compound steps run (`opt_geom`, `freq_calc`, `nmr_calc`, `vertical_redox`, `conf_search`), method, basis and solvent of each (`opt_method`, `opt_solvent`, `svp_method`, `property_solvent`, `nmr_method`, `nbo_run`), `extra_basis` (`{Ni: def2-tzvp}`: full def2-TZVP on Ni in the optimization and frequency steps, `{}` for none). No charge or multiplicity: both come from the ensemble |
| `slurm` | `setup` lines run at the start of every job (activate `gp_env`, `module load orca`, the NBO variables); `prepare` and `orca` profiles (`time`, `mem_per_cpu`, `ntasks`, `nodes`, `partition`, ...; `ntasks` becomes `%pal nprocs`, `mem_per_cpu` gives `%maxcore` at 75 %); `array_throttle` |
| `condense` | write `conf_N/orca_condensed.json` when each ORCA job ends (step 4 done by the job) |

The settings that define the results are written to `$LIB/params.json` by the first `submit`, and a later
submit with other values is refused with a diff: the transformation, method and level refined, the bceg
settings of that transformation, method and post-optimisation, and `dft`. A different level of theory is a
different `library_dir`. `orca`, `run_bceg`, the paths, both `slurm` blocks and `condense` are recorded but
may change freely, and so may the other methods of the `source.config` block. The shipped example carries
the level of theory of the LNiH2 library. The compound job's structure is
[desc_utils/submit/calcParam.inp.tmpl](desc_utils/submit/calcParam.inp.tmpl); give `dft.calc_param_template`
your own copy only to change which steps exist.

**3.2 Submit.**

```bash
conda activate gp_env
orcadesc submit <LIB>.yaml --dry-run     # what would run: molecules per state, bceg tasks pending; writes nothing
orcadesc submit <LIB>.yaml               # or: python -m desc_utils.submit submit <LIB>.yaml
```

`submit` writes `$LIB/params.json` (first time), freezes the resolved config, the template and the git
commit under `$LIB/_jobs/<timestamp>/`, runs bceg's submission for the input list (its job arrays, scripts
and logs land under `_jobs/<timestamp>/bceg/`), then queues a `prepare` array, one element per molecule,
`afterany` the bceg jobs. Each prepare element reads the molecule's ensemble `raw_1_<lot>.xyz` (replicate 1
only), checks its `params.json` hash, splits it into `$LIB/<InChIKey>/conf_N/input.xyz` inside the energy
window, renders `calcParam.inp` with the ensemble's charge and multiplicity, writes the record, and submits
that molecule's ORCA array, one element per conformer. Each ORCA element records its outcome and condenses.

Running the same command again is how a library grows or recovers: molecules recorded `complete` or
`in_progress` are skipped, the others are prepared again, and inside a molecule a `completed` conformer is
never resubmitted while a `failed` or `timeout` one is. `--force` also takes the molecules marked
`in_progress` (a job killed before it wrote its end leaves them there). An ensemble regenerated by bceg
with other parameters is refused for a library built on the old one (`ensemble_changed`).

**3.3 Check the jobs.**

```bash
orcadesc status <LIB>.yaml                                                                       # per molecule: state, ORCA stage, conformer counts
squeue -u $USER                                                                                  # what SLURM really holds
python3 /groups/ReismanLab/orca_descriptors/scripts-for-orca-computations/check_jobs.py $LIB      # per conformer, from squeue and the ORCA outputs
python3 -m json.tool $LIB/<InChIKey>/metadata.json                                               # the full record of one molecule
less $LIB/_jobs/<timestamp>/slurm_logs/orca_prepare_<jobid>_<index>.out                          # a prepare log; orca_<InChIKey>_... for ORCA
```

| Molecule state | Meaning |
|---|---|
| `absent` | nothing recorded yet |
| `in_progress` | prepare queued, or ORCA jobs `submitted` or `running` |
| `incomplete` | ensemble missing, failed or not applicable, or a conformer `failed`, `timeout`, or never submitted |
| `complete` | every conformer `completed` or skipped by the energy filter |

The `search` and `post_opt` columns are bceg's own records for the ensemble (`missing`, `running`, `done`,
`failed`, `not_applicable`) and `n_conf` the size of the ensemble, so a molecule waiting for its ensemble
shows `in_progress` / `queued` with the search still `running`.
The ORCA stage of a molecule (`orca_stage.status`) is `queued`, `prepared`, `ensemble_missing` (bceg not
done, or failed; the next `submit` picks it up), `not_applicable` (the transformation does not apply) or
`ensemble_changed`. A conformer is `completed` only if ORCA exited 0 *and* printed `ORCA TERMINATED
NORMALLY`; a walltime kill is recorded as `timeout`. States come from the records, not from `squeue`:
`check_jobs.py` shows whether anything is really queued (`QUEUED`, `RUNNING`, `OK`, `FAILED` with the
reason, `CHECK`, `NOT STARTED`).

**3.4 What is recorded, and how to reproduce a library.** `$LIB/params.json` holds the compared settings;
`_jobs/<timestamp>/` the frozen config, template, bceg config, task lists and scripts of each submission;
`molecule_registry.json` maps every InChIKey folder to the SMILES as submitted, its line in the list, the
canonical SMILES and the bceg method folder with its hyper-parameter hashes (an InChIKey is a hash and
cannot be turned back into a molecule). Per molecule, `metadata.json` records the ensemble (bceg directory,
method, level, replicate, `params.json` hashes, program versions), the clone and git commit, the DFT set-up
(charge, multiplicity, energy window, template, MD5 of the rendered input, `nprocs`, `maxcore`, the
`level_of_theory` values and `compound_steps`: method, solvent, extra basis, charge and multiplicity of
every compound step as it ran) and one record per conformer (energy above the minimum, SLURM ids, times,
exit code, termination banner, run time, condense result). To reproduce a library: copy its YAML file, and
the frozen template if you customised it, next to a new SMILES list.

The previous workflow, [submission_scripts_v3](submission_scripts_v3) (RDKit, CREST, UMA and ORCA from
scripts copied into a batch directory), is kept for the batches still running under it; its records are
not read by `orcadesc`.

Whatever the procedure, it must leave, somewhere on the cluster, one folder per conformer holding
the output of an ORCA 6.1 compound job run with `calcParam.inp`:

```
$LIB/<InChIKey>/conf_N/
```

### 4. Cluster: condense the ORCA outputs

Each conformer folder holds about 70 MB. This step writes one `orca_condensed.json` (0.4 to 1.5 MB)
per conformer with everything the descriptors need, on a compute node:

```bash
cd $LIB                                                            # the log lands where you submit from
sbatch /groups/ReismanLab/orca_descriptors/scripts-for-orca-computations/slurm-condense.sh $LIB  # 8 cores, 2 h; raise the #SBATCH lines for a big library
squeue -u $USER                                                    # wait until the "condense" job is gone
tail -n 4 condense_<jobid>.out   # expected: "... failed 0", "job status: completed N", "checked N file(s): N ok, 0 failed"
```

Every conformer that got past its singlepoint gets a file, finished or not, and the file records the
state of the ORCA job (`job.status`: `completed`, `running` or `failed`). The `job status:` line of the
log counts them: `completed N` alone means the library is done. Re-running the same command rebuilds
the files of the unfinished jobs and skips the completed ones, so it can be repeated after every
resubmission or new batch. A `failed` line names a conformer with nothing to condense yet (no
singlepoint), see [If something goes wrong](#if-something-goes-wrong). Jobs submitted with step 3 condense at
the end of every ORCA job (`condense: true` in the library config, recorded per conformer under `condensed`),
so for such a library this step is only needed for the conformers whose job died before the end or whose
file the check rejects; re-running it skips the completed files.

### 5. Laptop: copy the condensed files, extract and compile the descriptors

From the repository root, with `gp_env` active.

**5.1 rsync.** Only the geometry and the condensed file travel, about 0.5 MB per conformer. The
library folder lands as `DATA/<LIB>/`: no slash after `<LIB>` on the cluster side, or its content
spills straight into `DATA/`. The folders of jobs that produced nothing yet come along empty, on
purpose: that is how the laptop knows the molecule has conformers still to come.

```bash
conda activate gp_env
rsync -avz --include='*/' --include='optimization.xyz' --include='orca_condensed.json' \
      --exclude='*' <user>@<cluster>:/path/to/<LIB>  DATA/
```

**5.2 Descriptor extraction and compilation.**

```bash
cp specs/lnih2_pyox.yaml specs/<LIB>.yaml        # closest example; small organics: specs/test_molecules_at1_at2.yaml
# edit specs/<LIB>.yaml, fields in the table below
python -m desc_utils.run all --spec specs/<LIB>.yaml --jobs 4
#   -> <today>_<LIB>_atom_numbering_results.csv (per conformer: atom indices), <today>_<LIB>_combined_descriptors.csv
#      (per conformer: descriptors), <today>_<LIB>_descriptors.csv (per molecule: the spreadsheet)
#   -> <today>_<LIB>_conformers_to_check.csv only if some conformer needs a look, with a WARNING on the terminal
```

`--jobs N` spreads the per-conformer work (Open Babel and SMARTS numbering, morfeus geometry, ORCA and
DescriPyTor extraction) over N processes; `--jobs 0` takes every core, the default is one. The tables
are the same whatever N. The first run of a library is the one to parallelise: the morfeus results are
cached in `<conf>/xyz.json` (`<conf>/xyz_<stem>.json` when `data.geometry_file` is not `optimization.xyz`)
and read back on the next runs.

| Spec field | Set to |
|---|---|
| `name` | `<LIB>` (used in the output file name) |
| `data.root` | `DATA/<LIB>` |
| `atoms` | one label per atom of interest, `label: element` (`null` = any element) |
| `numbering` | how each label is found: `smarts` stages (see the LNiH2 spec), `first_n` (labels = atoms 1..N of the xyz), or `manual` with `csv:` pointing to your own table of 1-based indices |
| `descriptors.orca.*`, `descriptors.descripytor` | label lists / label tuples; delete the terms you do not need |
| `filter.metal_label` | the metal label, or `null` for organic molecules |

**Geometry-only descriptors from UMA structures.** The morfeus family (`descriptors.orca.xyz`: SASA,
%Vbur, bond lengths, angles, dihedrals, Sterimol, pyramidalisation, quadrants) needs nothing but an xyz
file, so a library pre-optimised with UMA can be described before, or without, any ORCA job and without
step 4 (no `orca_condensed.json`). In a library prepared by step 3 the UMA geometry of every conformer is
`conf_N/input.xyz`, whose comment line carries `energy=<Hartree>`: add `--include='input.xyz'` to the rsync of
step 5.1, set `data.geometry_file: input.xyz`, keep `descriptors.orca.xyz` as the only family, and take the
energy for the Boltzmann weighting from that comment line:

```yaml
numbering:
  energy: {source: xyz_comment, key: energy, units: hartree, column: uma_energy_hartree}
```

`specs/test_molecules_uma_only.yaml` is the template; it reads the extended xyz of the previous workflow
(`optimized_uma-s-1p1.xyz`, `units: eV`), the only difference being the file name and the units.

`column` is mandatory here and may not be `SCF_energy`: the value is a UMA energy converted to Hartree, and
the table must say so. The other families (`charges`, `nbo`, `redox`, `descripytor`) read ORCA output and
cannot run on an xyz; when none of them is requested the tables carry no `orca_job_status`, no
`all_conformers_completed` flag, and nothing is listed as "ORCA job missing".

The run never stops for a bad conformer. One that could not be numbered, or whose ORCA job is
running, failed or absent, is left out or NaN in the tables, the molecule gets
`all_conformers_completed = False` in the spreadsheet, and the conformer is listed in
`<today>_<LIB>_conformers_to_check.csv` with the reason (`numbering_status`, `numbering_notes`,
`orca_job_status`). Correcting a few atom indices by hand in the numbering table is a normal outcome;
rerun with the corrected table instead of re-deriving it:

```bash
python -m desc_utils.run all --spec specs/<LIB>.yaml --numbering <date>_<LIB>_atom_numbering_results.csv
```

Output names carry the date of the run and the spec name so that nothing is overwritten from one run
to the next (`output` section of the spec).

What the output files contain is in [What you get](#what-you-get).

## Details

### Re-run after changing the spec or adding conformers

| You changed | Run |
|---|---|
| Descriptor terms only | `python -m desc_utils.run extract --spec S` then `aggregate --spec S` (morfeus geometry is cached in `<conf>/xyz.json`; add `--overwrite-xyz-json` if you changed `vbur_radii` or `quadrants`) |
| Atom labels or numbering rules | `all --spec S`, or `number --spec S` then `all --spec S --numbering <csv>` after hand corrections |
| Added conformer folders on the cluster | step 4 (existing files are skipped), 5.1, then `all --spec S` |
| Only the aggregation temperature | `aggregate --spec S` |
| You want the noise filter (correlation / variance pruning) for a modelling set | `aggregate --spec S --filter`, or set `filter.enabled: true` in the spec; needs enough molecules for correlations to mean anything (with two molecules every column is perfectly correlated and one survives) |

### What you get

| File | One row per | Content |
|---|---|---|
| `<date>_<name>_atom_numbering_results.csv` | conformer | 1-based index of every label, SCF energy, SMILES, `numbering_status` |
| `<date>_<name>_combined_descriptors.csv` | conformer | `orca_job_status` (`completed`, `running`, `failed`; `unknown` for files condensed before 2026-09-23, `missing` for a folder with no file), `n_conformer_folders` of its molecule, then every descriptor of both engines, columns named by label (`charge_NBO_Ni`, `%Vbur_atom_Ni_radius_3.5`, `B1_Ni-N1`, ...) |
| `<date>_<name>_descriptors.csv` | molecule | the spreadsheet: `inchikey`, `smiles`, `smiles_noH` (implicit hydrogens), `n_conformers` (rows aggregated), `n_conformers_completed`, `n_conformer_folders` (folders on disk), `all_conformers_completed` (`True` / `False`; empty when a status is `unknown`), then `<descriptor>_min`, `_max`, `_lowE` (lowest-energy conformer) and `_boltz` (Boltzmann average at `temperature_K`); every descriptor is kept unless `--filter` is given |
| `<date>_<name>_conformers_to_check.csv` | conformer needing a look | only written when there is one (today's copy is removed otherwise): `path`, `reason`, `numbering_status`, `numbering_notes`, `orca_job_status` |

Conventions worth knowing when reading the spreadsheet: energies are in Hartree, HOMO/LUMO in eV,
IP and EA are positive for bound electrons (`IP = E(cation) - E(neutral)`), Fukui functions follow
`f+ = q(N) - q(N+1)`, `cm5_*` equals `hirshfeld_*` (the CM5 correction is not implemented), and
DescriPyTor Sterimol values (`B1_*`, `B5_*`, `L_*`) differ from the morfeus ones (`Sterimol_*`).
The full list is under [Descriptors provided](#descriptors-provided).

### If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `check_jobs.py` reports `FAILED ... killed by SLURM: time limit` or `out of memory`, or `status` shows `timeout` / `failed` | the SLURM allocation was too small for this conformer | raise `slurm.orca.time` or `mem_per_cpu` in `<LIB>.yaml` (recorded, not compared) and run `orcadesc submit` again: only the failed conformers are resubmitted |
| `orcadesc submit` prints `REFUSED` with a diff of `source` or `dft` | a result-defining setting changed since the library was created | restore it, or put the new settings in a new `library_dir` |
| `orcadesc submit` prints `bceg PARAMS MISMATCH` | a parameter of an existing bceg method changed in `source.config` | rename the method (and point `source.method` at it) or restore the parameter |
| `status` says `in_progress` but `squeue` shows nothing | a job died (walltime, node failure) before writing its end | `orcadesc submit <LIB>.yaml --force` |
| `status` says `ensemble_missing` after the bceg jobs finished | bceg failed for that molecule; its `run_1*.json` in the method folder holds the error | fix the cause, run `orcadesc submit` again (bceg reruns only the failed replicate) |
| a prepare log ends with `orca not found on PATH` or an ORCA job's `calcParam.out` is empty | the `slurm.setup` lines do not load the ORCA module, or NBO variables are missing | add `module load orca/6.1.0` (and the `NBOEXE` / `GENEXE` exports) to `slurm.setup`, resubmit |
| `status` shows `post_opt failed` and `run_1_<lot>.json` says `LocalEntryNotFoundError` or `CERTIFICATE_VERIFY_FAILED` | the job tried to download the UMA weights; conda's OpenSSL cannot verify huggingface.co | `orcadesc fetch-weights` once on the login node into the shared cache (step 2), keep `FAIRCHEM_CACHE_DIR` / `HF_HUB_OFFLINE=1` in `slurm.setup`, then `orcadesc submit` again: only the post-optimisation reruns |
| `ImportError: /lib64/libstdc++.so.6: version CXXABI_1.3.15 not found` when importing fairchem or `sqlite3` | pip-installed torch loaded the old system libstdc++ before the env's | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (in `slurm.setup` in the example); in Python, `import sqlite3` before torch, which bceg does |
| UMA post-optimisation fails with a Hugging Face authentication error on the login node | the UMA weights are gated | `huggingface-cli login` once in your account, then fetch the weights again |
| `all_conformers_completed` is `False` | a conformer of that molecule is `running`, `failed` or `missing` in `orca_job_status`; its late descriptors (NMR, redox, Fukui) are NaN and left out of the averages | finish or resubmit the job, step 4 again (rebuilds that file), 5.1, then `all` |
| `all_conformers_completed` is empty and `orca_job_status` is `unknown` | the condensed files predate the job status (before 2026-09-23) | step 4 on the cluster rebuilds them, then 5.1; or leave it, the descriptors are unaffected |
| `orca_job_status` is `failed` with `terminated normally, then a later run failed` | the folder was resubmitted after a successful run and the second run died | `tail calcParam.out`; if the first run's outputs are intact the descriptors are fine, otherwise resubmit |
| `condense` prints `failed ... no complete NBO block (NBO started but never returned)` | the ORCA job aborted; `tail calcParam.out` shows the error | rerun the ORCA job or drop the conformer folder |
| `condense` prints `failed ... optimization.xyz missing` / `singlepoint.property.json missing` | the job did not reach that step, or a different input file was used | same as above; the pipeline needs the `calcParam.inp` steps |
| `numbering_status` is `failed` or `incomplete` | no SMARTS stage matched this conformer | check the SMILES in the table, adjust the stage patterns, or switch to `backend: manual` with your own indices |
| `numbering_status` is `element_mismatch` | the matched atom is not the element declared in `atoms` | fix the SMARTS role positions or the `atoms` element |
| `numbering_status` is `inconsistent` | conformers of one molecule got different indices | usually a symmetric ligand; set `numbering.smarts.match_selection: lowest_indices` or add an `orient` rule |
| `descripytor_notes` says `no orca_condensed.json in ...` | the folder holds raw files only or was not transferred | run step 4 / 5.1, or `python -m desc_utils.run condense --spec S` locally when the raw files are present |
| `descripytor_notes` says `expected 1 molecule file ..., loaded 0` | the condensed file is unreadable | `python -m desc_utils.condense --root DATA/<LIB> --check` and rebuild with `--overwrite` (`sbatch slurm-condense.sh $LIB --overwrite` on the cluster) |
| `nbo_extraction_error` column is not empty | the NBO block of that conformer is missing or truncated | same as the `condense ... NBO block` row |
| `No module named desc_utils.run` | the clone predates the pipeline | `git pull` and run the commands from the repository root |
| every row has `numbering_status` `failed` with `No module named 'openbabel'` | Open Babel missing from the environment (needed to convert xyz to sdf) | `pip install openbabel-wheel==3.1.1.23` in `gp_env`, then rerun `number` |
| `extract` stops with `No module named 'networkx'` or `'plotly'` | environment created from an older `gp_env.yml` | `pip install networkx==3.6.1 plotly==7.1.0` in `gp_env` |
| `ImportError: DescriPyTor not found at ...` | submodule not checked out | `git submodule update --init` |
| `numpy.dtype size changed` or similar import errors | a `pip install` of DescriPyTor downgraded numpy | never pip-install DescriPyTor; recreate `gp_env` from `gp_env.yml` |
| `validate` lists `unknown label` | a descriptor term uses a label that is not in `atoms` | add the label or fix the typo; labels are case-sensitive |

## Repository layout

| Path | Content |
|---|---|
| [DATA](DATA) | Where the libraries go on your laptop (step 5.1): `DATA/<LIB>/<InChIKey>/conf_N/`. Empty in the repository, ignored by git. |
| [specs](specs) | One YAML spec per molecule family. `lnih2_pyox.yaml` (LNiH2 / PyOx ligand library) and `test_molecules_at1_at2.yaml` (small organic test set) are the shipped examples. |
| [desc_utils](desc_utils) | The pipeline: `spec.py` (spec loading and validation), `numbering/` (atom numbering backends), `extract/` (ORCA and DescriPyTor adapters), `aggregate.py` (Boltzmann aggregation, filtering), `run.py` (CLI), `orca_parse.py` (standard-library ORCA parsers, also run on the cluster), `condense.py` (condensed per-conformer file: producer, checker, loader), `orca_descriptors.py` and `data_extraction.py` (morfeus geometry, extraction functions); `submit/` (the ORCA stage of step 3, the `orcadesc` command: `config.py`, `source.py`, `template.py` and `calcParam.inp.tmpl`, `records.py`, `slurm.py`, `tasks.py`, `status.py`). |
| [DescriPyTor](DescriPyTor) | Git submodule, [Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor). |
| [bceg](bceg) | Conformer ensembles (imported from [ReismanLab/B-CEG-MCD](https://github.com/ReismanLab/B-CEG-MCD) with its history): transformations (`transforms_builtin/ni_x2.py`), searchers (`search/`: CREST, GOAT, molassembler, RDKit ETKDG, LoQI), UMA post-optimisation (`refine/`), hyper-parameter logs (`provenance.py`), SLURM arrays (`slurm.py`), the `bceg` command. Its original documents are under [docs](docs). |
| [configs](configs) | `orca_library_example.yaml`: one library in one file, ensembles and ORCA (LNiH2 level of theory); copy it next to your SMILES list (step 3). Its `source.config` block is also a complete config for the standalone `bceg` command. |
| [inputs](inputs) | Example SMILES list. |
| [docs](docs) | bceg's README, SUBMITTING and PLAN as imported. [PLAN.md](PLAN.md) at the root is this repository's plan for the merger. |
| [submission_scripts_v3](submission_scripts_v3) | Legacy, replaced by step 3 and kept for the batches still running under it. The previous submission workflow, copied into every batch directory: `workflow_inputs.yaml` (every chosen setting), `submit_all_smiles.py` (one job per SMILES, skips finished ligands, checks the settings against the records), `launch_workflow.sh` (build, CREST, UMA and ORCA submission for one ligand), the stage scripts (`ligand_only_create_folder.py`, `crest.py`, `uma_opt_confs.py`, `launch_orca_all.py`), `calcParam.inp.tmpl` and `calc_param_template.py` (ORCA input rendered from the yaml), `slurm-calcParam.sh` (one ORCA job, records its outcome and condenses its output), `metadata_utils.py`, `record_step.py`, `mark_orca_done.py`, `check_inputs.py` (the `metadata.json` provenance and status record), `lookup_molecule.py` (InChIKey to submitted SMILES and RDKit molecule, from `molecule_registry.json`), `submit_aggregate.py`, `slurm-aggregate.sh` and `aggregate_metadata.py` (batch summary `<LIB>_metadata.json`: majority settings, flagged ligands). |
| [scripts-for-orca-computations](scripts-for-orca-computations) | Cluster tools for any library: `check_jobs.py` (queued / running / done / failed state of every ORCA job), `list_molecules.py` (InChIKey and conformer count), `slurm-condense.sh` (condense a whole library on a compute node); `calcParam.inp`, the fixed compound input of the LNiH2 library; and the older manual launcher (`launch_orca_all.py`, `slurm-calcParam.sh`), superseded by step 3. |
| [tests](tests) | Unit tests, and regression tests against the reference CSVs in `tests/reference/` (run when the LNiH2 library is under `DATA/`). |
| [notebooks](notebooks) | Exploratory notebooks (`make_descriptor_descripytor.ipynb` ran the pipeline before the CLI existed; ligand separation, substructure numbering, descriptor correlations). Not maintained with the code. |

## Pipeline in detail: from ORCA files to the final descriptors

```
ORCA compound job per conformer                                                            (cluster)
        │  optimization.xyz, *.property.json, singlepoint.json (GBW dump), calcParam.out (NBO, frequencies)
        ▼
[1] condense  ──►  <conf>/orca_condensed.json      (standard-library Python, ~0.4 to 1.5 MB per conformer)
        │  rsync optimization.xyz + orca_condensed.json                                        (laptop)
        ▼
[2] spec  ──►  [3] number  ──►  [4] extract  ──►  [5] aggregate (+ optional filter)
     <date>_<name>_atom_numbering_results.csv   <date>_<name>_combined_descriptors.csv   <date>_<name>_descriptors.csv
```

All stages read the same spec. From the repository root, in `gp_env`:

```bash
python -m desc_utils.run validate  --spec specs/lnih2_pyox.yaml
python -m desc_utils.run condense  --spec specs/lnih2_pyox.yaml            # -> <conf>/orca_condensed.json (normally run on the cluster)
python -m desc_utils.run number    --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_atom_numbering_results.csv
python -m desc_utils.run extract   --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_combined_descriptors.csv (per conformer)
python -m desc_utils.run aggregate --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_descriptors.csv
python -m desc_utils.run all       --spec specs/lnih2_pyox.yaml            # number -> extract -> aggregate
```

`python -m desc_utils.run <command> --help` lists the options (`--numbering` to reuse an existing table,
`--families` to run a subset of descriptor families, `--strict` to fail on any numbering problem,
`--filter` to apply the spec's noise filter, `--overwrite-xyz-json`).

### 0. ORCA calculations

Every conformer is computed with an ORCA 6.1 compound job, `calcParam.inp`. The step 3 workflow renders it
per molecule from [desc_utils/submit/calcParam.inp.tmpl](desc_utils/submit/calcParam.inp.tmpl) and the
`dft.level_of_theory` block of the library config, with charge and multiplicity from the ensemble, and runs
ORCA in every `conf_N/`.
[scripts-for-orca-computations/calcParam.inp](scripts-for-orca-computations/calcParam.inp) is the fixed input
the LNiH2 library was computed with; its level of theory is the table below, including full `def2-TZVP` on Ni in
the optimization and frequency steps (`extra_basis: {Ni: def2-tzvp}` in the YAML). The steps and the files read
downstream are:

| Step (`%base`) | Level of theory | Purpose | Files used by the pipeline |
|---|---|---|---|
| `conf_search` (optional) | GOAT / xTB | conformer search | – |
| `optimization` | ωB97M-D4 / def2-TZVP(-f), def2-TZVP on Ni | geometry | `optimization.xyz` (geometry for morfeus, numbering) |
| `frequency` | same as optimization | harmonic frequencies, thermochemistry | `calcParam.out` (frequencies, normal modes for DescriPyTor) |
| `singlepoint` | M06-D4 / def2-TZVP, `NBO NPA`, `HIRSHFELD CHELPG`, `%elprop` (dipole, polarizability) | electronic descriptors | `singlepoint.property.json` (energies, charges, MOs, dipole), `calcParam.out` (NBO 7 block 1) |
| `nmr` | M06-D4 / def2-TZVP, GIAO | shieldings | `nmr.property.json` |
| `oxidation`, `reduction` | as `singlepoint`, vertical (N±1 electrons) on the optimized geometry | redox and Fukui descriptors | `oxidation.property.json`, `reduction.property.json`, `calcParam.out` (NBO blocks 2 and 3) |

Solvent is empty by default; NBO requires the `NBOEXE`/`GENEXE` environment variables. Every file in the
last column is read through `orca_condensed.json` once that file exists (section 1); the raw files are
then only needed on the cluster.

### 1. Condensed file and transfer

A finished conformer folder holds about 70 MB, of which the pipeline reads about 37 MB: the 32 MB GBW
dump `singlepoint.json` (for two HOMO/LUMO numbers), the 4 MB `calcParam.out` (NBO tables, frequencies)
and the small `.property.json` files. `desc_utils/condense.py` extracts exactly what the readers consume
into one versioned JSON per conformer, `orca_condensed.json` (about 0.4 MB for 37 atoms, 1.5 MB for
90 atoms, most of it the normal modes). It needs only the Python standard library, so it runs on the
cluster right after ORCA (last lines of `slurm-calcParam.sh`) or over a finished library:

```bash
# cluster: a plain git clone of this repository, no environment
PYTHONPATH=/path/to/orca_descriptors python -m desc_utils.condense --root /path/to/LIB --jobs 4
PYTHONPATH=/path/to/orca_descriptors python -m desc_utils.condense --root /path/to/LIB --check
# laptop: the same producer driven by the spec
python -m desc_utils.run condense --spec specs/lnih2_pyox.yaml
```

Then pull the condensed files together with the 3 KB geometry; everything else stays on the cluster:

```bash
rsync -avz --include='*/' --include='optimization.xyz' --include='orca_condensed.json' \
      --exclude='*' user@cluster:/path/to/LIB  DATA/          # -> DATA/LIB/; no slash after LIB
```

Content (schema version 1; units in the key names; `null` when ORCA did not print a quantity):

| Section | Content | Source |
|---|---|---|
| `molecule` | atoms with Å coordinates, charge, multiplicity | `optimization.xyz` |
| `states.<singlepoint / oxidation / reduction / nmr>` | SCF energy, electron counts, Mulliken / Loewdin / CHELPG / Hirshfeld charges, dipole (a.u.), polarizability, thermochemistry, NMR `siso` / `saniso`, MO energies and occupations (`singlepoint` only by default, `--mo-states`) | `<state>.property.json`, `<state>.json` |
| `nbo.blocks[]` | `NBOProps.results` per NBO block: NPA, NAO occupancies, NBO summary, hybrids, NHO directionality, second-order perturbation, cycle info. Block 0 (neutral) in full; the cation and anion blocks keep NPA only unless `--nbo-full all` | `calcParam.out` |
| `properties`, `vibrations`, `charges` | DescriPyTor's JSON schema: dipole in Debye (singlepoint), polarizability, frequencies with unit-norm displacements per mode (translations and rotations dropped, IR intensities where printed), NPA / Hirshfeld / CM5 charges | derived from the above |
| `producer` | ORCA and NBO versions, host, time, git commit, size and mtime of every input file | |
| `job` | `status` (`completed`, `running`, `failed`), `detail` (ORCA error line or SLURM kill reason), `checked_utc` | `calcParam.out` termination lines, `error_<jobid>.txt` |

Every reader (`OrcaReader`, the NBO loader, `scf_energy` of the numbering stage, DescriPyTor's
`Molecules`) uses the condensed file when `<conf>/<data.condensed_file>` exists and the raw files
otherwise, so folders downloaded before this step keep working; `data.condensed_file: null` forces the
raw files. `python -m desc_utils.condense --check` validates existing files, and `tests/test_condense.py`
asserts that every in-house column is identical whichever source is read.

A file is written as soon as the singlepoint and its NBO block exist, so a job that died or is still
running afterwards gets one too, with the missing states as `null` and `job.status` saying so. The
condenser only skips files whose status is `completed`; any other stored status is rebuilt from the
raw files when they are there (`tests/test_job_status.py`). The extract stage copies the status into
`orca_job_status` and counts the conformer folders of the molecule (`n_conformer_folders`, empty
folders included); the aggregation turns the two into `n_conformers_completed` and
`all_conformers_completed`. Descriptors of unfinished conformers are still aggregated: their missing
values are simply absent from the min / max / Boltzmann statistics, as any NaN is.

### 2. Spec

A spec has these sections (defaults in `desc_utils/spec.py::DEFAULTS`):

| Section | Content |
|---|---|
| `data` | `root` folder, `layout: molecule/conformer`, `conformer_glob`, `geometry_file` (the xyz every stage reads: numbering, morfeus, comment-line energy), `molecule_id_column`, `condensed_file` (`null` = raw ORCA files only). |
| `atoms` | Ordered map `label -> expected element` (`null` = any). Labels are the only way descriptors refer to atoms. |
| `numbering` | `backend` (`smarts`, `manual`, `first_n`, `molalign`), `index_base`, per-backend block, optional `orient` rules, `annotators`, `smiles` mode, `energy` (`source: orca`, the SCF energy of `basename`, column `SCF_energy`; or `source: xyz_comment`, the `key=` value of the extended-xyz comment line in `units`, converted to Hartree, with a mandatory `column` name that may not be `SCF_energy`). |
| `descriptors.orca` | Four families with label lists: `xyz` (atoms, bonds, angles, dihedrals, quadrants, `vbur_radii`), `charges`, `nbo` (`homo_lumo_rank_range`, `nbo_block_index`, `spin`), `redox` (Fukui `prefixes`, `nbo_block_map`). |
| `descriptors.descripytor` | DescriPyTor "Schema B" keys with label tuples: `charges`, `charge_diff`, `sterimol`, `bond_length`, `bond_angle`, `stretching_vibration`, `bending_vibration`, `dipole`, plus thresholds, `radii`, `drop_atoms`. |
| `aggregate` | `group_by`, `energy_column` (defaults to `numbering.energy.column`), `temperature_K`, `kB_hartree_per_K`, `stats`, `carry_first`. |
| `filter` | Every option of `filter_numeric_descriptors`, plus `metal_label`. |
| `output` | File names of the three stage outputs and of `check_csv`, the list of conformers to look at; `{date}` and `{name}` are filled in at run time. |

`validate` reports every problem at once (unknown labels, wrong arity, SMARTS that do not parse, roles
outside the SMARTS, labels never assigned). Quote every SMARTS string: `#` starts a YAML comment.

### 3. Atom numbering

`number` writes one row per conformer folder with the 1-based index of every label, plus `path`,
`inchikey`, `SCF_energy`, `smiles` (all hydrogens explicit, as in the xyz), `smiles_noH` (implicit hydrogens),
annotator columns (e.g. `ligand_class`), `numbering_backend`,
`numbering_status` (`ok`, `incomplete`, `failed`, `element_mismatch`, `inconsistent`) and
`numbering_notes`. The table is a plain CSV so it can be inspected or corrected by hand; rows whose
status is not `ok` are skipped downstream.

Backends:

- `smarts`: ordered stages, each with one or several SMARTS alternatives and a `roles: {label: match position}`
  map; later stages override earlier labels (used to orient PyOx ligands). Conformers are converted
  `xyz -> sdf` with Open Babel and matched with RDKit.
- `manual`: read the indices from an existing CSV (`csv`, `key_column`).
- `first_n`: label the first N atoms in file order (small test molecules).
- `molalign`: reserved for alignment-based numbering, not implemented yet.

### 4. Extraction

`extract` resolves every labelled term of the spec to atom indices per conformer and calls the two
engines. Column names embed the labels, never the indices (`%Vbur_atom_Ni_radius_3.5`, `B1_Ni-N1`,
`charge_NBO_C1`, ...):

- ORCA side (`desc_utils/extract/orca.py`): `xyz` (morfeus on `optimization.xyz`, cached in
  `<conf>/xyz.json` and recomputed when the spec asks for columns the cache lacks), `charges`
  (`singlepoint.property.json`, `nmr.property.json`), `nbo` (NBO 7 block of `calcParam.out`),
  `redox` (`singlepoint`/`oxidation`/`reduction`); all of them from `orca_condensed.json` when it exists.
- DescriPyTor side (`desc_utils/extract/descripytor.py`): one `Molecules` object per conformer, loaded
  from `orca_condensed.json` (`input.kind: json`),
  one call per family; the `FAMILIES` table maps DescriPyTor's index-based column names to label-based
  names. `python -m desc_utils.run to-schema-b --spec ... --row 0` prints the index-based JSON that
  DescriPyTor's own tools consume for one conformer.

The result is `<date>_<name>_combined_descriptors.csv`, one row per conformer, merged on `path`.

### 5. Aggregation and filtering

`aggregate` groups conformers by molecule (`group_by: inchikey`) and writes, for every numeric descriptor,
`<name>_min`, `<name>_max`, `<name>_lowE` (lowest-energy conformer) and `<name>_boltz`, the Boltzmann
average at `temperature_K` with weights `exp(-ΔE / kB T)` computed from `energy_column` (by default the
column of `numbering.energy`: the SCF energy of the `singlepoint` step, no thermal correction, or the
comment-line energy of the geometry file for a geometry-only run; always Hartree). `n_conformers` and the `carry_first` columns are
kept. This complete table is the default output. Optionally, with `filter.enabled: true` in the spec or
`aggregate --filter`, `filter_numeric_descriptors` then removes: non-numeric and all-NaN columns, NBO orbitals whose
Boltzmann occupancy is below `occupancy_boltz_threshold`, Rydberg orbitals (except on `metal_label`),
metal 4f Rydbergs, core orbitals, anisotropic NMR terms, NHO φ/θ angles, near-constant columns
(`ratio_threshold`) and one member of every pair above `corr_threshold`. The output is
`<date>_<name>_descriptors.csv` with `inchikey_folder`, `inchikey`, `smiles`, `smiles_noH`, `ligand_class` in front.

### Adding a new molecule family

Copy a spec, then edit `atoms`, `numbering` (SMARTS stages, or `manual` with your CSV, or `first_n`),
the label lists under `descriptors`, and `filter.metal_label`. Run `validate`, then `number` and check
the `numbering_status` column before `extract`.

### Tests

```bash
pytest tests -m "not slow"     # unit tests: descriptors, bceg (tests/bceg), the ORCA stage (tests/test_submit.py); seconds
pytest tests -m slow           # regression: reproduces tests/reference/atom_numbering_results.csv, the DescriPyTor
                               # per-conformer reference and the reference aggregated CSVs (minutes)
```

Three bceg tests need `xtb` on `PATH` (start-geometry pre-optimisation) or the hydride SMILES of a newer
RDKit than `gp_env`'s; they fail on a laptop without xtb and pass in the cluster environment.

The regression tests and the condensed-file round trip need the LNiH2 library under
`DATA/LNiH2_library_001_to_download_2026_03_24/` (raw ORCA folders for the round trip) and the
submodule; they are skipped otherwise.

## Descriptors provided

Notation: `X` is an atom label from the spec, `Y` a %Vbur radius (default 2.0 to 5.0 Å in 0.5 steps),
`R` an NBO rank token relative to the highest occupied NBO (`homo`, `hm1`, `hm2`, ... below it, `lp1`,
`lp2`, ... above it; kept for the ranks in `homo_lumo_rank_range`). Every per-conformer descriptor
below appears four times in the aggregated CSV (`_min`, `_max`, `_lowE`, `_boltz`).

:white_check_mark: implemented and covered by the regression tests. :hourglass: implemented, not
enabled in the shipped specs. :x: not implemented yet.

### ORCA descriptors

**Fragment specific geometric descriptors** (morfeus on `optimization.xyz`, `descriptors.orca.xyz`)

- %Vbur (Bondi radii × 1.17, H included, sphere radius `Y`): `%Vbur_atom_X_radius_Y` :white_check_mark:
- Atomic SASA: `SASA_X` :white_check_mark:
- Pyramidalization P and P angle: `Pyr_X`, `Pyr_angle_X` :white_check_mark:
- Sterimol (morfeus, CRC van der Waals radii; L axis from first to second label): `Sterimol_X_X_L`, `Sterimol_X_X_Bmin`, `Sterimol_X_X_Bmax` :white_check_mark: / B1 and B5 do not match DescriPyTor `B1_X-X`, `B5_X-X` (different radii and slicing)
- Bond length: `Bond_X_X` :white_check_mark: / matches DescriPyTor `dist_X-X`
- Angle: `Angle_X_X_X` :white_check_mark: / matches DescriPyTor `angle_X-X-X`
- Dihedral: `Dihed_X_X_X_X` :white_check_mark:
- %Vbur quadrants and octants (`quadrants: [[metal, z1, z2, xz_plane, ...]]`): `%Vbur_X_X_X_X_quadrant_+,+`, ..., `%Vbur_X_X_X_X_octant_+,+,+`, ... :hourglass:
- Bite angle (morfeus `BiteAngle`, method present, not wired to the spec) :x: not implemented yet

**Molecular geometric descriptors**

- SASA and SASA volume: `SASA`, `SASA_Volume` :white_check_mark:

**Charges** (`singlepoint.property.json`, `descriptors.orca.charges`)

- Mulliken: `mulliken_X` :white_check_mark:
- Loewdin: `loewdin_X` :white_check_mark:
- CHELPG: `chelpG_X` :white_check_mark:
- Hirshfeld: `hirshfeld_X` :white_check_mark:
- CM5: `cm5_X` :x: not implemented yet (the current column is a stub equal to `hirshfeld_X`)
- NBO / NPA charge: `npa_X_charge` :white_check_mark: / matches DescriPyTor `charge_NBO_X`

**Orbital analysis** (NBO 7 block of `calcParam.out`, `descriptors.orca.nbo`)

Open-shell (UHF/UKS) NBO blocks hold two complete analyses, one per spin, and NBO 7 only prints the closing `$CHOOSE` after the beta set. `NBOProps` reads a single spin set per block so every NBO number occurs once: `spin: alpha` (default, the set the parsers always returned), `spin: beta`, or `spin: both` (alpha rows then beta rows, each tagged with a `spin` key; duplicate NBO numbers then get `_2` column suffixes). Closed-shell blocks ignore the option. NPA charges and NAO occupancies always come from the spin-summed tables. `nbo_block_index` selects which NBO block of the output file is read (default 0, the first).

- NPA populations: `npa_X_core`, `npa_X_valence`, `npa_X_rydberg`, `npa_X_total`, `npa_totals_*` :white_check_mark:
- Natural atomic orbital occupancies: `nao_X_Cor-1s_occupancy`, `nao_X_Val-2p_occupancy`, `nao_X_Ryd-3d_occupancy`, ... (`_2`, `_3` suffixes for degenerate sets) :white_check_mark:
- Natural bond orbitals on labelled atoms: `nbo_R_X_X_type`, `nbo_R_X_X_bond_index`, `nbo_R_X_X_occupancy`, `nbo_R_X_X_energy` (lone pairs: `nbo_R_X_*`) :white_check_mark:
- Bond orbital composition: `bond_orbital_details_R_X_X_{bond_type,bond_index,occupancy}`, `bond_orbital_details_R_X_X_contrib_k_X_{percentage,coefficient,s_percent,p_power,p_percent}` :white_check_mark:
- NHO directionality: `nho_directionality_R_X_X_{line_theta,line_phi,hybrid_theta,hybrid_phi,deviation}` :white_check_mark: (φ/θ dropped by the optional filter)
- Second-order perturbation E(2) between labelled orbitals: `second_order_R_<donor>_to_R_<acceptor>_{donor_type,donor_index,acceptor_type,acceptor_index,e2_energy,energy_diff,fock_element}` :white_check_mark: (NBO 7 numbered rows and the legacy `->` layout; only rows whose donor and acceptor atoms are all labelled are kept; both `R` tokens are subject to `homo_lumo_rank_range`, so widen or disable it to keep these columns)
- HOMO and LUMO energies (eV, from the singlepoint MOs): `HOMO_energy`, `LUMO_energy` :white_check_mark:
- Other frontier orbital descriptors (gap, orbital coefficients) :x: not implemented yet

**Redox and conceptual DFT** (vertical `oxidation` / `reduction` steps, `descriptors.orca.redox`)

- Vertical ionization potential and electron affinity (Hartree): `Vertical_Ionization_Potential`, `Vertical_Electron_Affinity` :white_check_mark: (computed as `E(cation) − E(neutral)` and `E(neutral) − E(anion)`, positive for bound electrons)
- Hardness, electronegativity, electrophilicity: `Hardness`, `Electronegativity`, `Electrophilicity` :white_check_mark:

**Condensed Fukui functions** (`Z` = charge scheme listed in `redox.prefixes`: `nbo_charge`, `cm5`, `mulliken`, `loewdin`, `chelpG`, `hirshfeld`)

- `fplus_Z_X`, `fminus_Z_X`, `fzero_Z_X` :white_check_mark: (`Z = cm5` inherits the CM5 stub above)

**NMR** (`nmr.property.json`, GIAO)

- Isotropic shielding: `siso_X` :white_check_mark:
- Anisotropic shielding: `saniso_X` :white_check_mark: (dropped by the optional filter)

**Molecular descriptors**

- SCF energy of the singlepoint (Hartree): `SCF_energy` :white_check_mark: (aggregation weight, not a descriptor)
- Enthalpy, Gibbs free energy: `Enthalpy`, `GibbsFreeEnergy` :x: not implemented yet (read from `singlepoint.property.json`, which has no thermochemistry; needs `frequency.property.json`)
- Polarizability, molecular and atomic dipoles from `%elprop` :x: not implemented yet (computed by ORCA, not extracted)
- Level of theory (`opt_method`, `svp_theory`) :x: not implemented yet (reader exists, not wired to the spec)

### DescriPyTor descriptors

Computed by DescriPyTor `Molecules` from the per-conformer `orca_condensed.json` (`descriptors.descripytor`).
Each key takes a list of label tuples; the shipped LNiH2 spec uses the same bonds and angles as the
ORCA families.

**Fragment specific geometric descriptors**

- Sterimol (CPK radii, whole complex, metal connectivity from Alvarez covalent radii): `L_X-X`, `B1_X-X`, `B5_X-X`, `loc_B5_X-X`, `B1-B5_angle_X-X` :white_check_mark: / `L` almost matches morfeus `Sterimol_X_X_L`; `B1`, `B5` do not
- Bond length: `dist_X-X` :white_check_mark: / matches `Bond_X_X`
- Angle: `angle_X-X-X` :white_check_mark: / matches `Angle_X_X_X`
- Cube Sterimol (from a density `.cube` file) :x: not implemented yet
- Buried volume, cone angle, dispersion from DescriPyTor's morfeus suite :x: not implemented yet (covered by the ORCA side)

**Charges**

- NBO charge: `charge_NBO_X` :white_check_mark: / matches `npa_X_charge`
- NBO charge difference: `charge_NBO_diff_X-X` :white_check_mark:
- Hirshfeld / CM5 charges through DescriPyTor :x: not implemented yet (the pseudo-Gaussian log does not carry them; use the ORCA side)

**Vibrations** (frequencies and normal modes of the `frequency` step)

- Stretching mode of a bond, within `[stretch_threshold, upper_stretch_threshold]` cm⁻¹: `vib_freq_X-X`, `vib_amp_X-X` :white_check_mark:
- Bending mode of two atoms sharing a centre, above `bend_threshold` cm⁻¹: `vib_freq_bend_X-X`, `vib_amp_bend_X-X` :white_check_mark:
- Ring vibrations (`cross` / `para`) :x: not implemented yet

**Dipole** (frame defined by `[origin, y-axis atom, xy-plane atom]`)

- Norm: `dipole` (one triple) or `dipole_X-X-X` (several) :white_check_mark:
- Components in the transformed frame: `dipole_x_X-X-X`, `dipole_y_X-X-X`, `dipole_z_X-X-X` :white_check_mark:

Polarizability and IR intensities are stored in `orca_condensed.json` but no DescriPyTor family
consumes them. Enthalpy and Gibbs free energy :x: not implemented yet (they live in
`frequency.property.json`, which neither the readers nor the condenser open for thermochemistry).

## References

Software

- **ORCA 6.1.** Neese, F. *WIREs Comput. Mol. Sci.* **2012**, *2*, 73, [10.1002/wcms.81](https://doi.org/10.1002/wcms.81); Neese, F. *WIREs Comput. Mol. Sci.* **2022**, *12*, e1606, [10.1002/wcms.1606](https://doi.org/10.1002/wcms.1606).
- **GOAT** conformer search. Zivkovic, A.; de Souza, B.; Neese, F.; Bistoni, G. *Angew. Chem. Int. Ed.* **2025**, e202418338, [10.1002/anie.202418338](https://doi.org/10.1002/anie.202418338).
- **NBO 7.** Glendening, E. D.; Landis, C. R.; Weinhold, F. *J. Comput. Chem.* **2019**, *40*, 2234, [10.1002/jcc.25873](https://doi.org/10.1002/jcc.25873).
- **DescriPyTor.** Milo group, [github.com/Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor), modelled after *MoleculaR* ([barkais.github.io](https://barkais.github.io/)).
- **morfeus.** Jorner, K. et al., [github.com/digital-chemistry-laboratory/morfeus](https://github.com/digital-chemistry-laboratory/morfeus) (Sterimol, buried volume, SASA, pyramidalization).
- **DBSTEP.** Luchini, G.; Paton, R. S., [github.com/patonlab/DBSTEP](https://github.com/patonlab/DBSTEP) (Sterimol scans).
- **RDKit**, [rdkit.org](https://www.rdkit.org); **Open Babel.** O'Boyle, N. M. et al. *J. Cheminf.* **2011**, *3*, 33, [10.1186/1758-2946-3-33](https://doi.org/10.1186/1758-2946-3-33).
- Descriptor selection strategy after the Sigman group's [Get_Properties](https://github.com/SigmanGroup/Get_Properties).

Descriptor definitions

- **Sterimol.** Verloop, A. In *Drug Design*; Ariëns, E. J., Ed.; Academic Press, 1976; Vol. III. Conformer-dependent Sterimol: Brethomé, A. V.; Hadlington, T. J.; Paton, R. S. *ACS Catal.* **2019**, *9*, 2313, [10.1021/acscatal.8b04043](https://doi.org/10.1021/acscatal.8b04043).
- **Buried volume, quadrants, octants.** Falivene, L.; Cao, Z.; Petta, A.; Serra, L.; Poater, A.; Oliva, R.; Scarano, V.; Cavallo, L. *Nat. Chem.* **2019**, *11*, 872, [10.1038/s41557-019-0319-5](https://doi.org/10.1038/s41557-019-0319-5).
- **Pyramidalization.** P: Radhakrishnan, T. P.; Agranat, I. *Struct. Chem.* **1991**, *2*, 107; P angle: Gavrish, S. P. *J. Comput. Chem.* **2012**, *33*, 2173, [10.1002/jcc.23051](https://doi.org/10.1002/jcc.23051).
- **SASA.** Shrake, A.; Rupley, J. A. *J. Mol. Biol.* **1973**, *79*, 351, [10.1016/0022-2836(73)90011-9](https://doi.org/10.1016/0022-2836(73)90011-9).
- **Covalent radii** (metal connectivity in DescriPyTor). Cordero, B. et al. *Dalton Trans.* **2008**, 2832, [10.1039/b801115j](https://doi.org/10.1039/b801115j).
- **Hirshfeld charges.** Hirshfeld, F. L. *Theor. Chim. Acta* **1977**, *44*, 129, [10.1007/BF00549096](https://doi.org/10.1007/BF00549096).
- **CHELPG.** Breneman, C. M.; Wiberg, K. B. *J. Comput. Chem.* **1990**, *11*, 361, [10.1002/jcc.540110311](https://doi.org/10.1002/jcc.540110311).
- **CM5** (target of the not-yet-implemented column). Marenich, A. V.; Jerome, S. V.; Cramer, C. J.; Truhlar, D. G. *J. Chem. Theory Comput.* **2012**, *8*, 527, [10.1021/ct200866d](https://doi.org/10.1021/ct200866d).
- **NPA / NBO.** Reed, A. E.; Weinstock, R. B.; Weinhold, F. *J. Chem. Phys.* **1985**, *83*, 735, [10.1063/1.449486](https://doi.org/10.1063/1.449486); Reed, A. E.; Curtiss, L. A.; Weinhold, F. *Chem. Rev.* **1988**, *88*, 899, [10.1021/cr00088a005](https://doi.org/10.1021/cr00088a005).
- **Fukui functions.** Parr, R. G.; Yang, W. *J. Am. Chem. Soc.* **1984**, *106*, 4049, [10.1021/ja00326a036](https://doi.org/10.1021/ja00326a036); condensed form: Yang, W.; Mortier, W. J. *J. Am. Chem. Soc.* **1986**, *108*, 5708, [10.1021/ja00279a008](https://doi.org/10.1021/ja00279a008).
- **Hardness, electronegativity, electrophilicity.** Parr, R. G.; Pearson, R. G. *J. Am. Chem. Soc.* **1983**, *105*, 7512, [10.1021/ja00364a005](https://doi.org/10.1021/ja00364a005); Parr, R. G.; Szentpály, L. v.; Liu, S. *J. Am. Chem. Soc.* **1999**, *121*, 1922, [10.1021/ja983494x](https://doi.org/10.1021/ja983494x).
