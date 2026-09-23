# orca_descriptors

Compile a per-molecule descriptor table from ORCA calculations of conformer ensembles.

The chemistry is not hard-coded. A **spec** (one YAML file per molecule family) declares which atoms
are labelled, how they are found in every conformer, which descriptors are computed on which labels,
and how the conformers are aggregated. The same code then serves any molecule family. The output is the
complete descriptor library; a noise filter for modelling sets exists but is off by default.
Two engines compute the descriptors: the in-house ORCA readers (`desc_utils`) and
[DescriPyTor](https://github.com/Milo-group/DescriPyTor) (git submodule).

## Contents

- [Installation](#installation)
- [Descriptor extraction guidelines](#descriptor-extraction-guidelines)
  - [Prerequisites](#prerequisites)
  - [A. Run the calculations on the cluster](#a-run-the-calculations-on-the-cluster)
  - [B. Condense the ORCA outputs, on the cluster](#b-condense-the-orca-outputs-on-the-cluster)
  - [C. Copy the condensed files to your laptop](#c-copy-the-condensed-files-to-your-laptop)
  - [D. Extract the descriptors, on your laptop](#d-extract-the-descriptors-on-your-laptop)
  - [Reproduce the shipped LNiH2 descriptors](#reproduce-the-shipped-lnih2-descriptors)
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

## Installation

```bash
git clone https://github.com/ReismanLab/orca_descriptors
cd orca_descriptors
git checkout code-review             # the spec-driven pipeline lives on this branch until it is merged into main
git submodule update --init          # DescriPyTor, pinned commit
conda env create -f gp_env.yml       # environment "gp_env"
conda activate gp_env
```

On the cluster the home folder is too small for the environment: use
[setup_gp_env.sh](setup_gp_env.sh) instead of the last two lines, it puts `gp_env` in the group
folder (see [Prerequisites](#prerequisites)).

An existing `gp_env` created before 2026-09-22 lacks three packages that were added to `gp_env.yml`
since (Open Babel for the xyz to sdf conversion of the numbering step, networkx and plotly imported by
DescriPyTor). Inside the environment:

```bash
pip install openbabel-wheel==3.1.1.23 networkx==3.6.1 plotly==7.1.0
```

DescriPyTor is imported from the `DescriPyTor/` folder; do **not** `pip install` it into `gp_env`
(its `numpy<2` pin would downgrade the environment). The submodule pin is the version the regression
tests were validated against (`f6bc24c`, 2026-08-31).

The ORCA jobs themselves need ORCA 6.1 and NBO 7 on the cluster (see
[scripts-for-orca-computations](scripts-for-orca-computations)).

## Descriptor extraction guidelines

A copy-paste recipe for reproducing or computing descriptors from the command line. You do not need
the rest of this document to run it; the [detailed pipeline](#pipeline-in-detail-from-orca-files-to-the-final-descriptors)
explains what each command does. Four steps: **A** and **B** run on the cluster, **C** and **D** on
your laptop. Placeholders: `<LIB>` is the library folder name, `<cluster>` the login node, `<user>`
your account.

### Prerequisites

Two machines, and they need very different things:

| Where | Steps | Needs |
|---|---|---|
| The cluster | A, B | ORCA 6.1 and NBO 7 for the calculations. For `condense`: a `git clone` of this repository and any Python 3.8 or newer, **no conda environment at all** (`desc_utils.condense` and `desc_utils.orca_parse` import the standard library only, so `module load python` is enough). |
| Your laptop | C, D | The `gp_env` environment below: rdkit, morfeus, DescriPyTor, pandas and the rest. |

Once, on your laptop:

```bash
git clone https://github.com/ReismanLab/orca_descriptors
cd orca_descriptors
git checkout code-review             # the pipeline is on this branch until it is merged into main
git submodule update --init          # DescriPyTor, pinned commit
conda env create -f gp_env.yml       # environment "gp_env" (includes Open Babel for the numbering step)
conda activate gp_env
pytest tests -m "not slow"           # expected: all passed, a few seconds
```

A conda environment does not fit in the home quota of the cluster, and one private copy per group
member is a waste. [setup_gp_env.sh](setup_gp_env.sh) builds a single `gp_env` in the group folder
instead; replace the last three lines of the block above with it:

```bash
bash setup_gp_env.sh                 # environment + package cache in /groups/ReismanLab/miniconda3, then the tests
conda activate gp_env
```

The first person to run it builds the environment, ten minutes or so. Everybody else in the group
runs the same command, and it only points their conda at the shared folder, a few seconds:

```bash
bash setup_gp_env.sh --no-test
conda activate gp_env
```

The script prepends the group `envs` and `pkgs` folders to `envs_dirs` and `pkgs_dirs` in your
`~/.condarc`, so `conda env create` and `conda activate <name>` keep being used exactly as above but
read and write `/groups/ReismanLab/miniconda3/envs` from now on, this environment and the next ones.
It then reports whether conda really picked the setting up, rather than assuming it did. The package
cache moves along with the environments on purpose: conda can only hardlink a cached package into an
environment on the same filesystem, and copies every file otherwise.

The two shared folders are left group-writable (setgid, so that new files keep the group) but the
environment itself is left read-only for the group: everyone runs it, only its owner can change it,
and a stray `pip install` cannot break it for the whole lab. Refreshing it after a change to
`gp_env.yml` is therefore the owner's job:

```bash
conda env update --file gp_env.yml --prefix /groups/ReismanLab/miniconda3/envs/gp_env --prune
```

Options: `--root PATH` for another shared folder, `--no-test` to skip the test suite, `--all-users`
to write the setting into `$ROOT/.condarc` rather than each `~/.condarc`, which only helps when
`$ROOT` is the conda installation the whole group runs (the script says so if it is not). Re-running
the script is harmless; it leaves an existing environment alone. If conda stops on an Anaconda Terms
of Service error, run the two `conda tos accept` commands it prints and start the script again. A
conda 22.x installation whose `conda env` subcommand is missing its `conda-env` executable is worked
around.

Every command of step D runs on your laptop, from the repository root, with `gp_env` active. Steps A
and B run on the cluster and need no environment.

### A. Run the calculations on the cluster

> **Placeholder, to be written by Audrey.** How to launch a descriptor campaign: from the list of
> ligands or SMILES to submitted ORCA jobs. What belongs here is the conformer search and its
> settings, the naming of the folders, the generation of the `calcParam.inp` files and how the jobs
> are submitted and monitored ([launch_orca_all.py](scripts-for-orca-computations/launch_orca_all.py),
> [slurm-calcParam.sh](scripts-for-orca-computations/slurm-calcParam.sh)), plus whatever has to be
> checked before a campaign of several hundred conformers is launched.

Whatever that procedure is, the rest of the pipeline expects it to produce, somewhere on the cluster:

```
<LIB>/<molecule InChIKey>/conf_N/     one folder per conformer, holding the output of an ORCA 6.1
                                      compound job run with calcParam.inp
```

Four things are read out of each conformer folder in step B: `optimization.xyz`, the
`*.property.json` files, the GBW dump `singlepoint.json` and `calcParam.out` (NBO and frequencies).
A different input file or a different folder layout is fine, but it then needs a spec of its own,
see [2. Spec](#2-spec).

### B. Condense the ORCA outputs, on the cluster

An ORCA compound job leaves about 70 MB per conformer. `condense` reduces that to a single
`orca_condensed.json` of 0.4 to 1.5 MB holding everything the descriptors need, so that this is all
that has to travel. Skip this step if [slurm-calcParam.sh](scripts-for-orca-computations/slurm-calcParam.sh)
already ran it at the end of each job.

**This is the one step that needs no environment**: `python -m desc_utils.condense` imports only the
Python standard library, so any interpreter 3.8 or newer will do and `gp_env` is irrelevant here. Do
not `conda activate` anything.

Condensing a library reads every raw ORCA output it has, which is not login-node work: submit
[slurm-condense.sh](scripts-for-orca-computations/slurm-condense.sh) instead.

```bash
ssh <user>@<cluster>
git clone https://github.com/ReismanLab/orca_descriptors              # once, anywhere; no environment, no pip install
git -C orca_descriptors checkout code-review
sbatch orca_descriptors/scripts-for-orca-computations/slurm-condense.sh /path/to/<LIB>
```

Nothing to edit in the script: it finds the clone from the path it was submitted with. If you keep
a copy of it outside the clone, name the clone yourself with
`sbatch --export=ALL,REPO=$HOME/orca_descriptors ...`.

One node, eight cores, two hours by default; raise the `#SBATCH` lines for a large library. The
condensed files themselves are written in place, one `orca_condensed.json` per `conf_N/` folder of
the library. The log of the run, `condense_<jobid>.out` and `.err`, lands in the directory you
submitted from (`sbatch -o /elsewhere/condense_%j.out -e ...` to put it somewhere else). It ends
with `condensed N folder(s), skipped 0, failed 0` and
`checked N file(s): N ok, 0 failed`, and it comes back FAILED if any conformer could not be
condensed or if a file it wrote does not check out. Folders that already have their
`orca_condensed.json` are skipped, so submitting the same job again after a new batch of ORCA runs
only does the new conformers. A `failed` line names a conformer whose ORCA run did not finish; see
[If something goes wrong](#if-something-goes-wrong).

These are the two commands the job runs, if you ever need them by hand (a couple of conformers, an
interactive allocation, raw folders sitting on your laptop):

```bash
PYTHONPATH=$PWD/orca_descriptors python -m desc_utils.condense --root /path/to/<LIB> --jobs 4
PYTHONPATH=$PWD/orca_descriptors python -m desc_utils.condense --root /path/to/<LIB> --check
```

`PYTHONPATH` is the folder that *contains* `desc_utils/`, `--root` the library folder.

### C. Copy the condensed files to your laptop

About 0.5 MB per conformer, against ~70 MB for the raw folders. From the repository root on your
laptop:

```bash
rsync -avz --prune-empty-dirs --include='*/' --include='optimization.xyz' --include='orca_condensed.json' \
      --exclude='*' <user>@<cluster>:/path/to/<LIB>/  DATA/<LIB>/
```

`optimization.xyz` comes along because the numbering step reads the geometry from it. Nothing else
from the cluster is needed again: from here on the pipeline only reads `DATA/<LIB>/`.

### D. Extract the descriptors, on your laptop

With `gp_env` active, from the repository root.

**D1. Write the spec.** Copy the closest example and edit the marked fields:

```bash
cp specs/lnih2_pyox.yaml specs/<LIB>.yaml        # Ni complexes with bidentate N,N ligands
# or: cp specs/test_molecules_at1_at2.yaml specs/<LIB>.yaml   # small molecules, first atoms labelled
```

| Field | Set to |
|---|---|
| `name` | `<LIB>` (used in the output file name) |
| `data.root` | `DATA/<LIB>` |
| `atoms` | one label per atom of interest, `label: element` (`null` = any element) |
| `numbering` | how each label is found: `smarts` stages (see the LNiH2 spec), `first_n` (labels = atoms 1..N of the xyz), or `manual` with `csv:` pointing to your own table of 1-based indices |
| `descriptors.orca.*`, `descriptors.descripytor` | label lists / label tuples; delete the terms you do not need |
| `filter.metal_label` | the metal label, or `null` for organic molecules |

```bash
python -m desc_utils.run validate --spec specs/<LIB>.yaml   # expected: "OK  specs/<LIB>.yaml"
```

**D2. Number the atoms.** Which atom of each conformer carries which label:

```bash
python -m desc_utils.run number --spec specs/<LIB>.yaml --strict   # -> atom_numbering_results.csv (name from the spec)
```

`--strict` stops if any conformer could not be numbered; the table it just wrote lists the offending
rows with the reason in `numbering_status`. Correcting a few indices by hand in that CSV is a normal
outcome, keep the file and pass it to the next step.

**D3. Extract and aggregate:**

```bash
python -m desc_utils.run all --spec specs/<LIB>.yaml            # -> combined_descriptors.csv, <today>_<LIB>_descriptors.csv
```

`all` is `number` then `extract` then `aggregate`. It re-runs the numbering, so use
`all --spec specs/<LIB>.yaml --numbering atom_numbering_results.csv` to keep a hand-corrected table,
or run `extract` and `aggregate` separately. What the three files contain is in
[What you get](#what-you-get).

### Reproduce the shipped LNiH2 descriptors

The LNiH2 / PyOx library (308 conformers, 38 ligands) is not in the repository. Copy it into
`DATA/LNiH2_library_001_to_download_2026_03_24/` either as the raw ORCA folders or, much smaller, as
the condensed files with the rsync command of step C. Then:

```bash
python -m desc_utils.run condense --spec specs/lnih2_pyox.yaml --jobs 4   # ~20 s; skipped for folders that already have orca_condensed.json
python -m desc_utils.run all --spec specs/lnih2_pyox.yaml --numbering atom_numbering_results.csv   # ~6 min
```

Result: `<today>_LNiH2_library_001_to_download_2026_03_24_descriptors.csv`, 38 rows and about 2450
columns, numerically equal to the committed `20260922_LNiH2_library_001_to_download_2026_03_24_descriptors.csv`. To have the test
suite confirm it column by column:

```bash
pytest tests -m slow                 # ~10 min; also re-derives the atom numbering and the DescriPyTor block
```

### Re-run after changing the spec or adding conformers

| You changed | Run |
|---|---|
| Descriptor terms only | `python -m desc_utils.run extract --spec S` then `aggregate --spec S` (morfeus geometry is cached in `<conf>/xyz.json`; add `--overwrite-xyz-json` if you changed `vbur_radii` or `quadrants`) |
| Atom labels or numbering rules | `number --spec S --strict`, then `all --spec S --numbering <csv>` |
| Added conformer folders on the cluster | B (existing files are skipped), C, then `all --spec S` |
| Only the aggregation temperature | `aggregate --spec S` |
| You want the noise filter (correlation / variance pruning) for a modelling set | `aggregate --spec S --filter`, or set `filter.enabled: true` in the spec; needs enough molecules for correlations to mean anything (with two molecules every column is perfectly correlated and one survives) |

### What you get

| File | One row per | Content |
|---|---|---|
| `atom_numbering_results.csv` | conformer | 1-based index of every label, SCF energy, SMILES, `numbering_status` |
| `combined_descriptors.csv` | conformer | every descriptor of both engines, columns named by label (`charge_NBO_Ni`, `%Vbur_atom_Ni_radius_3.5`, `B1_Ni-N1`, ...) |
| `<date>_<name>_descriptors.csv` | molecule | the spreadsheet: `inchi`, `smiles`, `n_conformers`, then `<descriptor>_min`, `_max`, `_lowE` (lowest-energy conformer) and `_boltz` (Boltzmann average at `temperature_K`); every descriptor is kept unless `--filter` is given |

Conventions worth knowing when reading the spreadsheet: energies are in Hartree, HOMO/LUMO in eV,
IP and EA are positive for bound electrons (`IP = E(cation) - E(neutral)`), Fukui functions follow
`f+ = q(N) - q(N+1)`, `cm5_*` equals `hirshfeld_*` (the CM5 correction is not implemented), and
DescriPyTor Sterimol values (`B1_*`, `B5_*`, `L_*`) differ from the morfeus ones (`Sterimol_*`).
The full list is under [Descriptors provided](#descriptors-provided).

### If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `condense` prints `failed ... no complete NBO block (NBO started but never returned)` | the ORCA job aborted; `tail calcParam.out` shows the error | rerun the ORCA job or drop the conformer folder |
| `condense` prints `failed ... optimization.xyz missing` / `singlepoint.property.json missing` | the job did not reach that step, or a different input file was used | same as above; the pipeline needs the `calcParam.inp` steps |
| `numbering_status` is `failed` or `incomplete` | no SMARTS stage matched this conformer | check the SMILES in the table, adjust the stage patterns, or switch to `backend: manual` with your own indices |
| `numbering_status` is `element_mismatch` | the matched atom is not the element declared in `atoms` | fix the SMARTS role positions or the `atoms` element |
| `numbering_status` is `inconsistent` | conformers of one molecule got different indices | usually a symmetric ligand; set `numbering.smarts.match_selection: lowest_indices` or add an `orient` rule |
| `descripytor_notes` says `no orca_condensed.json in ...` | the folder holds raw files only or was not transferred | run step B / C, or `python -m desc_utils.run condense --spec S` locally when the raw files are present |
| `descripytor_notes` says `expected 1 molecule file ..., loaded 0` | the condensed file is unreadable | `python -m desc_utils.condense --root DATA/<LIB> --check` and rebuild with `--overwrite` |
| `nbo_extraction_error` column is not empty | the NBO block of that conformer is missing or truncated | same as the first row |
| `No module named desc_utils.run` | the clone is on `main`, which predates the pipeline | `git checkout code-review` and run the commands from the repository root |
| every row has `numbering_status` `failed` with `No module named 'openbabel'` | Open Babel missing from the environment (needed to convert xyz to sdf) | `pip install openbabel-wheel==3.1.1.23` in `gp_env`, then rerun `number` |
| `extract` stops with `No module named 'networkx'` or `'plotly'` | environment created from an older `gp_env.yml` | `pip install networkx==3.6.1 plotly==7.1.0` in `gp_env` |
| `ImportError: DescriPyTor not found at ...` | submodule not checked out | `git submodule update --init` |
| `numpy.dtype size changed` or similar import errors | a `pip install` of DescriPyTor downgraded numpy | never pip-install DescriPyTor; recreate `gp_env` from `gp_env.yml` |
| `validate` lists `unknown label` | a descriptor term uses a label that is not in `atoms` | add the label or fix the typo; labels are case-sensitive |

## Repository layout

| Path | Content |
|---|---|
| [DATA](DATA) | Calculation outputs: one folder per molecule (InChIKey), one sub-folder per conformer (`conf_*`). |
| [specs](specs) | One YAML spec per molecule family. `lnih2_pyox.yaml` (LNiH2 / PyOx ligand library) and `test_molecules_at1_at2.yaml` (small organic test set) are the shipped examples. |
| [desc_utils](desc_utils) | The pipeline: `spec.py` (spec loading and validation), `numbering/` (atom numbering backends), `extract/` (ORCA and DescriPyTor adapters), `aggregate.py` (Boltzmann aggregation, filtering), `run.py` (CLI), `orca_parse.py` (standard-library ORCA parsers, also run on the cluster), `condense.py` (condensed per-conformer file: producer, checker, loader), `orca_descriptors.py` and `data_extraction.py` (morfeus geometry, extraction functions). |
| [DescriPyTor](DescriPyTor) | Git submodule, [Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor). |
| [orca2gaussian](orca2gaussian) | Deprecated. Former converter writing a pseudo-Gaussian log per conformer for DescriPyTor (unit and layout bugs, see CODE_REVIEW C3 to C7). Replaced by the condensed JSON; only used with `descriptors.descripytor.input.kind: feather`. |
| [scripts-for-orca-computations](scripts-for-orca-computations) | ORCA compound input (`calcParam.inp`), SLURM scripts and launcher used on the cluster: `slurm-calcParam.sh` (one ORCA job), `launch_orca_all.py`, `slurm-condense.sh` (condense a whole library on a compute node). |
| [tests](tests) | Unit tests and regression tests against the committed descriptor CSVs. |
| `setup_gp_env.sh` | One-shot installer for `gp_env` in the group folder rather than the home folder (`envs_dirs` and `pkgs_dirs`, environment, tests). |
| [review](review) | Notes and scripts from the DescriPyTor upgrade (old vs new Sterimol comparison, per-conformer reference CSVs). |
| `make_descriptor_descripytor.ipynb` | Notebook running the same pipeline interactively. |
| `atom_numbering_results.csv`, `combined_descriptors.csv`, `<date>_<name>_descriptors.csv` | Outputs of the three stages for the LNiH2 library (regression references). |

## Pipeline in detail: from ORCA files to the final descriptors

```
ORCA compound job per conformer                                                            (cluster)
        │  optimization.xyz, *.property.json, singlepoint.json (GBW dump), calcParam.out (NBO, frequencies)
        ▼
[1] condense  ──►  <conf>/orca_condensed.json      (standard-library Python, ~0.4 to 1.5 MB per conformer)
        │  rsync optimization.xyz + orca_condensed.json                                        (laptop)
        ▼
[2] spec  ──►  [3] number  ──►  [4] extract  ──►  [5] aggregate (+ optional filter)
                atom_numbering_results.csv   combined_descriptors.csv   <date>_<name>_descriptors.csv
```

All stages read the same spec. From the repository root, in `gp_env`:

```bash
python -m desc_utils.run validate  --spec specs/lnih2_pyox.yaml
python -m desc_utils.run condense  --spec specs/lnih2_pyox.yaml            # -> <conf>/orca_condensed.json (normally run on the cluster)
python -m desc_utils.run number    --spec specs/lnih2_pyox.yaml            # -> atom_numbering_results.csv
python -m desc_utils.run extract   --spec specs/lnih2_pyox.yaml            # -> combined_descriptors.csv (per conformer)
python -m desc_utils.run aggregate --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_descriptors.csv
python -m desc_utils.run all       --spec specs/lnih2_pyox.yaml            # number -> extract -> aggregate
```

`python -m desc_utils.run <command> --help` lists the options (`--numbering` to reuse an existing table,
`--families` to run a subset of descriptor families, `--strict` to fail on any numbering problem,
`--filter` to apply the spec's noise filter, `--overwrite-xyz-json`).

### 0. ORCA calculations

Every conformer is computed with the compound job
[scripts-for-orca-computations/calcParam.inp](scripts-for-orca-computations/calcParam.inp)
(ORCA 6.1, `launch_orca_all.py` creates one `conf_N/` folder per conformer of an xyz file and submits
`slurm-calcParam.sh`). The steps and the files read downstream are:

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
rsync -avz --prune-empty-dirs --include='*/' --include='optimization.xyz' --include='orca_condensed.json' \
      --exclude='*' user@cluster:/path/to/LIB/  DATA/LIB/
```

Content (schema version 1; units in the key names; `null` when ORCA did not print a quantity):

| Section | Content | Source |
|---|---|---|
| `molecule` | atoms with Å coordinates, charge, multiplicity | `optimization.xyz` |
| `states.<singlepoint / oxidation / reduction / nmr>` | SCF energy, electron counts, Mulliken / Loewdin / CHELPG / Hirshfeld charges, dipole (a.u.), polarizability, thermochemistry, NMR `siso` / `saniso`, MO energies and occupations (`singlepoint` only by default, `--mo-states`) | `<state>.property.json`, `<state>.json` |
| `nbo.blocks[]` | `NBOProps.results` per NBO block: NPA, NAO occupancies, NBO summary, hybrids, NHO directionality, second-order perturbation, cycle info. Block 0 (neutral) in full; the cation and anion blocks keep NPA only unless `--nbo-full all` | `calcParam.out` |
| `properties`, `vibrations`, `charges` | DescriPyTor's JSON schema: dipole in Debye (singlepoint), polarizability, frequencies with unit-norm displacements per mode (translations and rotations dropped, IR intensities where printed), NPA / Hirshfeld / CM5 charges | derived from the above |
| `producer` | ORCA and NBO versions, host, time, git commit, size and mtime of every input file | |

Every reader (`OrcaReader`, the NBO loader, `scf_energy` of the numbering stage, DescriPyTor's
`Molecules`) uses the condensed file when `<conf>/<data.condensed_file>` exists and the raw files
otherwise, so folders downloaded before this step keep working; `data.condensed_file: null` forces the
raw files. `python -m desc_utils.condense --check` validates existing files, and `tests/test_condense.py`
asserts that every in-house column is identical whichever source is read.

### 2. Spec

A spec has these sections (defaults in `desc_utils/spec.py::DEFAULTS`):

| Section | Content |
|---|---|
| `data` | `root` folder, `layout: molecule/conformer`, `conformer_glob`, `geometry_file`, `molecule_id_column`. |
| `atoms` | Ordered map `label -> expected element` (`null` = any). Labels are the only way descriptors refer to atoms. |
| `numbering` | `backend` (`smarts`, `manual`, `first_n`, `molalign`), `index_base`, per-backend block, optional `orient` rules, `annotators`, `smiles` mode, `energy` source. |
| `descriptors.orca` | Four families with label lists: `xyz` (atoms, bonds, angles, dihedrals, quadrants, `vbur_radii`), `charges`, `nbo` (`homo_lumo_rank_range`, `nbo_block_index`, `spin`), `redox` (Fukui `prefixes`, `nbo_block_map`). |
| `descriptors.descripytor` | DescriPyTor "Schema B" keys with label tuples: `charges`, `charge_diff`, `sterimol`, `bond_length`, `bond_angle`, `stretching_vibration`, `bending_vibration`, `dipole`, plus thresholds, `radii`, `drop_atoms`. |
| `aggregate` | `group_by`, `energy_column`, `temperature_K`, `kB_hartree_per_K`, `stats`, `carry_first`. |
| `filter` | Every option of `filter_numeric_descriptors`, plus `metal_label`. |
| `output` | File names of the three stage outputs. |

`validate` reports every problem at once (unknown labels, wrong arity, SMARTS that do not parse, roles
outside the SMARTS, labels never assigned). Quote every SMARTS string: `#` starts a YAML comment.

### 3. Atom numbering

`number` writes one row per conformer folder with the 1-based index of every label, plus `path`,
`inchi`, `SCF_energy`, `smiles`, annotator columns (e.g. `ligand_class`), `numbering_backend`,
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
  from `orca_condensed.json` (`input.kind: json`; `feather` selects the deprecated orca2gaussian path),
  one call per family; the `FAMILIES` table maps DescriPyTor's index-based column names to label-based
  names. `python -m desc_utils.run to-schema-b --spec ... --row 0` prints the index-based JSON that
  DescriPyTor's own tools consume for one conformer.

The result is `combined_descriptors.csv`, one row per conformer, merged on `path`.

### 5. Aggregation and filtering

`aggregate` groups conformers by molecule (`group_by: inchi`) and writes, for every numeric descriptor,
`<name>_min`, `<name>_max`, `<name>_lowE` (lowest-energy conformer) and `<name>_boltz`, the Boltzmann
average at `temperature_K` with weights `exp(-ΔE / kB T)` computed from `energy_column` (the SCF energy
of the `singlepoint` step, no thermal correction). `n_conformers` and the `carry_first` columns are
kept. This complete table is the default output. Optionally, with `filter.enabled: true` in the spec or
`aggregate --filter`, `filter_numeric_descriptors` then removes: non-numeric and all-NaN columns, NBO orbitals whose
Boltzmann occupancy is below `occupancy_boltz_threshold`, Rydberg orbitals (except on `metal_label`),
metal 4f Rydbergs, core orbitals, anisotropic NMR terms, NHO φ/θ angles, near-constant columns
(`ratio_threshold`) and one member of every pair above `corr_threshold`. The output is
`<date>_<name>_descriptors.csv` with `inchi_folder`, `inchi`, `smiles`, `ligand_class` in front.

### Adding a new molecule family

Copy a spec, then edit `atoms`, `numbering` (SMARTS stages, or `manual` with your CSV, or `first_n`),
the label lists under `descriptors`, and `filter.metal_label`. Run `validate`, then `number` and check
the `numbering_status` column before `extract`.

### Tests

```bash
pytest tests -m "not slow"     # unit tests, seconds (includes the condensed-file round trip on the tracked fixtures)
pytest tests -m slow           # regression: reproduces atom_numbering_results.csv, the DescriPyTor
                               # per-conformer reference and the committed aggregated CSVs (minutes)
```

The slow tests are skipped when `DATA/` or the submodule is absent.

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
- Sterimol (morfeus, CRC van der Waals radii; L axis from first to second label): `Sterimol_X_X_L`, `Sterimol_X_X_Bmin`, `Sterimol_X_X_Bmax` :white_check_mark: / B1 and B5 do not match DescriPyTor `B1_X-X`, `B5_X-X` (different radii and slicing; see [review/DESCRIPYTOR_UPGRADE.md](review/DESCRIPYTOR_UPGRADE.md))
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
- CM5: `cm5_X` :x: not implemented yet (the current column is a stub equal to `hirshfeld_X`, see [CODE_REVIEW.md](CODE_REVIEW.md) C1)
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

Level of theory

- **ωB97M-V.** Mardirossian, N.; Head-Gordon, M. *J. Chem. Phys.* **2016**, *144*, 214110, [10.1063/1.4952647](https://doi.org/10.1063/1.4952647); **D4.** Caldeweyher, E. et al. *J. Chem. Phys.* **2019**, *150*, 154122, [10.1063/1.5090222](https://doi.org/10.1063/1.5090222).
- **M06.** Zhao, Y.; Truhlar, D. G. *Theor. Chem. Acc.* **2008**, *120*, 215, [10.1007/s00214-007-0310-x](https://doi.org/10.1007/s00214-007-0310-x).
- **def2 basis sets.** Weigend, F.; Ahlrichs, R. *Phys. Chem. Chem. Phys.* **2005**, *7*, 3297, [10.1039/b508541a](https://doi.org/10.1039/b508541a).
