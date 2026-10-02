# orca_descriptors

From a list of SMILES to a per-molecule descriptor table: conformer ensembles, one ORCA compound job per
conformer, and descriptors computed from the ORCA output, driven by two YAML files.

- A **library config** (one per campaign of calculations) says how the ensembles are generated and what ORCA
  runs. `orcadesc submit <LIB>.yaml` on the cluster does the rest and records everything needed to reproduce
  the library.
- A **spec** (one per molecule family) says which atoms are labelled, how they are found in every conformer
  and which descriptors are computed on them. `python -m desc_utils.run all --spec specs/<LIB>.yaml` on your
  laptop writes the tables.

No chemistry is hard-coded: the same code serves any molecule family. The ensembles come from `bceg`
(imported from [ReismanLab/B-CEG-MCD](https://github.com/ReismanLab/B-CEG-MCD)); the descriptors from two
engines, the in-house ORCA readers (`desc_utils`) and [DescriPyTor](https://github.com/Milo-group/DescriPyTor)
(git submodule).

```
cluster   orcadesc submit <LIB>.yaml :  SMILES ─► ensembles (bceg) ─► pruning ─► ORCA job per conformer ─► orca_condensed.json
                                                                                                                   │ rsync
laptop    desc_utils.run all --spec  :  number ─► extract ─► aggregate ─► <date>_<LIB>_descriptors.csv  ◄──────────┘
```

## Install (laptop, once)

```bash
git clone https://github.com/ReismanLab/orca_descriptors
cd orca_descriptors
git submodule update --init          # DescriPyTor, pinned commit
conda env create -f gp_env.yml       # environment "gp_env"
conda activate gp_env
pip install -e . --no-deps           # `orcadesc` and `bceg` on PATH, imports work from anywhere
pytest tests -m "not slow"           # seconds; a few bceg tests are skipped without xtb
```

Never `pip install` DescriPyTor into `gp_env`: it is imported from the `DescriPyTor/` folder, and its
`numpy<2` pin would downgrade the environment.

On the cluster nothing is installed: the shared `gp_env` and the group clone exist. The first time, fetch
the UMA weights into the shared cache ([docs/cluster_setup.md](docs/cluster_setup.md)).

## Run a library

`<LIB>` is the library name (one campaign, e.g. `LNiH2_library_002`); `$LIB` its folder on the cluster, laid
out as `<LIB>/<InChIKey>/conf_N/`.

### On the cluster: ensembles and ORCA

```bash
source /groups/ReismanLab/miniconda3/etc/profile.d/conda.sh && conda activate gp_env
mkdir -p ~/libraries/<LIB> && cd ~/libraries/<LIB>
cp /groups/ReismanLab/orca_descriptors/configs/orca_library_example.yaml <LIB>.yaml
nano ligands.txt                          # one SMILES per line, optional name after it
nano <LIB>.yaml                           # one block per stage, see the table below
orcadesc submit <LIB>.yaml --dry-run      # what would run; writes nothing
orcadesc submit <LIB>.yaml                # ensembles, pruning, one ORCA job per conformer, condensing
orcadesc status <LIB>.yaml                # one row per molecule, one column per stage
orcadesc cost   <LIB>.yaml                # core-hours and GPU-hours per stage
```

The config has one block per stage. Every block is documented with all its options and defaults:

| Stage | Block | Chooses |
|---|---|---|
| 1 | `transformation` | the structure computed from each SMILES: as is, or L-NiH2 via `ni_x2.py`, or your script; charge and multiplicity ([docs/stages/1_transformation.md](docs/stages/1_transformation.md)) |
| 2 | `pre_refinement` | the pre-optimisation of the start geometry: `xtb`, `uma` or `none` ([docs/stages/2_pre_refinement.md](docs/stages/2_pre_refinement.md)) |
| 3 | `conformer_search` | the searcher (`crest`, `goat`, `molassembler`, `rdkit_etkdg`, `loqi`), its parameters, replicates ([docs/stages/3_conformer_search.md](docs/stages/3_conformer_search.md)) |
| 4 | `post_optimization` | the refiner every conformer goes through: `uma`, `gxtb`, `aimnet2` or none ([docs/stages/4_post_optimization.md](docs/stages/4_post_optimization.md)) |
| 5 | `pruning` | which conformers go to ORCA: all, or an energy window ([docs/stages/5_pruning.md](docs/stages/5_pruning.md)) |
| 6 | `dft` | the compound steps and their level of theory; `orca: off` stops after the ensembles ([docs/stages/6_dft.md](docs/stages/6_dft.md)) |
|   | `slurm`, `condense` | job set-up lines and resource profiles; condense each ORCA output when its job ends |

The six stage blocks define the results and are frozen in `$LIB/params.json`: a later submit with other
values is refused. Running `submit` again is how a library grows or recovers: finished molecules and
conformers are skipped, failed ones are resubmitted. Everything else (what each command prints, the status
values, what is recorded, how to reproduce a library) is in
[docs/library_workflow.md](docs/library_workflow.md).

ORCA jobs condense their own output. If a job died before the end, rebuild the missing files on a compute
node (finished ones are skipped) before transferring ([docs/condense.md](docs/condense.md)):

```bash
cd $LIB && sbatch /groups/ReismanLab/orca_descriptors/scripts-for-orca-computations/slurm-condense.sh $LIB
```

### On the laptop: descriptors

Only the geometry and the condensed file travel (about 0.5 MB per conformer). No slash after `<LIB>`.

```bash
conda activate gp_env
rsync -avz --include='*/' --include='optimization.xyz' --include='orca_condensed.json' \
      --exclude='*' <user>@<cluster>:/path/to/<LIB>  DATA/
cp specs/lnih2_pyox.yaml specs/<LIB>.yaml         # closest example; small organics: specs/test_molecules_at1_at2.yaml
nano specs/<LIB>.yaml                              # fields below
python -m desc_utils.run all --spec specs/<LIB>.yaml --jobs 4
```

| Spec field | Set to |
|---|---|
| `name` | `<LIB>` (used in the output file names) |
| `data.root` | `DATA/<LIB>` |
| `atoms` | one label per atom of interest, `label: element` (`null` = any element) |
| `numbering` | how each label is found: `smarts` stages (see the LNiH2 spec), `first_n` (labels = atoms 1..N of the xyz), or `manual` with `csv:` pointing to your own table of 1-based indices |
| `descriptors.orca.*`, `descriptors.descripytor` | label lists / label tuples; delete the terms you do not need |
| `filter.metal_label` | the metal label, or `null` for organic molecules |

Three tables come out, named with the date and the spec name so nothing is overwritten:

| File | One row per | Content |
|---|---|---|
| `<date>_<LIB>_atom_numbering_results.csv` | conformer | the index of every label, SCF energy, SMILES, `numbering_status`; correct it by hand and rerun with `--numbering <csv>` |
| `<date>_<LIB>_combined_descriptors.csv` | conformer | every descriptor of both engines, named by label (`charge_NBO_Ni`, `%Vbur_atom_Ni_radius_3.5`), the ORCA job status and cost |
| `<date>_<LIB>_descriptors.csv` | molecule | the spreadsheet: `_min`, `_max`, `_lowE` and `_boltz` of every descriptor, `all_conformers_completed`, DFT core-hours |

A conformer that could not be numbered or whose ORCA job is unfinished never stops the run: it is left
out, the molecule is flagged, and `<date>_<LIB>_conformers_to_check.csv` lists it with the reason. Which
command to rerun after changing a spec, the geometry-only mode for UMA structures (descriptors without any
ORCA job), the column conventions and the optional noise filter are in
[docs/descriptor_pipeline.md](docs/descriptor_pipeline.md).

## Documentation

| Read | For |
|---|---|
| [docs/cluster_setup.md](docs/cluster_setup.md) | the shared environment, fetching UMA / AIMNet2 / LoQI weights into the offline caches, environment pitfalls |
| [docs/stages/](docs/stages) | every option and parameter of the six stages of a library config, with examples |
| [docs/library_workflow.md](docs/library_workflow.md) | the config in full, what `submit` does, the `status` columns and states, what is recorded, reproducing a library |
| [docs/condense.md](docs/condense.md) | the ORCA compound job and its files, condensing, transfer, the schema of `orca_condensed.json` |
| [docs/descriptor_pipeline.md](docs/descriptor_pipeline.md) | the spec reference, numbering backends, extraction, aggregation and filtering, output columns, re-run rules, tests |
| [docs/descriptors.md](docs/descriptors.md) | every descriptor column of both engines, with its status, and the references |
| [docs/troubleshooting.md](docs/troubleshooting.md) | symptom, cause and fix, cluster and laptop |
| [configs/orca_library_example.yaml](configs/orca_library_example.yaml), [specs/](specs) | the example library config and the shipped specs |

## Repository layout

| Path | Content |
|---|---|
| [desc_utils](desc_utils) | the descriptor pipeline (`spec.py`, `numbering/`, `extract/`, `aggregate.py`, `run.py`, `condense.py`, the ORCA parsers) and `submit/`, the ORCA stage behind the `orcadesc` command |
| [bceg](bceg) | conformer ensembles: transformations, searchers (`search/`), refiners (`refine/`), hyper-parameter logs, SLURM arrays, the `bceg` command |
| [DescriPyTor](DescriPyTor) | git submodule, [Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor) |
| [configs](configs), [specs](specs), [inputs](inputs) | the example library config, one spec per molecule family, an example SMILES list |
| [docs](docs) | the documents above |
| [scripts-for-orca-computations](scripts-for-orca-computations) | cluster tools for any library: `check_jobs.py`, `list_molecules.py`, `slurm-condense.sh`, the fixed `calcParam.inp` of the LNiH2 library |
| [tests](tests) | unit tests (seconds) and regression tests against `tests/reference/` (minutes, need the LNiH2 library under `DATA/`) |
| [DATA](DATA), [notebooks](notebooks) | where the libraries land on the laptop (ignored by git); exploratory notebooks, not maintained |

## References

ORCA 6.1, NBO 7, GOAT, DescriPyTor, morfeus, DBSTEP, RDKit and Open Babel, and the papers defining every
descriptor, are listed in [docs/descriptors.md](docs/descriptors.md#references).
