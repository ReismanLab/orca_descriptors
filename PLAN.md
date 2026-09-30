# Plan: conformer ensembles from B-CEG-MCD, ORCA and descriptors from orca_descriptors

Branch `conformer-orca-merger`, started 2026-09-30 from `submission-fixes` (PR #6). Nothing below is
implemented yet; this file records the design and the decisions so the work can be done in steps.

## Goal

Stop maintaining two implementations of SMILES → embedding → CREST → UMA. On the cluster both projects
live in `/groups/ReismanLab`:

- **B-CEG-MCD** (`bceg`, `/groups/ReismanLab/B-CEG-MCD`, shared env `bceg`) generates conformer
  ensembles: transformation, search (CREST, GOAT, ...), UMA post-optimisation, replicates,
  `params.json` hyper-parameter logs, `molecules.csv`.
- **orca_descriptors** (`/groups/ReismanLab/orca_descriptors`, shared env `gp_env`) takes an ensemble,
  runs one ORCA compound job per conformer, condenses the outputs and computes the descriptors.

The `build`, `crest` and `uma` stages of `submission_scripts_v3/` are replaced by bceg's output. The
ORCA stage is rewritten as a package module with a CLI, borrowing bceg's design where it is better.

## Decisions (project owner, 2026-09-30)

| topic | decision |
|---|---|
| Split | bceg: ensembles. orca_descriptors: ORCA jobs, condense, descriptors. |
| Environment | `gp_env` for everything on the orca_descriptors side (rdkit, pyyaml, pandas already in it), not `nickel_oa`. |
| Code location | run from the shared clone (later: installed into `gp_env`), never copied into the library; the git commit is recorded. |
| Settings | one config file per library; `submit` freezes the resolved copy and the rendered `calcParam.inp` under `<LIB>/_jobs/<timestamp>/`, and the jobs read that snapshot. |
| Changed settings | refused with a diff (bceg's rule). Different settings mean a different library directory. No interactive confirmation. |
| Conformer folders | `conf_N/` are created at the ORCA stage only, from the ensemble xyz; bceg keeps its multi-conformer `raw_<i>_<lot>.xyz`. |
| Replicates | bceg config used for ORCA libraries defaults to `replicates: 1`. |
| Skip rule | keep orca_descriptors': the record decides (`complete` / `in_progress` are skipped, `--force` resubmits); a stuck `running` stage is not silently rerun. |
| Per-run record | bceg-style record per conformer job, plus the ORCA set-up when present: module, rendered-input MD5, `compound_steps`, SLURM ids, exit code, termination banner, run time, condense result. |
| Status | a `status` command printing a table per stage, not `--dry-run`. |
| SLURM | job arrays for the ORCA jobs (one array per submission, throttled); resource profiles in the config. |
| Molecule identity | InChIKey of the canonical input SMILES, the same key on both sides; the registry carries the submitted SMILES and bceg's `molecules.csv` row. |

## Architecture

```
bceg submit ...                         (B-CEG-MCD, env bceg)
  DATA_bceg/<transformation>/<InChIKey>/<method>/raw_1_uma-s-1p2.xyz, params.json, params_<lot>.json, run_1*.json
  DATA_bceg/molecules.csv

python -m desc_utils.submit submit <config>          (orca_descriptors, env gp_env)
  <LIB>/_jobs/<timestamp>/config.yaml, calcParam.inp, orca.sbatch, tasks.jsonl, slurm_logs/
  <LIB>/molecule_registry.json                       InChIKey -> submitted SMILES, bceg source, params sha256
  <LIB>/<InChIKey>/metadata.json                     per-molecule record (as today, ORCA part only)
  <LIB>/<InChIKey>/conf_N/                           input.xyz, calcParam.inp, calcParam.out, orca_condensed.json, ...

python -m desc_utils.submit status <config>          table: molecules x {pending, submitted, running, completed, failed, timeout, skipped}
python -m desc_utils.run all --spec specs/<LIB>.yaml (laptop, unchanged)
```

The ORCA library keeps the `<LIB>/<InChIKey>/conf_N/` layout so the descriptor pipeline, `check_jobs.py`
and `slurm-condense.sh` work unchanged. The link to the ensemble is recorded, not implied by nesting:
`metadata.json` and the registry store the bceg results directory, transformation, method, level of
theory, replicate, and the `sha256` of `params.json` / `params_<lot>.json`.

### Config (`<LIB>/orca_library.yaml`, one per library)

```yaml
library_dir: /groups/ReismanLab/DATA/orca/<LIB>          # absolute
source:
  kind: bceg                                             # or xyz: {file: ensemble.xyz} for one molecule
  results_dir: /groups/ReismanLab/DATA/bceg
  transformation: NiH2
  method: crest_gfn2
  lot: uma-s-1p2                                         # which raw_<i>_<lot>.xyz to take
  replicate: 1
  smiles: ligands.txt                                    # subset of molecules.csv to submit; default: all with a finished ensemble
dft:
  ewin_kcal_mol: 5.0                                     # on the energies of the ensemble comment lines (Hartree)
  calc_param_template: calcParam.inp.tmpl                # relative to this file; default: the repository's
  level_of_theory: {...}                                 # as today, extra_basis included
  charge: 0
  spin_multiplicity: 1                                   # must equal bceg's structure multiplicity; checked
slurm:
  setup: [source /groups/ReismanLab/miniconda3/etc/profile.d/conda.sh, conda activate gp_env, module load orca/6.1.0]
  orca: {time: "160:00:00", mem_per_cpu: 4G, ntasks: 16, nodes: 1}
  array_throttle: 50
condense: true
```

### What each ORCA array task does

1. Read the frozen config. Locate the molecule's ensemble file; refuse if `params.json` sha256 differs
   from the one recorded in the registry at submission (the ensemble changed under the library).
2. On first contact with a molecule: split the ensemble into `conf_N/input.xyz` (comment-line energy kept),
   apply the energy window, write `metadata.json` (`source`, `dft`, `compound_steps`, conformer list with
   `skipped_energy_filter`), register the molecule.
3. Run ORCA in `conf_N/`, record `submitted` → `completed | failed | timeout` with the banner check,
   condense, record the condense result.

Steps 1–2 run in the array task rather than at submission so `submit` stays fast for hundreds of molecules;
`submit` only decides which (molecule, conformer) pairs are pending from the records.

## Steps

1. `desc_utils/submit/`: config loader with freeze, bceg source reader (ensemble file, `molecules.csv`,
   `params*.json`), registry, metadata records (port `metadata_utils`, `record_step`, `mark_orca_done`),
   template renderer (port `calc_param_template`), array generation, `submit` / `status` / `run-task` CLI.
   Unit tests with a fake bceg tree.
2. `slurm-calcParam.sh` becomes a generated array script calling `python -m desc_utils.submit run-task`.
   Condense stays in-process. TERM trap for `timeout` kept.
3. README step 3 rewritten: bceg first, then `desc_utils.submit`. `submission_scripts_v3/` kept one
   release for batches in flight, then removed (history and tag `archive/submission-scripts`).
4. Later: `pip install -e .` of orca_descriptors into the shared `gp_env` (pyproject.toml), so the
   command is `orcadesc submit` and the git commit is the installed version.

## Open questions

- One command or two: run bceg and the ORCA stage as two submissions (ORCA picks up finished ensembles on
  each re-run), or have `submit` also queue the ORCA array `afterany` the bceg arrays? Proposal: two, first.
- Replicates > 1: one ORCA library per replicate (`<LIB>_rep2`), or `conf_N` numbered across replicates
  with `replicate` recorded per conformer? Proposal: one library per replicate.
- `charge`: bceg's `Structure` carries charge and multiplicity per transformation; take both from the
  ensemble's `params.json` rather than from the ORCA config, and only allow an override with a new library?
- Keep `check_jobs.py` (squeue-based) as the independent cross-check of `status`? Proposal: yes.
