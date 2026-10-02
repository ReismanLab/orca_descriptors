# Troubleshooting

One table, cluster first then laptop. The commands named here are explained in
[library_workflow.md](library_workflow.md), [condense.md](condense.md) and
[descriptor_pipeline.md](descriptor_pipeline.md); environment problems are also in
[cluster_setup.md](cluster_setup.md).

## On the cluster: `orcadesc submit`, `status`, the jobs

| Symptom | Cause | Fix |
|---|---|---|
| `check_jobs.py` reports `FAILED ... killed by SLURM: time limit` or `out of memory`, or `status` shows `timeout` / `failed` | the SLURM allocation was too small for this conformer | raise `slurm.orca.time` or `mem_per_cpu` in `<LIB>.yaml` (recorded, not compared) and run `orcadesc submit` again: only the failed conformers are resubmitted |
| `orcadesc submit` prints `REFUSED` with a diff of a stage block | a result-defining setting changed since the library was created (also after updating the code: older libraries have the previous layout of `params.json`) | restore it, or put the new settings in a new `library_dir`; for a test library, delete `$LIB/params.json` |
| `orcadesc submit` prints `bceg PARAMS MISMATCH` | a parameter of `conformer_search`, `pre_refinement`, `transformation` or `post_optimization` changed while the method folder exists | give `conformer_search.name` a new value or restore the parameter |
| `orcadesc submit` prints `CONFIG ERROR ... unknown parameter(s)` | a misspelt or unknown key in a stage block | the stage document lists the keys: [stages](stages) |
| `status` says `in_progress` but `squeue` shows nothing | a job died (walltime, node failure) before writing its end | `orcadesc submit <LIB>.yaml --force` |
| `status` says `ensemble_missing` after the bceg jobs finished | bceg failed for that molecule; its `run_1*.json` in the method folder holds the error | fix the cause, run `orcadesc submit` again (bceg reruns only the failed replicate) |
| a prepare log ends with `orca not found on PATH` or an ORCA job's `calcParam.out` is empty | the `slurm.setup` lines do not load the ORCA module, or NBO variables are missing | add `module load orca/6.1.0` (and the `NBOEXE` / `GENEXE` exports) to `slurm.setup`, resubmit |
| `status` shows `post_opt failed` and `run_1_<lot>.json` says `a post-optimisation worker process died` | a `uma` / `aimnet2` worker was killed, usually by the job's memory limit: every worker holds a copy of the model | raise the step's `mem` or set `post_optimization.params.workers` lower ([stages/4_post_optimization.md](stages/4_post_optimization.md#cpu-parallelism-workers-and-threads)) |
| `status` shows `post_opt failed` and `run_1_<lot>.json` says `LocalEntryNotFoundError` or `CERTIFICATE_VERIFY_FAILED` | the job tried to download the UMA weights; conda's OpenSSL cannot verify huggingface.co | `orcadesc fetch-weights` once on the login node into the shared cache ([cluster_setup.md](cluster_setup.md)), keep `FAIRCHEM_CACHE_DIR` / `HF_HUB_OFFLINE=1` in `slurm.setup`, then `orcadesc submit` again: only the post-optimisation reruns |
| `status` shows `post_opt failed` and `run_1_<lot>.json` says `AIMNet2 weights ... not in the cache` | the job tried to download the AIMNet2 weights | `aimnet download <model>` on the login node with `AIMNET_CACHE_DIR` set to the shared cache, keep the `export` in `slurm.setup`, `orcadesc submit` again |
| `search` is `failed` in `status` and the method folder's `run_1.json` says `'loqi' not found on PATH` | the shared `gp_env` was built before LoQI was added to it (2026-10-01) or its pip step failed | maintainer, in `gp_env`: `pip install "loqi @ git+https://github.com/isayevlab/LoQI@v1.1"` (or `conda env update -f gp_env_cluster.yml`), then `which loqi`; `orcadesc submit` again |
| `search` is `failed` and `run_1.json` says `LoQI failed` ... `is not in the cache` | the job tried to download the LoQI checkpoint | `loqi download --model <name>` on the login node with `LOQI_CACHE_DIR` set to the shared cache, keep the `export` in `slurm.setup`, `orcadesc submit` again |
| `status` shows `post_opt failed` and `run_1_<lot>.json` says `does not know --gxtb` | the `xtb` the job found is the standard one, not the g-xTB build | put the g-xTB `bin/` first on PATH in `slurm.setup` or set `post_optimization.params.executable` to its `xtb` |
| `ImportError: /lib64/libstdc++.so.6: version CXXABI_1.3.15 not found` when importing fairchem or `sqlite3` | pip-installed torch loaded the old system libstdc++ before the env's | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (in `slurm.setup` in the example); in Python, `import sqlite3` before torch, which bceg does |
| UMA post-optimisation fails with a Hugging Face authentication error on the login node | the UMA weights are gated | `huggingface-cli login` once in your account, then fetch the weights again |
| `condense` prints `failed ... no complete NBO block (NBO started but never returned)` | the ORCA job aborted; `tail calcParam.out` shows the error | rerun the ORCA job or drop the conformer folder |
| `condense` prints `failed ... optimization.xyz missing` / `singlepoint.property.json missing` | the job did not reach that step, or a different input file was used | same as above; the pipeline needs the `calcParam.inp` steps |

## On the laptop: `desc_utils.run`

| Symptom | Cause | Fix |
|---|---|---|
| `all_conformers_completed` is `False` | a conformer of that molecule is `running`, `failed` or `missing` in `orca_job_status`; its late descriptors (NMR, redox, Fukui) are NaN and left out of the averages | finish or resubmit the job, condense again (rebuilds that file), rsync, then `all` |
| `all_conformers_completed` is empty and `orca_job_status` is `unknown` | the condensed files predate the job status (before 2026-09-23) | condense on the cluster rebuilds them, then rsync; or leave it, the descriptors are unaffected |
| `orca_job_status` is `failed` with `terminated normally, then a later run failed` | the folder was resubmitted after a successful run and the second run died | `tail calcParam.out`; if the first run's outputs are intact the descriptors are fine, otherwise resubmit |
| `numbering_status` is `failed` or `incomplete` | no SMARTS stage matched this conformer | check the SMILES in the table, adjust the stage patterns, or switch to `backend: manual` with your own indices |
| `numbering_status` is `element_mismatch` | the matched atom is not the element declared in `atoms` | fix the SMARTS role positions or the `atoms` element |
| `numbering_status` is `inconsistent` | conformers of one molecule got different indices | usually a symmetric ligand; set `numbering.smarts.match_selection: lowest_indices` or add an `orient` rule |
| `descripytor_notes` says `no orca_condensed.json in ...` | the folder holds raw files only or was not transferred | condense and rsync, or `python -m desc_utils.run condense --spec S` locally when the raw files are present |
| `descripytor_notes` says `expected 1 molecule file ..., loaded 0` | the condensed file is unreadable | `python -m desc_utils.condense --root DATA/<LIB> --check` and rebuild with `--overwrite` (`sbatch slurm-condense.sh $LIB --overwrite` on the cluster) |
| `nbo_extraction_error` column is not empty | the NBO block of that conformer is missing or truncated | same as the `condense ... NBO block` row above |
| `No module named desc_utils.run` | the clone predates the pipeline | `git pull` and run the commands from the repository root |
| every row has `numbering_status` `failed` with `No module named 'openbabel'` | Open Babel missing from the environment (needed to convert xyz to sdf) | `pip install openbabel-wheel==3.1.1.23` in `gp_env`, then rerun `number` |
| `extract` stops with `No module named 'networkx'` or `'plotly'` | environment created from an older `gp_env.yml` | `pip install networkx==3.6.1 plotly==7.1.0` in `gp_env` |
| `ImportError: DescriPyTor not found at ...` | submodule not checked out | `git submodule update --init` |
| `numpy.dtype size changed` or similar import errors | a `pip install` of DescriPyTor downgraded numpy | never pip-install DescriPyTor; recreate `gp_env` from `gp_env.yml` |
| `validate` lists `unknown label` | a descriptor term uses a label that is not in `atoms` | add the label or fix the typo; labels are case-sensitive |
