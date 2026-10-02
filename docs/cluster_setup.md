# Cluster set-up: environment and model weights

Everything the cluster side of the pipeline needs, once per user (and once per model), before the first
`orcadesc submit`. The step-by-step use of the cluster is in the [README](../README.md); the stages of a
library and their options are under [stages](stages).

## The shared environment

The repository is cloned in the group folder and the shared environment `gp_env` exists (built by its
maintainer from [gp_env_cluster.yml](../gp_env_cluster.yml): the laptop environment plus xtb, CREST, UMA,
molassembler, LoQI, and this clone installed editable). Nothing to install:

```bash
ssh <user>@<cluster>
source /groups/ReismanLab/miniconda3/etc/profile.d/conda.sh
conda activate gp_env
orcadesc --help                      # sanity check: the ORCA-stage command of this clone
```

Condensing alone ([condense.md](condense.md)) needs only the Python of `module load python`.

For the maintainer: `conda env create -f gp_env_cluster.yml` from the root of
`/groups/ReismanLab/orca_descriptors` (its `-e .` line installs that clone, so `git pull` there updates
everyone). LoQI is part of that environment.

## Model weights: fetched once on the login node, read offline by the jobs

The compute nodes cannot download anything, so every machine-learned potential or generator keeps its
weights in a cache shared by the group, filled once from the login node and exported in `slurm.setup`
of the library config ([configs/orca_library_example.yaml](../configs/orca_library_example.yaml)).

### UMA (`post_optimization.refiner: uma`, `pre_refinement.method: uma`, CREST `+uma`)

The UMA weights are gated on Hugging Face:

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
The jobs then run with `FAIRCHEM_CACHE_DIR` and `HF_HUB_OFFLINE=1` exported in `slurm.setup`.

### AIMNet2 (`post_optimization.refiner: aimnet2`)

```bash
export AIMNET_CACHE_DIR=/groups/ReismanLab/aimnet_cache
aimnet download aimnet2                                    # once per model, on the login node
```

Keep the same `export` in `slurm.setup` ([stages/4_post_optimization.md](stages/4_post_optimization.md)).

### g-xTB (`post_optimization.refiner: gxtb`)

No weights: only the g-xTB build of `xtb` on PATH or in `post_optimization.params.executable`
([stages/4_post_optimization.md](stages/4_post_optimization.md)).

### LoQI (`conformer_search.searcher: loqi`)

```bash
export LOQI_CACHE_DIR=/groups/ReismanLab/loqi_cache
loqi download --model loqi_flow_v1.1.0                     # once per model, on the login node
```

Keep the same `export` in `slurm.setup` ([stages/3_conformer_search.md](stages/3_conformer_search.md)).

## Known environment pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: /lib64/libstdc++.so.6: version CXXABI_1.3.15 not found` when importing fairchem or `sqlite3` | pip-installed torch loaded the old system libstdc++ before the env's | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (in `slurm.setup` in the example); in Python, `import sqlite3` before torch, which bceg does |
| `CERTIFICATE_VERIFY_FAILED` or `LocalEntryNotFoundError` from a job | the job tried to download weights; conda's OpenSSL cannot verify huggingface.co and compute nodes have no network | fetch the weights on the login node as above, keep the cache variables and `HF_HUB_OFFLINE=1` in `slurm.setup` |
| UMA fails with a Hugging Face authentication error on the login node | the weights are gated | `hf auth login` (or `huggingface-cli login`) once in your account, then fetch again |
| `'loqi' not found on PATH` in a search record | the shared `gp_env` predates LoQI (added 2026-10-01) or its pip step failed | maintainer, in `gp_env`: `pip install "loqi @ git+https://github.com/isayevlab/LoQI@v1.1"` (or `conda env update -f gp_env_cluster.yml`), then `which loqi` |
| `does not know --gxtb` in a post-optimisation record | the `xtb` found is the standard build, not g-xTB | put the g-xTB `bin/` first on PATH in `slurm.setup` or set `post_optimization.params.executable` |

The full symptom table, for every stage, is in [troubleshooting.md](troubleshooting.md).
