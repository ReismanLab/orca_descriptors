# Stage 2: `pre_refinement`

CREST, GOAT and molassembler start from **one** 3D geometry. bceg builds it per replicate: an ETKDG
embedding of the transformed structure (seed = replicate seed), a pre-optimisation at a cheap level,
then a check that no bond to a metal is longer than `max_metal_ligand_distance`, no other bond is
broken and no two non-bonded atoms clash. A rejected geometry is re-embedded with the next seed
(`seed + attempt * 10007`), so the result is deterministic. This block chooses the pre-optimisation.
RDKit ETKDG and LoQI have no start geometry: for them the block must be `none` (the default).

What ran is recorded in the method folder's `run_1.json` under `details.start_geometry`: `source`
(the embedding and its seed), `attempts`, `preopt`, `preopt_level`, `preopt_energy`, `preopt_program`,
`preopt_version`, `rejected` (every rejected seed with its problems), `rmsd_preopt_vs_embedded`,
`metal_geometry` (ligand-metal-ligand angles and tau4) and `rmsd_to_replicates`. The accepted
geometry is kept as `logs/raw_1_start_geometry.xyz`; the attempts of a failed run under
`logs/raw_1_failed/preopt_<n>/`.

## Keys

| Key | Default | Meaning | bceg parameter | Hashed |
|---|---|---|---|---|
| `method` | `xtb` (searchers with a start step), `none` (others) | `xtb`, `uma` or `none` | `start_preopt` | yes |
| `params` | per method, below | the pre-optimiser's settings | `start_preopt_level` / `start_preopt_params` | yes |
| `start_geometry` | `etkdg` | `etkdg`: an RDKit ETKDG embedding seeded by `seed_offset + replicate` (`seed_offset` is a `conformer_search.params` key, default 0; with the default `conformer_search.replicates: 1` the one replicate has seed 1), re-seeded with `seed + attempt * 10007` after a rejection, hence reproducible by default; `input`: the coordinates returned by the transformation | `start_geometry` | yes |
| `start_ff` | `auto` | force field relaxing the raw embedding before the pre-optimisation: `auto` (MMFF94s if parametrised, else none), `MMFF94`, `MMFF94s`, `UFF`, `none` | `start_ff` | yes |
| `max_metal_ligand_distance` | `2.5` | Angstrom; a longer metal-ligand bond rejects the geometry (decoordination) | `max_metal_ligand_distance` | yes |
| `max_attempts` | `10` | re-embeddings before the search fails | `start_max_attempts` | yes |
| `xtb_executable` | `xtb` | which xtb binary; the path and version that ran are recorded | `xtb_executable` | no (resource) |

The last column names the parameters of the searcher that the block is translated into
(`bceg/search/base.py`, `START_DEFAULTS`); `bceg`'s own config takes them inside `methods.<name>.params`.

## Options

### `xtb` (default): GFN-xTB optimisation

| `params` | Default | Values |
|---|---|---|
| `gfn` | `gfn2` | `gfn2`, `gfn1`, `gfn0`, `gfnff` |
| `level` | `normal` | xtb `--opt` level: `crude`, `sloppy`, `loose`, `lax`, `normal`, `tight`, `vtight`, `extreme` |

```yaml
pre_refinement:
  method: xtb
  params: {gfn: gfn2, level: normal}
```

Threads: the job's `cpus_per_task` (CREST's `threads` or GOAT's `nprocs` when set). xtb comes from
conda (`gp_env_cluster.yml`).

### `uma`: optimisation with the UMA machine-learned potential

The same refiner as stage 4 ([4_post_optimization.md](4_post_optimization.md)) run on the single
start geometry. `params` takes the UMA keys except `lot` and `keep_unconverged`:

| `params` | Default | Meaning |
|---|---|---|
| `model` | `uma-s-1p2` | checkpoint name (weights fetched once with `orcadesc fetch-weights`) |
| `task` | `omol` | UMA task head |
| `optimizer` | `LBFGS` | ASE optimiser: `LBFGS`, `BFGS`, `FIRE` |
| `fmax` | `0.05` | convergence, eV/Angstrom |
| `max_steps` | `500` | optimiser steps |
| `device` | `cpu` | `cpu` or `cuda` (resource; `cuda` sends the whole bceg job to `slurm.ensembles.gpu`) |
| `threads` | job's CPUs | torch threads (resource) |

```yaml
pre_refinement:
  method: uma
  params: {model: uma-s-1p2, task: omol, fmax: 0.05, max_steps: 500}
```

The model is loaded once per bceg job; a chained job (search then post-optimisation) loads it a
second time for stage 4.

### `none`

```yaml
pre_refinement: none
```

The checked ETKDG embedding (after `start_ff`) is handed to the search as it is. Required for
`rdkit_etkdg` and `loqi`, which embed their own conformers.

### Starting from the transformation's coordinates

```yaml
pre_refinement:
  method: xtb
  start_geometry: input          # the script returned coords
  max_metal_ligand_distance: 2.8
```

## Status

`pre_refinement` in `orcadesc status`: `n/a` when the pipeline has no such step, `queued` while the
bceg job waits, `running` while the search task is in this phase (`run_1.json` says
`phase: pre_refinement`), `done` once the geometry was accepted, `failed` when no acceptable geometry
was found within `max_attempts` (the `reason` column carries the error; the attempts are under
`logs/raw_1_failed/`).
