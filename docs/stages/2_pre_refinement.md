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
| `constrained` | `false` | `true`: impose `conformer_search.constraints` on the embedding, then pre-optimise with their atoms fixed, see [imposing constraints](#imposing-constraints) | `start_constraints` | yes, when `true` |
| `impose_steps` | `10` | with `constrained`: steps of the ramp that brings the embedding onto the imposed values | `start_impose_steps` | yes, when `constrained` |

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
start geometry. `params` takes the UMA keys except `lot`, `keep_unconverged` and `workers` (one geometry: no pool):

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

With `conformer_search.constraints` the UMA geometry is also what the search keeps fixed, e.g. the
metal and its first shell, which xtb describes poorly
([3_conformer_search.md](3_conformer_search.md#constraints)).

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

## Imposing constraints

`constrained: true` makes the start geometry satisfy `conformer_search.constraints`
([3_conformer_search.md](3_conformer_search.md#constraints)) instead of merely measuring them on it:
the planes, and every bond, angle and dihedral with a `value`, are reached before the pre-optimisation,
and every atom any entry names (the *held atoms*) stays fixed during it (xtb `$fix`, ASE `FixAtoms` for
UMA). The CREST search then freezes them, and stage 4 and the DFT optimisation keep them fixed when they
say `constrained: true` too. For example LSmI2 with a linear I-Sm-I, or a square-planar Ni:

```yaml
pre_refinement:
  method: uma
  constrained: true
conformer_search:
  name: crest_gfn2_SmI2lin
  searcher: crest
  constraints:
    freeze: [{element: Sm, neighbors: 1}]
    angles: [{smarts: "[I:1]~[Sm:2]~[I:3]", value: 180}]
  params: {shake: 0, tstep: 2.0, crossing: false}
```

The values are reached gradually (`bceg/impose.py`), never in one jump on the ETKDG geometry:

1. **Ramp**: `impose_steps` restrained minimisations, step s pulling every target s/`impose_steps` of the
   way from its embedded value. The rest of the molecule follows through an elastic network built on the
   embedding: every bond and 1-3 distance is held near its embedded length (except the ones the targets
   must change, e.g. the 1-3 distances around the apex of an imposed angle), and non-bonded atoms are
   kept at least their covalent-radii sum apart. The network needs no force-field atom types, so it works
   for any metal (UFF types neither Sm(III) nor the Ni of these complexes). An angle is restrained
   through the distance between its end atoms, which is smooth at 180 degrees.
2. **Snap**: once every target is within 3 degrees / 0.05 Angstrom / a plane RMS of 0.05 Angstrom, the
   exact values are set by rigid moves (the smaller side of the bond is rotated or translated; a plane
   group is projected onto its least-squares plane). A target inside a ring is set by a stiff restraint
   instead.
3. **Settle**: the network is minimised again with the held atoms fixed.

The connectivity is checked on the embedding, after every ramp step, after the imposition and after the
pre-optimisation: no metal-ligand bond longer than `max_metal_ligand_distance`, no other bond broken, no
non-bonded clash, and no tetrahedral stereocentre inverted. Any problem, or a ramp that ends outside the
snap tolerance, rejects the embedding and the next seed is tried; `rejected` names the `stage`
(`imposed` or `preopt`). `details.start_geometry` records `held_atoms` and `imposed`: the targets'
values at the `start`, `after_ramp` and `final`.

The held atoms are fixed in Cartesian coordinates from here on: their bond lengths stay what the
pre-optimisation (with them fixed, so effectively the imposition) gave, also through the DFT optimisation
when it is constrained. Pick the start geometry's level accordingly.

## Status

`pre_refinement` in `orcadesc status`: `n/a` when the pipeline has no such step, `queued` while the
bceg job waits, `running` while the search task is in this phase (`run_1.json` says
`phase: pre_refinement`), `done` once the geometry was accepted, `failed` when no acceptable geometry
was found within `max_attempts` (the `reason` column carries the error; the attempts are under
`logs/raw_1_failed/`).

## Cost

The time this stage took is kept apart from the search's: `run_<i>.json` has, next to the task's total
`wall_time_s` / `cpu_hours` / `gpu_hours`, the same three figures per phase under `phases.pre_refinement`
(embedding, force field and pre-optimisation, every rejected attempt included) and `phases.search`. A
task that died before a geometry was accepted has only `phases.pre_refinement`. `orcadesc cost` sums
them per molecule (`pre_refinement_cpu_h`, `pre_refinement_gpu_h`), so the price of `xtb` against `uma`
can be read off the records after the fact.
