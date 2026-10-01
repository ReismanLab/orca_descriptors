# Stage 4: `post_optimization`

Every conformer of the search output is optimised at a better level and written to
`raw_1_<lot>.xyz` (energies in Hartree on the comment lines) next to `params_<lot>.json` (the
refiner's parameters, hashed together with the search's `params.json` hash) and `run_1_<lot>.json`.
The ORCA stage reads `raw_1_<lot>.xyz`; its energies are what stage 5 prunes on. In chained mode
(the default, `slurm.ensembles.mode: chained`) the post-optimisation runs in the same job as the
search, right after it.

## Keys

| Key | Default | Meaning | Hashed |
|---|---|---|---|
| `refiner` | `uma` | the only refiner today | yes |
| `params` | `{}` | the refiner's parameters, below | yes, except the resource keys |
| `slurm` | `{}` | overrides of `slurm.ensembles.cpu`/`.gpu` for this step (split mode) or merged under the search's overrides (chained mode) | no |

## `uma`: fairchem UMA through ASE

| `params` | Default | Meaning |
|---|---|---|
| `model` | `uma-s-1p2` | checkpoint name; the weights are gated on Hugging Face and fetched once on a login node with `orcadesc fetch-weights --model <name>` into `FAIRCHEM_CACHE_DIR` (README step 2) |
| `task` | `omol` | UMA task head (`omol` for molecules) |
| `optimizer` | `LBFGS` | ASE optimiser: `LBFGS`, `BFGS`, `FIRE` |
| `fmax` | `0.05` | convergence criterion on the forces, eV/Angstrom |
| `max_steps` | `500` | optimiser steps per conformer |
| `keep_unconverged` | `true` | keep conformers that did not reach `fmax` (flagged `converged: false` on their comment line); `false` drops them |
| `lot` | model name | label used in the file names (`raw_1_<lot>.xyz`) |
| `device` | `cpu` | resource: `cpu` or `cuda`; `cuda` sends the step (and, chained, its whole job) to `slurm.ensembles.gpu` |
| `threads` | job's CPUs | resource: torch threads (`SLURM_CPUS_PER_TASK`) |

Recorded per run (`run_1_<lot>.json`): `n_conformers`, `details.n_input`, `details.n_unconverged`,
`versions` (`fairchem-core`, `ase`), wall time and core-hours. Charge and multiplicity come from the
transformation (stage 1).

```yaml
post_optimization:
  refiner: uma
  params: {model: uma-s-1p2, task: omol, optimizer: LBFGS, fmax: 0.05, max_steps: 500, device: cpu}
```

On a GPU partition:

```yaml
post_optimization:
  refiner: uma
  params: {model: uma-s-1p2, device: cuda}
slurm:
  ensembles:
    mode: split                                  # the search stays on CPUs, the post-optimisation goes to the gpu profile
    cpu: {time: "24:00:00", cpus_per_task: 8, mem: 16G}
    gpu: {partition: gpu, time: "02:00:00", gres: "gpu:1", cpus_per_task: 4, mem: 16G}
```

## `null`: no post-optimisation

```yaml
post_optimization: null
```

ORCA then refines `raw_1.xyz`, the search output, with the energies the searcher gave (CREST, GOAT,
RDKit with `ff_optimize`) or none (molassembler, LoQI, RDKit without force field). Without energies,
`pruning.energy_window_kcal_mol` must be `null`.

## Status

`post_opt` in `orcadesc status`: `n/a` without a refiner, `queued` while the search has not finished,
`running`, `done`, `failed` (the error from `run_1_<lot>.json` is in `reason`; a missing UMA weight
cache or an SSL error at download time are the usual causes, see the README's troubleshooting table).
`n_conf` then counts the post-optimised conformers.
