# Stage 4: `post_optimization`

Every conformer of the search output is optimised at a better level and written to
`raw_1_<lot>.xyz` (energies in Hartree on the comment lines) next to `params_<lot>.json` (the
refiner's parameters, hashed together with the search's `params.json` hash) and `run_1_<lot>.json`.
The ORCA stage reads `raw_<i>_<lot>.xyz` of every replicate; its energies are what stage 5 prunes on. In chained mode
(the default, `slurm.ensembles.mode: chained`) the post-optimisation runs in the same job as the
search, right after it.

## Keys

| Key | Default | Meaning | Hashed |
|---|---|---|---|
| `refiner` | `uma` | `uma` (fairchem UMA through ASE), `gxtb` (g-xTB, one `xtb --gxtb` run per conformer, CPU) or `aimnet2` (AIMNet2 through ASE); a refiner is a class in `bceg/refine/` registered in `REFINERS`, see the end of this page | yes |
| `params` | `{}` | the refiner's parameters, below | yes, except the resource keys |
| `slurm` | `{}` | overrides of `slurm.ensembles.cpu`/`.gpu` for this step (split mode) or merged under the search's overrides (chained mode) | no |

Recorded per run for every refiner (`run_1_<lot>.json`): `n_conformers`, `details.n_input`,
`details.n_unconverged`, `versions`, wall time and core-hours. Each conformer's comment line carries
`converged=True|False` and `source_index` (its index in `raw_1.xyz`). Charge and multiplicity come from
the transformation (stage 1).

## `uma`: fairchem UMA through ASE

| `params` | Default | Meaning |
|---|---|---|
| `model` | `uma-s-1p2` | checkpoint name; the weights are gated on Hugging Face and fetched once on a login node with `orcadesc fetch-weights --model <name>` into `FAIRCHEM_CACHE_DIR` ([cluster_setup.md](../cluster_setup.md)) |
| `task` | `omol` | UMA task head (`omol` for molecules) |
| `optimizer` | `LBFGS` | ASE optimiser: `LBFGS`, `BFGS`, `FIRE` |
| `fmax` | `0.05` | convergence criterion on the forces, eV/Angstrom |
| `max_steps` | `500` | optimiser steps per conformer |
| `keep_unconverged` | `true` | keep conformers that did not reach `fmax` (flagged `converged: false` on their comment line); `false` drops them |
| `lot` | model name | label used in the file names (`raw_1_<lot>.xyz`) |
| `device` | `cpu` | resource: `cpu` or `cuda`; `cuda` sends the step (and, chained, its whole job) to `slurm.ensembles.gpu` |
| `workers` | allocation's CPUs | resource: worker processes the conformers are shared out over (`cpu`: one per CPU of the allocation, at most 8 outside SLURM; `cuda`: 1), see [CPU parallelism](#cpu-parallelism-workers-and-threads) |
| `threads` | CPUs / `workers` | resource: torch threads per worker (all the allocation's CPUs for a single worker) |

`versions`: `fairchem-core`, `ase`. Per conformer: `steps`, `fmax` (residual force). `details`:
`workers` and `threads` as resolved.

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

### CPU parallelism: `workers` and `threads`

On CPUs the conformers of a molecule are independent, so `uma` and `aimnet2` optimise them in a pool
of `workers` processes, each loading the model once and running torch on `threads` threads. One
forward pass on a molecule of a few dozen atoms gains little from many threads; one single-threaded
worker per core keeps every core busy. The defaults do that: with `cpus_per_task: 8`, eight workers of
one thread each (with `ntasks: 4, cpus_per_task: 1`, the GOAT allocation, four workers). `workers: 1`
restores the previous behaviour, one in-process optimisation at a time on all the threads. On a GPU
the default is one worker: processes would only queue on the device.

Every worker holds its own copy of the model, roughly 1 to 2 GB each with torch: size the step's `mem`
for `workers` copies (the chained job's `mem` is shared with the search). A worker killed by the memory
limit fails the step with a message naming the pool; lower `workers` or raise `mem`. Each worker writes
its own `opt_<i>.log`; the conformers keep their input order (`source_index`) whatever finished first.
`run_1_<lot>.json` records `details.workers` and `details.threads`, so comparing `wall_time_s` of a
run with `workers: 1` against the default on one molecule is the benchmark:

```yaml
post_optimization:
  refiner: uma
  params: {model: uma-s-1p2, workers: 1}        # serial reference; drop `workers` for the pool
```

## `gxtb`: g-xTB (Grimme group), CPU

[g-xTB](https://github.com/grimme-lab/g-xTB) approximates ωB97M-V/def2-TZVPPD for Z = 1–103. It is
shipped as a statically linked build of `xtb` (6.7.1 with a modified tblite) that runs the method when
`--gxtb` is given; there is no separate program and no parameter file. Unpack the Linux binary from the
repository's `binaries/` folder somewhere the compute nodes mount and either put that folder first on
PATH in `slurm.setup` or set `executable` to the binary. The standard `xtb` (conda, `module load xtb`)
does not know `--gxtb`: the refiner recognises its "unknown option" warning and fails the step instead
of accepting a GFN2-xTB result.

Each conformer is optimised by its own `xtb start.xyz --gxtb --opt <level> --chrg <charge>` run in a
subfolder `conf_<i>/` of the job's scratch (`xtb.out` kept there until the job ends); `--uhf` is passed
only for open shells (with the flag g-xTB always runs unrestricted). The energy is the `TOTAL ENERGY`
of the output; a conformer whose optimisation stops before convergence is taken from `xtblast.xyz`.

| `params` | Default | Meaning |
|---|---|---|
| `level` | `normal` | xtb `--opt` level: `crude`, `sloppy`, `loose`, `lax`, `normal`, `tight`, `vtight`, `extreme` |
| `cycles` | xtb's default | optimiser cycles per conformer (xtb `--cycles`) |
| `keep_unconverged` | `true` | keep conformers that did not converge within the cycles (flagged `converged: false`); `false` drops them |
| `lot` | `gxtb` | label used in the file names (`raw_1_gxtb.xyz`) |
| `executable` | `xtb` | resource: the g-xTB build of xtb, a name on PATH or an absolute path |
| `threads` | job's CPUs | resource: OpenMP threads of each xtb run (`SLURM_CPUS_PER_TASK`); the conformers run one after the other |

`versions`: `xtb` (the `--version` line; the g-xTB build reports 6.7.1 like the standard one, so
`details.executable`, the resolved path, is what tells them apart). Per conformer: `cycles` when the
optimisation converged. The refiner has no `device`: g-xTB runs on CPUs only.

```yaml
post_optimization:
  refiner: gxtb
  params: {level: tight, executable: /groups/ReismanLab/gxtb/bin/xtb}
```

## `aimnet2`: AIMNet2 through ASE

[AIMNet2](https://github.com/isayevlab/aimnetcentral) (package `aimnet`, in `gp_env_cluster.yml`)
covers organic and element-organic molecules; a conformer with an element the model was not trained
on fails the step (`Some species are not implemented`). The closed-shell models ignore the
multiplicity, so a transformation with `multiplicity` > 1 is refused unless the model is an open-shell
one (`aimnet2-nse`).

| `params` | Default | Meaning |
|---|---|---|
| `model` | `aimnet2` | registry alias: `aimnet2` (ωB97M-D3), `aimnet2-2025` (B97-3c, better intermolecular interactions), `aimnet2-nse` (open shell), `aimnet2-pd` (Pd), `aimnet2-rxn` (reactive) |
| `optimizer` | `LBFGS` | ASE optimiser: `LBFGS`, `BFGS`, `FIRE` |
| `fmax` | `0.05` | convergence criterion on the forces, eV/Angstrom |
| `max_steps` | `500` | optimiser steps per conformer |
| `keep_unconverged` | `true` | keep conformers that did not reach `fmax` (flagged `converged: false`); `false` drops them |
| `lot` | model name | label used in the file names (`raw_1_<lot>.xyz`) |
| `device` | `cpu` | resource: `cpu` or `cuda`; `cuda` sends the step (and, chained, its whole job) to `slurm.ensembles.gpu` |
| `workers` | allocation's CPUs | resource: worker processes, as for `uma` ([CPU parallelism](#cpu-parallelism-workers-and-threads)) |
| `threads` | CPUs / `workers` | resource: torch threads per worker |

`versions`: `aimnet`, `ase`. Per conformer: `steps`, `fmax` (residual force). `details`: `workers`, `threads`.

The weights are downloaded on first use into `AIMNET_CACHE_DIR` (default `~/.cache/aimnet`). Compute
nodes have no internet access: fetch them once on a login node into a shared cache and export the same
variable in `slurm.setup`:

```bash
export AIMNET_CACHE_DIR=/groups/ReismanLab/aimnet_cache
aimnet download aimnet2            # or aimnet2-nse, ...; `aimnet download --all` for every model
```

```yaml
post_optimization:
  refiner: aimnet2
  params: {model: aimnet2, fmax: 0.05, max_steps: 500, device: cuda}
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
`running`, `done`, `failed` (the error from `run_1_<lot>.json` is in `reason`; a missing weight cache
(UMA, AIMNet2), an SSL error at download time or a standard `xtb` where the g-xTB build was expected
are the usual causes, see [troubleshooting.md](../troubleshooting.md)). `n_conf` then counts the post-optimised
conformers.

## Adding a refiner

A refiner is a subclass of `bceg.refine.base.Refiner` in `bceg/refine/`: `name`, `defaults` (every
parameter with its default), `resource_keys` (the ones not hashed: device, threads, executable), a
`lot` property (the label of `raw_1_<lot>.xyz`) and `run(structure, conformers, workdir)` returning a
`RefineResult` with the conformers, their energies in Hartree and the program versions. Registering the
class in `REFINERS` (`bceg/refine/__init__.py`) makes it a valid `post_optimization.refiner`; the
config, hashing and status need no change. Add its parameter table to this page (the test suite
checks that every default of every registered refiner is documented). `gxtb` is the model for a
program called per conformer, `uma` and `aimnet2` (both through `bceg/refine/ase_opt.py`) for an ASE
calculator whose `device` is routed to the gpu profile. An ASE refiner gives the shared loop a
`make_calculator(structure)` method and the `optimizer`, `fmax`, `max_steps`, `keep_unconverged`,
`device`, `workers` and `threads` parameters; the loop rebuilds the refiner from its parameters inside
every worker process (spawned, so the refiner must be picklable before its model is loaded).
