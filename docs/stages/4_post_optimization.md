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
| `refiner` | `uma` | `uma` (fairchem UMA through ASE), `xtb` (GFN-n, one `xtb --opt` run per conformer, CPU), `gxtb` (g-xTB, the same driver with `--gxtb`, CPU) or `aimnet2` (AIMNet2 through ASE); a refiner is a class in `bceg/refine/` registered in `REFINERS`, see the end of this page | yes |
| `params` | `{}` | the refiner's parameters, below | yes, except the resource keys |
| `slurm` | `{}` | overrides of `slurm.ensembles.cpu`/`.gpu` for this step (split mode) or merged under the search's overrides (chained mode) | no |
| `continue_from` | — | the lot of an existing refinement in the same method folder: this step starts from `raw_<i>_<lot>.xyz` instead of the search output, so a library can be refined further without recomputing stages 1-3, see [below](#refining-further-continue_from) | yes, when set |
| `constrained` | `false` | `true`: the atoms `conformer_search.constraints` names stay fixed in every conformer's optimisation (ASE `FixAtoms` for `uma` and `aimnet2`; xtb `$fix` with the L-BFGS optimiser, `$opt engine=lbfgs`, for `gxtb`: xtb's default ANC optimiser ignores `$fix`), see below | yes, when `true` |

Recorded per run for every refiner (`run_1_<lot>.json`): `n_conformers`, `details.n_input`,
`details.n_unconverged`, `versions`, wall time and core-hours. Each conformer's comment line carries
`converged=True|False` and `source_index` (its index in `raw_1.xyz`). Charge and multiplicity come from
the transformation (stage 1), unless the refiner's `multiplicity` sets another spin state ([below](#at-another-spin-state-multiplicity)).

### At another spin state: `multiplicity`

Every refiner takes `params.multiplicity` (default `null`). Set, the step optimises the conformers at that
multiplicity instead of the transformation's (checked against the electron count), records it in
`run_<i>_<lot>.json` (`structure.multiplicity`) and in `params_<lot>.json` (it is hashed; `null` is left
out, so the lots written before it keep their hash), and hands it to stage 6: a library built on that lot
runs ORCA or g-xTB at that multiplicity. The search, the start geometry (`pre_refinement` refuses the key)
and the other lots keep the transformation's.

Its use is a `continue_from` lot at another spin state on an existing ensembles tree, e.g. the septet of
Sm(II) complexes searched as quintets, without searching again:

```yaml
library_dir: ../DATA/smI2_septet                # new
ensembles_dir: ../DATA/smI2_quintet/ensembles   # the existing tree; stages 1-3 copied verbatim
post_optimization:
  refiner: uma
  continue_from: uma-s-1p2                      # the quintet lot
  params: {model: uma-s-1p2, task: omol, fmax: 0.01, max_steps: 50000, device: cpu,
           multiplicity: 7, lot: uma-s-1p2-m7}
```

Because the `conf_N` numbering follows the source lot, the quintet and septet libraries compare conformer
by conformer.

### A job that runs out of time: the checkpoint

Every refiner appends each optimised conformer to `raw_<i>_<lot>.partial.jsonl` as soon as it is
done, so a job killed by its time limit, its memory limit, a node failure or `scancel` loses only the
conformers in flight. The bceg jobs get SIGTERM 10 min before their time limit (`--signal=B:TERM@600`): the
run record is then written `failed` with an `Interrupted` error and `n_checkpointed`, the conformers kept.
After a hard kill the record stays `running`, and `orcadesc status` / `submit` ask `sacct` whether its job
(`slurm_job_id`) has ended: if so the step counts as failed. Either way `orcadesc submit` runs it again,
without `--force`, and the run optimises only the conformers not yet in the checkpoint; the file is
removed once `raw_<i>_<lot>.xyz` is written (`details.n_resumed`: how many came from it). Its first line
is a key, the hash of `raw_<i>.xyz` and the refiner's parameters: a checkpoint of another search output or
other parameters is ignored.

`xtb` and `gxtb` record a conformer that did not converge too, and apply `keep_unconverged` once the
loop is over, so resuming yields exactly the conformers an uninterrupted run would have produced.

### Refining further: `continue_from`

A refinement is identified by its `lot`, and `lot` names the files (`raw_<i>_<lot>.xyz`,
`params_<lot>.json`, `run_<i>_<lot>.json`). Several lots therefore live side by side in one method
folder, and `continue_from` makes one of them the *input* of the next:

```yaml
post_optimization:
  refiner: uma
  continue_from: uma-s-1p2                      # reads raw_<i>_uma-s-1p2.xyz
  params: {model: uma-s-1p2, lot: uma-tight, fmax: 0.005, max_steps: 2000}
```

```
<ensembles_dir>/<transformation>/<InChIKey>/<method>/
    raw_1.xyz                 search output            (untouched)
    raw_1_uma-s-1p2.xyz       fmax 0.05, 500 steps     (untouched, and still the input of its own library)
    raw_1_uma-tight.xyz       fmax 0.005, 2000 steps   <- continued from the file above
```

This is how an already computed library is refined further: the conformer search and the optimiser
steps already spent are kept, and only the extra steps are paid for. The usual way to use it is a second
library config pointing at the same `ensembles_dir`, see
[library_workflow.md](../library_workflow.md#5-a-second-library-on-an-existing-ensembles-tree).

- `params.lot` is **required** with `continue_from`, and must differ from it: the default lot is the
  model name, which would make the step overwrite the file it reads.
- The refiner may differ from the one that produced the source lot, so `gxtb` -> `uma` and `uma` ->
  `gxtb` cascades are written the same way.
- The step is never [chained](3_conformer_search.md) behind a search (`slurm.ensembles.mode` is ignored
  for it): its input is a lot an earlier run produced.
- `params_<lot>.json` records `compared.continue_from` = the source lot and the sha256 of *its*
  `params_<lot>.json`, so the provenance chains search -> source lot -> this one, and all of it is
  checked before the step runs. A source lot that was recomputed with other parameters fails the step
  with `ParamsMismatch` rather than silently refining the new geometries.
- `run_<i>_<lot>.json` records `continued_from: {lot, file, n_input}`.
- Without `continue_from` the key is left out of the hash, so **every `params_<lot>.json` written before
  this feature keeps its hash** and no existing library is invalidated.
- A replicate whose source lot has no finished run yet simply has nothing pending: `orcadesc submit`
  warns and skips it (the library reports it as `ensemble_missing`), and the rest of the submission goes
  ahead. Run the config that produces the source lot first.
- The checkpoint above is keyed on the file this step actually reads, so a checkpoint of the source lot
  is never mistaken for one of this step.
- Conformer order is preserved, but `source_index` in the output counts positions in
  `raw_<i>_<source-lot>.xyz`, not in `raw_<i>.xyz`. The two coincide while `keep_unconverged: true` (the
  default); with `keep_unconverged: false` the source lot has already dropped conformers.

### Constrained post-optimisation

With `constrained: true` the refiner gets the search's constraint block as its `constraints` parameter
(bceg's own config takes `post_optimization.params.constraints` directly) and keeps fixed, in Cartesian
coordinates, every atom an entry names: the frozen groups, the plane groups and the atoms of every bond,
angle and dihedral ([2_pre_refinement.md](2_pre_refinement.md#imposing-constraints)). The residual
force that decides `converged` ignores them. `details.fixed_atoms` lists them (0-based). Without
constraints the parameter is left out of `params_<lot>.json`, so unconstrained ensembles keep their hash.

## `uma`: fairchem UMA through ASE

| `params` | Default | Meaning |
|---|---|---|
| `model` | `uma-s-1p2` | checkpoint name; the weights are gated on Hugging Face and fetched once on a login node with `orcadesc fetch-weights --model <name>` into `FAIRCHEM_CACHE_DIR` ([cluster_setup.md](../cluster_setup.md)) |
| `task` | `omol` | UMA task head (`omol` for molecules) |
| `optimizer` | `LBFGS` | ASE optimiser: `LBFGS`, `BFGS`, `FIRE` |
| `fmax` | `0.05` | convergence criterion on the forces, eV/Angstrom |
| `max_steps` | `500` | optimiser steps per conformer |
| `keep_unconverged` | `true` | keep conformers that did not reach `fmax` (flagged `converged: false` on their comment line); `false` drops them |
| `stall_steps` | `null` (off) | stop a conformer whose lowest energy went down by less than `stall_energy` over the last `stall_steps` steps (an optimiser oscillating between two geometries); it is then unconverged (`keep_unconverged` keeps or drops it) and flagged `stalled: true`; `details.n_stalled` counts them. Hashed only when set. `100` stopped the one oscillating SmI2 conformer at step 437 and none of 2359 converged ones |
| `stall_energy` | `0.001` | eV; the energy drop below which `stall_steps` counts as no progress (hashed only with `stall_steps`) |
| `multiplicity` | `null` | spin state of this refinement: `null` keeps the transformation's; a number re-optimises at that multiplicity ([at another spin state](#at-another-spin-state-multiplicity)); every refiner takes it |
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

Every worker holds its own copy of the model, and the job's own process may hold one more (a `uma`
pre-refinement). uma-s-1p2 on CPU takes ~2.0 GB per process up to ~100 atoms, then roughly
+15 kB x atoms² (+0.7 GB at 200 atoms, +1.6 GB at 300); loading the checkpoint briefly takes ~5 GB, so
the workers load it one at a time. For `uma` on `cpu`, a job profile without `mem` gets it from its
CPUs: `(workers + 1) x 3.6 GB + 2 GB`, enough up to ~300 atoms, `workers` being one per CPU unless set,
i.e. `35G` for `cpus_per_task: 8` (the chained job's `mem` is shared with the search, which needs less). An explicit `mem` is kept; `submit` warns when it is below
that. A worker killed by the memory limit fails the step with a message naming the pool, the other
workers are killed and the job ends; raise `mem` or lower `workers`. Each worker writes
its own `opt_<i>.log`; the conformers keep their input order (`source_index`) whatever finished first.
`run_1_<lot>.json` records `details.workers` and `details.threads`, so comparing `wall_time_s` of a
run with `workers: 1` against the default on one molecule is the benchmark:

```yaml
post_optimization:
  refiner: uma
  params: {model: uma-s-1p2, workers: 1}        # serial reference; drop `workers` for the pool
```

## `xtb`: GFN-n xtb, CPU

The cheap refiner, and the usual first half of a cascade. The searchers that embed rather than
optimise (`rdkit_etkdg`, `loqi`, `molassembler`) hand on geometries that GFN2-xTB tidies for a small
fraction of what the better method costs; the better method then continues from that lot with
[`continue_from`](#refining-further-continue_from) instead of starting from the raw search output.

Each conformer is optimised by its own `xtb start.xyz --gfn <n> --opt <level> --chrg <charge> --uhf <uhf>`
run in a subfolder `conf_<i>/` of the job's scratch (`xtb.out` kept there until the job ends), exactly
as for `gxtb` above — same program, different method flags. `--uhf` is always passed (unlike g-xTB,
xtb reads it for closed shells too). The energy is the `TOTAL ENERGY` of the output; a conformer whose
optimisation stops before convergence is taken from `xtblast.xyz`.

This is the same method as `pre_refinement.method: xtb` ([2_pre_refinement.md](2_pre_refinement.md)),
which pre-optimises the single start geometry a search begins from; here every conformer the search
produced is optimised instead.

| `params` | Default | Meaning |
|---|---|---|
| `gfn` | `gfn2` | method: `gfn2`, `gfn1`, `gfn0` or `gfnff` (the force field). `gfn0` additionally needs its parameter file, see below |
| `level` | `normal` | xtb `--opt` level: `crude`, `sloppy`, `loose`, `lax`, `normal`, `tight`, `vtight`, `extreme` |
| `cycles` | xtb's default | optimiser cycles per conformer (xtb `--cycles`) |
| `keep_unconverged` | `true` | keep conformers that did not converge within the cycles (flagged `converged: false`); `false` drops them |
| `lot` | the `gfn` value | label used in the file names (`raw_1_gfn2.xyz`), so two GFN levels never collide |
| `executable` | `xtb` | resource: a name on PATH or an absolute path |
| `threads` | job's CPUs | resource: OpenMP threads of each xtb run (`SLURM_CPUS_PER_TASK`); the conformers run one after the other |

`versions`: `xtb` (the `--version` line); `details.executable` is the resolved path. Per conformer:
`cycles` when the optimisation converged. The refiner has no `device`: xtb runs on CPUs only. It is
resumable, like every other refiner ([the checkpoint](#a-job-that-runs-out-of-time-the-checkpoint)).

#### `gfn0` needs its parameter file on the search path

GFN1, GFN2 and GFN-FF are compiled into xtb, but **GFN0 reads its parameters from
`param_gfn0-xtb.txt` at run time**, and xtb looks for that file only in `XTBPATH` or `XTBHOME` — not
next to the binary, even though the build ships it under `<prefix>/share/xtb`. With neither set, a
`gfn0` job dies on the compute node with `param_gfn0-xtb.txt not found!` long after it was queued, so
the refiner refuses `gfn: gfn0` while the config is read unless the file is reachable. Export it in
`slurm.setup`:

```yaml
slurm:
  setup:
    - export XTBPATH=$CONDA_PREFIX/opt/gxtb-2.0.1/share/xtb
```

The cascade, as two library configs over one `ensembles_dir`
([library_workflow.md](../library_workflow.md#5-a-second-library-on-an-existing-ensembles-tree)):

```yaml
post_optimization:                              # first config: clean up the embedded geometries
  refiner: xtb
  params: {gfn: gfn2, level: normal}            # -> raw_<i>_gfn2.xyz
```

```yaml
post_optimization:                              # second config: pay for UMA only from there on
  refiner: uma
  continue_from: gfn2
  params: {model: uma-s-1p2, lot: uma-s-1p2, fmax: 0.01}
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
subfolder `conf_<i>/` of the job's scratch (`xtb.out` kept there until the job ends), through the same
driver as `xtb` above; `--uhf` is passed only for open shells (with the flag g-xTB always runs
unrestricted). The energy is the `TOTAL ENERGY`
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
| `stall_steps` | `null` (off) | stop a conformer whose lowest energy went down by less than `stall_energy` over the last `stall_steps` steps (an optimiser oscillating between two geometries); it is then unconverged (`keep_unconverged` keeps or drops it) and flagged `stalled: true`; `details.n_stalled` counts them. Hashed only when set. `100` stopped the one oscillating SmI2 conformer at step 437 and none of 2359 converged ones |
| `stall_energy` | `0.001` | eV; the energy drop below which `stall_steps` counts as no progress (hashed only with `stall_steps`) |
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
checks that every default of every registered refiner is documented).

There are two shared loops to build on rather than a bare `run()`. `xtb` and `gxtb` go through
`bceg/refine/xtb_opt.py`: subclass `XtbRefiner`, take `XTB_REFINER_DEFAULTS` into `defaults` and define
`command(structure, exe)` (plus `check_output(stdout, exe)` to reject the wrong binary, and `validate()`
for a parameter the command line would otherwise hand to xtb unchecked) — the per-conformer subfolder,
the `xtbopt.xyz`/`xtblast.xyz` fallback, the energy and cycle parsing and the `$fix` constraints come
for free. `uma` and `aimnet2` go through `bceg/refine/ase_opt.py` for an ASE calculator whose `device`
is routed to the gpu profile. An ASE refiner gives the shared loop a
`make_calculator(structure)` method and the `optimizer`, `fmax`, `max_steps`, `keep_unconverged`,
`device`, `workers` and `threads` parameters; the loop rebuilds the refiner from its parameters inside
every worker process (spawned, so the refiner must be picklable before its model is loaded).
