# Stage 3: `conformer_search`

The conformer ensemble of the transformed structure, written by bceg to
`<ensembles_dir>/<transformation>/<InChIKey>/<name>/raw_<i>.xyz`, one file per replicate `i` (energies
in Hartree on the comment lines when the searcher gives them). The folder also holds `params.json`
(every parameter with the defaults filled in, hashed), `run_<i>.json` (status, phase, SLURM ids, host,
wall time and core-hours in total and per phase under `phases` — `pre_refinement` and `search`, see
[stage 2](2_pre_refinement.md#cost) —, program versions, `details`) and `logs/` (`raw_<i>_crest.out`,
`raw_<i>_input.toml`, `raw_<i>_start_geometry.xyz`, ...; a failed run keeps its small files under
`logs/raw_<i>_failed/`). With the default `replicates: 1` the only files are `raw_1.xyz` and `run_1.json`.

**The `name` is the parameter set.** Every parameter of the searcher is logged in `params.json`;
running again under the same name with any parameter changed is refused with a diff (`REFUSED` from
`orcadesc submit`, `bceg PARAMS MISMATCH` from bceg). Two searches that differ by one value need two
names, e.g. `crest_gfn2_ewin6` and `crest_gfn2_ewin3`; their folders sit side by side and each library
YAML points at one of them.

## Keys

| Key | Default | Meaning | Hashed |
|---|---|---|---|
| `name` | required | method folder; letters, digits, `_`, `-`, `.` | yes |
| `searcher` | required | `crest`, `goat`, `molassembler`, `rdkit_etkdg`, `loqi` | yes |
| `params` | `{}` | the searcher's parameters, tables below; an unknown key is refused at load | yes, except the resource keys marked |
| `replicates` | `1` | how many times the search runs, each replicate from its own seed (and start geometry); the ORCA stage pools them, see below | yes, in the library's `params.json` (not in the method folder's) |
| `slurm` | `{}` | overrides of `slurm.ensembles.cpu` (or `.gpu`) for this step: `partition`, `time`, `cpus_per_task`, `ntasks`, `mem`, `mem_per_cpu`, `nodes`, `gres`, `extra`, ... | no |

Every searcher also takes `seed_offset` (default `0`): the replicate seed is `seed_offset + replicate`.
The pre-refinement keys (`start_preopt`, ...) are set from the `pre_refinement` block, not here
([2_pre_refinement.md](2_pre_refinement.md)).

## Replicates

```yaml
conformer_search:
  name: crest_gfn2
  searcher: crest
  replicates: 3
```

bceg runs the search `replicates` times for every molecule, replicate `i` seeded by `seed_offset + i`
(a new ETKDG embedding each for CREST, GOAT and molassembler, so the searches start from different
geometries; see [2_pre_refinement.md](2_pre_refinement.md)); each replicate is one cluster task that
writes `raw_<i>.xyz`, `run_<i>.json` and, after stage 4, `raw_<i>_<lot>.xyz`. The `prepare` task of
the ORCA stage then **pools the replicates, in order, into one ensemble**: the conformers of `raw_1`,
then those of `raw_2`, ... Stage 5 prunes that pooled list against its overall minimum, and `conf_N`
numbers positions in it. Every conformer records its `replicate` and `index_in_replicate` in
`metadata.json` and on the comment line of its `input.xyz` (`replicate=2 replicate_index=1`). Nothing
removes duplicates across replicates: two replicates finding the same minimum give it two ORCA jobs.

Every replicate must be finished before a molecule is prepared: one without a record yet is
`ensemble_missing` (the prepare task waits for the next `submit`), one that failed is `ensemble_failed`
with the replicate named in `reason`. The library's content therefore never depends on which bceg jobs
happened to be done. `orcadesc submit` again resubmits only the unfinished replicates.

`replicates` is part of the library's `params.json` (a library with 1 and one with 3 are two libraries),
but not of the method folder's: bceg records the count without comparing it, so a shared `ensembles_dir`
can hold three replicates of a method and serve a library that pools one as well as one that pools three.

## `crest`: CREST iMTD-GC (metals and organics)

CREST 3.x driven by a TOML input (`crest --input input.toml`); starts from the pre-refined geometry.

| `params` | Default | Meaning |
|---|---|---|
| `runtype` | `imtd-gc` | CREST runtype |
| `workhorse` | `gfn2` | level of the search: a method name (`gfn2`, `gfn1`, `gfn0`, `gfnff`) or a full `[[calculation.level]]` mapping |
| `workhorse_extra` | `{}` | extra keys of the workhorse level (e.g. `{alpb: toluene}`) |
| `refine` | `null` | CREST's inline refinement of the final ensemble: `null`, `sp`, `opt` (`add`, `freq`) |
| `refine_level` | UMA through CREST's `mlip` calculator (`mlip_backend: uma`, `mlip_uma_task: omol`, `mlip_uma_model: uma-s-1p2`) | level used by `refine`; needs `fmlip-relay-server` on PATH |
| `toml_extra` | `{}` | deep-merged into the TOML document, e.g. `{cregen: {ewin: 6.0}}` for CREST's own energy window (kcal/mol) |
| `seed_offset` | `0` | see above |
| `threads` | job's CPUs | resource: CREST threads (`SLURM_CPUS_PER_TASK`) |
| `refine_threads` | `null` | resource: threads per refinement-level instance |
| `mlip_device` | `cpu` | resource: device of the `mlip` levels; `cuda` sends the job to `slurm.ensembles.gpu` when an `mlip` level is used |
| `executable` | `crest` | resource: which CREST binary |

Output: `crest_conformers.xyz` (the full ensemble, not `crest_clustered.xyz`), so `n_conf` can be
large; stage 5 and CREST's `ewin` are how the count going to ORCA is kept in hand. CREST prints an
OpenBLAS warning per thread change unless `OPENBLAS_NUM_THREADS=1`, which bceg sets.

```yaml
conformer_search:
  name: crest_gfn2
  searcher: crest
  params:
    workhorse: gfn2
    toml_extra: {cregen: {ewin: 6.0}}
```

```yaml
conformer_search:
  name: crest_gfnff_uma_sp
  searcher: crest
  params: {workhorse: gfnff, refine: sp}
```

## `goat`: ORCA GOAT (metals and organics)

`! GOAT <method>` in ORCA 6; starts from the pre-refined geometry. ORCA wants MPI ranks: give the
step `slurm: {ntasks: N, cpus_per_task: 1, nodes: 1}` and `nprocs: N`.

| `params` | Default | Meaning |
|---|---|---|
| `method` | `XTB` | level of the GOAT run (simple-input keyword) |
| `keywords` | `""` | additional simple-input keywords |
| `goat_block` | `{}` | `key: value` lines inside `%goat ... end`, e.g. `{MaxEn: 6.0}` |
| `seed_offset` | `0` | see above |
| `nprocs` | job's CPUs | resource: `%pal nprocs` |
| `maxcore` | `2000` | resource: `%maxcore`, MB per process |
| `orca_cmd` | `orca` | resource: the ORCA binary; use the absolute path for parallel runs |

```yaml
conformer_search:
  name: goat_xtb
  searcher: goat
  params: {method: XTB, nprocs: 4, goat_block: {MaxEn: 6.0},
           orca_cmd: /central/software9/external/orca/orca_6_1_0_linux_x86-64_shared_openmpi418_avx2/orca}
  slurm: {ntasks: 4, cpus_per_task: 1, nodes: 1}
```

In chained mode the post-optimisation inherits this `slurm` block: its worker pool then uses the
`ntasks` CPUs of the allocation ([stage 4](4_post_optimization.md#cpu-parallelism-workers-and-threads)); use
`slurm.ensembles.mode: split` to give it its own.

## `molassembler`: SCINE Molassembler distance geometry (transition-metal complexes)

The molecular graph comes from the RDKit template; local shapes and stereo (square-planar vs
tetrahedral Ni) are perceived from the pre-refined start geometry, hence stage 2 matters here.

| `params` | Default | Meaning |
|---|---|---|
| `n_confs` | `100` | ensemble size requested |
| `dg_config` | `{}` | attributes of `scine_molassembler.dg.Configuration` |
| `fixed_start_seed` | `false` | `true`: one start geometry (seeded by `seed_offset`) for every replicate |
| `seed_offset` | `0` | see above |

No energies: with `post_optimization: null` the conformers reach ORCA without energies and
`pruning.energy_window_kcal_mol` must stay `null`.

```yaml
conformer_search:
  name: masm100
  searcher: molassembler
  params: {n_confs: 100}
  slurm: {cpus_per_task: 1}
```

## `rdkit_etkdg`: RDKit ETKDGv3 (organics)

Embeds `n_confs` conformers, optionally force-field optimised (energies then in Hartree). No start
geometry: `pre_refinement` must be `none`.

| `params` | Default | Meaning |
|---|---|---|
| `n_confs` | `300` | conformers embedded |
| `prune_rms` | `-1.0` | RMSD pruning threshold, Angstrom; negative = none |
| `ff_optimize` | `none` | `none`, `auto` (MMFF94s if parametrised, else none), `MMFF94`, `MMFF94s`, `UFF` |
| `max_iters` | `2000` | force-field iterations |
| `use_small_ring_torsions` | `true` | ETKDG option |
| `use_macrocycle_torsions` | `true` | ETKDG option |
| `use_random_coords` | `false` | ETKDG option |
| `seed_offset` | `0` | see above |
| `threads` | job's CPUs | resource |

```yaml
pre_refinement: none
conformer_search:
  name: etkdg300_mmff
  searcher: rdkit_etkdg
  params: {n_confs: 300, ff_optimize: MMFF94s}
```

## `loqi`: LoQI conformer generator (organics only)

Called through its CLI (`loqi sample`, the console script of the `loqi` pip package in the shared env);
runs on the GPU profile unless `device: cpu`. A molecule that is not organic is `not_applicable` for this
searcher. No start geometry: `pre_refinement` must be `none`. The replicate seed is passed as `--seed`
(upstream would otherwise sample every replicate with seed 42). Flags checked against upstream
`src/loqi/cli.py` (loqi 1.1.0) on 2026-10-02.

| `params` | Default | Meaning |
|---|---|---|
| `n_confs` | `100` | conformers sampled |
| `model` | `loqi_flow_v1.1.0` | registered checkpoint (`loqi`, `loqi_flow`, `loqi_flow_v1.1.0`) or the path of a `.ckpt` file |
| `config` | `null` | bundled config name or YAML path, only for a checkpoint outside the registry |
| `steps` | `null` | sampling steps (`null`: the checkpoint's training value) |
| `add_hs` | `true` | LoQI adds the hydrogens of the SMILES (`false`: `--no-add-hs`) |
| `extra_args` | `[]` | additional CLI arguments, appended verbatim |
| `seed_offset` | `0` | see above |
| `device` | `null` | resource: torch device (`null`: cuda if available, else cpu; `cpu` sends the step to the cpu profile) |
| `batch_atoms` | `null` | resource: atom budget per batch (GPU memory) |
| `threads` | job's CPUs | resource: CPU threads of the torch side (`SLURM_CPUS_PER_TASK`) |
| `executable` | `loqi` | resource: the `loqi` on PATH |

**Weights.** LoQI downloads a registered checkpoint on first use into `$LOQI_CACHE_DIR` (else
`~/.cache/loqi`). Compute nodes cannot, so fetch it once on a login node into the shared cache and keep the
same `export` in `slurm.setup` (the example config has it):

```bash
export LOQI_CACHE_DIR=/groups/ReismanLab/loqi_cache
loqi download --model loqi_flow_v1.1.0        # ~360 MB, SHA-256 verified; prints the cached path
```

`versions`: `loqi` (the package version). `details`: `seed`, `model`, `checkpoint`, `n_sampled`,
`n_failed_samples` (samples upstream could not turn into a geometry). The conformers carry no energies.

```yaml
pre_refinement: none
conformer_search:
  name: loqi100
  searcher: loqi
  params: {n_confs: 100}
slurm:
  setup:
    - export LOQI_CACHE_DIR=/groups/ReismanLab/loqi_cache
  ensembles:
    gpu: {partition: gpu, time: "04:00:00", gres: "gpu:1", cpus_per_task: 4, mem: 16G}
```

## Status

`search` in `orcadesc status`: `queued` while the bceg job waits or is still in the pre-refinement
phase, `running`, `done`, `failed` (error in `reason`, files under `logs/raw_<i>_failed/`), `n/a` when
the searcher does not apply to the molecule (LoQI on a complex). `n_conf` is the ensemble size, summed
over the replicates. With several replicates one record stands for all: `failed` if any replicate
failed, else `running` if any runs, `done` only once every replicate is done, `-` while one has no
record yet.
