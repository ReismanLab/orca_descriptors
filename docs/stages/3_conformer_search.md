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
| `constraints` | none | atoms frozen and bonds, angles, dihedrals held during the search (`crest`, `goat`), [below](#constraints) | yes, when given |
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

## Constraints

A cheap search level can distort what it describes badly, e.g. GFN2-xTB or GFN-FF around a transition
metal. The usual remedy: pre-optimise with UMA ([stage 2](2_pre_refinement.md), `method: uma`), then
search with the metal and its first shell held where UMA put them. `constraints` says which atoms, in
terms of the molecule (SMARTS or elements), never as atom numbers, so one block serves the whole
library. CREST and GOAT take it; the other searchers refuse it at load.

```yaml
pre_refinement:
  method: uma
conformer_search:
  name: crest_gfn2_fixNi1              # constraints are part of the parameter set: a new set needs a new name
  searcher: crest
  constraints:
    freeze:
      - {element: Ni, neighbors: 1}    # Ni and every atom bound to it (dative bonds included)
```

| Key | Default | Meaning |
|---|---|---|
| `freeze` | `[]` | atoms kept at their start-geometry positions: every atom of every match, plus those within `neighbors` bonds |
| `bonds` | `[]` | one distance constraint per match of a 2-atom selection |
| `angles` | `[]` | one angle constraint per match of a 3-atom selection |
| `dihedrals` | `[]` | one dihedral constraint per match of a 4-atom selection |
| `planes` | `[]` | one plane group per match (its atoms plus those within `neighbors` bonds, at least 4), frozen in the search; made planar by `pre_refinement: constrained` |
| `force_constant` | `null` | strength of every constraint below that sets none, see [strength](#strength) |

Each entry of the four lists is a SMARTS string, or a mapping:

| Entry key | Default | Meaning |
|---|---|---|
| `smarts` | | the atoms, e.g. `"[Ni]"`, `"[Ni]~[#7]"`; with atom maps (`"[#7:1]~[Ni:2]~[#7:3]"`) only the mapped atoms count, in map order. Bonds to a metal are dative, so write them `~` (any bond): `-` (single) matches none of them |
| `element` | | instead of `smarts`: a symbol or a list of symbols, e.g. `Ni` or `[Ni, Pd]` |
| `neighbors` | `0` | `freeze` and `planes` only: also take the atoms within this many bonds of the matched ones |
| `value` | start geometry | `bonds`, `angles`, `dihedrals`: the target, Angstrom or degrees; by default the value in the start geometry |
| `force_constant` | top-level `force_constant` | this entry's strength |
| `optional` | `false` | `true`: a molecule where the selection matches nothing runs unconstrained; `false`: its search fails with the selection named |

The selection is resolved per molecule when the search runs, on the start geometry handed to the
search. A bond, angle or dihedral whose atoms are all frozen is dropped (it is fixed already). The
outcome is recorded in `run_<i>.json` under `details.constraints`: `freeze` (0-based atom indices, the
order of every xyz file), `bonds`, `angles`, `dihedrals` (`atoms`, `value`, `force_constant`), and
`labels` (`"4": "Ni4"`, ...). The searcher's input file in `logs/` shows what the program received.

By default the constraints act on the search only: the stage 2 pre-optimisation produces the
geometry that is frozen (pair the constraints with `pre_refinement: uma` or with `start_geometry: input`;
with `xtb` the atoms are frozen where xtb put them), and stages 4 and 6 relax every conformer freely.
Each of those stages can keep them with `constrained: true`:

| Stage | `constrained: true` | bceg parameter |
|---|---|---|
| `pre_refinement` | the planes and the values given are imposed on the embedding and kept by the pre-optimisation, which relaxes the rest, bond lengths included ([imposing constraints](2_pre_refinement.md#imposing-constraints)) | `start_constraints`, `start_impose_steps` |
| `post_optimization` | the held atoms stay fixed in every conformer's optimisation | the refiner's `constraints` |
| `dft` | `{ C i C }` for every held atom in the ORCA optimisation | none |

The *held atoms* are every atom an entry names: the frozen groups (with or without a force constant), the
plane groups and the atoms of every bond, angle and dihedral. A plane, or a linear I-Sm-I, is then kept
by freezing its atoms in the search, e.g. for a square-planar Ni:

```yaml
pre_refinement: {method: uma, constrained: true}
conformer_search:
  name: crest_gfn2_NiPlanar
  searcher: crest
  constraints:
    planes: [{element: Ni, neighbors: 1}]
  params: {shake: 0, tstep: 2.0, crossing: false}
post_optimization: {refiner: uma, constrained: true}
dft: {level_of_theory: {...}, constrained: true}
```

`start_constraints` and `start_impose_steps` (searcher parameters, set from the pre_refinement block)
enter the method folder's hash only when the constraints are imposed.

### Strength

Without a force constant every constraint is as strict as the program makes it: frozen atoms are
exact (ORCA's `{C i C}`; CREST's `freeze` zeroes their gradient, but see SHAKE below); bonds,
angles and dihedrals are exact in GOAT and harmonic in CREST at its default 0.01. `force_constant`
(top level for all entries, per entry to override) makes them harmonic restraints of that strength, in
Eh/Bohr² for distances and Eh/rad² for angles and dihedrals; for reference, xtb's `$constrain`
default is 0.5. A frozen group with a force constant becomes a restraint on every distance within
the group (xtb's `$constrain atoms:`), so it keeps its shape but may move and turn as a whole.

GOAT's constraints are always exact: a `force_constant` with `searcher: goat` is refused.

With CREST's defaults, frozen atoms are **not** exact: CREST 3.0.2 runs the MD with SHAKE on every
bond except bonds to a metal, and SHAKE moves both atoms of a bond, frozen or not. A frozen atom
bonded to a free one drifts (the N donors of a frozen Ni first shell: Ni-N 1.83-2.15 Å from a
2.00 Å start), and the genetic crossing (GC) mixes the coordinates of conformers, frozen atoms
included (up to 0.02 Å). bceg leaves CREST's defaults alone; for exact frozen atoms set all three
CREST parameters (measured on a pyox-NiH2 with GFN2, 16 cores: exact, 8 min):

```yaml
conformer_search:
  searcher: crest
  constraints: {freeze: [{element: Ni, neighbors: 1}]}
  params:
    shake: 0          # SHAKE off
    tstep: 2.0        # fs; without SHAKE, CREST's 5 fs step for GFN2 blows the MD up
    crossing: false   # crest -nocross: the ensemble comes from the MTDs alone
```

| | `freeze` | `freeze` + `force_constant` | `bonds` / `angles` / `dihedrals` | + `force_constant` |
|---|---|---|---|---|
| `crest` | `[calculation] freeze` (exact with `shake: 0`, `crossing: false`, below) | `[[calculation.constraints]]` bonds within the group | `[[calculation.constraints]]`, CREST's default strength | that strength |
| `goat` | `%geom Constraints { C i C }` | refused | `{ B i j d C }`, `{ A ... }`, `{ D ... }` (exact) | refused |

```yaml
constraints:
  force_constant: 0.5                              # Eh/Bohr^2, Eh/rad^2
  freeze: [{smarts: "[Ni]", neighbors: 1}]         # restrained, not frozen, because of force_constant
  bonds: ["[Ni]~[#7]"]                             # every Ni-N distance at its start value
  angles: [{smarts: "[#7]~[Ni]~[#7]", value: 90}]  # the bite angle, at 90 degrees
  dihedrals:
    - {smarts: "[#7:1]~[#6:2]~[#6:3]~[#7:4]", force_constant: 1.0, optional: true}
```

## `crest`: CREST iMTD-GC (metals and organics)

CREST 3.x driven by a TOML input (`crest --input input.toml`); starts from the pre-refined geometry.

| `params` | Default | Meaning |
|---|---|---|
| `runtype` | `imtd-gc` | CREST runtype |
| `workhorse` | `gfn2` | level of the search: a method name (`gfn2`, `gfn1`, `gfn0`, `gfnff`) or a full `[[calculation.level]]` mapping |
| `workhorse_extra` | `{}` | extra keys of the workhorse level (e.g. `{alpb: toluene}`) |
| `refine` | `null` | CREST's inline refinement of the final ensemble: `null`, `sp`, `opt` (`add`, `freq`) |
| `refine_level` | UMA through CREST's `mlip` calculator (`mlip_backend: uma`, `mlip_uma_task: omol`, `mlip_uma_model: uma-s-1p2`) | level used by `refine`; needs `fmlip-relay-server` on PATH |
| `shake` | `null` | `[dynamics] shake`: `0` off, `1` bonds to H, `2` all bonds; `null` leaves CREST's choice (all bonds except to a metal) |
| `tstep` | `null` | `[dynamics] tstep`, MD time step in fs; `null` leaves CREST's choice (5 fs for GFN2, which needs SHAKE) |
| `crossing` | `true` | genetic crossing (GC) after the MTDs; `false` runs `crest -nocross` |
| `reftopo` | `true` | CREST's check that every conformer keeps the topology of the input; `false` runs `crest --noreftopo` (needed when the start geometry's bonding is ambiguous, e.g. a hydride on a metal; may let rearranged structures through) |
| `cluster` | `null` | `null`: hand on `crest_conformers.xyz`; `true`: `crest --cluster`, CREST's PCA/k-means clustering picks the number of clusters, and the clustered subset `crest_clustered.xyz` is handed on instead; a number `N`: `--cluster N`, N clusters. The full ensemble is kept as `logs/raw_<i>_crest_conformers.xyz` |
| `toml_extra` | `{}` | deep-merged into the TOML document, e.g. `{cregen: {ewin: 6.0}}` for CREST's own energy window (kcal/mol); applied last, so it wins over `shake` and `tstep` |
| `seed_offset` | `0` | see above |
| `threads` | job's CPUs | resource: CREST threads (`SLURM_CPUS_PER_TASK`) |
| `refine_threads` | `null` | resource: threads per refinement-level instance |
| `mlip_device` | `cpu` | resource: device of the `mlip` levels; `cuda` sends the job to `slurm.ensembles.gpu` when an `mlip` level is used |
| `executable` | `crest` | resource: which CREST binary |
| `md_retries` | `2` | resource: when CREST's trial MD fails (`Automatic MD restart failed`, e.g. a hydride leaving the Ni at GFN2), rerun from a new start geometry up to this many times; the failed logs are kept as `raw_<i>_crest_md_failed_<k>.out`, the attempts in `details.md_retries` |

Output: `crest_conformers.xyz` (the full ensemble), so `n_conf` can be large; stage 5, CREST's
`ewin` and `cluster` are how the count going to ORCA is kept in hand. With `cluster` it is
`crest_clustered.xyz`, one representative per cluster. `reftopo` and `cluster` are command-line
flags (CREST's TOML has no keys for them) and enter the method's hash only when they are not
their defaults. CREST prints an
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
  name: crest_gfn2_ewin15_clustered       # the pre-orcadesc LNiH2 libraries (CREST 2.12 then, 3.x now)
  searcher: crest
  params:
    workhorse: gfn2
    reftopo: false
    cluster: true
    toml_extra: {cregen: {ewin: 15.0}}
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
| `n_confs` | `100` | ensemble size requested: a number, or a [formula in `n_rot`](#n_confs-as-a-function-of-rotatable-bonds) |
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
| `n_confs` | `300` | conformers embedded: a number, or a [formula in `n_rot`](#n_confs-as-a-function-of-rotatable-bonds) |
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

### `n_confs` as a function of rotatable bonds

For `molassembler`, `rdkit_etkdg` and `loqi`, `n_confs` can be a quoted formula in `n_rot`, the
molecule's number of rotatable bonds, instead of a fixed number:

```yaml
conformer_search:
  name: etkdg_rot30
  searcher: rdkit_etkdg
  params: {n_confs: "(n_rot + 2) * 30"}       # 0 rotors: 60, 3 rotors: 150
```

Allowed: numbers, `n_rot`, `+ - * / // **`, `min`, `max` (e.g. `"min(500, (n_rot + 2) * 30)"`); the
result is rounded to an integer of at least 1. `n_rot` is RDKit's strict rotatable-bond count on the
heavy-atom graph (`bceg.chem.rotatable_bonds`): methyl groups, bonds in rings (chelate rings through
the metal included), amide C-N bonds and dative bonds are not counted. Each run records `n_rot` and the
resulting `n_confs` in its `details`. The formula string is what enters the params hash, so a library
run with `n_confs: 300` is not the same method as one run with a formula.

## `loqi`: LoQI conformer generator (organics only)

Called through its CLI (`loqi sample`, the console script of the `loqi` pip package in the shared env);
runs on the GPU profile unless `device: cpu`. A molecule that is not organic is `not_applicable` for this
searcher. No start geometry: `pre_refinement` must be `none`. The replicate seed is passed as `--seed`
(upstream would otherwise sample every replicate with seed 42). Flags checked against upstream
`src/loqi/cli.py` (loqi 1.1.0) on 2026-10-02.

| `params` | Default | Meaning |
|---|---|---|
| `n_confs` | `100` | conformers sampled: a number, or a [formula in `n_rot`](#n_confs-as-a-function-of-rotatable-bonds) |
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
