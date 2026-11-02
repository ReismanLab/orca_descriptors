# Stage 6: `dft` (and `orca`)

The electronic properties every descriptor is computed from. Two engines produce them, chosen with
`dft.engine`; both write the same `conf_N/orca_condensed.json`, so **the laptop side does not change**:
the same spec, the same commands and the same column names serve either one.

| `dft.engine` | What runs | Cost per conformer (35-atom L-NiH2 complex) |
|---|---|---|
| `orca` (default) | one ORCA 6.1 compound job | hours of core time |
| `gxtb` | one `xtb --gxtb` run per step ([below](#the-g-xtb-engine-dftengine-gxtb)) | 5.8 s for a single point plus both redox states |

Use `gxtb` to screen a library cheaply, and `orca` when you need the properties it alone computes
(NBO, NMR, polarizability, solvation). The rest of this page is the ORCA engine; the g-xTB one is at
the end.

One ORCA 6.1 compound job per kept conformer, in `<LIB>/<InChIKey>/conf_N/`. The `prepare` task
renders `calcParam.inp` once per molecule from the template and the `level_of_theory` block, with the
charge and multiplicity of the ensemble (stage 1) and `%pal nprocs` / `%maxcore` from `slurm.orca`,
copies it into every `conf_N/`, and submits one SLURM array per molecule (one element per conformer).
Each element records `running`, runs `orca calcParam.inp`, records `completed` only if ORCA exited 0
**and** printed `ORCA TERMINATED NORMALLY` (else `failed`; `timeout` when SLURM's TERM arrived 10 min
before the walltime), and condenses the outputs to `orca_condensed.json` when `condense: true`.

## Keys

| Key | Default | Meaning | Hashed |
|---|---|---|---|
| `orca` (top level) | `on` | `off`: stop after the ensembles; the same command later with `on` refines them | no |
| `dft.engine` | `orca` | `orca` (the compound job) or `gxtb` ([below](#the-g-xtb-engine-dftengine-gxtb)) | yes, but only stored when it is `gxtb`, so a library made before the key existed keeps its hash |
| `dft.level_of_theory` | required (ORCA) | the table below; may stay unset (`{...}`) while `orca: off`; refused with `engine: gxtb` | yes, but an unset one may later be set (and the rest of `dft` changed with it) |
| `dft.constrained` | `false` | `true`: every atom `conformer_search.constraints` names stays where stages 2 to 4 put it in the optimisation step: one `{ C i C }` line (0-based) per atom, in place of the template's `&{geom_constraints}` line; the frequency step has no constraints, so small imaginary modes along the frozen coordinates are expected | yes, when `true` |
| `dft.calc_param_template` | `desc_utils/submit/calcParam.inp.tmpl` | your own copy of the template, only to change which compound steps exist or how they chain | yes (its SHA-256; the copy is frozen under `_jobs/<timestamp>/`) |
| `slurm.orca` | `{time: "160:00:00", mem_per_cpu: 4G, ntasks: 16, nodes: 1}` | the ORCA job profile; `ntasks` becomes `%pal nprocs`, `mem_per_cpu` (or `mem`/`ntasks`) gives `%maxcore` at 75 % | no |
| `slurm.orca_cmd` | `orca` on PATH | the ORCA binary, when `module load` is not enough | no |
| `condense` (top level) | `true` | write `conf_N/orca_condensed.json` when each job ends | no |

### `level_of_theory`

All keys but `extra_basis` are required; an unknown key is refused.

| Key | Type | Compound step it feeds | Meaning |
|---|---|---|---|
| `opt_geom` | bool | `optimization` | run the geometry optimisation (else the input geometry is used as is) |
| `freq_calc` | bool | `frequency` | harmonic frequencies and thermochemistry on the optimised geometry |
| `nmr_calc` | bool | `nmr` | GIAO shieldings |
| `vertical_redox` | bool | `oxidation`, `reduction` | vertical N-1 / N+1 electron single points on the optimised geometry (multiplicity + 1) |
| `opt_method` | string | `optimization`, `frequency` | simple-input keywords, e.g. `"wb97m-d4 def2-tzvp(-f)"` |
| `opt_solvent` | string | `optimization`, `frequency` | solvation keyword, `""` for gas phase |
| `svp_method` | string | `singlepoint`, `oxidation`, `reduction` | e.g. `"m06 d4 def2-tzvp"` |
| `property_solvent` | string | `singlepoint`, `nmr`, `oxidation`, `reduction` | solvation keyword for the property steps |
| `nmr_method` | string | `nmr` | e.g. `"m06 d4 def2-tzvp"` |
| `nbo_run` | string | `singlepoint`, `oxidation`, `reduction` | population analysis keywords, `"NBO NPA"` (needs `NBOEXE`/`GENEXE` in `slurm.setup`) or `""` |
| `extra_basis` | mapping | `optimization`, `frequency` | `{element: basis}` rendered as a `%basis NewGTO` block, e.g. `{Ni: def2-tzvp}` for full def2-TZVP on Ni; `{}` or absent for none |

The `singlepoint` step always runs (`HIRSHFELD CHELPG`, `%elprop` dipole and polarizability are fixed
in the template). The conformer search is not a compound step any more (the ensemble is the search).

```yaml
orca: on
dft:
  level_of_theory:
    opt_geom: true
    freq_calc: true
    nmr_calc: true
    vertical_redox: true
    opt_method: "wb97m-d4 def2-tzvp(-f)"
    opt_solvent: ""
    svp_method: "m06 d4 def2-tzvp"
    property_solvent: ""
    nmr_method: "m06 d4 def2-tzvp"
    nbo_run: "NBO NPA"
    extra_basis: {Ni: def2-tzvp}
slurm:
  orca: {time: "160:00:00", mem_per_cpu: 4G, ntasks: 16, nodes: 1}
condense: true
```

A cheaper single-point-only library:

```yaml
dft:
  level_of_theory:
    opt_geom: false
    freq_calc: false
    nmr_calc: false
    vertical_redox: false
    opt_method: ""
    opt_solvent: ""
    svp_method: "r2scan-3c"
    property_solvent: ""
    nmr_method: ""
    nbo_run: ""
    extra_basis: {}
```

## What is recorded

`metadata.json` of the molecule, `dft`: `engine`, `orca_module`, `charge`, `multiplicity`, `pruning`,
`calc_param_template`, `calc_param_md5` (of the rendered input), `nprocs`, `maxcore_mb`,
`level_of_theory` as validated, and `compound_steps`: for every step (`optimization`, `frequency`,
`singlepoint`, `nmr`, `oxidation`, `reduction`) whether it runs, its method, solvent, extra basis,
charge and multiplicity; with `dft.constrained`, `fixed_atoms` (`{"4": "Ni4", ...}`, 0-based). Per conformer: SLURM ids, host, start and end, `exit_code`,
`orca_terminated_normally`, `orca_run_time` (ORCA's `TOTAL RUN TIME` line) and `orca_run_time_s` (the same
in seconds), `wall_time_s`, `cpus` (what the SLURM allocation holds on the node: `SLURM_CPUS_ON_NODE`,
else `ntasks` x `cpus_per_task`, else `nprocs`), `cpu_hours` (`wall_time_s` x `cpus`, the allocation's
cost as in bceg's records), `orca_steps_s` (wall time of every compound step by its `%base` name, from
ORCA's own `Timings for individual modules` tables; `null` for a step that did not finish), `condensed`
(`ok` or the error).

## Cost

Every record is written for a failed or timed-out job too: it was paid for. `orcadesc cost <LIB>.yaml`
sums the records per molecule and stage (`pre_refinement_cpu_h`, `search_cpu_h`, `post_opt_cpu_h`,
`dft_cpu_h`, one `dft_<step>_cpu_h` per compound step, the `_gpu_h` counterparts, `total_cpu_h`), prints
the table with totals and, with `--csv`, writes it, one row per molecule joinable on `inchikey` to the
descriptor spreadsheet. The condensed file carries the same step timings under `job.timings`
(`total_run_time_s`, `nprocs`, `steps`), which is how the laptop side gets them: the per-conformer table
has `orca_run_time_s`, `orca_nprocs`, `orca_cpu_hours` and `orca_step_<step>_s`, the aggregated spreadsheet
`dft_cpu_hours` and `dft_<step>_cpu_hours` summed over the molecule's conformers (bookkeeping columns, never
aggregated or filtered as descriptors). Records and files written before 2026-10-02 lack these fields.

## Status

`dft` in `orcadesc status`: `off` with `orca: off`, `queued` while the prepare task or the ORCA
array wait, `running` while any conformer runs, `done` when every kept conformer is `completed`,
`failed` when one is `failed`, `timeout` or was never submitted (the `reason` column lists them;
`orcadesc submit` resubmits only those). `n_dft` is kept/total conformers and `dft_done` completed/kept.

---

## The g-xTB engine (`dft.engine: gxtb`)

[g-xTB](https://github.com/grimme-lab/g-xTB) is the Grimme group's general-purpose tight-binding
method, shipped as a build of `xtb` 6.7.1 that runs it when `--gxtb` is given. With `dft.engine: gxtb`
each kept conformer gets one `xtb` run per step instead of an ORCA compound job, and
`desc_utils.gxtb_condense` writes the same `orca_condensed.json` from the output.

Nothing on the laptop side changes: run the same spec and the same `desc_utils.run` commands. The
descriptors g-xTB cannot produce come out as all-NaN columns, which `aggregate` drops, so they are
simply absent from the spreadsheet rather than wrong. That also means a g-xTB table and a DFT table of
the same library line up column-for-column wherever both engines have data.

### What you get, and what you lose

| Descriptor family | g-xTB |
|---|---|
| morfeus geometry (Sterimol, `%Vbur`, bond lengths, angles, dihedrals) | **yes** — read from `optimization.xyz`, never from the engine |
| charges, charge differences | **yes**, g-xTB's Mulliken-type `q` (see below) |
| HOMO / LUMO | **yes** |
| dipole | **yes** |
| vertical redox (IP, EA, hardness, electronegativity, electrophilicity, Fukui) | **yes**, with `vertical_redox` |
| stretching / bending vibrations | **yes**, with `freq_calc` — but read its caveat below |
| thermochemistry (enthalpy, Gibbs) | **yes**, with `freq_calc` |
| NBO (`_hmN_`, `_lpN_`, occupancies, 2nd-order perturbation) | **no** — g-xTB has no natural-orbital analysis |
| NMR shieldings | **no** |
| polarizability | **no** (`--alpha` is silently ignored under `--gxtb`) |
| solvation | **no** — ALPB/GBSA/COSMO are not parameterised for g-xTB, so a run is gas phase |

Extras ORCA does not give, stored under the condensed file's `gxtb` section: Wiberg/Mayer bond orders,
condensed Fukui indices and shell populations.

**The charges.** g-xTB computes one set of atomic charges and has no NPA, Hirshfeld or CM5 analysis.
Those charges are written to the `nbo` and `cm5` slots of the condensed file (and to `mulliken` inside
`states.*`), because those are the slots the descriptor families read. The columns are therefore still
called `charge_NBO_Ni` and `mulliken_Ni` — convenient for comparing engines, but **a `charge_NBO_*`
column in a g-xTB library holds a Mulliken-type charge, not an NPA one**. `charges.note` and
`producer.engine` in every condensed file record this.

### Keys

| Key | Default | Meaning |
|---|---|---|
| `dft.gxtb.opt_geom` | `false` | optimise the geometry with g-xTB first (`--opt`) before the properties; otherwise the stage-4 geometry is used as is |
| `dft.gxtb.opt_level` | `normal` | `--opt` level: `crude`, `sloppy`, `loose`, `lax`, `normal`, `tight`, `vtight`, `extreme` |
| `dft.gxtb.freq_calc` | `false` | `--hess`: frequencies, IR intensities and thermochemistry (see the caveat) |
| `dft.gxtb.vertical_redox` | `true` | N±1 single points, filling `states.oxidation` / `states.reduction` |
| `dft.gxtb.fukui` | `true` | `--vfukui`: condensed Fukui indices into the `gxtb` section |
| `dft.gxtb.accuracy` | `null` | `--acc`; `null` is xtb's own default (1.0), lower is tighter |
| `dft.gxtb.etemp` | `null` | `--etemp` in K; `null` is xtb's own default (300) |
| `dft.gxtb.executable` | `xtb` | the g-xTB build of xtb: a name on PATH or an absolute path. **Not hashed** — where the binary lives is a property of the machine, not of the results |
| `slurm.gxtb` | `{time: "04:00:00", mem_per_cpu: 2G, ntasks: 8, nodes: 1}` | the job profile, the `slurm.orca` counterpart |
| `slurm.gxtb_cmd` | `dft.gxtb.executable` | overrides the binary without touching the hashed block |

`dft.constrained` works as it does for ORCA: the atoms `conformer_search.constraints` names stay where
stages 2 to 4 put them. It binds the `opt_geom` step only — a single point has no geometry to hold — and
is applied through xtb's `$fix` with the L-BFGS optimiser, because xtb's default ANCopt ignores `$fix`
without a word (the same mechanism as `post_optimization.refiner: gxtb`).

```yaml
dft:
  engine: gxtb
  gxtb:
    opt_geom: false
    vertical_redox: true
    fukui: true
slurm:
  gxtb: {time: "04:00:00", mem_per_cpu: 2G, ntasks: 8, nodes: 1}
condense: true
```

Put the g-xTB `bin/` first on PATH in `slurm.setup`, or point `dft.gxtb.executable` at it. A standard
`xtb` warns about the flag it does not know and quietly runs GFN2 instead; both the runner and the
parsers refuse such output rather than passing a GFN2 result off as g-xTB.

### `freq_calc`: expensive, and unreliable on metal complexes

xtb takes the Hessian **numerically**, and it is by far the most expensive step. Measured on a 35-atom
L-NiH2 complex, where a single point takes 0.13 s: the Hessian took **13 min 38 s** and then **failed**.

Two separate failure modes, both worth knowing before you turn the flag on:

1. **On a geometry optimised at another level** (an ORCA or UMA geometry), the displaced-geometry SCFs
   do not converge and xtb stops with `hessian_numhess: Hessian on incompletely optimized geometry!`.
   The config refuses this combination at load time: set `dft.gxtb.opt_geom: true`, or optimise the
   ensemble with g-xTB in stage 4 (`post_optimization.refiner: gxtb`).
2. **Even on a g-xTB geometry**, a metal complex can still fail: the same system, optimised with g-xTB
   first, ran 10 min 20 s and aborted with `hessian_numhess: Hessian element ... is not symmetric`.
   The numerical second derivatives were not consistent enough. Small organics are fine.

A failed frequency step therefore **does not fail the conformer**. It is recorded in the conformer's
`gxtb_failed_steps`, the vibration descriptors come out empty, and the charges, orbitals, dipole and
redox states that did come out are condensed and used as normal.

### Layout, records and status

Each step runs in its own folder, because xtb always writes `charges`, `wbo` and `g98.out` under fixed
names:

```
conf_N/
    input.xyz            the stage-4 geometry
    optimization.xyz     the geometry the properties are on (the optimised one, or a copy of input.xyz)
    gxtb/
        run.json         the resolved options and the wall time of every step
        optimization/  singlepoint/  frequency/  oxidation/  reduction/
    orca_condensed.json
```

The conformer records use the **same keys as the ORCA engine** (`orca_terminated_normally`,
`orca_run_time_s`, `orca_steps_s`, `cpus`, `cpu_hours`, `condensed`), plus `gxtb_steps` and
`gxtb_failed_steps`. So `orcadesc status` and `orcadesc cost` work unchanged, and `dft_<step>_cpu_h`
breaks the cost down by g-xTB step exactly as it does by compound step. The molecule's `dft` record
holds `engine: g-xTB`, the resolved `gxtb` block and `compound_steps` (the steps that will run).
