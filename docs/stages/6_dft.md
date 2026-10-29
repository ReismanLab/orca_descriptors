# Stage 6: `dft` (and `orca`)

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
| `dft.level_of_theory` | required | the table below; may stay unset (`{...}`) while `orca: off` | yes, but an unset one may later be set (and the rest of `dft` changed with it) |
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
