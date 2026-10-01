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
| `dft.level_of_theory` | required | the table below | yes |
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
charge and multiplicity. Per conformer: SLURM ids, host, start and end, `exit_code`,
`orca_terminated_normally`, `orca_run_time`, `wall_time_s`, `condensed` (`ok` or the error).

## Status

`dft` in `orcadesc status`: `off` with `orca: off`, `queued` while the prepare task or the ORCA
array wait, `running` while any conformer runs, `done` when every kept conformer is `completed`,
`failed` when one is `failed`, `timeout` or was never submitted (the `reason` column lists them;
`orcadesc submit` resubmits only those). `n_dft` is kept/total conformers and `dft_done` completed/kept.
