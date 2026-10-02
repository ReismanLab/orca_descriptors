# Stage 5: `pruning`

Which conformers of the ensemble get an ORCA job. Run by the `prepare` task of the ORCA stage, once
per molecule, after bceg: it reads `raw_<i>_<lot>.xyz` (or `raw_<i>.xyz` without a post-optimisation)
for every replicate `i` of the search ([3_conformer_search.md](3_conformer_search.md), `replicates`,
default 1) and pools them in order into one ensemble, decides per conformer, writes the kept ones to
`<LIB>/<InChIKey>/conf_N/input.xyz` (N = position in the pooled ensemble, 1-based) and records every
conformer, kept or not, in `metadata.json` with its energy, its energy above the pooled minimum and
the replicate it came from.

## Keys

| Key | Default | Meaning | Hashed |
|---|---|---|---|
| `energy_window_kcal_mol` | `null` | `null`: every conformer goes to ORCA; a number: only the conformers within that many kcal/mol of the lowest energy of the file | yes |

The energies are those on the comment lines of the ensemble file: the post-optimisation's (stage 4),
or the search's without one. A window with an ensemble that carries no energies (molassembler or LoQI
with `post_optimization: null`) sends nothing to ORCA; keep `null` in that case.

## Options

### Everything (default)

```yaml
pruning: {energy_window_kcal_mol: null}
```

or no `pruning` block at all.

### Energy window

```yaml
pruning:
  energy_window_kcal_mol: 5.0
```

CREST's own `cregen.ewin` (stage 3, `toml_extra`) already bounds what the search keeps; this window
is applied after the post-optimisation, on better energies, so the two are complementary.

## What is recorded

Per conformer in `metadata.json` (`conformers.conf_N`): `energy_hartree`, `energy_above_min_kcal_mol`,
`replicate` and `index_in_replicate` (its position in that replicate's file), and a status: `prepared` (kept, then `submitted`, `running`, `completed`, `failed` or `timeout` as the
ORCA job proceeds), `skipped_energy_filter` (above the window) or `skipped_no_energy` (no energy on its
comment line while a window is set). The DFT record keeps the block as `dft.pruning`. The `prepare`
log says how many conformers were kept (`n_to_run`) out of how many (`n_conformers_in_ensemble`, over
`n_replicates` replicates).

## Adding a criterion

`desc_utils/submit/tasks.py`, `energy_window`, returns `(name, conformer, energy_above_min, status)`
per conformer and is the only place that decides; a new key of the `pruning` block (declared in
`PRUNING_KEYS` of `desc_utils/submit/config.py`, so it is hashed) would be read there.

## Status

`pruning` in `orcadesc status`: `queued` while the prepare task waits for the ensemble, `done` once
the molecule's `conf_N/` folders exist (`n_dft` is kept/total), `failed` when the ensemble's
`params.json` hash differs from the one the library was built on (`ensemble_changed`), `off` with
`orca: off`, `-` otherwise.
