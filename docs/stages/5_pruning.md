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
| `topology` | `null` (off) | the connectivity check below: `true` for its defaults, or a block of `allow`, `max_metal_ligand_distance`, `agostic_max`, `new_bond_tolerance` | yes, when set (a library without it keeps its hash) |

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

### Connectivity check

```yaml
pruning:
  energy_window_kcal_mol: 5.0
  topology:
    allow: [agostic]                 # kinds that do not disqualify a conformer (default)
    max_metal_ligand_distance: 2.5   # Angstrom (default)
    agostic_max: 2.3                 # Angstrom (default)
    new_bond_tolerance: 0.4          # Angstrom added to the covalent-radii sum (default)
```

(`topology: true` is the same.) Each conformer is compared with the molecular graph of stage 1
(`structure.mol`, the transformation's product: Ni-N and Ni-H are bonds of it) by `bceg/topology.py`.
The search (CREST with `reftopo: false` in particular), the post-optimisation, or both can turn a
structure into another species; the kinds of change are:

| Kind | Meaning |
|---|---|
| `decoordinated` | a graph bond to a metal longer than `max_metal_ligand_distance` (a Ni-N dissociated, a hydride gone) |
| `bond_broken` | another graph bond longer than 1.35 x the covalent-radii sum |
| `new_bond` | two non-bonded non-metal atoms closer than their radii sum + `new_bond_tolerance` (H-H after a reductive elimination, also when H2 stays sigma-bound to Ni; C-H, C-C) |
| `agostic` | a metal...H contact shorter than `agostic_max`, the H bonded to C, Si or B in the graph |
| `new_metal_contact` | an N-H or O-H closer to the metal than `agostic_max`, or a non-bonded heavy atom closer than `max_metal_ligand_distance` (an eta2-arene, an ether O) |
| `stereo` | a stereocentre assigned in the SMILES has the other configuration in 3D |

A conformer with a kind not in `allow` is `skipped_topology`. That happens **before** the energy
window: the minimum is taken over the intact conformers only, so a rearranged structure lower in
energy cannot push the real ones out of the window. `allow: []` drops agostic conformers too. The
check is cheap (distances and graph only) and needs the graph, so a library without a transformation
record (`structure`) cannot use it.

The same check on the DFT-optimised geometry, at descriptor time, is `numbering.topology` in the spec
([../descriptor_pipeline.md](../descriptor_pipeline.md)).

## What is recorded

Per conformer in `metadata.json` (`conformers.conf_N`): `energy_hartree`, `energy_above_min_kcal_mol`,
`replicate` and `index_in_replicate` (its position in that replicate's file), and a status: `prepared` (kept, then `submitted`, `running`, `completed`, `failed` or `timeout` as the
ORCA job proceeds), `skipped_topology` (another species than the graph, with `topology`),
`skipped_energy_filter` (above the window) or `skipped_no_energy` (no energy on its
comment line while a window is set). With `topology`, every conformer that has a change, allowed ones
included (an agostic contact of a kept conformer), lists them as `topology_changes`, and `orca_stage`
counts `n_skipped_topology`. The DFT record keeps the block as `dft.pruning`. The `prepare`
log says how many conformers were kept (`n_to_run`) out of how many (`n_conformers_in_ensemble`, over
`n_replicates` replicates).

## Adding a criterion

`desc_utils/submit/tasks.py`, `energy_window`, returns `(name, conformer, energy_above_min, status,
topology_changes)` per conformer and is the only place that decides (a new skipped status also goes
into `records.SKIPPED_CONFORMER`); a new key of the `pruning` block (declared in
`PRUNING_KEYS` of `desc_utils/submit/config.py`, so it is hashed) would be read there.

## Status

`pruning` in `orcadesc status`: `queued` while the prepare task waits for the ensemble, `done` once
the molecule's `conf_N/` folders exist (`n_dft` is kept/total), `failed` when the ensemble's
`params.json` hash differs from the one the library was built on (`ensemble_changed`), `off` with
`orca: off`, `-` otherwise.
