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
| `method` | `null` (off) | one of `rmsd`, `tfd`, `cregen`, `marc`, applied last to the conformers the check and the window left ([Methods](#methods)) | yes, when set, with its block |
| `rmsd`, `tfd`, `cregen`, `marc` | defaults | the settings of the chosen method; only the block of the chosen method may be given | with `method` |

Order: topology check, then energy window, then `method`. A rearranged structure or a conformer outside
the window therefore never becomes the conformer a group of duplicates collapses onto, and MARC only
clusters what the window kept.

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

### Methods

```yaml
pruning:
  energy_window_kcal_mol: 6.0
  method: rmsd            # null | rmsd | tfd | cregen | marc
  rmsd: {threshold_angstrom: 0.01}
```

A method runs inside the `prepare` task (`slurm.prepare`: 1 CPU, 2 GB, 30 min by default). The prepare
log prints how many conformers it kept and how long it took; `orca_stage.pruning_method` keeps the same
numbers. `conf_N` stays the position in the pooled ensemble whatever a method drops.

#### `rmsd`

Greedy leader clustering in ascending energy order on the symmetric heavy-atom RMSD: the lowest-energy
conformer of a group of duplicates is kept, the others are `skipped_duplicate` with `duplicate_of` and
`rmsd_to_leader_angstrom`. The same code as the descriptor-side prune below (`desc_utils.prune.dedupe`),
so the two count duplicates identically.

| Key | Default | Meaning |
|---|---|---|
| `threshold_angstrom` | `0.01` | two conformers closer than this are duplicates |
| `energy_gate_kcal_mol` | `0.2` | pairs further apart in energy are never compared. Measured safe below 0.02 A only: a larger threshold is refused unless the gate is set explicitly (`null` compares every pair) |
| `max_automorphisms` | `10000` | cap on the symmetries tried per pair; `orca_stage.pruning_method.at_automorphism_cap` says when it was reached |

Without energies on the ensemble (molassembler or LoQI with `post_optimization: null`) every conformer
has the same energy: the visit order is the ensemble's and the gate compares every pair.

#### `tfd`

The same clustering on RDKit's torsion fingerprint deviation (TFD, Schulz-Gasch et al., JCIM 2012):
the torsions of the rotatable bonds and the averaged torsions of each ring, each deviation divided by its
largest possible value, weighted by distance from the molecule's centre. It needs no superposition and
no symmetry enumeration, so a flexible ligand is compared rotor by rotor rather than through a global
alignment. It reads the stage-1 graph (`structure.mol`), so a library without a transformation record
cannot use it.

| Key | Default | Meaning |
|---|---|---|
| `threshold` | none, required | TFD (0 to 1) below which two conformers are duplicates; `skipped_duplicate` records `tfd_to_leader` |
| `use_weights` | `true` | RDKit's distance weights; `false` averages every torsion alike |
| `energy_gate_kcal_mol` | `0.2` | as for `rmsd` |

What it does not see: bond angles, the coordination geometry, and every atom without a torsion of its
own. On a Ni(N,N)H2 complex no rotatable torsion goes through Ni; the chelate ring is one averaged ring
torsion, and the orientation of the hydrides is invisible. Two conformers that differ only there are
duplicates for TFD and not for RMSD.

**Calibration, and why there is no default threshold.** On the pooled nih2 set (uma-conv geometries,
6 kcal/mol window, pairs within 0.2 kcal/mol), with the graph perceived from the geometry, TFD was
compared with the 0.01 A symmetric RMSD duplicates (2026-11-13):

| InChIKey | conformers | RMSD duplicate pairs | largest TFD of one | pairs with TFD < 0.01 | of which not RMSD duplicates |
|---|---|---|---|---|---|
| WVVLVBHMBBXGQX | 221 | 48 | 0.23 | 78 | 55 |
| NDFONMCZIFGPJP | 1002 | 16 | 0.037 | 230 | 226 |
| SOGUACKAVBIAPH | 395 | 72 | 0.068 | 1688 | 1650 |
| BCXWBFWXWGPUJW | 396 | 191 | 0.005 | 16640 | 16449 |
| CONTYZSPSXHNNJ | 2634 | 2494 | 0.27 | 6733 | 5952 |

No threshold reproduces the RMSD duplicates: RMSD duplicates reach a TFD of 0.27 (the RMSD is minimised
over the graph's symmetries, TFD's torsion atoms are fixed), while TFD below 0.01 also joins thousands of
pairs RMSD keeps apart (they differ where TFD has no torsion). The two are different notions of "the
same conformer"; `tfd` is there for that notion, with a threshold the user chooses.

#### `cregen`

CREST's CREGEN criteria, ported to Python from `src/cregen.f90` and `src/rotcompare.f90` of CREST
3.0.2 rather than calling the binary: the standalone `crest --cregen` of 3.0.2 sorts with an energy
window of 0 whatever `--ewin` or a TOML `[cregen]` block say, and keeps only the lowest conformer
(checked 2026-11-13, laptop and cluster builds). The energy window is this stage's own
`energy_window_kcal_mol`, so the port has none.

In energy order, conformer *i* is a duplicate of a lower one *j* when `E_i - E_j < ethr` and either the
RMSD is below `rthr`, or it is below `2 rthr` and the three rotational constants agree to half the
`bthr` threshold. The RMSD is CREGEN's: all atoms (or the heavy atoms), index-matched (no symmetry), the
smaller of the RMSD to *j* and to *j*'s mirror image. Among the survivors, *i* is a rotamer of *j*
(the same conformer with equivalent atoms permuted, e.g. a rotated methyl) when the energies agree within
`ethr`, the rotational constants at `bthr`, and the invariant sum of Z_a Z_b / r_ab^2 to 1e-3. The
`bthr` threshold grows with the anisotropy of the rotational constants, from `bthr` to `bthrmax`
(CREST's error-function form, centred by `bthrshift`).

| Key | Default | Meaning |
|---|---|---|
| `ethr_kcal_mol` | `0.05` | energy threshold of both passes |
| `rthr_angstrom` | `0.125` | RMSD threshold |
| `bthr` | `0.01` | relative rotational-constant threshold (1 %) for isotropic molecules |
| `bthrmax` | `0.025` | the threshold for strongly anisotropic ones |
| `bthrshift` | `0.5` | where the threshold switches between the two |
| `heavy_atoms` | `false` | RMSD on the heavy atoms only (CREST `--heavy`) |
| `mirror` | `true` | also compare with the mirror image (CREST does unless it runs an entropy mode) |
| `drop_rotamers` | `true` | `true`: one conformer per CREGEN group, as `crest_conformers.xyz`; `false`: only the duplicates are dropped |

A duplicate is `skipped_duplicate` with `duplicate_of`, `cregen: duplicate` and the RMSD; a rotamer is
`skipped_duplicate` with `cregen: rotamer`. CREGEN needs energies: an ensemble without them stops the
prepare task with an error.

#### `marc`

navicat-marc 0.2.8 (Laplaza et al., JPCL 2024; MIT): a dissimilarity matrix from RMSD, relative energy
and/or heavy-atom dihedrals, clustered, and one conformer kept per cluster. Unlike the three methods
above it does not only remove copies: it picks a representative subset, of `n_clusters` conformers or of
a number it chooses itself (gap statistic). The others are `skipped_not_representative` with
`marc_cluster` and `marc_representative`.

| Key | Default | Meaning |
|---|---|---|
| `metric` | `avg` | `rmsd`, `erel`, `da`, `ewrmsd`, `ewda`, `mix`, `avg` (MARC's `-m`) |
| `cluster` | `kmeans` | `kmeans`, `agglomerative`, `affprop` (MARC's `-c`) |
| `n_clusters` | `null` | conformers kept; `null`: MARC chooses. Not with `affprop` |
| `pick` | `lowest` | `lowest`: the lowest-energy member of each cluster (MARC's `-mine`); `central`: MARC's own most representative member |
| `include_h` | `false` | hydrogens in the RMSD and dihedrals (MARC's `-yesh`) |
| `scale_factor` | `1.10` | covalent-radius scale of MARC's bond perception (`-sf`) |

`navicat_marc` (and scikit-learn) must be installed in the environment the prepare task runs in; it is
imported only when `method: marc`. Its RMSD matrix is quadratic and slow: 46 s for 221 conformers of a
120-atom complex, so an ensemble of a few thousand needs a longer `slurm.prepare.time`. With
`n_clusters: null`, MARC 0.2.8 fails when two cluster representatives are identical; the error says to
set `n_clusters`. A metric whose dihedral or energy part has no spread over the ensemble makes MARC
produce NaN; the error says to use `rmsd` or `erel`.

## What is recorded

Per conformer in `metadata.json` (`conformers.conf_N`): `energy_hartree`, `energy_above_min_kcal_mol`,
`replicate` and `index_in_replicate` (its position in that replicate's file), and a status: `prepared` (kept, then `submitted`, `running`, `completed`, `failed` or `timeout` as the
ORCA job proceeds), `skipped_topology` (another species than the graph, with `topology`),
`skipped_energy_filter` (above the window), `skipped_no_energy` (no energy on its
comment line while a window is set), `skipped_duplicate` (`rmsd`, `tfd`, `cregen`) or
`skipped_not_representative` (`marc`). With `topology`, every conformer that has a change, allowed ones
included (an agostic contact of a kept conformer), lists them as `topology_changes`, and `orca_stage`
counts `n_skipped_topology`. The DFT record keeps the block as `dft.pruning`. The `prepare`
log says how many conformers were kept (`n_to_run`) out of how many (`n_conformers_in_ensemble`, over
`n_replicates` replicates).

## Adding a criterion

`desc_utils/submit/tasks.py`, `energy_window`, decides the topology check and the window; a method is a
function of `desc_utils/submit/pruning.py` returning `(status, info)` per conformer, registered in
`METHODS` and `DEFAULTS` there and validated in `check_params`. A new skipped status also goes into
`records.SKIPPED_CONFORMER`, and every new key into this document (`tests/test_stage_docs.py` checks).

## Status

`pruning` in `orcadesc status`: `queued` while the prepare task waits for the ensemble, `done` once
the molecule's `conf_N/` folders exist (`n_dft` is kept/total), `failed` when the ensemble's
`params.json` hash differs from the one the library was built on (`ensemble_changed`), `off` with
`orca: off`, `-` otherwise.

---

## The other prune: choosing conformers of a library that is already computed

Everything above is about which conformers get a *job*. A separate, later question is which
conformers of a finished library are **described**, and it is answered on the laptop by the
`prune` block of the descriptor spec (`desc_utils/prune.py`), not here.

They are deliberately two things. Stage 5 is hashed into the library's identity: a library built
with a 5 kcal/mol window is a different library from one built without. The descriptor-side prune
is a property of the descriptor run, so the same library can be extracted twice under two windows
and both tables are legitimate — which is the normal case when a library was computed with
`energy_window_kcal_mol: null` (every conformer) precisely so the choice could be made later.

```yaml
prune:
  enabled: true
  energy_column: uma_energy_hartree   # default: numbering.energy.column
  energy_window_kcal_mol: 6.0         # kcal/mol above each molecule's own lowest conformer
  rmsd_threshold_angstrom: 0.01       # symmetric heavy-atom RMSD
  energy_gate_kcal_mol: 0.2           # pairs further apart in energy are never compared
  max_automorphisms: 10000
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | `false`: every conformer is described, and the block is inert |
| `energy_column` | `numbering.energy.column` | the energy the window and the clustering order use. It defaults to the numbering table's own column so that the prune, `_lowE` and the Boltzmann weights cannot end up on different energies without the spec saying so |
| `energy_window_kcal_mol` | `null` | per molecule, against that molecule's lowest conformer |
| `rmsd_threshold_angstrom` | `null` | greedy leader clustering in **ascending energy order**, so the lowest-energy conformer of a group of duplicates is the one kept; the distance is the heavy-atom RMSD minimised over the automorphisms of the heavy-atom graph, so a relabelling of equivalent atoms is not a different structure |
| `energy_gate_kcal_mol` | `0.2` | pairs further apart than this in `energy_column` are not compared at all. An empirical shortcut, not a bound: measured on this project's own data, the largest energy gap of a pair below 0.02 Å is 0.04 kcal/mol. Above a threshold of 0.02 Å it is **refused** rather than extrapolated — set it explicitly, or `null` to compare every pair |
| `max_automorphisms` | `10000` | cap on the permutations tried per pair |

The window runs first, so a conformer outside it never becomes the leader of one inside it.

### What is recorded

`<date>_<name>_pruning.csv`, one row per conformer of the numbering table, kept or dropped:
`path`, the molecule id, the energy, `prune_delta_e_kcal_mol` (above that molecule's minimum),
`prune_status` (`kept`, `above_energy_window`, `duplicate`, `no_energy`), and for a duplicate the
`prune_leader` it collapsed onto with `prune_rmsd_to_leader`. A count in a provenance block that
cannot be checked afterwards is not provenance, which is why the whole table is written and not
only the survivors.

The prune runs between `number` and `extract` (`desc_utils/run.py::_apply_prune`), so only the
survivors are described and the per-conformer table holds exactly the conformers the library is
built from.

### The automorphism cap is reported, not hidden

A molecule whose heavy-atom graph has more symmetries than `max_automorphisms` is compared against
an arbitrary subset of them, so its surviving count is an upper bound and can move with the RDKit
version. The record carries `prune_n_automorphisms` per conformer and the run names every molecule
that reached the cap.

The default was 1000 until 2026-11-13 and is now 10000. On the pooled nih2 set (0.01 A, 6 kcal/mol,
0.2 kcal/mol gate) three molecules were at 1000, and the subset was far from harmless:

| InChIKey | symmetries | kept at 1000 | kept at 10000 | kept, all symmetries |
|---|---|---|---|---|
| NDFONMCZIFGPJP | 4608 | 994 / 1002 | 990 | 990 |
| CONTYZSPSXHNNJ | 2304 | 2070 / 2634 | 1859 | 1859 |
| WVVLVBHMBBXGQX | 20736 | 217 / 221 | 193 (still capped) | 183 |

The cost is the permutations tried on pairs that pass the pre-filters, so it grows with the number of
symmetries actually used: CONTYZSPSXHNNJ took 16 s at 1000 and 31 s with all 2304. The other molecules
have at most 432 symmetries and are unchanged. The project's measured ladder
(`ConfAnalysis-v1/output/rmsd_dedupe_descriptors/.../kept_counts_umaconv_only_*.csv`) was made with
the 1000 cap, so it still matches for every molecule below 1000 symmetries, not for these three.
