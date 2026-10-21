# From condensed ORCA files to the descriptor tables

The laptop half of the pipeline: one **spec** per molecule family drives atom numbering, extraction with
the two engines and the Boltzmann aggregation. The short version is in the [README](../README.md); this
page is the reference. The spec sections and defaults live in `desc_utils/spec.py::DEFAULTS`.

```
ORCA compound job per conformer                                                            (cluster)
        │  optimization.xyz, *.property.json, singlepoint.json (GBW dump), calcParam.out (NBO, frequencies)
        ▼
[1] condense  ──►  <conf>/orca_condensed.json      (standard-library Python, ~0.4 to 1.5 MB per conformer)
        │  rsync optimization.xyz + orca_condensed.json                                        (laptop)
        ▼
[2] spec  ──►  [3] number  ──►  [4] extract  ──►  [5] aggregate (+ optional filter)
     <date>_<name>_atom_numbering_results.csv   <date>_<name>_combined_descriptors.csv   <date>_<name>_descriptors.csv
```

## Commands

All stages read the same spec. From the repository root, in `gp_env`:

```bash
python -m desc_utils.run validate  --spec specs/lnih2_pyox.yaml
python -m desc_utils.run condense  --spec specs/lnih2_pyox.yaml            # -> <conf>/orca_condensed.json (normally run on the cluster)
python -m desc_utils.run number    --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_atom_numbering_results.csv
python -m desc_utils.run extract   --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_combined_descriptors.csv (per conformer)
python -m desc_utils.run aggregate --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_descriptors.csv
python -m desc_utils.run all       --spec specs/lnih2_pyox.yaml --jobs 4   # number -> extract -> aggregate
```

`python -m desc_utils.run <command> --help` lists the options (`--numbering` to reuse an existing table,
`--families` to run a subset of descriptor families, `--strict` to fail on any numbering problem,
`--filter` to apply the spec's noise filter, `--overwrite-xyz-json`).

`--jobs N` spreads the per-conformer work (Open Babel and SMARTS numbering, morfeus geometry, ORCA and
DescriPyTor extraction) over N processes; `--jobs 0` takes every core, the default is one. The tables
are the same whatever N. The first run of a library is the one to parallelise: the morfeus results are
cached in `<conf>/xyz.json` (`<conf>/xyz_<stem>.json` when `data.geometry_file` is not `optimization.xyz`)
and read back on the next runs.

Output names carry the date of the run and the spec name so that nothing is overwritten from one run
to the next (`output` section of the spec).

### Re-run after changing the spec or adding conformers

| You changed | Run |
|---|---|
| Descriptor terms only | `python -m desc_utils.run extract --spec S` then `aggregate --spec S` (morfeus geometry is cached in `<conf>/xyz.json`; add `--overwrite-xyz-json` if you changed `vbur_radii` or `quadrants`) |
| Atom labels or numbering rules | `all --spec S`, or `number --spec S` then `all --spec S --numbering <csv>` after hand corrections |
| Added conformer folders on the cluster | condense on the cluster (existing files are skipped), rsync, then `all --spec S` |
| Only the aggregation temperature | `aggregate --spec S` |
| You want the noise filter (correlation / variance pruning) for a modelling set | `aggregate --spec S --filter`, or set `filter.enabled: true` in the spec; needs enough molecules for correlations to mean anything (with two molecules every column is perfectly correlated and one survives) |

### Conformers that need a look

The run never stops for a bad conformer. One that could not be numbered, or whose ORCA job is
running, failed or absent, is left out or NaN in the tables, the molecule gets
`all_conformers_completed = False` in the spreadsheet, and the conformer is listed in
`<today>_<LIB>_conformers_to_check.csv` with the reason (`numbering_status`, `numbering_notes`,
`orca_job_status`). Correcting a few atom indices by hand in the numbering table is a normal outcome;
rerun with the corrected table instead of re-deriving it:

```bash
python -m desc_utils.run all --spec specs/<LIB>.yaml --numbering <date>_<LIB>_atom_numbering_results.csv
```

## The spec

A spec has these sections:

| Section | Content |
|---|---|
| `name` | the library name, used in the output file names |
| `data` | `root` folder (`DATA/<LIB>`), `layout: molecule/conformer`, `conformer_glob`, `geometry_file` (the xyz every stage reads: numbering, morfeus, comment-line energy), `molecule_id_column`, `condensed_file` (`null` = raw ORCA files only). |
| `atoms` | Ordered map `label -> expected element` (`null` = any). Labels are the only way descriptors refer to atoms. |
| `numbering` | `backend` (`smarts`, `manual`, `first_n`, `molalign`), `index_base`, per-backend block, optional `orient` rules, `annotators`, `smiles` mode, `energy` (`source: orca`, the SCF energy of `basename`, column `SCF_energy`; or `source: xyz_comment`, the `key=` value of the extended-xyz comment line in `units`, converted to Hartree, with a mandatory `column` name that may not be `SCF_energy`). |
| `descriptors.orca` | Four families with label lists: `xyz` (atoms, bonds, angles, dihedrals, quadrants, `vbur_radii`), `charges`, `nbo` (`homo_lumo_rank_range`, `nbo_block_index`, `spin`), `redox` (Fukui `prefixes`, `nbo_block_map`). |
| `descriptors.descripytor` | DescriPyTor "Schema B" keys with label tuples: `charges`, `charge_diff`, `sterimol`, `bond_length`, `bond_angle`, `stretching_vibration`, `bending_vibration`, `dipole`, plus thresholds, `radii`, `drop_atoms`. |
| `aggregate` | `group_by`, `energy_column` (defaults to `numbering.energy.column`), `temperature_K`, `kB_hartree_per_K`, `stats`, `carry_first`. |
| `filter` | Every option of `filter_numeric_descriptors`, plus `metal_label` (the metal label, or `null` for organic molecules). |
| `output` | File names of the three stage outputs and of `check_csv`, the list of conformers to look at; `{date}` and `{name}` are filled in at run time. |

`validate` reports every problem at once (unknown labels, wrong arity, SMARTS that do not parse, roles
outside the SMARTS, labels never assigned). Quote every SMARTS string: `#` starts a YAML comment.

### Adding a new molecule family

Copy the closest shipped spec (`specs/lnih2_pyox.yaml` for metal complexes, `specs/test_molecules_at1_at2.yaml`
for small organics), then edit `atoms`, `numbering` (SMARTS stages, or `manual` with your CSV, or `first_n`),
the label lists under `descriptors` (delete the terms you do not need), and `filter.metal_label`. Run
`validate`, then `number` and check the `numbering_status` column before `extract`.

### Geometry-only descriptors from UMA structures

The morfeus family (`descriptors.orca.xyz`: SASA, %Vbur, bond lengths, angles, dihedrals, Sterimol,
pyramidalisation, quadrants) needs nothing but an xyz file, so a library pre-optimised with UMA can be
described before, or without, any ORCA job and without condensing (no `orca_condensed.json`). In a library
prepared by `orcadesc` the UMA geometry of every conformer is `conf_N/input.xyz`, whose comment line carries
`energy=<Hartree>`: add `--include='input.xyz'` to the rsync, set `data.geometry_file: input.xyz`, keep
`descriptors.orca.xyz` as the only family, and take the energy for the Boltzmann weighting from that
comment line:

```yaml
numbering:
  energy: {source: xyz_comment, key: energy, units: hartree, column: uma_energy_hartree}
```

`specs/test_molecules_uma_only.yaml` is the template; it reads the extended xyz of the previous workflow
(`optimized_uma-s-1p1.xyz`, `units: eV`), the only difference being the file name and the units.

`column` is mandatory here and may not be `SCF_energy`: the value is a UMA energy converted to Hartree, and
the table must say so. The other families (`charges`, `nbo`, `redox`, `descripytor`) read ORCA output and
cannot run on an xyz; when none of them is requested the tables carry no `orca_job_status`, no
`all_conformers_completed` flag, and nothing is listed as "ORCA job missing".

## Atom numbering

`number` writes one row per conformer folder with the 1-based index of every label, plus `path`,
`inchikey`, `SCF_energy`, `smiles`, `smiles_noH` (implicit hydrogens),
annotator columns (e.g. `ligand_class`), `numbering_backend`,
`numbering_status` (`ok`, `incomplete`, `failed`, `element_mismatch`, `inconsistent`) and
`numbering_notes`.

Where the two SMILES columns come from is `numbering.smiles.source`:

| `source` | `smiles` / `smiles_noH` |
|---|---|
| `geometry` (default) | perceived from the xyz (Open Babel bond orders), all hydrogens explicit / removed; with `mode: strip_atoms` the `strip` labels are removed first and the ligand fragment is kept |
| `registry` | the SMILES the molecule was submitted with, from `<data.root>/molecule_registry.json` (written by `orcadesc submit`) / its RDKit canonical form |
| `file` | the same, from the SMILES list `numbering.smiles.file` (`.csv` with a `smiles` column, or one SMILES per line), matched to the folders by InChIKey |

Use `registry` (or `file`) for metal complexes: bond orders perceived from a complex's geometry are
unreliable. An N bound to the metal is taken for an amine, so after `strip_atoms` an L-NiH2 pyox ligand
comes out as a dihydro tautomer with two N–H, and its InChIKey no longer matches the folder. Atom
numbering is not affected: its SMARTS use `~` (any bond). A molecule missing from the source gets no
SMILES, and a note in `numbering_notes`. The table is a plain CSV so it can be inspected or corrected by hand; rows whose
status is not `ok` are skipped downstream.

Backends:

- `smarts`: ordered stages, each with one or several SMARTS alternatives and a `roles: {label: match position}`
  map; later stages override earlier labels (used to orient PyOx ligands). Conformers are converted
  `xyz -> sdf` with Open Babel and matched with RDKit.
- `manual`: read the indices from an existing CSV (`csv`, `key_column`).
- `first_n`: label the first N atoms in file order (small test molecules).
- `molalign`: reserved for alignment-based numbering, not implemented yet.

## Extraction

`extract` resolves every labelled term of the spec to atom indices per conformer and calls the two
engines. Column names embed the labels, never the indices (`%Vbur_atom_Ni_radius_3.5`, `B1_Ni-N1`,
`charge_NBO_C1`, ...):

- ORCA side (`desc_utils/extract/orca.py`): `xyz` (morfeus on `optimization.xyz`, cached in
  `<conf>/xyz.json` and recomputed when the spec asks for columns the cache lacks), `charges`
  (`singlepoint.property.json`, `nmr.property.json`), `nbo` (NBO 7 block of `calcParam.out`),
  `redox` (`singlepoint`/`oxidation`/`reduction`); all of them from `orca_condensed.json` when it exists.
- DescriPyTor side (`desc_utils/extract/descripytor.py`): one `Molecules` object per conformer, loaded
  from `orca_condensed.json` (`input.kind: json`),
  one call per family; the `FAMILIES` table maps DescriPyTor's index-based column names to label-based
  names. `python -m desc_utils.run to-schema-b --spec ... --row 0` prints the index-based JSON that
  DescriPyTor's own tools consume for one conformer.

The result is `<date>_<name>_combined_descriptors.csv`, one row per conformer, merged on `path`.

## Aggregation and filtering

`aggregate` groups conformers by molecule (`group_by: inchikey`) and writes, for every numeric descriptor,
`<name>_min`, `<name>_max`, `<name>_lowE` (lowest-energy conformer) and `<name>_boltz`, the Boltzmann
average at `temperature_K` with weights `exp(-ΔE / kB T)` computed from `energy_column` (by default the
column of `numbering.energy`: the SCF energy of the `singlepoint` step, no thermal correction, or the
comment-line energy of the geometry file for a geometry-only run; always Hartree). `n_conformers`, the `carry_first` columns,
the completeness columns and the cost columns (`dft_cpu_hours`, `dft_<step>_cpu_hours`: the per-conformer
`orca_*` cost columns summed, never aggregated as descriptors) are kept. This complete table is the default output. Optionally, with `filter.enabled: true` in the spec or
`aggregate --filter`, `filter_numeric_descriptors` then removes: non-numeric and all-NaN columns, NBO orbitals whose
Boltzmann occupancy is below `occupancy_boltz_threshold`, Rydberg orbitals (except on `metal_label`),
metal 4f Rydbergs, core orbitals, anisotropic NMR terms, NHO φ/θ angles, near-constant columns
(`ratio_threshold`) and one member of every pair above `corr_threshold`. The output is
`<date>_<name>_descriptors.csv` with `inchikey_folder`, `inchikey`, `smiles`, `smiles_noH`, `ligand_class` in front.

## What you get

| File | One row per | Content |
|---|---|---|
| `<date>_<name>_atom_numbering_results.csv` | conformer | 1-based index of every label, SCF energy, SMILES, `numbering_status` |
| `<date>_<name>_combined_descriptors.csv` | conformer | `orca_job_status` (`completed`, `running`, `failed`; `unknown` for files condensed before 2026-09-23, `missing` for a folder with no file), `n_conformer_folders` of its molecule, what its ORCA job cost (`orca_run_time_s`, `orca_nprocs`, `orca_cpu_hours`, `orca_step_<step>_s` per compound step; empty for files condensed before 2026-10-02), then every descriptor of both engines, columns named by label (`charge_NBO_Ni`, `%Vbur_atom_Ni_radius_3.5`, `B1_Ni-N1`, ...) |
| `<date>_<name>_descriptors.csv` | molecule | the spreadsheet: `inchikey`, `smiles`, `smiles_noH` (implicit hydrogens), `n_conformers` (rows aggregated), `n_conformers_completed`, `n_conformer_folders` (folders on disk), `all_conformers_completed` (`True` / `False`; empty when a status is `unknown`), `dft_cpu_hours` and `dft_<step>_cpu_hours` (core-hours of the molecule's ORCA jobs, summed; the ensemble's cost is in `orcadesc cost --csv`, joinable on `inchikey`), then `<descriptor>_min`, `_max`, `_lowE` (lowest-energy conformer) and `_boltz` (Boltzmann average at `temperature_K`); every descriptor is kept unless `--filter` is given |
| `<date>_<name>_conformers_to_check.csv` | conformer needing a look | only written when there is one (today's copy is removed otherwise): `path`, `reason`, `numbering_status`, `numbering_notes`, `orca_job_status` |

Conventions worth knowing when reading the spreadsheet: energies are in Hartree, HOMO/LUMO in eV,
IP and EA are positive for bound electrons (`IP = E(cation) - E(neutral)`), Fukui functions follow
`f+ = q(N) - q(N+1)`, `cm5_*` equals `hirshfeld_*` (the CM5 correction is not implemented), and
DescriPyTor Sterimol values (`B1_*`, `B5_*`, `L_*`) differ from the morfeus ones (`Sterimol_*`).
The full list of columns is in [descriptors.md](descriptors.md).

## Tests

```bash
pytest tests -m "not slow"     # unit tests: descriptors, bceg (tests/bceg), the ORCA stage (tests/test_submit.py); seconds
pytest tests -m slow           # regression: reproduces tests/reference/atom_numbering_results.csv, the DescriPyTor
                               # per-conformer reference and the reference aggregated CSVs (minutes)
```

Three bceg tests need `xtb` on `PATH` (start-geometry pre-optimisation) or the hydride SMILES of a newer
RDKit than `gp_env`'s; they are skipped on a laptop without xtb and run in the cluster environment.

The regression tests and the condensed-file round trip need the LNiH2 library under
`DATA/LNiH2_library_001_to_download_2026_03_24/` (raw ORCA folders for the round trip) and the
submodule; they are skipped otherwise.
