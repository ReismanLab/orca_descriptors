# Can we build the descriptor database with the latest DescriPyTor?

Assessed 2026-09-10 on branch `code-review`. **Update 2026-09-10 (later):** the submodule swap is done and the full-library comparison is in the last section, against upstream
[Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor) at commit `f6bc24c` (2026-08-31).

## Short answer

Yes. The upstream `Molecules` API the notebook uses is unchanged, it runs in the existing
`gp_env` environment without touching a single installed package, and on our existing
`.feather` files it reproduces charges, bond lengths, angles, vibrations and dipoles
exactly. Two descriptor families change value on purpose (see "Behaviour changes").

The bigger opportunity is that upstream now reads a **structured JSON** molecule file
natively, so the `orca2gaussian` fake-Gaussian-log step (source of findings C3 to C6 in
`CODE_REVIEW.md`) can be replaced by a direct ORCA to JSON converter.

## What the vendored copy is

`DescriPyTor/MolFeatures/` is a byte-for-byte copy of upstream commit `e7765f6`
(2025-11-01) plus one local fix in `feather_extractor.py` (`os.chdir(feather_dir)` instead
of the hard-coded `os.chdir('feather_files')`). Nothing else was modified locally, so
there is no local logic to port.

## What changed upstream since then (27 commits, June to August 2026)

| Area | Change | Impact on us |
|------|--------|--------------|
| Layout | `MolFeatures/` removed; packages now live at repo root (`M2_data_extractor`, `utils`, `M3_modeler`, `descripytor`) and install via `pyproject.toml` | Import paths change, see below |
| Input | `Molecules` loads `*.feather` **and** `*.json` (file must contain `"molecule"` and `"atoms"`); loader is `utils.help_functions.json_file_handler` | Lets us skip the fake Gaussian log |
| Connectivity | `extract_connectivity` auto-detects metals with a 2.8 Å metal threshold and Alvarez 2013 van der Waals radii (Ni 1.97 Å; vendored used 2.00 Å) | Ni now bonded to N/C donors; Sterimol on Ni changes |
| Sterimol | B1 row selection is now `argmin(B1)`; the old code matched `avs` against `b1s` and could pick the wrong row for the B1-B5 angle | `B1-B5_angle_*` columns in existing CSVs are unreliable |
| Dipole | `get_dipole_dict(..., center_atoms=None)` new optional argument | Backward compatible |
| Removed | NPA descriptor family | Not used by the notebook |
| New | `MetalComplex` / `MetalComplexEnsemble` (XYZ + xTB metal-complex descriptors, %Vbur, Boltzmann weighting) | Optional, relevant to the LNiH2 library |
| Perf | `Molecules` caches loaded folders (keyed on file mtimes and sizes) and loads in threads | Faster notebook loops |

The `json_file_handler` docstring mentions a "cclib hybrid converter" that produces the JSON.
That converter is **not** shipped upstream; only the reader is. We would write the producer.

## Verification performed

Script: `review/compare_descripytor_versions.py`. Run in `gp_env` with upstream cloned
to a scratch folder, on two existing feathers:

| Descriptor | Ni complex (IJHJNYMO.../conf_1) | Organic (NYVSDPNX.../conf_1) |
|-----------|-------------------------------|------------------------------|
| NBO charges | identical | identical |
| Bond lengths, angle | identical | identical |
| Stretch / bend frequency | identical | identical |
| Dipole (x, y, z, total) | identical | identical |
| Sterimol L | identical | identical |
| Sterimol B1 / B5 | 1.919 / 1.920 → 1.969 / 1.970 (Ni radius, metal bonds) | identical |
| B1-B5 angle | 34.5 → 0.0 | 180.0 → 59.8 |
| Ni neighbours found | [1, 3] → [1, 3, 4, 5, 12] | n/a |

Environment: `gp_env` has numpy 2.3.4, pandas 2.3.3, rdkit 2025.09, morfeus 0.7.2. Upstream
imported and ran with these versions. Only `chardet` and `flask-cors` are missing, and both
are only needed by the GUI.

## Behaviour changes to be aware of

1. **Sterimol on metal-bonded pairs** shifts by about 0.05 Å because of the Ni radius and
   because the Ni-N/C bonds are now part of the substructure. This is a correction, not a
   regression, but any model trained on the old `B1_*`/`B5_*` columns will not transfer.
2. **B1-B5 angle** was computed from the wrong row in the vendored version. Treat existing
   `B1-B5_angle_*` columns as invalid.
3. Stretch amplitudes and bend cross-magnitudes are `0.0` in both versions. That is the
   transposed normal-mode matrix from `orca2gaussian` (`CODE_REVIEW.md` C5), not a
   DescriPyTor issue, and is another reason to move to the JSON path.

## Installation caveats

- `pip install <upstream>` pins `numpy<2` and would **downgrade numpy to 1.26.4** in
  `gp_env`. The pin exists for their Docker build; our numpy 2 stack works. Recommended:
  do not pip install into `gp_env`. Either add upstream as a git submodule and put its
  root on `sys.path` (mirrors the current notebook pattern), or install with
  `pip install --no-deps`.
- Upstream installs a top-level package literally named `utils`. That will shadow or be
  shadowed by any other `utils` package in the environment. Another reason to prefer the
  submodule + `sys.path` approach over a site-packages install.
- cclib 1.8.1 (in `gp_env`) returns nothing for our `calcParam.out` files (ORCA 6.1
  compound jobs), so cclib is not a shortcut for the converter.

## Proposed migration

1. Replace the `DescriPyTor/` snapshot with a git submodule pinned to upstream `f6bc24c`.
2. In `make_descriptor_descripytor.ipynb`, replace the four `sys.path.append(...)` lines
   and the bare imports with:
   ```python
   sys.path.insert(0, str(Path("DescriPyTor").resolve()))
   from M2_data_extractor.data_extractor import Molecules
   from M2_data_extractor.feather_extractor import logs_to_feather
   ```
   Everything downstream of `Molecules(...)` is unchanged.
3. Write `orca2json.py` that emits the upstream JSON schema per conformer folder from
   `singlepoint.property.json` (coordinates in Å, Hirshfeld and NBO charges, dipole,
   polarizability, energy) and from `calcParam.out` (frequencies, IR intensities, normal
   modes in per-mode layout). Schema keys expected by the reader:
   `molecule.atoms[{atom,x,y,z}]`, `properties.dipole{x,y,z,total}`,
   `properties.polarizability{iso,aniso}`, `properties.energies.final_single_point_energy_hartree_orca`,
   `vibrations.modes[{frequency_cm-1, ir_intensity_km_per_mol, displacements[{dx,dy,dz}]}]`,
   `charges.{nbo,hirshfeld,cm5}`. This retires `orca2gaussian` and fixes C3 to C6.
4. Regenerate the descriptor CSVs and re-check `B1_*`, `B5_*`, `B1-B5_angle_*`, and all
   vibration amplitude columns against the old files before retraining.

## Full-library comparison after the submodule swap

`DescriPyTor/` is now a git submodule pinned to `f6bc24c`. The notebook import cell was
updated to the flattened layout. `review/extract_library_descriptors.py` re-runs the
notebook's `Molecules` loop (same atom, bond, angle lists; numbering from
`atom_numbering_results.csv`) over all 308 conformers of the LNiH2 library, and
`review/compare_library_descriptors.py` diffs the two runs and, after the notebook's
Boltzmann aggregation, diffs each against `20260423_LNiH2_library_001_to_download_2026_03_24_descriptors.csv`.
Raw outputs are in `review/results/`.

**Sanity check.** The vendored re-run reproduces the 20260423 reference on all 141 shared
columns to better than 1e-8. The harness is faithful to the notebook.

**Upstream vs vendored, per conformer (308 rows):**

| Family | Columns | Values changed | Largest change | Cause |
|--------|---------|----------------|----------------|-------|
| NBO charges, charge differences | 17 | 0 % | 0 | none |
| Bond lengths, angles | 17 | 0 % | 0 | none |
| Dipole | 4 | 0 % | 0 | none |
| Bend vibrations | 6 | 0 % | 0 | none |
| Stretch vibrations | 20 | 25 % | NaN → value | `vib_freq_Ni-N1`, `vib_freq_Ni-N2` (and amplitudes) were NaN for all 308 conformers in the old version because Ni-N was not a recognised bond. Upstream returns values. |
| Sterimol L | 10 | 2.5 % | 1.55 Å | metal connectivity (below) |
| Sterimol B1 | 10 | 89 % | 3.0 Å | metal connectivity + Ni radius |
| Sterimol B5 | 10 | 87 % | 7.5 Å | metal connectivity |
| Sterimol loc_B5 | 10 | 26 % | 14 Å | metal connectivity |
| B1-B5 angle | 10 | 100 % | 177° | row-selection fix |

**Root cause of the Sterimol shift, verified by experiment.** Upstream now bonds Ni to its
donor atoms (2.8 Å metal threshold). With `sub_structure=True`, the fragment behind a
bond such as C1→N1 is found by walking the connectivity graph, and that walk now crosses
Ni into the whole complex. Re-running upstream on one conformer with metal bonding
disabled (`extract_connectivity(..., metals=[])`) reproduces the vendored numbers exactly
(B1 1.699, B5 4.312, L 4.606 for C1→N1), while the default gives B1 1.878, B5 8.454.
So for bonds pointing toward the metal, the old columns described the ligand arm only and
the new columns describe the entire complex. Bonds pointing away from Ni (N1→C1, N2→C2)
change B5 by nothing and B1 by about 0.1 Å (Ni radius only). Ni-H and Ni-N pairs change
by about 0.05 Å (Ni radius 2.00 → 1.97 Å).

**B1-B5 angle.** Vendored values cluster at a handful of numbers (35° appears in 616 of
3080 cells), consistent with the wrong row being picked. Upstream values are spread
continuously between 0° and 86°. The old columns were not meaningful.

**Impact on the 20260423 reference table.** After aggregation, 0 of the 62 NBO, geometry,
dipole and vibration columns change; all 102 Sterimol and B1-B5-angle columns change.
Any model that used a Sterimol column needs retraining on regenerated descriptors.

**Decision needed.** Whether "ligand arm only" or "whole complex" is the intended Sterimol
fragment for the C→N bonds is a chemistry choice. If the ligand-only behaviour is wanted,
pass `metals=[]` through `Molecules`/`Molecule` (upstream exposes the argument on
`extract_connectivity` but not yet on the class constructors, so a one-line wrapper or an
upstream PR is needed). Either way the Ni-N vibrations and the B1-B5 angle fix are
straightforward gains.

## Update 2026-09-17: spec-driven platform

The numbering and extraction code was reorganised (see `README.md`, `specs/`, `desc_utils/run.py`).
`review/extract_library_descriptors.py` was retired: its role is now played by
`tests/test_regression.py::test_descripytor_block_reproduces_reference`, which reproduces
`review/results/library_per_conformer_upstream_f6bc24c.csv` from `specs/lnih2_pyox.yaml` to 1e-8.
One DescriPyTor quirk found on the way: passing `drop_atoms=[]` to `get_sterimol_dict` changes B1 and
the B1-B5 angle compared with the default `None` (their own `get_molecules_features_set` passes `[]`);
the adapter only forwards the option when atoms are listed.
