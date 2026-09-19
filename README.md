# orca_descriptors

Compile a per-molecule descriptor table from ORCA calculations of conformer ensembles.

The chemistry is not hard-coded. A **spec** (one YAML file per molecule family) declares which atoms
are labelled, how they are found in every conformer, which descriptors are computed on which labels,
and how the conformers are aggregated and filtered. The same code then serves any molecule family.
Two engines compute the descriptors: the in-house ORCA readers (`desc_utils`) and
[DescriPyTor](https://github.com/Milo-group/DescriPyTor) (git submodule).

## Contents

- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Pipeline: from ORCA files to the final descriptors](#pipeline-from-orca-files-to-the-final-descriptors)
  - [0. ORCA calculations](#0-orca-calculations)
  - [1. Spec](#1-spec)
  - [2. Atom numbering](#2-atom-numbering)
  - [3. Preprocessing for DescriPyTor](#3-preprocessing-for-descripytor)
  - [4. Extraction](#4-extraction)
  - [5. Aggregation and filtering](#5-aggregation-and-filtering)
  - [Adding a new molecule family](#adding-a-new-molecule-family)
  - [Tests](#tests)
- [Descriptors provided](#descriptors-provided)
  - [ORCA descriptors](#orca-descriptors)
  - [DescriPyTor descriptors](#descripytor-descriptors)
- [References](#references)

## Installation

```bash
git clone https://github.com/ReismanLab/orca_descriptors
cd orca_descriptors
git submodule update --init          # DescriPyTor, pinned commit
conda env create -f gp_env.yml       # environment "gp_env"
conda activate gp_env
```

DescriPyTor is imported from the `DescriPyTor/` folder; do **not** `pip install` it into `gp_env`
(its `numpy<2` pin would downgrade the environment). The submodule pin is the version the regression
tests were validated against (`f6bc24c`, 2026-08-31).

The ORCA jobs themselves need ORCA 6.1 and NBO 7 on the cluster (see
[scripts-for-orca-computations](scripts-for-orca-computations)).

## Repository layout

| Path | Content |
|---|---|
| [DATA](DATA) | Calculation outputs: one folder per molecule (InChIKey), one sub-folder per conformer (`conf_*`). |
| [specs](specs) | One YAML spec per molecule family. `lnih2_pyox.yaml` (LNiH2 / PyOx ligand library) and `test_molecules_at1_at2.yaml` (small organic test set) are the shipped examples. |
| [desc_utils](desc_utils) | The pipeline: `spec.py` (spec loading and validation), `numbering/` (atom numbering backends), `extract/` (ORCA and DescriPyTor adapters), `aggregate.py` (Boltzmann aggregation, filtering), `run.py` (CLI), `orca_descriptors.py` and `data_extraction.py` (ORCA readers and morfeus geometry). |
| [DescriPyTor](DescriPyTor) | Git submodule, [Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor). |
| [orca2gaussian](orca2gaussian) | Converter writing a pseudo-Gaussian log per conformer so DescriPyTor can read ORCA results. To be replaced by a direct ORCA to DescriPyTor-JSON converter. |
| [scripts-for-orca-computations](scripts-for-orca-computations) | ORCA compound input (`calcParam.inp`), SLURM script and launcher used on the cluster. |
| [tests](tests) | Unit tests and regression tests against the committed descriptor CSVs. |
| [review](review) | Notes and scripts from the DescriPyTor upgrade (old vs new Sterimol comparison, per-conformer reference CSVs). |
| `make_descriptor_descripytor.ipynb` | Notebook running the same pipeline interactively. |
| `atom_numbering_results.csv`, `combined_descriptors.csv`, `<date>_<name>_descriptors.csv` | Outputs of the three stages for the LNiH2 library (regression references). |

## Pipeline: from ORCA files to the final descriptors

```
ORCA compound job per conformer            (cluster)
        │  optimization.xyz, singlepoint/oxidation/reduction/nmr .property.json, calcParam.out (NBO)
        ▼
[1] spec  ──►  [2] number  ──►  [3] orca2gaussian + logs_to_feather  ──►  [4] extract  ──►  [5] aggregate + filter
                atom_numbering_results.csv   conf_N.log, conf_N.feather        combined_descriptors.csv   <date>_<name>_descriptors.csv
```

All stages read the same spec. From the repository root, in `gp_env`:

```bash
python -m desc_utils.run validate  --spec specs/lnih2_pyox.yaml
python -m desc_utils.run number    --spec specs/lnih2_pyox.yaml            # -> atom_numbering_results.csv
python -m desc_utils.run extract   --spec specs/lnih2_pyox.yaml            # -> combined_descriptors.csv (per conformer)
python -m desc_utils.run aggregate --spec specs/lnih2_pyox.yaml            # -> <date>_<name>_descriptors.csv
python -m desc_utils.run all       --spec specs/lnih2_pyox.yaml            # number -> extract -> aggregate
```

`python -m desc_utils.run <command> --help` lists the options (`--numbering` to reuse an existing table,
`--families` to run a subset of descriptor families, `--strict` to fail on any numbering problem,
`--no-filter`, `--overwrite-xyz-json`).

### 0. ORCA calculations

Every conformer is computed with the compound job
[scripts-for-orca-computations/calcParam.inp](scripts-for-orca-computations/calcParam.inp)
(ORCA 6.1, `launch_orca_all.py` creates one `conf_N/` folder per conformer of an xyz file and submits
`slurm-calcParam.sh`). The steps and the files read downstream are:

| Step (`%base`) | Level of theory | Purpose | Files used by the pipeline |
|---|---|---|---|
| `conf_search` (optional) | GOAT / xTB | conformer search | – |
| `optimization` | ωB97M-D4 / def2-TZVP(-f), def2-TZVP on Ni | geometry | `optimization.xyz` (geometry for morfeus, numbering) |
| `frequency` | same as optimization | harmonic frequencies, thermochemistry | `calcParam.out` (frequencies, normal modes for DescriPyTor) |
| `singlepoint` | M06-D4 / def2-TZVP, `NBO NPA`, `HIRSHFELD CHELPG`, `%elprop` (dipole, polarizability) | electronic descriptors | `singlepoint.property.json` (energies, charges, MOs, dipole), `calcParam.out` (NBO 7 block 1) |
| `nmr` | M06-D4 / def2-TZVP, GIAO | shieldings | `nmr.property.json` |
| `oxidation`, `reduction` | as `singlepoint`, vertical (N±1 electrons) on the optimized geometry | redox and Fukui descriptors | `oxidation.property.json`, `reduction.property.json`, `calcParam.out` (NBO blocks 2 and 3) |

Solvent is empty by default; NBO requires the `NBOEXE`/`GENEXE` environment variables.

### 1. Spec

A spec has these sections (defaults in `desc_utils/spec.py::DEFAULTS`):

| Section | Content |
|---|---|
| `data` | `root` folder, `layout: molecule/conformer`, `conformer_glob`, `geometry_file`, `molecule_id_column`. |
| `atoms` | Ordered map `label -> expected element` (`null` = any). Labels are the only way descriptors refer to atoms. |
| `numbering` | `backend` (`smarts`, `manual`, `first_n`, `molalign`), `index_base`, per-backend block, optional `orient` rules, `annotators`, `smiles` mode, `energy` source. |
| `descriptors.orca` | Four families with label lists: `xyz` (atoms, bonds, angles, dihedrals, quadrants, `vbur_radii`), `charges`, `nbo` (`homo_lumo_rank_range`), `redox` (Fukui `prefixes`, `nbo_block_map`). |
| `descriptors.descripytor` | DescriPyTor "Schema B" keys with label tuples: `charges`, `charge_diff`, `sterimol`, `bond_length`, `bond_angle`, `stretching_vibration`, `bending_vibration`, `dipole`, plus thresholds, `radii`, `drop_atoms`. |
| `aggregate` | `group_by`, `energy_column`, `temperature_K`, `kB_hartree_per_K`, `stats`, `carry_first`. |
| `filter` | Every option of `filter_numeric_descriptors`, plus `metal_label`. |
| `output` | File names of the three stage outputs. |

`validate` reports every problem at once (unknown labels, wrong arity, SMARTS that do not parse, roles
outside the SMARTS, labels never assigned). Quote every SMARTS string: `#` starts a YAML comment.

### 2. Atom numbering

`number` writes one row per conformer folder with the 1-based index of every label, plus `path`,
`inchi`, `SCF_energy`, `smiles`, annotator columns (e.g. `ligand_class`), `numbering_backend`,
`numbering_status` (`ok`, `incomplete`, `failed`, `element_mismatch`, `inconsistent`) and
`numbering_notes`. The table is a plain CSV so it can be inspected or corrected by hand; rows whose
status is not `ok` are skipped downstream.

Backends:

- `smarts`: ordered stages, each with one or several SMARTS alternatives and a `roles: {label: match position}`
  map; later stages override earlier labels (used to orient PyOx ligands). Conformers are converted
  `xyz -> sdf` with Open Babel and matched with RDKit.
- `manual`: read the indices from an existing CSV (`csv`, `key_column`).
- `first_n`: label the first N atoms in file order (small test molecules).
- `molalign`: reserved for alignment-based numbering, not implemented yet.

### 3. Preprocessing for DescriPyTor

DescriPyTor reads Gaussian logs. `orca2gaussian/orca_2_gauss.py --orca-folder DATA/<library>` writes a
pseudo-Gaussian `conf_N.log` per conformer from the ORCA outputs, and DescriPyTor's `logs_to_feather`
turns each log into the `conf_N.feather` file that `Molecules` loads. The notebook has a labelled
preprocessing section for this; `extract` reports the exact command for any conformer whose feather is
missing. This step will disappear once the direct ORCA to DescriPyTor-JSON converter lands
(`descriptors.descripytor.input.kind: json` is reserved for it). cclib cannot read ORCA 6.1 compound
outputs, so the converter parses the `.property.json` files and `calcParam.out` directly.

### 4. Extraction

`extract` resolves every labelled term of the spec to atom indices per conformer and calls the two
engines. Column names embed the labels, never the indices (`%Vbur_atom_Ni_radius_3.5`, `B1_Ni-N1`,
`charge_NBO_C1`, ...):

- ORCA side (`desc_utils/extract/orca.py`): `xyz` (morfeus on `optimization.xyz`, cached in
  `<conf>/xyz.json` and recomputed when the spec asks for columns the cache lacks), `charges`
  (`singlepoint.property.json`, `nmr.property.json`), `nbo` (NBO 7 block of `calcParam.out`),
  `redox` (`singlepoint`/`oxidation`/`reduction`).
- DescriPyTor side (`desc_utils/extract/descripytor.py`): one `Molecules` object per conformer feather,
  one call per family; the `FAMILIES` table maps DescriPyTor's index-based column names to label-based
  names. `python -m desc_utils.run to-schema-b --spec ... --row 0` prints the index-based JSON that
  DescriPyTor's own tools consume for one conformer.

The result is `combined_descriptors.csv`, one row per conformer, merged on `path`.

### 5. Aggregation and filtering

`aggregate` groups conformers by molecule (`group_by: inchi`) and writes, for every numeric descriptor,
`<name>_min`, `<name>_max`, `<name>_lowE` (lowest-energy conformer) and `<name>_boltz`, the Boltzmann
average at `temperature_K` with weights `exp(-ΔE / kB T)` computed from `energy_column` (the SCF energy
of the `singlepoint` step, no thermal correction). `n_conformers` and the `carry_first` columns are
kept. Then `filter_numeric_descriptors` removes: non-numeric and all-NaN columns, NBO orbitals whose
Boltzmann occupancy is below `occupancy_boltz_threshold`, Rydberg orbitals (except on `metal_label`),
metal 4f Rydbergs, core orbitals, anisotropic NMR terms, NHO φ/θ angles, near-constant columns
(`ratio_threshold`) and one member of every pair above `corr_threshold`. The output is
`<date>_<name>_descriptors.csv` with `inchi_folder`, `inchi`, `smiles`, `ligand_class` in front.

### Adding a new molecule family

Copy a spec, then edit `atoms`, `numbering` (SMARTS stages, or `manual` with your CSV, or `first_n`),
the label lists under `descriptors`, and `filter.metal_label`. Run `validate`, then `number` and check
the `numbering_status` column before `extract`.

### Tests

```bash
pytest tests -m "not slow"     # unit tests, seconds
pytest tests -m slow           # regression: reproduces atom_numbering_results.csv, the DescriPyTor
                               # per-conformer reference and the committed aggregated CSVs (minutes)
```

The slow tests are skipped when `DATA/` or the submodule is absent.

## Descriptors provided

Notation: `X` is an atom label from the spec, `Y` a %Vbur radius (default 2.0 to 5.0 Å in 0.5 steps),
`R` an NBO rank token relative to the highest occupied NBO (`homo`, `hm1`, `hm2`, ... below it, `lp1`,
`lp2`, ... above it; kept for the ranks in `homo_lumo_rank_range`). Every per-conformer descriptor
below appears four times in the aggregated CSV (`_min`, `_max`, `_lowE`, `_boltz`).

:white_check_mark: implemented and covered by the regression tests. :hourglass: implemented, not
enabled in the shipped specs. :x: not implemented yet.

### ORCA descriptors

**Fragment specific geometric descriptors** (morfeus on `optimization.xyz`, `descriptors.orca.xyz`)

- %Vbur (Bondi radii × 1.17, H included, sphere radius `Y`): `%Vbur_atom_X_radius_Y` :white_check_mark:
- Atomic SASA: `SASA_X` :white_check_mark:
- Pyramidalization P and P angle: `Pyr_X`, `Pyr_angle_X` :white_check_mark:
- Sterimol (morfeus, CRC van der Waals radii; L axis from first to second label): `Sterimol_X_X_L`, `Sterimol_X_X_Bmin`, `Sterimol_X_X_Bmax` :white_check_mark: / B1 and B5 do not match DescriPyTor `B1_X-X`, `B5_X-X` (different radii and slicing; see [review/DESCRIPYTOR_UPGRADE.md](review/DESCRIPYTOR_UPGRADE.md))
- Bond length: `Bond_X_X` :white_check_mark: / matches DescriPyTor `dist_X-X`
- Angle: `Angle_X_X_X` :white_check_mark: / matches DescriPyTor `angle_X-X-X`
- Dihedral: `Dihed_X_X_X_X` :white_check_mark:
- %Vbur quadrants and octants (`quadrants: [[metal, z1, z2, xz_plane, ...]]`): `%Vbur_X_X_X_X_quadrant_+,+`, ..., `%Vbur_X_X_X_X_octant_+,+,+`, ... :hourglass:
- Bite angle (morfeus `BiteAngle`, method present, not wired to the spec) :x: not implemented yet

**Molecular geometric descriptors**

- SASA and SASA volume: `SASA`, `SASA_Volume` :white_check_mark:

**Charges** (`singlepoint.property.json`, `descriptors.orca.charges`)

- Mulliken: `mulliken_X` :white_check_mark:
- Loewdin: `loewdin_X` :white_check_mark:
- CHELPG: `chelpG_X` :white_check_mark:
- Hirshfeld: `hirshfeld_X` :white_check_mark:
- CM5: `cm5_X` :x: not implemented yet (the current column is a stub equal to `hirshfeld_X`, see [CODE_REVIEW.md](CODE_REVIEW.md) C1)
- NBO / NPA charge: `npa_X_charge` :white_check_mark: / matches DescriPyTor `charge_NBO_X`

**Orbital analysis** (NBO 7 block of `calcParam.out`, `descriptors.orca.nbo`)

- NPA populations: `npa_X_core`, `npa_X_valence`, `npa_X_rydberg`, `npa_X_total`, `npa_totals_*` :white_check_mark:
- Natural atomic orbital occupancies: `nao_X_Cor-1s_occupancy`, `nao_X_Val-2p_occupancy`, `nao_X_Ryd-3d_occupancy`, ... (`_2`, `_3` suffixes for degenerate sets) :white_check_mark:
- Natural bond orbitals on labelled atoms: `nbo_R_X_X_type`, `nbo_R_X_X_bond_index`, `nbo_R_X_X_occupancy`, `nbo_R_X_X_energy` (lone pairs: `nbo_R_X_*`) :white_check_mark:
- Bond orbital composition: `bond_orbital_details_R_X_X_{bond_type,bond_index,occupancy}`, `bond_orbital_details_R_X_X_contrib_k_X_{percentage,coefficient,s_percent,p_power,p_percent}` :white_check_mark:
- NHO directionality: `nho_directionality_R_X_X_{line_theta,line_phi,hybrid_theta,hybrid_phi,deviation}` :white_check_mark: (φ/θ dropped by the default filter)
- Second-order perturbation E(2) between labelled orbitals: `second_order_R_<donor>_to_R_<acceptor>_{donor_type,donor_index,acceptor_type,acceptor_index,e2_energy,energy_diff,fock_element}` :white_check_mark: (NBO 7 numbered rows and the legacy `->` layout; only rows whose donor and acceptor atoms are all labelled are kept; both `R` tokens are subject to `homo_lumo_rank_range`, so widen or disable it to keep these columns)
- HOMO and LUMO energies (eV, from the singlepoint MOs): `HOMO_energy`, `LUMO_energy` :white_check_mark:
- Other frontier orbital descriptors (gap, orbital coefficients) :x: not implemented yet

**Redox and conceptual DFT** (vertical `oxidation` / `reduction` steps, `descriptors.orca.redox`)

- Vertical ionization potential and electron affinity (Hartree): `Vertical_Ionization_Potential`, `Vertical_Electron_Affinity` :white_check_mark: (computed as `E(neutral) − E(cation)` and `E(anion) − E(neutral)`, i.e. with the opposite sign of the usual convention; see CODE_REVIEW C2)
- Hardness, electronegativity, electrophilicity: `Hardness`, `Electronegativity`, `Electrophilicity` :white_check_mark: (same sign caveat)

**Condensed Fukui functions** (`Z` = charge scheme listed in `redox.prefixes`: `nbo_charge`, `cm5`, `mulliken`, `loewdin`, `chelpG`, `hirshfeld`)

- `fplus_Z_X`, `fminus_Z_X`, `fzero_Z_X` :white_check_mark: (`Z = cm5` inherits the CM5 stub above)

**NMR** (`nmr.property.json`, GIAO)

- Isotropic shielding: `siso_X` :white_check_mark:
- Anisotropic shielding: `saniso_X` :white_check_mark: (dropped by the default filter)

**Molecular descriptors**

- SCF energy of the singlepoint (Hartree): `SCF_energy` :white_check_mark: (aggregation weight, not a descriptor)
- Enthalpy, Gibbs free energy: `Enthalpy`, `GibbsFreeEnergy` :x: not implemented yet (read from `singlepoint.property.json`, which has no thermochemistry; needs `frequency.property.json`)
- Polarizability, molecular and atomic dipoles from `%elprop` :x: not implemented yet (computed by ORCA, not extracted)
- Level of theory (`opt_method`, `svp_theory`) :x: not implemented yet (reader exists, not wired to the spec)

### DescriPyTor descriptors

Computed by DescriPyTor `Molecules` from the per-conformer feather (`descriptors.descripytor`).
Each key takes a list of label tuples; the shipped LNiH2 spec uses the same bonds and angles as the
ORCA families.

**Fragment specific geometric descriptors**

- Sterimol (CPK radii, whole complex, metal connectivity from Alvarez covalent radii): `L_X-X`, `B1_X-X`, `B5_X-X`, `loc_B5_X-X`, `B1-B5_angle_X-X` :white_check_mark: / `L` almost matches morfeus `Sterimol_X_X_L`; `B1`, `B5` do not
- Bond length: `dist_X-X` :white_check_mark: / matches `Bond_X_X`
- Angle: `angle_X-X-X` :white_check_mark: / matches `Angle_X_X_X`
- Cube Sterimol (from a density `.cube` file) :x: not implemented yet
- Buried volume, cone angle, dispersion from DescriPyTor's morfeus suite :x: not implemented yet (covered by the ORCA side)

**Charges**

- NBO charge: `charge_NBO_X` :white_check_mark: / matches `npa_X_charge`
- NBO charge difference: `charge_NBO_diff_X-X` :white_check_mark:
- Hirshfeld / CM5 charges through DescriPyTor :x: not implemented yet (the pseudo-Gaussian log does not carry them; use the ORCA side)

**Vibrations** (frequencies and normal modes of the `frequency` step)

- Stretching mode of a bond, within `[stretch_threshold, upper_stretch_threshold]` cm⁻¹: `vib_freq_X-X`, `vib_amp_X-X` :white_check_mark:
- Bending mode of two atoms sharing a centre, above `bend_threshold` cm⁻¹: `vib_freq_bend_X-X`, `vib_amp_bend_X-X` :white_check_mark:
- Ring vibrations (`cross` / `para`) :x: not implemented yet

**Dipole** (frame defined by `[origin, y-axis atom, xy-plane atom]`)

- Norm: `dipole` (one triple) or `dipole_X-X-X` (several) :white_check_mark:
- Components in the transformed frame: `dipole_x_X-X-X`, `dipole_y_X-X-X`, `dipole_z_X-X-X` :white_check_mark:

**Not available from DescriPyTor's feather path**: polarizability, IR intensities, Enthalpy and Gibbs
free energy (:x: not implemented yet; planned with the JSON converter).

## References

Software

- **ORCA 6.1.** Neese, F. *WIREs Comput. Mol. Sci.* **2012**, *2*, 73, [10.1002/wcms.81](https://doi.org/10.1002/wcms.81); Neese, F. *WIREs Comput. Mol. Sci.* **2022**, *12*, e1606, [10.1002/wcms.1606](https://doi.org/10.1002/wcms.1606).
- **GOAT** conformer search. Zivkovic, A.; de Souza, B.; Neese, F.; Bistoni, G. *Angew. Chem. Int. Ed.* **2025**, e202418338, [10.1002/anie.202418338](https://doi.org/10.1002/anie.202418338).
- **NBO 7.** Glendening, E. D.; Landis, C. R.; Weinhold, F. *J. Comput. Chem.* **2019**, *40*, 2234, [10.1002/jcc.25873](https://doi.org/10.1002/jcc.25873).
- **DescriPyTor.** Milo group, [github.com/Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor), modelled after *MoleculaR* ([barkais.github.io](https://barkais.github.io/)).
- **morfeus.** Jorner, K. et al., [github.com/digital-chemistry-laboratory/morfeus](https://github.com/digital-chemistry-laboratory/morfeus) (Sterimol, buried volume, SASA, pyramidalization).
- **DBSTEP.** Luchini, G.; Paton, R. S., [github.com/patonlab/DBSTEP](https://github.com/patonlab/DBSTEP) (Sterimol scans).
- **RDKit**, [rdkit.org](https://www.rdkit.org); **Open Babel.** O'Boyle, N. M. et al. *J. Cheminf.* **2011**, *3*, 33, [10.1186/1758-2946-3-33](https://doi.org/10.1186/1758-2946-3-33).
- Descriptor selection strategy after the Sigman group's [Get_Properties](https://github.com/SigmanGroup/Get_Properties).

Descriptor definitions

- **Sterimol.** Verloop, A. In *Drug Design*; Ariëns, E. J., Ed.; Academic Press, 1976; Vol. III. Conformer-dependent Sterimol: Brethomé, A. V.; Hadlington, T. J.; Paton, R. S. *ACS Catal.* **2019**, *9*, 2313, [10.1021/acscatal.8b04043](https://doi.org/10.1021/acscatal.8b04043).
- **Buried volume, quadrants, octants.** Falivene, L.; Cao, Z.; Petta, A.; Serra, L.; Poater, A.; Oliva, R.; Scarano, V.; Cavallo, L. *Nat. Chem.* **2019**, *11*, 872, [10.1038/s41557-019-0319-5](https://doi.org/10.1038/s41557-019-0319-5).
- **Pyramidalization.** P: Radhakrishnan, T. P.; Agranat, I. *Struct. Chem.* **1991**, *2*, 107; P angle: Gavrish, S. P. *J. Comput. Chem.* **2012**, *33*, 2173, [10.1002/jcc.23051](https://doi.org/10.1002/jcc.23051).
- **SASA.** Shrake, A.; Rupley, J. A. *J. Mol. Biol.* **1973**, *79*, 351, [10.1016/0022-2836(73)90011-9](https://doi.org/10.1016/0022-2836(73)90011-9).
- **Covalent radii** (metal connectivity in DescriPyTor). Cordero, B. et al. *Dalton Trans.* **2008**, 2832, [10.1039/b801115j](https://doi.org/10.1039/b801115j).
- **Hirshfeld charges.** Hirshfeld, F. L. *Theor. Chim. Acta* **1977**, *44*, 129, [10.1007/BF00549096](https://doi.org/10.1007/BF00549096).
- **CHELPG.** Breneman, C. M.; Wiberg, K. B. *J. Comput. Chem.* **1990**, *11*, 361, [10.1002/jcc.540110311](https://doi.org/10.1002/jcc.540110311).
- **CM5** (target of the not-yet-implemented column). Marenich, A. V.; Jerome, S. V.; Cramer, C. J.; Truhlar, D. G. *J. Chem. Theory Comput.* **2012**, *8*, 527, [10.1021/ct200866d](https://doi.org/10.1021/ct200866d).
- **NPA / NBO.** Reed, A. E.; Weinstock, R. B.; Weinhold, F. *J. Chem. Phys.* **1985**, *83*, 735, [10.1063/1.449486](https://doi.org/10.1063/1.449486); Reed, A. E.; Curtiss, L. A.; Weinhold, F. *Chem. Rev.* **1988**, *88*, 899, [10.1021/cr00088a005](https://doi.org/10.1021/cr00088a005).
- **Fukui functions.** Parr, R. G.; Yang, W. *J. Am. Chem. Soc.* **1984**, *106*, 4049, [10.1021/ja00326a036](https://doi.org/10.1021/ja00326a036); condensed form: Yang, W.; Mortier, W. J. *J. Am. Chem. Soc.* **1986**, *108*, 5708, [10.1021/ja00279a008](https://doi.org/10.1021/ja00279a008).
- **Hardness, electronegativity, electrophilicity.** Parr, R. G.; Pearson, R. G. *J. Am. Chem. Soc.* **1983**, *105*, 7512, [10.1021/ja00364a005](https://doi.org/10.1021/ja00364a005); Parr, R. G.; Szentpály, L. v.; Liu, S. *J. Am. Chem. Soc.* **1999**, *121*, 1922, [10.1021/ja983494x](https://doi.org/10.1021/ja983494x).

Level of theory

- **ωB97M-V.** Mardirossian, N.; Head-Gordon, M. *J. Chem. Phys.* **2016**, *144*, 214110, [10.1063/1.4952647](https://doi.org/10.1063/1.4952647); **D4.** Caldeweyher, E. et al. *J. Chem. Phys.* **2019**, *150*, 154122, [10.1063/1.5090222](https://doi.org/10.1063/1.5090222).
- **M06.** Zhao, Y.; Truhlar, D. G. *Theor. Chem. Acc.* **2008**, *120*, 215, [10.1007/s00214-007-0310-x](https://doi.org/10.1007/s00214-007-0310-x).
- **def2 basis sets.** Weigend, F.; Ahlrichs, R. *Phys. Chem. Chem. Phys.* **2005**, *7*, 3297, [10.1039/b508541a](https://doi.org/10.1039/b508541a).
