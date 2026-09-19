# Code review — `orca_descriptors`

Scope: in-house code only (`desc_utils`, `mlr_utils`, `orca2gaussian`, `scripts-for-orca-computations`, `DATA/test_molecules`, plus the aggregation cell in `make_descriptor_descripytor.ipynb`). The vendored `DescriPyTor` package (from [Milo-group/DescriPyTor](https://github.com/Milo-group/DescriPyTor)) was not reviewed, though its parser behaviour is used below to confirm findings in `orca2gaussian`.

Review type: correctness and bugs. Style, naming and formatting are out of scope.

Abbreviations used: **MLR** multivariate linear regression; **CV** cross-validation; **NBO** natural bond orbital; **NPA** natural population analysis; **NAO** natural atomic orbital; **IP** ionization potential; **EA** electron affinity; **SASA** solvent-accessible surface area; **%Vbur** percent buried volume; **SCF** self-consistent field; **ZPE** zero-point energy; **PFI** permutation feature importance; **SMARTS** SMILES arbitrary target specification; **RDKit** the cheminformatics toolkit; **KS** Kennard-Stone; **SFS** sequential feature selection.

---

## Summary

Six findings were confirmed against the shipped data files or by running the converter, not just by reading code:

| # | Finding | Evidence |
|---|---------|----------|
| C1 | CM5 charges are numerically identical to Hirshfeld charges | 18/18 pairs in `combined_descriptors.csv` agree to < 1e-12 |
| C2 | IP, EA, hardness, electronegativity, electrophilicity all sign-flipped (**fixed 2026-09-18**, `RedoxSet.get_redox_properties`) | every `Vertical_Ionization_Potential` in `combined_descriptors.csv` is negative |
| C3 | Coordinates written to the fake Gaussian log are in Bohr, labelled Angstroms | C–O = 2.673 (Bohr) vs 1.414 Å for methanol |
| C4 | Dipole written in atomic units, labelled Debye | factor 2.5417 vs `calcParam.out` |
| C5 | Normal-mode matrix is transposed | `norm(MODES[:,0:6]) == 0` matches `FREQ[0:6] == 0` |
| C6 | Hirshfeld and CM5 columns parse out 100% empty via the fake log | measured all-NaN, no exception raised |

The highest-risk area is `orca2gaussian`: any descriptor that reached your models through the fake-Gaussian path (vibrations, dipole, NBO charge, polarizability, IR intensity) should be treated as suspect until re-extracted. The second-highest is `mlr_utils/feature_selection.py`, where the reported validation score is an upward-biased maximum over many candidates and the CV splitter is unseeded.

---

## Critical

### C1. CM5 charges are a stub — every `cm5_*` descriptor equals `hirshfeld_*`

`desc_utils/orca_descriptors.py:314-335`

```python
atomradius = np.zeros(119)
atomradius[:] = 0.7
alpha = 2.474
def _tij(i, j):
    return 0.0
...
    s += tij * bij
self.results[f"cm5_{label}"] = hirshfeld_charges[i] + s
```

`_tij` returns 0.0 unconditionally, so `s == 0.0` and `cm5 == hirshfeld` exactly. The Dq parameter table that defines CM5 is absent, and `atomradius` is a flat 0.7 for every element rather than the Bragg-Slater radii CM5 uses.

Verified: in `combined_descriptors.csv`, `cm5_at1`/`cm5_at2` match `hirshfeld_at1`/`hirshfeld_at2` for all 18 comparable values to within 1e-12.

Downstream impact is larger than it looks. `RedoxSet.__init__` defaults to `descriptorPrefixes=["cm5"]` (line 992), so **every Fukui descriptor in the shipped dataset** — `fplus_cm5_*`, `fminus_cm5_*`, `fzero_cm5_*` — is Hirshfeld-based. Any model that selected a `cm5_*` feature over a `hirshfeld_*` one (or vice versa) picked between two identical columns; the correlation filter will also have dropped one of each pair arbitrarily by column order.

Fix: either implement CM5 properly (needs the pairwise Dq parameters and Bragg-Slater radii) or delete the method and stop emitting `cm5_*`. The README currently marks CM5 as checked (`:white_check_mark:`) — that should be corrected either way.

### C2. IP and EA are sign-flipped, propagating to hardness, electronegativity and electrophilicity — fixed 2026-09-18

`desc_utils/orca_descriptors.py:1046-1047`

```python
self.results['Vertical_Ionization_Potential'] = self.neutralReader.results.get('SCF_energy') - self.cationReader.results.get('SCF_energy') ...
self.results['Vertical_Electron_Affinity'] = self.anionReader.results.get('SCF_energy') - self.neutralReader.results.get('SCF_energy') ...
```

IP is defined as `E(cation) − E(neutral)` and EA as `E(neutral) − E(anion)`. Both are computed backwards.

Verified in `combined_descriptors.csv`: every `Vertical_Ionization_Potential` is negative (−0.32 to −0.27 Hartree, i.e. −8.7 to −7.4 eV). A neutral closed-shell organometallic has a positive IP of roughly that magnitude.

Consequences at lines 1051-1055: with both inputs flipped, `Hardness`, `Electronegativity` and `Electrophilicity` come out with correct magnitude but negative sign (all confirmed negative in the CSV). Because a global sign flip is absorbed into a linear coefficient, this has probably not corrupted MLR *fits* — but any reported value, plot axis, or chemical interpretation of these five descriptors is wrong, and `Electrophilicity` in particular is meaningless as a negative number.

Secondary issue, same block: these five are left in **Hartree** while `HOMO_energy`/`LUMO_energy` are converted to eV at lines 417/419 via `h_to_eV`. The descriptor table mixes units with no unit suffix in the column names.

Fix:
```python
ip = cation_scf - neutral_scf
ea = neutral_scf - anion_scf
```
and multiply by `h_to_eV` for consistency with HOMO/LUMO.

Note: the Fukui functions immediately below (lines 1070-1072) use the *correct* convention — `f_plus = qN - qA`, `f_minus = qC - qN`, `f_zero = (f_plus + f_minus)/2`. Only the redox block is wrong.

### C3. Coordinates from ORCA JSON written in Bohr but labelled Angstroms

`orca2gaussian/orca_to_gaussian_log.py:292`, header at `:476`

```python
x, y, z = _to_float(row[1]), _to_float(row[2]), _to_float(row[3])
```
```python
" Center     Atomic      Atomic             Coordinates (Angstroms)",
```

`Geometry.Coordinates.Cartesians` in `*.property.json` is in Bohr. Verified on `DATA/test_molecules/OKKJLVBELUTLKV-UHFFFAOYSA-N/conf_1`: methanol C–O comes out 2.673, i.e. 1.414 Å × 1.889. The missing `× 0.5291772` makes every emitted per-job geometry 1.89× too large.

This is currently masked because DescriPyTor takes the *last* `Standard orientation:` block, which happens to be the correctly-scaled block read from `calcParam.out` (line 751). Any conformer directory without `calcParam.out` silently gets Bohr geometry — and `_extract_data` guards that path with `if calcparam_path.exists()` (line 370), so it fails open, not closed.

### C4. Dipole written in atomic units but labelled Debye

`orca2gaussian/orca_to_gaussian_log.py:314-319`, `:491`, and again in `_format_computation_summary` at `:572-576`

```python
" Dipole moment (field-independent basis, Debye):",
f"    X= {dx: .6f}    Y= {dy: .6f}    Z= {dz: .6f}    Tot= {dmag: .6f}",
```

Verified against `calcParam.out:3163-3166`: ORCA's `dipoleMagnitude` is 0.7828 a.u. = 1.9897 Debye. No `× 2.541746` is applied, so every dipole descriptor coming through this path is 2.5418× too small.

### C5. Normal-mode matrix is transposed — all vibration descriptors are garbage

`orca2gaussian/orca_to_gaussian_log.py:383-387`

```python
modes = fr_geom.get("Hessian", {}).get("MODES", [])
for mode in modes:
    if isinstance(mode, list):
        mode_vectors.append([_to_float(v, 0.0) for v in mode])
```

`MODES` is stored **column-major**: mode *j* is `MODES[:, j]`, not `MODES[j]`. Verified on the real `frequency.property.json` (18×18): `norm(MODES[:, 0:6]) == 0.0` exactly, matching `FREQ[0:6] == 0.0`, while the row norms are nonzero.

The generated log shows the symptom directly — atoms 1 and 2 (C, O) have all-zero displacements in every block, while the 0.0000 cm⁻¹ block carries nonzero displacements.

Fix: `mode_vectors = [[MODES[i][j] for i in range(len(MODES))] for j in range(len(MODES[0]))]`, or transpose with numpy.

### C6. Hirshfeld and CM5 columns come back 100% NaN, silently

`orca2gaussian/orca_to_gaussian_log.py:587`

```python
lines.append(" Atom      Symbol   Mulliken     Hirshfeld     CM5-Charges")
```

DescriPyTor's `feather_extractor.py:270-277` locates the charge block with `matches("Hirshfeld")[-2].end()` and `matches("Hirshfeld charges with hydrogens ")[-1].start()`. This extra occurrence of the word `Hirshfeld` in the summary header shifts `[-2]` onto the terminator line, so `start > end`, the slice is empty, `reshape(-1, 6)` succeeds on an empty array, and the parser still prints its success message. Measured result: `hirshfeld_charge` and `cm5_charge` are entirely NaN.

Fix: remove the word `Hirshfeld` from the summary header line.

### C7. NBO charge block terminator collides with the writer's own `=` banners

`orca2gaussian/orca_to_gaussian_log.py:507` vs `:560`, `:597`, `:702`, `:708-710`, `:745-747`, and the raw NBO dump at `:644-649`

```python
lines.append(" =====")                                      # :507  intended terminator
lines = [" COMPUTATION OUTCOMES SUMMARY", " " + "=" * 70]    # :559-560
```

DescriPyTor takes `starts[-1]` of `"Summary of"` and `ends[-1]` of `"====="` over the *whole file*. In the generated log the last `"Summary of"` is inside the verbatim NBO text, as is the last `"====="`, so the parsed slice is ~500 lines of NBO output.

Measured on the test molecule: `nbo_charge = [-0.27479, -0.72920, 0.16906, 0.18640, 0.16906, 0.47948, 3.99998, 4.0, 18.0, 1.1, ...]` — 427 rows for a 6-atom molecule — instead of the true values `[-0.1214, -0.4252, 0.0982, 0.1060, 0.0982, 0.2442]`.

Fix: use a unique terminator, and do not emit `"=" * 70` banners or the raw NBO dump after the charge block.

### C8. `--force_relaunch_orca` uses `type=bool` and then deletes every completed result

`DATA/test_molecules/launch_orca_all.py:10`, `:52-58`; same pattern in `make_xyz_and_create_folder.py:10`

```python
parser.add_argument("--force_relaunch_orca", type=bool, ... default=False)
...
subprocess.run('rm calcParam*', check=True, shell=True)
subprocess.run('rm frequency*' , check=True, shell=True)
```

With `type=bool`, `--force_relaunch_orca False` / `0` / `no` all evaluate to `True` (any non-empty string is truthy). Triggering it `rm`s `calcParam*`, `frequency*`, `nmr*`, `singlepoint*`, `oxidation*`, `reduction*` and `optimization*` — irreversibly destroying every finished calculation in the tree.

Compounding it, the force branch never regenerates `input.xyz` (the `cut -c 1-57 ... > input.xyz` step present in the `elif` branch at line 70 is missing from the force branch), while line 58 has already deleted `optimization*`. The resubmitted job runs on a stale or absent input.

Fix: `action="store_true"`. Consider requiring an explicit `--yes-i-mean-it` for the delete, or moving files to a `trash/` directory instead of `rm`.

### C9. `relauncn_all_orcas.py` resubmits every conformer unconditionally and overwrites good data

`DATA/test_molecules/relauncn_all_orcas.py:17-21`

```python
if 'input.xyz' in os.listdir('.'):
    subprocess.run('cp input.xyz original_input.xyz', ...)
subprocess.run('cp optimization.xyz input.xyz', ...)
subprocess.run('sbatch slurm-calcParam.sh', ...)
```

No check of convergence (`grep NORMALLY`) or of queue state. Every invocation double-submits every conformer, and each new ORCA job overwrites `calcParam.out` and all `*.json` in place. Additionally `original_input.xyz` is overwritten with the *previous* run's `optimization.xyz`, so the true starting geometry is lost on the second invocation.

Recommend not running this script again in its current form.

### C10. SMILES injected into a `sed s///` expression

`DATA/test_molecules/submit_all_smiles.py:8-10`

```python
subprocess.run(f"sed -i 's/smiles_ligand/{ligand}/g' launch_workflow_{i}.sh", shell=True)
subprocess.run(f"sbatch launch_workflow_{i}.sh", shell=True)
```

SMILES routinely contain `/` and `\` (E/Z bond specifications, e.g. `C/C=C/C`), which break the `s///` delimiter, and `&`, which `sed` expands to the whole match. A `'` escapes the shell quoting entirely. No `check=True` anywhere, so a failed `sed` is silent and `sbatch` still runs on a corrupt or unsubstituted script.

Fix: write the substituted script with Python string replacement rather than `sed`, and add `check=True`.

### C11. `alter_results_dict` references undefined globals

`desc_utils/data_extraction.py:604-659`

```python
if subkey.startswith("Bond_"):
    for bond in bonds:
```

`bonds`, `angles`, `dihedrals` and `atoms` are neither parameters nor module-level names. Any call raises `NameError`. Either the function is dead code (in which case delete it) or a caller is silently never reaching it.

### C12. Core_5 ligand classes are silently dropped from the returned dataframe

`desc_utils/atom_numbering.py:292-296`, consumed at `:324-331`

```python
if len(other_ligands) > 0:
    other_lig_df = df5.loc[df5['xyz_name'].isin(other_ligands)].copy()
    other_lig_df.loc[other_lig_df['xyz_name'].isin(biox_ligands), 'ligand_class'] = 'biox'
```

`other_ligands` is only appended to in the classification-*failure* branch (line 282). `bpy_ligands`, `biox_ligands` and `biim_ligands` are never added to it, so those `.loc[... .isin(...)]` assignments are no-ops on an empty frame. `df5_pyox_corrected` is built from `pyox_lig_df + other_lig_df` only, so every *correctly classified* non-pyox 5-ring ligand disappears from the output.

### C13. `pyr6` atom indices come out 2 too large

`desc_utils/atom_numbering.py:395-396`, `:400-401`

```python
pyr6_lig_df.loc[i, 'N1'], pyr6_lig_df.loc[i, 'N2']  = N2_index + 1, N1_index + 1
```

`find_pyox_ligands_and_renumber` operates on 0-based RDKit indices throughout (see its own `GetAtomWithIdx(N1_index)` calls at 266-269, 353-357, 387-391). Only the pyr6 branch adds `+1`; the pyox branch (317-320) and box branch write raw 0-based values. `compile_final_nn1` then adds `+1` to every row (599-605 / 629-635), so pyr6 `N1`/`N2`/`C1`/`C2` end up 2 too large while every other class is correct.

Two further problems in the same block: the increment is applied even in the no-swap-needed case, and the comment on line 393 ("df is correct as is") contradicts the code, which swaps.

### C14. `number_H1_and_H2` assumes 1-based input but is called with 0-based

`desc_utils/atom_numbering.py:436-441`, `:458-462`, called at `:595`

```python
Ni_index_rdkit = row['Ni'] - 1
Ni_atom = mol.GetAtomWithIdx(Ni_index_rdkit)
```

The call site precedes the `indexing == 1` conversion, so `row['Ni']` is already 0-based and `Ni_index_rdkit` points at the wrong atom. Usually `H_indices` ends up empty and the function falls through to the warning at line 464 leaving H1/H2 unassigned; but if the atom at `Ni-1` happens to have two H neighbours, it assigns *another atom's* hydrogens. It then writes `H_indices[0] + 1` (1-based) into an otherwise 0-based dataframe, which `compile_final_nn1` increments again.

### C15. Flattening all substructure matches, then indexing positionally

`desc_utils/atom_numbering.py:87-88`, `:101-102`, `:113`

```python
atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
... 'N2': atoms_idx_in_substructure_rdkit[7]})
```

`GetSubstructMatches` returns *every* match; flattening concatenates them end to end. The `substructure_5` SMARTS have 7 atoms, but the retry branch at line 102 indexes `[5]`, `[6]`, `[7]` — index 7 is the first atom of the *second, symmetry-equivalent* match. So an H on Ni can be labelled `N2`, and `C2` receives the value that should have been `N2`. If only one match exists it raises `IndexError` instead. (Your own inline comments at those lines — "why is there C3 here?", "I don't understand how the list ordering works" — mark exactly this spot.)

Fix: iterate over matches explicitly and index within a single match tuple; use canonical ranking or `useChirality` to disambiguate symmetric matches deterministically.

---

## High

### H1. Reported validation score is a biased maximum over many candidates

`mlr_utils/feature_selection.py:334-337`, `:404-406`, `:445-448`

```python
sorted_history = sorted(history, key=lambda r: r['val_score'], reverse=True)
top = sorted_history[:top_k]
best_output = top[0] if top else None
```

Every candidate feature combination — up to tens of thousands in exhaustive mode, and in `sfs` mode whenever `top_k > 1` — is scored on the *same* fixed `cv_splits`, and the maximum is then reported as `val_score`. The max of many noisy CV estimates is strongly upward-biased. The same selection happens again across models at 445-448. Nothing in the module computes an outer-loop score, so any headline Q²/"Val R²" from this code is optimistic, potentially by a lot on the 20-40 compound datasets typical of ee work.

Fix: wrap the whole selection in an outer CV loop and report the outer score, or hold out a test set that selection never touches.

### H2. Unseeded shuffled splitter re-partitions on every call

`mlr_utils/feature_selection.py:94`, `:97`

```python
return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
```

With the default `random_state=None`, a `KFold(shuffle=True)` *object* reshuffles each time `split()` is called. That object is reused across `sfs.fit()` (line 270), the follow-up `cross_validate` (273), and every combination `cross_validate` (311/380). Candidates are therefore ranked against *different* fold partitions — the comparison is not like-for-like, the winner is partly whoever drew the luckiest split, and results are irreproducible run to run. Same for `ShuffleSplit`.

Fix: materialize the splits once (`list(KFold(...).split(X))`) and require an integer `random_state`.

### H3. `val_score` is a mean of per-fold R², not Q²

`mlr_utils/feature_selection.py:282-283`, `:390-391`; `mlr_utils/plot_utils.py:102-103`

```python
mean_train = np.mean(cv_res['train_score'])
mean_test  = np.mean(cv_res['test_score'])
```

Each fold's R² is referenced to *that fold's own* mean, then averaged. The MLR convention (Q²) is a single pooled `1 - PRESS/TSS` against the full-data mean. With 5 folds on ~20-40 compounds the per-fold means are unstable, so candidate ranking is partly driven by fold-mean artifacts.

This becomes fatal for `split_method='extreme'` (`:132-143`): the test set is the top/bottom 20% of y and has near-zero internal variance, so R² referenced to that slice's mean is strongly negative even for an excellent extrapolating model. Extrapolation tests need RMSE, or R² referenced to the *training* mean.

### H4. Parity plots raise for three of the five split methods, and plot a different partition than they report

`mlr_utils/plot_utils.py:100`, `:233`

```python
y_pred = cross_val_predict(model, X, y, cv=splits, n_jobs=-1)
```

`cross_val_predict` requires the splits to form a partition. `random` (ShuffleSplit), `kennard_stone` and `extreme` all return a single or incomplete split, so this raises — meaning the parity plot has never been exercised for the extrapolation splits you would most want to plot.

Separately, when it *does* work, `cross_val_predict` and `cross_validate` each call `splits.split()` independently; per H2 with an unseeded splitter those are two different partitions. The scatter, `metrics['r2']`, `metrics['rmse']` and the annotated `Val R2` describe different experiments while being displayed as one result.

### H5. Three different R²-like numbers in two annotation boxes, none unambiguously labelled

`mlr_utils/plot_utils.py:74`, `:114-122`, `:154`; `mlr_utils/feature_selection.py:49`, `:65`, `:76`

`val_r2` is a mean per-fold R²; `metrics['r2']` is the pooled out-of-fold R² (the actual Q²); `parity_plot` has already stamped a second box containing another `R2=`; and the full-data training R² is printed only to stdout. In `feature_selection.report_mlr_equation`, `model.score(X, y)` (a training R²) lands in the returned dict as plain `'r2'` next to CV-derived scores. High risk of the wrong number reaching a figure caption.

### H6. Boltzmann weighting uses bare SCF energies

`make_descriptor_descripytor.ipynb` cell 29; `desc_utils/data_extraction.py:1125-1136`

```python
T = 298.15
kB_hartree_per_K = 3.166811563e-6  # Hartree/K
dE = e - e_min
w = np.exp(-(dE / (kB_hartree_per_K * T)))
w = w / np.nansum(w)
```

The mechanics here are **correct**: ORCA SCF energies are Hartree, `kB` in Hartree/K is right, energies are referenced to the minimum so there is no overflow, and `np.average` renormalizes over the valid subset internally.

The chemistry is the problem: `orca_descriptors.py:434` exposes `GibbsFreeEnergy`, but the ensemble is weighted on electronic energies with no ZPE, thermal or entropic correction. For a conformer ensemble at 298 K that is the wrong ensemble — conformers differing in entropy (e.g. more or less hindered rotors) are mis-weighted. kB·T here is 0.593 kcal/mol, so a 1 kcal/mol error in relative energy changes a weight by a factor of ~5.

Also: if any `SCF_energy` is NaN, `e.min()` is NaN and *all* weights become NaN silently.

### H7. Two divergent aggregation implementations, and `_lowE` exists in only one of them

`desc_utils/data_extraction.py:1141-1144` vs `make_descriptor_descripytor.ipynb` cell 29

`descriptor_stats()` in `data_extraction.py` emits `_min`, `_max`, `_mean`. The notebook emits `_min`, `_max`, `_lowE`, `_boltz`. The shipped CSVs use the notebook's naming, so `descriptor_stats()` appears to be an unused older path — but it is still importable, and `suffix_priority` (`:898-908`) ranks `_boltz`/`_boltzmann`/`_lowE` and would assign `_mean` the fallback priority 4. Whichever path a future user picks changes both column names and the correlation-pruning tie-break.

`descriptor_stats()` additionally has a hardcoded `df.iloc[:, 15:]` column offset (line 1141) combined with a conditional `df.insert(5, 'Relative_energy', ...)` that shifts everything by one on first call — so *which* descriptors get aggregated depends on how many metadata columns happen to precede them. It also mutates the caller's dataframe in place.

Recommend deleting `descriptor_stats()` and moving the notebook's aggregation into `data_extraction.py` as the single implementation.

### H8. `min`/`max` and `_boltz` disagree about NaN

`desc_utils/data_extraction.py:1142-1144`

```python
descriptor_stats[f'{column}_min'] = df[column].min()
descriptor_stats[f'{column}_mean'] = np.average(df[column], weights=normalized_weights)
```

`Series.min()`/`.max()` skip NaN, so a conformer missing that descriptor is silently dropped and the reported min/max is over a *different* conformer subset for each descriptor — while `np.average` propagates NaN and poisons the column. One failed conformer therefore yields a valid-looking min/max and a NaN mean.

The notebook path handles this better (`valid = v.notna() & w.notna()`), but still silently averages over a different conformer subset per descriptor without recording how many contributed. Recommend emitting an `n_conformers_used` column per descriptor family.

### H9. Ground-state dipole, HOMO/LUMO gap and NBO charges are taken from the reduced species

`orca2gaussian/orca_to_gaussian_log.py:705-741`, with `:418`

```python
for prefix, label in [("singlepoint", "Ground State"), ("oxidation", "Oxidized"), ("reduction", "Reduced")]:
```

DescriPyTor uses last-occurrence semantics for dipole, charges and HOMO/LUMO. Because the reduction job is emitted last, all three come from the anion.

Measured on the test molecule: the written ground-state dipole is `X=-0.387758 Y=-0.680019 Tot=0.782805`, but the parsed values are `dip_x=-2.176855, dip_y=+1.449131, total=2.615087` — the reduction job's, **including a sign flip on y**. Parsed `energy` (HOMO-LUMO gap) is 9.215 eV, also the anion's.

Fix: emit the ground state last, or write one job per file.

### H10. The six trivial zero-frequency modes are written as real frequencies

`orca2gaussian/orca_to_gaussian_log.py:380-381`, `:609-618`

ORCA's `FREQ` array has 3N entries whose first six are exactly 0.0. Gaussian never prints the trivial translational/rotational modes. Measured parsed `Frequency` column: `[0, 0, 0, 0, 0, 0, 294.66, 1055.26, ...]`. Any "lowest frequency" descriptor returns 0.0, and any imaginary-frequency check is defeated.

Related, `:612-615`: frequency blocks are zero-padded to a multiple of three, so up to two spurious 0.0 cm⁻¹ modes are appended whenever `n_modes % 3 != 0`. Real Gaussian writes a short final block.

### H11. Polarizability inflated 1000× by the emitted number format

`orca2gaussian/orca_to_gaussian_log.py:494-496`

```python
" Polarizability tensor (a.u.)",
f" iso {iso:.6f} 0.000000 0.000000 0.000000 0.000000 {aniso:.6f}",
```

DescriPyTor multiplies by 1000 because real Gaussian prints these in `D+0n` mantissa notation, from which its `FLOAT_POL` regex strips the exponent. Writing plain decimals means the ×1000 is applied to the full value. Measured: `iso = 22948.525` where ORCA's `isotropicPolar = 22.9485`.

Also `:325`: `anisotropic_pol = max(diag_vals) - min(diag_vals)` matches neither Gaussian's `α_zz − (α_xx+α_yy)/2` nor the conventional invariant `sqrt(½Σ(α_i−α_j)²)`, so the descriptor is not comparable to Gaussian-derived datasets.

### H12. IR intensities and force constants are hardcoded dummies

`orca2gaussian/orca_to_gaussian_log.py:620-622`

```python
lines.append(" Red. masses --     1.0000     1.0000     1.0000")
lines.append(" Frc consts  --     1.0000     1.0000     1.0000")
lines.append(" IR Inten    --     0.0000     0.0000     0.0000")
```

Measured parsed `IR` column is `1.0` for every mode. ORCA does emit IR intensities in `calcParam.out`/`.hess`; they are simply never extracted. Every IR-intensity descriptor is a constant and will be dropped by the zero-variance filter — which is the benign outcome, but worth knowing.

### H13. Fabricated HOMO/LUMO fallbacks substituted silently

`orca2gaussian/orca_to_gaussian_log.py:151`, `:248`, `:443-446`

```python
if sp.get("homo") is None:
    sp["homo"] = -0.2
if sp.get("lumo") is None:
    sp["lumo"] = _to_float(sp.get("homo"), -0.2) + 0.01
```

Produces a plausible-looking 0.272 eV gap with no warning. Easy to hit: `_parse_labeled_orbital_energies_from_text` (`:185`) depends on the regex `r"Storing NBOs:\s*(singlepoint|oxidation|reduction)\.nbo"`, which matches nothing if the ORCA input names its NBO files differently, and `_extract_single_point_data` returns `homo=None` whenever `<prefix>.json` is absent.

Related, `:526-532`: only HOMO and LUMO are real; the other four emitted eigenvalues are `homo ± 0.10/0.20`. Recommend raising, or emitting no MO block, rather than inventing values.

### H14. Mulliken charges emitted under the NBO header; CHELPG emitted as CM5

`orca2gaussian/orca_to_gaussian_log.py:501-508`, `:308-312`

```python
" Summary of Natural Population Analysis:",
```
```python
cm5_like = [_to_float(item, 0.0) for item in chelpg]
```

Descriptors named `nbo_charge` receive Mulliken values and `cm5_charge` receives CHELPG — different physical quantities on different scales. ORCA does write real NBO charges (`singlepoint.nbo`, and the NBO summary is already captured in `nbo_text`); they are just never parsed out. Currently moot for both columns because of C6 and C7, but it will bite as soon as those are fixed.

### H15. NMR shieldings zero-padded and aligned by position, ignoring `NUC`

`orca2gaussian/orca_to_gaussian_log.py:666-669`

```python
nmr_siso = (nmr_siso + [0.0] * n_atoms)[:n_atoms]
...
nmr_nuc = (nmr_nuc + list(range(n_atoms)))[:n_atoms]
```

`Chemical_Shift.NUC` holds the actual ORCA atom indices of the computed nuclei. With a subset selection (`%eprnmr Nuclei = all C`), `siso` is shorter than `n_atoms`, is padded at the end, and shieldings land on the **wrong atoms** with zeros for the tail. Alignment must go through `NUC`.

### H16. NBO extraction failures swallowed by a bare `except`

`desc_utils/data_extraction.py:341-359`

```python
except Exception:
    # Silently skip if NBO extraction fails
    pass
```

A missing file, wrong block index, parse failure or bad atom index all produce a row with no `nbo_charge_*`/`npa_*_charge` keys at all, which later becomes NaN or a missing column with no diagnostic. At minimum log the molecule and the exception.

### H17. Arbitrary `.out` file fallback silently mixes redox states

`desc_utils/data_extraction.py:38-46`

```python
matching = [p for p in out_files if "nbo" in p.name.lower()]
if matching:
    return matching[0]
return out_files[0]
```

Called with `strict_basename=False` from `_extract_nbo_charges_for_reader` (344) and `get_all_nbo_properties` (505). When `oxidation.out` is absent this returns e.g. `reduction.out` (alphabetically first) and then reads `nbo_block_index=1` from it — cation descriptors silently taken from the anion calculation. Combined with H16 this is undetectable.

### H18. `NBOProps` methods crash when no NBO block was found

`desc_utils/orca_descriptors.py:507-517` then `:538`

```python
if not self.nbo_blocks:
    print("Could not find any complete NBO section in the file")
    self.nbo_section = None
...
npa_start = self.nbo_section.find("Summary of Natural Population Analysis:")
```

Every `get_*` method calls `.find()` on `self.nbo_section` without a None check, so a file without a complete NBO block prints a warning and then raises `AttributeError`. Add an early return, or raise a clear exception in `__init__`.

### H19. `aggregate_results` leaves `propertyjson` pointing at the NMR file

`desc_utils/orca_descriptors.py:480-487`

```python
original_basename = self.basename
if original_basename != "nmr":
    self.basename = "nmr"
    self.parse_property_json()
self.get_saniso()
self.get_siso()
self.basename = original_basename
```

`basename` is restored but `self.propertyjson` is not. Any subsequent call to `get_all_charges()`, `get_scfenergy()`, `get_enthalpy()` or `get_gfe()` on the same reader reads the *NMR* job's geometry and properties. Fix: save and restore `propertyjson` too, or use a separate reader for NMR.

### H20. `SCF_energy` created as `object` dtype, then excluded from numeric filtering

`desc_utils/atom_numbering.py:539-553`

```python
df['SCF_energy'] = '' # add empty column for scf energy
...
df.at[i, 'SCF_energy'] = singlepoint.results['SCF_energy']
```

The column stays `object`, so `df.select_dtypes(include=[np.number])` in `filter_numeric_descriptors:784` **excludes SCF_energy entirely**, and any row whose parse failed keeps `''`, which makes `df['SCF_energy'] - df['SCF_energy'].min()` raise or produce garbage. A missing `'SCF_energy'` key also raises an unhandled `KeyError`. Initialize with `np.nan` and cast to float.

### H21. Bare `except:` with no `continue` leaves a stale molecule in scope

`desc_utils/atom_numbering.py:257-258`, `:311-312`, `:350-351`, `:384-385`, `:502-503`, `:528-536`, `:186-188`

```python
except:
    print (f'bad input file for {sdf_path}')
```

Execution falls through to `mol.GetSubstructMatch(...)` / `Chem.RWMol(mol)` using the *previous iteration's* molecule, silently writing another ligand's atom indices into this row (and raising `NameError` on the first iteration). Note also that `SDMolSupplier` does not raise for an unparseable record — it yields `None`, so `mol is None` bypasses the `except` entirely and then raises an uncaught `AttributeError`. Add `continue` and an explicit `if mol is None` check.

### H22. `lig_name` derived by splitting a full filesystem path

`desc_utils/atom_numbering.py:142`

```python
df['lig_name'] = df['xyz_name'].str.split('_').str[0]
```

`xyz_name` values come from `get_ligand_core_type_xyz` line 44 (`remove_path.split('.')[0]`, i.e. `"optimization"`), or from `get_xyz_ids` line 33 which writes absolute paths. Either every row gets the same `lig_name`, or the split happens on the first `_` anywhere in the path. Ensemble grouping is wrong either way. `split('.')[0]` also breaks on any path containing a dot.

### H23. Multi-frame XYZ collapses to one conformer

`desc_utils/atom_numbering.py:25-26`

```python
for mol in pybel.readfile("xyz", xyz_path):
    mol.write("sdf", sdf_path, overwrite=True)
```

`overwrite=True` inside the loop rewrites the file each frame, so a trajectory `optimization.xyz` leaves only the last frame on disk. Downstream "ensemble" logic then silently operates on one conformer per directory.

### H24. Substring atom-token matching is ambiguous

`desc_utils/data_extraction.py:571-598`, `:611-656`

```python
atom = col[:-1]
dict_atoms[row.Index][col] = atom + str(number)
...
if all(bk in subkey for bk in bond_key):
```

Token `N1` matches any descriptor name containing `N12`, so one ligand's descriptor can be mapped onto another atom's label. `make_dict_atoms` also mangles any column merely *starting with* `H1`/`C1`/`Ni` (e.g. `C1_boltz`, `Ni_charge`) via `col[:-1]`. Use anchored regex with non-alphanumeric boundaries, as `remove_atom_labeled_columns` (line 693) already does correctly.

### H25. Data-dependent column renaming makes column sets non-comparable across runs

`desc_utils/data_extraction.py:526-562`

```python
non_nan_per_row = df_out[members].notna().sum(axis=1)
if non_nan_per_row.max() <= 1:
    _new_cols[base] = df_out[members].bfill(axis=1).iloc[:, 0]
```

Whether `nbo_lp3_H1_energy` survives or is merged into `nbo_lp_H1_energy` depends on which molecules happen to be in the batch. The same molecule run in two different libraries gets different column names, so downstream joins and models silently misalign. An all-NaN group has `max() == 0` and is merged into a useless all-NaN column.

### H26. NBO "HOMO-relative" rank tokens assume `nbo_num` is energy-ordered

`desc_utils/data_extraction.py:141-158` — needs confirmation

NBO numbers orbitals by *type block* (BD, BD\*, CR, LP, RY\*, …), not by energy, so `nbo_num - homo_num` is not an energy rank and the `homo`/`hm{n}`/`lp{n}` tokens are chemically meaningless as HOMO-relative labels. Every `nbo_*`, `bond_orbital_details_*`, `nho_directionality_*` and `second_order_*` column name derives from this. The `>= 0.1` occupancy cutoff is also arbitrary — strongly delocalized valence antibonds routinely exceed it. Worth checking against a known case before trusting these column names.

### H27. `bridge_star_centers` has a guaranteed `KeyError` and a dead sort

`desc_utils/data_extraction.py:1053-1056`

```python
aps_sorted = sorted(list(aps), key=_rank_key, reverse=True)
to_remove_now = set(aps_sorted)
...
for rm in to_remove_now:
    for nb in list(work_neighbors.get(rm, set())):
        work_neighbors[nb].discard(rm)
```

The sort is pointless — the entire articulation-point set is removed, contradicting the comment "Drop the least preferred articulation points first", and iterating until no articulation points remain can strip nearly every descriptor. `work_neighbors[nb]` also raises `KeyError` when `nb` is another member of `to_remove_now` already popped earlier in the loop, which is guaranteed for any component with two adjacent articulation points.

This is the mode the notebook actually uses (`corr_pruning_mode="bridge_star_centers"` in cell 32), so this path matters.

Related, `:944-977`: the articulation-point DFS is recursive and unbounded; a large connected component in the correlation graph will hit `RecursionError`.

### H28. Empty `grep` output treated as converged

`DATA/test_molecules/check_convergence.py:13-17`; `make_downloadable_folder.py:29-34`

```python
conv = subprocess.run(f"grep 'NORMALLY' calcParam.out | wc -l ", capture_output=True, shell=True, text=True).stdout.strip()
if conv == '0':
```

`elif conv > '1'` is a **string** comparison. In `make_downloadable_folder.py`, the `else` branch (which performs the copy) is reached for any value that is neither `'0'` nor string-greater than `'1'`, including an empty string from a failed pipe — so unconverged or absent results get archived as good.

### H29. `check=True` on a dozen `cp` calls aborts the archiving run mid-tree

`DATA/test_molecules/make_downloadable_folder.py:40-57`

Any conformer missing any one of the 12 files kills the script. Worse, lines 54-57 sit *inside* the `for c` loop after `os.chdir('..')`, so the crest files are re-copied once per conformer, and line 56 targets a directory that is only created inside the `else` branch — so for any non-converged conformer the destination does not exist and `check=True` raises.

### H30. Cluster scripts that cannot run as written

- `scripts-for-orca-computations/launch_orca_all.py:6`: `subprocess.run(["module", "load", "openbabel"], check=True)` — `module` is a shell function, not an executable, so this raises `FileNotFoundError` immediately. Even with `shell=True` it could not modify the parent environment, so the subsequent `obabel` call would still fail.
- `DATA/test_molecules/launch_orca_all.py:9`, `:47`: `--ewin` is `type=str` but compared with `>` against a float, so any explicit `--ewin 5` raises `TypeError`. Only the un-overridden `default=8` works.
- `DATA/test_molecules/launch_orca_all.py:29-31`: `energies[f] = np.nan` with `numpy` never imported — the recovery path raises `NameError` and aborts the whole submission run.
- `DATA/test_molecules/check_num_confs.py:9`: globs `conformer*/*xyz` while every other script uses `conf_*`, so the reported count is always 0.

### H31. `cut -c 1-57` can silently truncate a coordinate

`DATA/test_molecules/launch_orca_all.py:70`

```python
subprocess.run("cut -c 1-57 optimized_uma-s-1*.xyz > input.xyz", shell=True, check=True)
```

Fixed-width truncation of ASE extxyz. A wide row (three negative coordinates, a two-letter element symbol, or >99 atoms shifting the layout) has its z coordinate cut mid-number — which still parses as a valid float, producing a silently wrong geometry. Parse the file properly instead.

### H32. `os.system` chain destroys the only geometry when xtb fails

`DATA/test_molecules/make_xyz_and_create_folder.py:49-52`

```python
os.system("xtb mol.xyz --opt --gfn 2")
os.system(f"mv xtbopt.xyz {output}")
os.system("rm mol.xyz")
```

Return codes ignored: if xtb fails, `mv` fails and `rm mol.xyz` still deletes the input, leaving an empty folder that downstream scripts treat as "already generated" (`:26`). `{output}` is also unquoted.

---

## Medium

### M1. Four methods defined twice in `OrcaReader`

`desc_utils/orca_descriptors.py`: `_atom_key` at 128 and 171; `set_outfile` at 133 and 179; `set_nbo_block_index` at 144 and 190; `set_nbo_job_number` at 155 and 201. The second definition silently wins. The two `_atom_key` bodies differ subtly — the first uses `.get(atom_index, str(atom_index))`, the second tests label truthiness — so an empty-string label behaves differently depending on which one you thought you were reading. Delete the duplicates.

### M2. `_clear_nbo_data` clears keys that are never written

`desc_utils/orca_descriptors.py:163-168`

```python
nbo_keys = [k for k in self.results.keys() if k.startswith("nbo_charge_")]
```

The NBO charge keys are written as `npa_<label>_charge` (per the README and `RedoxSet._pull_charge` line 1029), so this clear is a no-op and stale NBO charges survive a `set_outfile`/`set_nbo_job_number` change.

### M3. HOMO/LUMO occupancy filtering can misalign energies and occupancies

`desc_utils/orca_descriptors.py:408-409`

```python
mo_energies = np.array([e for e in mo_energies if e is not None])
mo_occupancy = np.array([o for o in mo_occupancy if o is not None])
```

The two lists are filtered independently, so a single MO with a present energy and a missing occupancy shifts every subsequent pairing. Filter as pairs. Separately, `homo_mask = (mo_occupancy == 1.0) | (mo_occupancy == 2.0)` (line 416) uses exact float equality and will miss any fractional occupancy (finite-temperature smearing, fractional-occupation DFT), silently returning `None` for HOMO.

### M4. Cached `xyz.json` reuse gated on `len(results) > 1`, never validated against the atom map

`desc_utils/data_extraction.py:272-278`

```python
if len(P.results) > 1 and not overwrite_json:
    if P.results is not None:
```

A partially written JSON with as few as two keys is accepted as complete, so descriptors are silently missing. The cached results are also never checked against the current `atom_map` — if atom numbering was recomputed, stale descriptors keyed to the old numbering are reused. (The inner `is not None` check is dead, and its `else` branch unreachable.)

### M5. `_normalize_nbo_block_map` heuristic silently shifts valid 0-based input

`desc_utils/data_extraction.py:362-372`

```python
if vals and all(int(v) >= 1 for v in vals) and all(int(v) != 0 for v in vals):
    return {k: int(v) - 1 for k, v in normalized.items()}
```

A legitimate 0-based map that happens not to include block 0 (e.g. `{"oxidation": 1, "reduction": 2}`) is decremented, reading the wrong NBO blocks with no warning. The intent is unrecoverable from the values alone — make the convention an explicit parameter.

### M6. All-NaN and single-row columns reported as "zero std"

`desc_utils/data_extraction.py:886-889`

`std` of an all-NaN column, or of any column when `df` has one row, is NaN; `NaN > 0` is False, so the column is dropped and mislabelled as zero-variance. With a single compound in `df`, *every* descriptor is dropped and the report says they had zero variance.

Related, `:894`: columns whose mean is exactly 0 are mapped to NaN and then **kept**, so the std/mean ratio rule is skipped precisely for the columns most likely to be noise around zero.

### M7. Correlation filter over-drops via transitive chains and ignores relevance to y

`mlr_utils/feature_selection.py:20`

```python
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
```

All drops are computed in one pass against the original matrix. If A~B and B~C but A≁C, both B and C are dropped even though C carried independent information. Correct behaviour requires iterative removal with recomputation. The survivor is also chosen by *column order*, not by correlation with the target, so which of two near-duplicate descriptors survives is an artifact of ordering — which matters directly given C1 (every `cm5_*`/`hirshfeld_*` pair is perfectly correlated). Constant columns give NaN correlations that never trip the `>` test and are silently retained.

### M8. No guard on features-per-sample; no complexity penalty

`mlr_utils/feature_selection.py:259-261`, `:364-366`

The only bound on `k` is the number of *features*, not the number of *training samples*. With the default `n_features_to_select=[1..5]` and a 15-20 compound dataset, `k=5` leaves very few residual degrees of freedom, and `top` is chosen across all `k` purely by `val_score` with no penalty. Recommend enforcing something like `k <= n_train/5` and reporting adjusted R² or an information criterion.

### M9. `max_combinations` is a no-op; `top_k > 1` silently changes the algorithm

`mlr_utils/feature_selection.py:300-306`, `:369-375`, `:298-332`

`max_combinations` only issues a `UserWarning` and then evaluates everything anyway, contradicting its docstring. And `top_k > 1` in `'sfs'` mode silently switches to a full exhaustive scan of `C(n_features, k)` — a different and far more selection-biased experiment than the mode name implies. The SFS-chosen set is also present in the exhaustive combinations, so `history` contains it twice (with possibly *different* scores, per H2), and `top` can return the same model twice.

### M10. Kennard-Stone distances on unstandardized descriptors, and an off-by-one

`mlr_utils/feature_selection.py:110-119`

```python
dist = pairwise_distances(X.values, metric='euclidean')
...
selected = [i0, j0]
while len(selected) < test_count and available:
```

Descriptor columns span wildly different units and magnitudes (volumes vs NBO charges), so the Euclidean distance — and hence the entire KS design — is dominated by whichever descriptor has the largest numeric range. `Pipeline` is imported at line 11 and never used; there is no `StandardScaler` anywhere in the module.

Also, `selected` is seeded with two points, so for `test_count <= 2` the loop never runs and the test set has exactly 2 members regardless of `test_size`. Note the KS-selected (maximally spread) points go to *test*, not train, inverting the usual convention — needs confirmation whether that is an intentional extrapolation design.

### M11. Reported coefficients are unstandardized but presented as importances

`mlr_utils/feature_selection.py:38-40` and its docstring at `:25` ("feature importance coefficients")

Raw OLS coefficients on unscaled descriptors are not comparable in magnitude. If these are used to rank descriptor importance, the ranking reflects units, not influence.

### M12. Unknown elements silently drop atoms and misalign every per-atom array

`orca2gaussian/orca_to_gaussian_log.py:8-23`, `:218`, `:294`

```python
atomic_numbers = [SYMBOL_TO_Z.get(symbol, 0) for symbol, _, _, _ in coords]
```

The table omits Fe, Cu, Zn, Pd, Mg, Li, Se, Sn and others. `Z=0` is written into the standard-orientation block; DescriPyTor maps it to NaN and then drops the row, so the atom count shrinks while charge and NMR arrays do not — shifting every per-atom descriptor. Raise on unknown symbols instead.

Related, `:38-51`: `_to_float`/`_to_int` convert malformed data to 0.0, which combined with the `[0.0]*n_atoms` padding at `:512-513`, `:589-591`, `:725` turns a missing population-analysis block into a full column of physically plausible zero charges rather than an error.

### M13. Non-converged optimizations silently fall back to the last geometry

`orca2gaussian/orca_to_gaussian_log.py:212-215`, `:235-238`

```python
if marker_idx >= 0:
    chosen = next((item for item in coords_blocks if item[0] > marker_idx), coords_blocks[-1])
else:
    chosen = coords_blocks[-1]
```

If `FINAL ENERGY EVALUATION AT THE STATIONARY POINT` is absent (opt failed or job killed), the last printed geometry is used with no warning — and per C3 that is the geometry DescriPyTor actually parses.

### M14. `convert_tree` output path collision via `with_suffix`

`orca2gaussian/orca_to_gaussian_log.py:780`

```python
output_log = output_root / rel.with_suffix(".log")
```

`with_suffix` replaces everything after the last dot in the final component, so `conf_1.2/` becomes `conf_1.log`, colliding with `conf_1/`. Use `rel.parent / (rel.name + ".log")`.

Related, `orca2gaussian/orca_2_gauss.py:23`: `Path("orca2gaussian/orca_to_gaussian_log.py").resolve()` only works from the repo root; use `Path(__file__).with_name(...)`.

### M15. `df.loc[row.Index, ...]` under `itertuples`, results keyed by index

`desc_utils/data_extraction.py:269-270`, `:336`, `:385-386`, `:433`

With a duplicated index label, `df.loc[label, col]` returns a Series (so `int(...)` raises) and `results_dict[row.Index]` overwrites, silently losing rows. `int(df.loc[..., atom])` also raises `ValueError` on a NaN atom index without reporting which molecule failed. Present in `get_xyz_descriptors`, `get_redox_descriptors`, `get_all_charges`, `get_nbo_descriptors` and `get_all_nbo_properties`. Add a `reset_index` guard.

### M16. Nothing validates that the atom-index dataframe is 1-based

`desc_utils/data_extraction.py:284`, `:292`, `:302-304`, `:314-318`, `:327`

`PropertiesXYZ` is 1-based throughout (`orca_descriptors.py:1147`, `:1447`) and morfeus `BuriedVolume`/`Sterimol`/`SASA.atom_areas` are 1-indexed, but nothing checks the incoming column values. A dataframe built with `compile_final_nn1(..., indexing=0)` silently produces descriptors for neighbouring atoms. Given C13 and C14 leave *mixed* conventions in one dataframe, this will not fail loudly. Add an assertion that indices are in `[1, n_atoms]` and that the element at each index matches the label's element symbol.

### M17. Duplicate atom indices silently collapse labels

`desc_utils/data_extraction.py:106-107`

```python
atom_index_to_label = {v: k for k, v in atom_index_map.items()}
```

If two labels resolve to the same atom index — a real possibility given the numbering bugs above — the inverted dict keeps only the last label. Every NBO descriptor for that atom is emitted under one label and the other label's columns are missing, with no warning.

### M18. Silent no-op when two numbering sets are not a clean N1/N2 swap

`desc_utils/atom_numbering.py:147-154`

If there are exactly two differing numbering sets but they are not a pure N1/N2 swap, neither the swap branch nor the `len > 2` interactive branch fires — the inconsistency is left in place with no warning. The swap branch also resolves ties by `iloc[0]`, i.e. by row order, so the N1/N2 convention across ligands is arbitrary (later only partly fixed for pyox).

Related, `:130-139`: no `else` branch, so if no ligand matched any SMARTS you get `UnboundLocalError` on `results`. And `:177-222`: `verify_atom_numbering` only prints mismatches and returns `None`; both callers (592, 624) ignore the result, so a detected numbering error does not stop the pipeline.

### M19. `GetSubstructMatch` (singular) silently takes the first pyox mapping

`desc_utils/atom_numbering.py:314`

```python
sub_structure = mol.GetSubstructMatch(pyox_structure)
```

No `useChirality`, no check for multiple matches, no canonical-rank tie-break. Where the SMARTS maps onto the molecule in more than one way, `N1`/`N2`/`C1`/`C2` come from whichever mapping RDKit returns first — not guaranteed stable across RDKit versions or across conformers of the same ligand.

### M20. Aliasing makes the 1-based conversion conditionally vanish

`desc_utils/atom_numbering.py:596-597`, `:606`; same defect at `:656-660`

```python
df_with_smiles = get_smiles(df_pyox_corrected, path)
df_with_energy = get_scf_energy(df_with_smiles)
if indexing == 1:
    df_with_smiles['H1'] = df_with_smiles['H1'] + 1
df_final = df_with_energy
```

The increments are applied to `df_with_smiles` but `df_final` is `df_with_energy`. This only works because `get_scf_energy` usually returns the same object — but when `'SCF_energy'` is already present it does `df = df.drop(columns=['SCF_energy'])` (line 541), returning a **copy**, and the entire 1-based conversion is silently lost.

### M21. Stray `C3` NaN column defeats `drop_duplicates`

`desc_utils/atom_numbering.py:88` vs `:102`, `:131`

The core_5 retry branch adds a `C3` key that the primary branch does not, so `results5` gains a mostly-NaN `C3` column that is never dropped (only `results6`'s is). That NaN column then participates in `drop_duplicates()` at line 145, where `NaN != NaN` makes otherwise identical rows look distinct.

### M22. No-Ni and ambiguous-ring-size molecules crash or vanish

`desc_utils/atom_numbering.py:53-63`

```python
Ni_atom = [atom for atom in original_mol.GetAtoms() if atom.GetAtomicNum() == 28]
Ni_atom_idx = Ni_atom[0].GetIdx()
```

`IndexError` if no Ni is present. And a Ni in both a 5- and a 6-membered ring (or in neither) matches neither branch at 58/61, so the molecule is dropped from `mols_5`, `mols_6` *and* `results` with no message.

### M23. The `inchi` column is not an InChI

`desc_utils/atom_numbering.py:511-512`, `:533-534`

```python
inchi = os.path.basename(os.path.dirname(os.path.normpath(path)))
```

This is a directory name (an InChIKey at best), later surfaced as `"INCHI"` in `descriptor_stats:1149`. Rename it, or compute the real InChI from the molecule.

Related, `:506-513`: after removing Ni, the first polyatomic fragment wins — so any complex with a bridging solvent, counter-ion or second ligand silently records the wrong SMILES, since `GetMolFrags` order is not a chemically meaningful ranking.

### M24. Column set differs between molecules on the multi-SMILES path

`desc_utils/data_extraction.py:1147-1151`

The failure branch still *returns* the dataframe, just without `SMILES`/`INCHI`, so concatenating results across molecules produces misaligned identifier columns. Since `get_smiles` leaves `''` on failure, one bad conformer makes `nunique() == 2` and trips this branch.

### M25. Non-atom-mapped descriptors dropped when they span a selected and an unselected atom

`desc_utils/data_extraction.py:163`, `:179`, `:215`, `:239`

```python
if not atoms_list or not all(atom in selected_indices for atom in atoms_list):
    continue
```

Requiring *all* atoms to be selected discards e.g. every Ni-to-substituent bond orbital. Probably intended, but it means the descriptor set depends on the `atoms` list in a non-obvious way and nothing records what was skipped.

### M26. `descriptor_stats` mutates the caller's dataframe and can raise on narrow frames

`desc_utils/data_extraction.py:1124-1127`

```python
df.insert(5, 'Relative_energy', df['SCF_energy'] - df['SCF_energy'].min())
```

In-place insert on the argument; `insert(5, ...)` also raises `IndexError` if `df` has fewer than 5 columns.

### M27. `_lowE` selection breaks on a duplicated index

`make_descriptor_descripytor.ipynb` cell 29

```python
idx_low = e.idxmin()
out[f"{col}_lowE"] = g.loc[idx_low, col] if idx_low in g.index else np.nan
```

If the outer merge produced a duplicated index, `g.loc[idx_low, col]` returns a Series rather than a scalar. More importantly, the merge at the top of that cell is `how="outer"` on `conf_dir` — if either side has duplicate `conf_dir` values, conformers are duplicated in `g` and therefore **double-counted in the Boltzmann average**. Recommend asserting `combined_df["conf_dir"].is_unique` before grouping.

### M28. `print_ml_equation=True` is the default but `coef_` is accessed unguarded

`mlr_utils/plot_utils.py:143`

```python
coefs = model.coef_
```

Unlike `report_mlr_equation`, which checks `hasattr`, this raises `AttributeError` for any tree or kernel model — after the figure has been built and all the CV work done.

### M29. `sed` substitution treats the filename as a regex

`scripts-for-orca-computations/launch_orca_all.py:29`

```python
subprocess.run(["sed", "-i", f"s/inp.xyz/{f}/g", "calcParam.inp"], check=True)
```

`inp.xyz` matches `inpXxyz`; a `/` in `f` breaks the delimiter; `sed -i` without an argument fails on BSD/macOS. Line 30 also passes a stray `"."` as an argument to `sbatch`.

### M30. Silent element substitution changes the molecule relative to its directory key

`DATA/test_molecules/make_xyz_and_create_folder.py:43-44`

```python
if atom.GetSymbol() == 'Be':
    f.write(f"H {pos.x:.6f} ...")
```

The written molecule differs from the SMILES printed at line 36 and from the InChIKey used as the directory name (line 23), so the directory key no longer identifies the structure that was computed. Also `AllChem.EmbedMolecule(mol)` at line 20 has its `-1` failure return ignored (so `mol.GetConformer()` raises on embedding failure) and uses no seed or ETKDG, making geometries non-reproducible.

### M31. `min()` over a dict that may contain NaN

`DATA/test_molecules/launch_orca_all.py:33`

```python
min_e = min(energies.values())
```

If any energy is NaN, `min` can return NaN depending on iteration order, making every relative energy NaN so all conformers are skipped by the guard at line 44 with only per-conformer prints. Raises `ValueError` if `energies` is empty. (The `23.061` factor at line 37 is correct for ASE's eV output; it would be wrong — should be 627.51 — if the source were ever switched to Hartree.)

---

## Low

- **`desc_utils/orca_descriptors.py:338-376`** — `get_saniso` and `get_siso` are byte-identical apart from one string literal. Collapse into one parameterized method.
- **`desc_utils/orca_descriptors.py:232`, `:244`, `:1104`** — `json.load(open(...))` leaks file handles; use a context manager.
- **`desc_utils/orca_descriptors.py:230`, `:242`** — `os.listdir(self.directory)` raises `FileNotFoundError` if the directory is missing, rather than reporting it like the missing-file case does. Use `Path.exists()`.
- **`desc_utils/orca_descriptors.py:1093-1095`** — `path.endswith('/')` assumes `str`; a `Path` argument raises `AttributeError`. Same for `self.path + 'optimization.xyz'`.
- **`desc_utils/orca_descriptors.py:1110`** — `self.results['path'] = self.path` puts a filesystem path into the descriptor namespace, which then flows into the JSON and the dataframe.
- **`desc_utils/orca_descriptors.py:1526-1573`** — quadrant/octant key ordering (`bv_quadrants[1..4]`, `bv_octants[0..7]`) matches morfeus's `QUADRANT_SIGNS`/`OCTANT_SIGNS` as I remember them, but I could not verify offline. Worth one assertion against a known-asymmetric test case, since a mislabelled quadrant is invisible in the numbers.
- **`desc_utils/orca_descriptors.py:573`** — `NBOProps.get_NPA` writes integer atom keys, which become *strings* after a JSON round-trip. Confirm that downstream lookups in `data_extraction.py` do not index with ints.
- **`desc_utils/orca_descriptors.py:992`** — `descriptorPrefixes=["cm5"]` is a mutable default argument. Not mutated today, but use `None` plus the existing `or` fallback.
- **`desc_utils/data_extraction.py:49-67`** — `_flatten_nested_results` is dead code, and the only place that maps empty values to `np.nan`.
- **`desc_utils/data_extraction.py:83-84`, `:93`** — the atom-number regex `([A-Z][a-z]?)\s+(\d+)` requires whitespace, so an NBO field formatted `C1`/`N12` yields no matches and the entry is silently skipped at line 239. `[A-Z][a-z]?` also matches the `C` of `CR`/`BD` type tokens. Needs confirmation against real `NBOProps` output.
- **`desc_utils/data_extraction.py:851`, `:862`** — unanchored substring filters (`"aniso" in c`, `"phi" in c`) will drop unrelated descriptors containing those letters.
- **`desc_utils/data_extraction.py:663-667`** — `df.at` accumulation writes `None` (not `np.nan`) into freshly enlarged columns, making them `object` dtype so `select_dtypes(include=[np.number])` excludes them entirely rather than treating them as NaN. Column presence also depends on iteration order.
- **`desc_utils/atom_numbering.py:165`** — `int(input(...))` is unguarded and blocks any batch run.
- **`desc_utils/atom_numbering.py:359`, `:393`, `:398`** — `if x is True` breaks silently if the value ever becomes `numpy.bool_`.
- **`desc_utils/atom_numbering.py:143-173`** — writes into `df` while iterating `df.groupby(...)`; safe only because the prefix groups happen to be disjoint. Needs confirmation that no `lig_name` is a prefix of another.
- **`mlr_utils/feature_selection.py:250`** — `if top_k < 1 or not isinstance(top_k, int)` compares before type-checking, so a non-numeric `top_k` raises `TypeError` instead of `ValueError`; `isinstance(True, int)` is `True`, so `top_k=True` is accepted as 1.
- **`mlr_utils/feature_selection.py:426-428`** — dead branch: `model` was already replaced with `LinearRegression()` at 226-227.
- **`mlr_utils/feature_selection.py:223-224`** — no NaN/inf validation on `X_numeric` or `y`; NaNs surface as an opaque sklearn error deep inside a tqdm loop over thousands of combinations.
- **`mlr_utils/feature_selection.py:260-261`, `:365-366`** — if every requested `k` is skipped, the function returns a well-formed dict with `'selected_features': []` and no error.
- **`mlr_utils/plot_utils.py:180`, `:193`, `:216`** — `except Exception: pass` in the structure-rendering helpers; a systematically failing InChI parse looks like missing data. Line 190 only attempts conversion when the string starts with `InChI=`, so prefix-less InChIs silently yield `''`.
- **`orca2gaussian/orca_to_gaussian_log.py:327-330`** — the `DFT_Energy` fallback is evaluated eagerly and assumes a dict, while every sibling block in ORCA's property.json is a *list* of dicts. Version-fragile, and an `AttributeError` here is uncaught.
- **`orca2gaussian/orca_to_gaussian_log.py:492-497`, `:604-605`, `:644-646`, `:661-664`** — placeholder blocks written unconditionally produce a structurally valid log that parses to nothing, rather than an error.
- **All of `check_convergence.py`, `check_num_confs.py`, `launch_orca_all.py`, `relauncn_all_orcas.py`, `make_downloadable_folder.py`** — walk the tree with `os.chdir` and no `try/finally`, so any exception leaves the process in an arbitrary directory. Dangerous given the `rm`/`cp` commands run on relative paths.
- **Hardcoded values** — `crest.py:7` `sys.path.append('/resnick/groups/ReismanLab/...')`; `crest.py:18` and `uma_opt_confs.py:49` hardcode `charge=0, spin=1` (wrong for the charged Ni complexes in `DATA`, and the latter clobbers any info read from the file); `make_downloadable_folder.py:10` embeds the date literal `'_to_download_2026_04_08'`; `uma_opt_confs.py:56` `dyn.run(fmax=0.05)` with no `steps=` cap.
- **`crest.py:19`** — calls `job.CREST("output.xyz")`, but `output.xyz` only exists when `--optimize` was set; otherwise the file is `mol.xyz`.

---

## Repository health

Not code correctness, but relevant to reproducing any of the above.

**`.git` is 19 GB (`size-pack: 18.98 GiB`) for 161 tracked files.** Large calculation outputs and `ROBERT` artifacts have been committed. Cloning this repo is impractical, and the README's install instructions start with `git clone`. Consider `git-filter-repo` to excise the large blobs, and move `DATA`/`ROBERT` outputs to Zenodo or an institutional store with a pointer file in the repo.

**`DescriPyTor` is vendored as a plain directory, not a submodule or a pinned dependency.** `orca2gaussian/orca_to_gaussian_log.py` is written against a specific version of `feather_extractor.py`'s parsing behaviour — findings C6, C7, H9 and H11 are all cases where the converter and the parser are coupled through undocumented last-occurrence and regex conventions. An upstream change to DescriPyTor will silently alter your descriptors. Either add it as a submodule pinned to a commit, or `pip install` it from a pinned git ref in the README's install block, and add a round-trip test that asserts a known molecule's parsed descriptors.

**No tests.** Given how many of the bugs above are silent (wrong-by-a-constant-factor, wrong atom, empty column that still parses), a small set of golden-file tests would have caught most of them. The highest-value ones, in order:

1. A known molecule (methanol is already in `DATA`) through `orca_to_gaussian_log.py` → DescriPyTor, asserting bond lengths in Å, dipole in Debye, the number of real frequencies (`3N-6`), and that no descriptor column is entirely NaN.
2. `cm5_* != hirshfeld_*` for at least one atom.
3. `Vertical_Ionization_Potential > 0` and `Hardness > 0`.
4. Atom-numbering round-trip: for each ligand class, assert the element at each labelled index matches the label (`N1` is an N, `C1` is a C, `Ni` is Ni). This single check catches C13, C14, C15, M16 and M17 at once.
5. Boltzmann weights sum to 1 and match a hand-computed two-conformer case.

**One-line assertion worth adding now**, before anything else, since it is cheap and catches the whole atom-numbering family:

```python
for label, idx in atom_map.items():
    expected = re.match(r"([A-Z][a-z]?)", label).group(1)
    assert P.elements[idx - 1] == expected, f"{label} -> index {idx} is {P.elements[idx-1]}, expected {expected}"
```

---

## Suggested order of work

1. **C2** (IP/EA sign) — one line, and it invalidates five reported descriptors.
2. **The atom-numbering assertion above**, then **C13, C14, C15** — everything downstream depends on the labels being right.
3. **C1** (CM5) — decide whether to implement or remove; either way stop shipping duplicate columns.
4. **C3, C4, C5, C6, C7, H9, H10, H11** (`orca2gaussian`) — these are all in one file and all verified. After fixing, re-extract and diff against the current `combined_descriptors.csv` to see which model features actually move.
5. **H1, H2** (`feature_selection`) — before any statistic from this code goes into a manuscript.
6. **C8, C9, C10** (destructive cluster scripts) — no rush on correctness, but do not run them again as written.
