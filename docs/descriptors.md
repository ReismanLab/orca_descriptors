# Descriptors provided

Every column the two engines can write, with its spec family and its status. How the columns are produced
and aggregated is in [descriptor_pipeline.md](descriptor_pipeline.md).

Notation: `X` is an atom label from the spec, `Y` a %Vbur radius (default 2.0 to 5.0 Å in 0.5 steps),
`R` an NBO rank token relative to the highest occupied NBO (`homo`, `hm1`, `hm2`, ... below it, `lp1`,
`lp2`, ... above it; kept for the ranks in `homo_lumo_rank_range`). Every per-conformer descriptor
below appears four times in the aggregated CSV (`_min`, `_max`, `_lowE`, `_boltz`).

:white_check_mark: implemented and covered by the regression tests. :hourglass: implemented, not
enabled in the shipped specs. :x: not implemented yet.

## ORCA descriptors

**Fragment specific geometric descriptors** (morfeus on `optimization.xyz`, `descriptors.orca.xyz`)

- %Vbur (Bondi radii × 1.17, H included, sphere radius `Y`): `%Vbur_atom_X_radius_Y` :white_check_mark:
- Atomic SASA: `SASA_X` :white_check_mark:
- Pyramidalization P and P angle: `Pyr_X`, `Pyr_angle_X` :white_check_mark:
- Sterimol (morfeus, CRC van der Waals radii; L axis from first to second label): `Sterimol_X_X_L`, `Sterimol_X_X_Bmin`, `Sterimol_X_X_Bmax` :white_check_mark: / B1 and B5 do not match DescriPyTor `B1_X-X`, `B5_X-X` (different radii and slicing)
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
- CM5: `cm5_X` :x: not implemented yet (the current column is a stub equal to `hirshfeld_X`)
- NBO / NPA charge: `npa_X_charge` :white_check_mark: / matches DescriPyTor `charge_NBO_X`

**Orbital analysis** (NBO 7 block of `calcParam.out`, `descriptors.orca.nbo`)

Open-shell (UHF/UKS) NBO blocks hold two complete analyses, one per spin, and NBO 7 only prints the closing `$CHOOSE` after the beta set. `NBOProps` reads a single spin set per block so every NBO number occurs once: `spin: alpha` (default, the set the parsers always returned), `spin: beta`, or `spin: both` (alpha rows then beta rows, each tagged with a `spin` key; duplicate NBO numbers then get `_2` column suffixes). Closed-shell blocks ignore the option. NPA charges and NAO occupancies always come from the spin-summed tables. `nbo_block_index` selects which NBO block of the output file is read (default 0, the first).

- NPA populations: `npa_X_core`, `npa_X_valence`, `npa_X_rydberg`, `npa_X_total`, `npa_totals_*` :white_check_mark:
- Natural atomic orbital occupancies: `nao_X_Cor-1s_occupancy`, `nao_X_Val-2p_occupancy`, `nao_X_Ryd-3d_occupancy`, ... (`_2`, `_3` suffixes for degenerate sets) :white_check_mark:
- Natural bond orbitals on labelled atoms: `nbo_R_X_X_type`, `nbo_R_X_X_bond_index`, `nbo_R_X_X_occupancy`, `nbo_R_X_X_energy` (lone pairs: `nbo_R_X_*`) :white_check_mark:
- Bond orbital composition: `bond_orbital_details_R_X_X_{bond_type,bond_index,occupancy}`, `bond_orbital_details_R_X_X_contrib_k_X_{percentage,coefficient,s_percent,p_power,p_percent}` :white_check_mark:
- NHO directionality: `nho_directionality_R_X_X_{line_theta,line_phi,hybrid_theta,hybrid_phi,deviation}` :white_check_mark: (φ/θ dropped by the optional filter)
- Second-order perturbation E(2) between labelled orbitals: `second_order_R_<donor>_to_R_<acceptor>_{donor_type,donor_index,acceptor_type,acceptor_index,e2_energy,energy_diff,fock_element}` :white_check_mark: (NBO 7 numbered rows and the legacy `->` layout; only rows whose donor and acceptor atoms are all labelled are kept; both `R` tokens are subject to `homo_lumo_rank_range`, so widen or disable it to keep these columns)
- HOMO and LUMO energies (eV, from the singlepoint MOs): `HOMO_energy`, `LUMO_energy` :white_check_mark:
- Other frontier orbital descriptors (gap, orbital coefficients) :x: not implemented yet

**Redox and conceptual DFT** (vertical `oxidation` / `reduction` steps, `descriptors.orca.redox`)

- Vertical ionization potential and electron affinity (Hartree): `Vertical_Ionization_Potential`, `Vertical_Electron_Affinity` :white_check_mark: (computed as `E(cation) − E(neutral)` and `E(neutral) − E(anion)`, positive for bound electrons)
- Hardness, electronegativity, electrophilicity: `Hardness`, `Electronegativity`, `Electrophilicity` :white_check_mark:

**Condensed Fukui functions** (`Z` = charge scheme listed in `redox.prefixes`: `nbo_charge`, `cm5`, `mulliken`, `loewdin`, `chelpG`, `hirshfeld`)

- `fplus_Z_X`, `fminus_Z_X`, `fzero_Z_X` :white_check_mark: (`Z = cm5` inherits the CM5 stub above)

**NMR** (`nmr.property.json`, GIAO)

- Isotropic shielding: `siso_X` :white_check_mark:
- Anisotropic shielding: `saniso_X` :white_check_mark: (dropped by the optional filter)

**Molecular descriptors**

- SCF energy of the singlepoint (Hartree): `SCF_energy` :white_check_mark: (aggregation weight, not a descriptor)
- Enthalpy, Gibbs free energy: `Enthalpy`, `GibbsFreeEnergy` :x: not implemented yet (read from `singlepoint.property.json`, which has no thermochemistry; needs `frequency.property.json`)
- Polarizability, molecular and atomic dipoles from `%elprop` :x: not implemented yet (computed by ORCA, not extracted)
- Level of theory (`opt_method`, `svp_theory`) :x: not implemented yet (reader exists, not wired to the spec)

## DescriPyTor descriptors

Computed by DescriPyTor `Molecules` from the per-conformer `orca_condensed.json` (`descriptors.descripytor`).
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

Polarizability and IR intensities are stored in `orca_condensed.json` but no DescriPyTor family
consumes them. Enthalpy and Gibbs free energy :x: not implemented yet (they live in
`frequency.property.json`, which neither the readers nor the condenser open for thermochemistry).

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
