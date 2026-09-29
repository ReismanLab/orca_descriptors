# Plan: B-CEG-MCD

Benchmark conformer-ensemble generation and refinement, and their impact on the descriptors used to
rationalise reactivity. The requirements are in [README.md](README.md#requirements-project-specification).

Remote: `https://github.com/ReismanLab/B-CEG-MCD.git` (private). Package: `bceg` (`src/bceg/`).

## Stage 1: ensembles (implemented)

Pipeline, for every (transformation, input molecule, method, replicate 1..3):

```
SMILES file ──canonicalise──► InChIKey (depth-2 folder)
            ──transformation (none | NiF2 | NiH2 | user script)──► Structure (SMILES, charge, multiplicity)
            ──start geometry (single-start searchers)──► ETKDG(seed) → GFN2-xTB pre-opt → coordination check
            ──conformer search──► raw_<i>.xyz
            ──post-search optimisation (UMA-S)──► raw_<i>_<lot>.xyz
```

### Decisions (from the project owner)

| topic | decision |
|---|---|
| Input | text file of SMILES, given at submission with the transformation(s): `bceg submit CONFIG --smiles FILE -t NiF2` |
| Transformations | `none`, `NiF2`, `NiH2` (`transforms/ni_x2.py`, adapted from `scripts_4_structure/make_Ni{F2,H2}.py`); other scripts can be added |
| Searchers | CREST, GOAT, molassembler, RDKit ETKDG, LoQI |
| CREST inline steps | `refine: sp` / `refine: opt` (composite workflow); default level UMA (`mlip`, `uma-s-1p2`, task `omol`) |
| Post-search optimisation | user-specified; default UMA-S (`uma-s-1p2`) |
| Replicates | 3 per workflow; each replicate starts from a different geometry |
| Start geometry | pre-optimised (GFN2-xTB); the N-Ni-N angle may vary freely; Ni-N (any metal-ligand bond) ≤ 2.5 Å, otherwise re-embed with a new seed |
| Spin | multiplicity 1 by default; set per transformation (`multiplicity:`) or with `--multiplicity` (folder `<name>_m<mult>`); used by xtb, CREST, GOAT and UMA, and logged |
| Layout | `DATA/<transformation>/<InChIKey of input SMILES>/<method>/raw_{1,2,3}.xyz` + `raw_{1,2,3}_<lot>.xyz` |
| Hyper-parameter log | `params.json` / `params_<lot>.json` per method folder; exact match → skip; mismatch → refuse, rename the method |
| Cluster | SLURM + GPUs; each job runs search, then post-optimisation (`mode: chained`); `mode: split` is available |
| Data | `DATA/` exists in the repo, its contents are git-ignored |

### Implementation notes

- Atom order in every xyz file is that of `AddHs(MolFromSmiles(<transformed SMILES>))`. Energies are in Hartree on the comment line.
- RDKit UFF collapses dative Ni complexes (H-H in NiH2), so the `auto` force field never uses UFF.
- Raw ETKDG places one Ni-F at ~2.6 Å in most NiF2 embeddings, so the xtb pre-optimisation is required
  (`start_preopt: gfn2`, the default).
- LoQI is organics only, so for Ni complexes it is recorded as `not_applicable`.
- `bceg submit` freezes the resolved config in `DATA/_jobs/<timestamp>/config.yaml`; the jobs read that copy.
- A replicate count change is recorded but does not count as a hyper-parameter change.

### To verify on the first cluster run

Suggested test: 1 ligand × NiF2 × {crest_gfn2, crest_gfn2+uma-opt, molassembler} × 3 replicates.

- [ ] xtb pre-optimisation of NiF2/NiH2 complexes passes the coordination check (see `run_<i>.json` → `details.start_geometry`)
- [ ] CREST 3.1 accepts the generated TOML (`logs/raw_<i>_input.toml`), and `fmlip-relay-server` is found for UMA inline steps
- [ ] GOAT writes `goat.finalensemble.xyz` with the energy as the first float of each comment line
- [ ] molassembler Python API calls (`interpret.molecules`, `dg.generate_ensemble`) match the installed version
- [ ] LoQI CLI options (`--n-confs`, seeding)
- [ ] UMA post-optimisation on GPU (fairchem `pretrained_mlip.get_predict_unit`, charge/spin via `atoms.info`)
- [ ] atom order identical across the 6 xyz files of each method folder; `bceg status` all done

## Open items

- Cluster details: partitions, account, `setup:` lines (conda, ORCA module) in the config.
- Defaults to confirm: CREST workhorse (GFN2), GOAT method (XTB), UMA convergence `fmax` (0.05 eV/Å), metal-ligand cutoff (2.5 Å).
- Should Ni(II) complexes also run as triplets (e.g. `NiF2_m3`) as part of the benchmark?
- Chained vs split mode: chained jobs hold a GPU even during CPU-only CREST GFN2 searches.
- Method folder naming convention for variants (current: `<searcher>_<level>[+<inline-lot>-<sp|opt>]`).

## Stage 2: descriptors (not started)

Extract reactivity descriptors from the ensembles and evaluate how the ensemble method, the refinement
and the replicate variability affect them. Specification to come.
