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
| Transformations | `none`, `NiF2`, `NiH2` (`src/bceg/transforms_builtin/ni_x2.py`, adapted from `scripts_4_structure/make_Ni{F2,H2}.py`); other scripts can be added |
| Searchers | CREST, GOAT, molassembler, RDKit ETKDG, LoQI |
| CREST inline steps | `refine: sp` / `refine: opt` (composite workflow); default level UMA (`mlip`, `uma-s-1p2`, task `omol`) |
| Post-search optimisation | user-specified; default UMA-S (`uma-s-1p2`) |
| Replicates | 3 per workflow; each replicate starts from a different geometry |
| Start geometry | new random ETKDG embedding per replicate, full GFN2-xTB pre-optimisation (`start_preopt_level: normal`); Ni-N (any metal-ligand bond) ≤ 2.5 Å, otherwise re-embed with a new seed |
| Bite angle | N-Ni-N varies from ligand to ligand, and should be relatively consistent for a single ligand (across replicates) |
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
- CREST, GOAT and molassembler replicate *i* use the **same** start geometry (same seed, same structure).
  This is deliberate: it gives a paired comparison of the searchers.
- Methods that do not apply (LoQI for metal complexes) create no folder and no task.
- Threads default to `SLURM_CPUS_PER_TASK`. These are resource keys, so changing them never changes `params.json`.
- A failed replicate keeps its working directory (files ≤ 10 MB) in `logs/raw_<i>_failed/`.
- Stored SMILES for the Ni complexes has no explicit C-H hydrogens (only `[H-]` hydrides), e.g.
  `[H-]->[Ni+2]1(<-[H-])<-[n]2ccccc2-c2cccc[n]->12`.
- CREST mlip keys checked against the [docs](https://crest-lab.github.io/crest-docs/page/examples/mlip.html):
  `method="mlip"`, `mlip_backend="uma"`, `mlip_uma_task="omol"`, `mlip_uma_model="uma-s-1p2"`, `mlip_device`.
  This needs CREST ≥ 3.1 and `fmlip-relay-server` on PATH. `uma-s-1p2` is fairchem's id for HuggingFace `uma-s-1.2`.

### Start-geometry diversity (decided)

Measured with real GFN2-xTB (`tests/test_geometry.py::test_real_xtb_preopt_nix2_bpy`, and review runs):

| complex | replicate start RMSD after pre-opt | at Ni |
|---|---|---|
| bpy-NiF2 | 0.01 Å (same minimum, identical energies) | square planar, τ4 0.04, N-Ni-N 85.1° |
| bpy-NiH2 | 0.03-0.21 Å (only the hydrides move) | square planar, τ4 0.06-0.07, N-Ni-N 86.6° |
| tBu-pyox-NiF2 | 0.0-0.4 Å (tBu rotamers) | |
| dtbbpy, tBu-pyox (review) | 0.7-1.3 Å | |

Decisions (owner):
- The N-Ni-N bite angle varies from ligand to ligand, and should be relatively consistent for a single
  ligand. The consistency across replicates seen above is therefore expected.
- **Identical starts are accepted.** For rigid ligands the three replicates may start from the same
  geometry, and any variability comes from the searches themselves (CREST's metadynamics, GOAT,
  Molassembler).
- No looser pre-optimisation (`crude`/`loose` rejected), and no artificial twisting at Ni. The
  pre-optimisation stays a full GFN2 optimisation.
- `run_<i>.json` keeps recording the start RMSDs, the angles at Ni and τ4 for monitoring. Stage 2 can
  check bite-angle consistency per ligand from these records.

### To verify on the first cluster run

Suggested test: 1 ligand × NiF2 × {crest_gfn2, crest_gfn2+uma-opt, molassembler} × 3 replicates.

- [x] xtb pre-optimisation of NiF2/NiH2 complexes passes the coordination check (bpy, tBu-pyox; locally with xtb 6.7.1)
- [x] CREST accepts the generated TOML (`crest --input input.toml`, `charge`/`uhf` keys; CREST 3.0.2 locally, ethanol and NH4+)
- [ ] CREST 3.1 `mlip` level for the inline UMA steps, with `fmlip-relay-server` found
- [ ] **CREST run-to-run variability**: run CREST twice from the *same* start geometry (flexible and
      rigid ligand) and compare `crest_conformers.xyz`. The iMTD-GC workflow has no documented seed
      (only the `ttconf` runtype has `-ttseed`). Since identical starts are accepted for rigid ligands,
      replicate variability relies on CREST not being deterministic. If it is deterministic, the rigid-ligand
      replicates will be identical and this decision needs revisiting. (Not testable locally: the local
      CREST 3.0.2 build fails to optimize the snapshots for 1-hexanol.)
- [ ] Same question for GOAT (does ORCA expose/record a seed?). Molassembler takes the replicate seed explicitly.
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
