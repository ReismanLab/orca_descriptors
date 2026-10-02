# The ORCA job and its condensed file

What every conformer folder contains after ORCA, what the descriptors read from it, and how
`orca_condensed.json` carries exactly that to the laptop.

## The ORCA compound job

Every conformer is computed with an ORCA 6.1 compound job, `calcParam.inp`. `orcadesc` renders it per
molecule from [desc_utils/submit/calcParam.inp.tmpl](../desc_utils/submit/calcParam.inp.tmpl) and the
`dft.level_of_theory` block of the library config ([stages/6_dft.md](stages/6_dft.md)), with charge and
multiplicity from the ensemble, and runs ORCA in every `conf_N/`.
[scripts-for-orca-computations/calcParam.inp](../scripts-for-orca-computations/calcParam.inp) is the fixed
input the LNiH2 library was computed with; its level of theory is the table below, including full `def2-TZVP`
on Ni in the optimization and frequency steps (`extra_basis: {Ni: def2-tzvp}` in the YAML). The steps and the
files read downstream are:

| Step (`%base`) | Level of theory | Purpose | Files used by the pipeline |
|---|---|---|---|
| `optimization` | ωB97M-D4 / def2-TZVP(-f), def2-TZVP on Ni | geometry | `optimization.xyz` (geometry for morfeus, numbering) |
| `frequency` | same as optimization | harmonic frequencies, thermochemistry | `calcParam.out` (frequencies, normal modes for DescriPyTor) |
| `singlepoint` | M06-D4 / def2-TZVP, `NBO NPA`, `HIRSHFELD CHELPG`, `%elprop` (dipole, polarizability) | electronic descriptors | `singlepoint.property.json` (energies, charges, MOs, dipole), `calcParam.out` (NBO 7 block 1) |
| `nmr` | M06-D4 / def2-TZVP, GIAO | shieldings | `nmr.property.json` |
| `oxidation`, `reduction` | as `singlepoint`, vertical (N±1 electrons) on the optimized geometry | redox and Fukui descriptors | `oxidation.property.json`, `reduction.property.json`, `calcParam.out` (NBO blocks 2 and 3) |

The conformer search happens before ORCA (stages 1 to 5 of the library config; the fixed LNiH2 input had
an optional GOAT step instead). Solvent is empty by default; NBO requires the `NBOEXE`/`GENEXE` environment
variables. Every file in the last column is read through `orca_condensed.json` once that file exists; the
raw files are then only needed on the cluster.

## Condensing a library

A finished conformer folder holds about 70 MB, of which the pipeline reads about 37 MB: the 32 MB GBW
dump `singlepoint.json` (for two HOMO/LUMO numbers), the 4 MB `calcParam.out` (NBO tables, frequencies)
and the small `.property.json` files. `desc_utils/condense.py` extracts exactly what the readers consume
into one versioned JSON per conformer, `orca_condensed.json` (about 0.4 MB for 37 atoms, 1.5 MB for
90 atoms, most of it the normal modes). It needs only the Python standard library.

Jobs submitted with `orcadesc submit` condense at the end of every ORCA job (`condense: true` in the
library config, recorded per conformer under `condensed`). The whole-library pass below is then only
needed for the conformers whose job died before the end or whose file the check rejects; it skips the
completed files. For a library produced any other way it is the step that follows ORCA:

```bash
cd $LIB                                                            # the log lands where you submit from
sbatch /groups/ReismanLab/orca_descriptors/scripts-for-orca-computations/slurm-condense.sh $LIB  # 8 cores, 2 h; raise the #SBATCH lines for a big library
squeue -u $USER                                                    # wait until the "condense" job is gone
tail -n 4 condense_<jobid>.out   # expected: "... failed 0", "job status: completed N", "checked N file(s): N ok, 0 failed"
```

Every conformer that got past its singlepoint gets a file, finished or not, and the file records the
state of the ORCA job (`job.status`: `completed`, `running` or `failed`). The `job status:` line of the
log counts them: `completed N` alone means the library is done. Re-running the same command rebuilds
the files of the unfinished jobs and skips the completed ones, so it can be repeated after every
resubmission or new batch. A `failed` line names a conformer with nothing to condense yet (no
singlepoint), see [troubleshooting.md](troubleshooting.md).

The same producer runs without SLURM, or on the laptop when the raw files are there:

```bash
# cluster: a plain git clone of this repository, no environment
PYTHONPATH=/path/to/orca_descriptors python -m desc_utils.condense --root /path/to/LIB --jobs 4
PYTHONPATH=/path/to/orca_descriptors python -m desc_utils.condense --root /path/to/LIB --check
# laptop: the same producer driven by the spec
python -m desc_utils.run condense --spec specs/lnih2_pyox.yaml
```

## Transfer

Only the geometry and the condensed file travel, about 0.5 MB per conformer; everything else stays on the
cluster. The library folder lands as `DATA/<LIB>/`: no slash after `<LIB>` on the cluster side, or its
content spills straight into `DATA/`. The folders of jobs that produced nothing yet come along empty, on
purpose: that is how the laptop knows the molecule has conformers still to come.

```bash
rsync -avz --include='*/' --include='optimization.xyz' --include='orca_condensed.json' \
      --exclude='*' <user>@<cluster>:/path/to/<LIB>  DATA/
```

Add `--include='input.xyz'` to describe the UMA geometries without ORCA
([descriptor_pipeline.md](descriptor_pipeline.md#geometry-only-descriptors-from-uma-structures)).

## Content of `orca_condensed.json`

Schema version 1; units in the key names; `null` when ORCA did not print a quantity.

| Section | Content | Source |
|---|---|---|
| `molecule` | atoms with Å coordinates, charge, multiplicity | `optimization.xyz` |
| `states.<singlepoint / oxidation / reduction / nmr>` | SCF energy, electron counts, Mulliken / Loewdin / CHELPG / Hirshfeld charges, dipole (a.u.), polarizability, thermochemistry, NMR `siso` / `saniso`, MO energies and occupations (`singlepoint` only by default, `--mo-states`) | `<state>.property.json`, `<state>.json` |
| `nbo.blocks[]` | `NBOProps.results` per NBO block: NPA, NAO occupancies, NBO summary, hybrids, NHO directionality, second-order perturbation, cycle info. Block 0 (neutral) in full; the cation and anion blocks keep NPA only unless `--nbo-full all` | `calcParam.out` |
| `properties`, `vibrations`, `charges` | DescriPyTor's JSON schema: dipole in Debye (singlepoint), polarizability, frequencies with unit-norm displacements per mode (translations and rotations dropped, IR intensities where printed), NPA / Hirshfeld / CM5 charges | derived from the above |
| `producer` | ORCA and NBO versions, host, time, git commit, size and mtime of every input file | |
| `job` | `status` (`completed`, `running`, `failed`), `detail` (ORCA error line or SLURM kill reason), `checked_utc` | `calcParam.out` termination lines, `error_<jobid>.txt` |

Every reader (`OrcaReader`, the NBO loader, `scf_energy` of the numbering stage, DescriPyTor's
`Molecules`) uses the condensed file when `<conf>/<data.condensed_file>` exists and the raw files
otherwise, so folders downloaded before this step keep working; `data.condensed_file: null` forces the
raw files. `python -m desc_utils.condense --check` validates existing files, and `tests/test_condense.py`
asserts that every in-house column is identical whichever source is read.

A file is written as soon as the singlepoint and its NBO block exist, so a job that died or is still
running afterwards gets one too, with the missing states as `null` and `job.status` saying so. The
condenser only skips files whose status is `completed`; any other stored status is rebuilt from the
raw files when they are there (`tests/test_job_status.py`). The extract stage copies the status into
`orca_job_status` and counts the conformer folders of the molecule (`n_conformer_folders`, empty
folders included); the aggregation turns the two into `n_conformers_completed` and
`all_conformers_completed`. Descriptors of unfinished conformers are still aggregated: their missing
values are simply absent from the min / max / Boltzmann statistics, as any NaN is.
