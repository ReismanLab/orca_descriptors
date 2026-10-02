"""ORCA stage of a descriptor library: from conformer ensembles (bceg) to one ORCA compound job
per conformer, with the record needed to reproduce the library.

    python -m desc_utils.submit submit  configs/orca_library.yaml [--dry-run] [--force] [--no-sbatch]
    python -m desc_utils.submit status  configs/orca_library.yaml
    python -m desc_utils.submit cost    configs/orca_library.yaml [--csv cost.csv]

Layout of a library (``library_dir``)::

    <LIB>/params.json                 compared settings of the library (the six stage blocks); a change is refused
    <LIB>/molecule_registry.json      InChIKey -> submitted SMILES, bceg source and hyper-parameter hashes
    <LIB>/_jobs/<timestamp>/          frozen config, rendered template, task lists, sbatch scripts, logs
    <LIB>/<InChIKey>/metadata.json    per-molecule record: ensemble, DFT set-up, one status per conformer
    <LIB>/<InChIKey>/conf_N/          input.xyz, calcParam.inp, calcParam.out, orca_condensed.json, ...

The config has one block per stage: transformation, pre_refinement, conformer_search, post_optimization
(run by bceg), pruning, dft (run here). One ``submit`` runs bceg (ensembles) and then, ``afterany`` its
jobs, a ``prepare`` array with one element per molecule that prunes the ensemble, splits it into
``conf_N/`` folders and submits that molecule's ORCA array. ``orca: off`` stops after bceg; the same
command later with ``orca: on`` refines the finished ensembles. The search's replicates
(``conformer_search.replicates``, default 1) are pooled into one ensemble before pruning.
``status`` reports one status per stage (queued / running / done / failed) per molecule; ``cost`` the
core-hours and GPU-hours each molecule has taken per stage (and per ORCA compound step), from the
same records, so a library's accuracy can be weighed against its price.
"""
