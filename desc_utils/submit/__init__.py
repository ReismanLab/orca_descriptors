"""ORCA stage of a descriptor library: from conformer ensembles (bceg) to one ORCA compound job
per conformer, with the record needed to reproduce the library.

    python -m desc_utils.submit submit  configs/orca_library.yaml [--dry-run] [--force] [--no-sbatch]
    python -m desc_utils.submit status  configs/orca_library.yaml

Layout of a library (``library_dir``)::

    <LIB>/params.json                 compared settings of the library (source + dft); a change is refused
    <LIB>/molecule_registry.json      InChIKey -> submitted SMILES, bceg source and hyper-parameter hashes
    <LIB>/_jobs/<timestamp>/          frozen config, rendered template, task lists, sbatch scripts, logs
    <LIB>/<InChIKey>/metadata.json    per-molecule record: ensemble, DFT set-up, one status per conformer
    <LIB>/<InChIKey>/conf_N/          input.xyz, calcParam.inp, calcParam.out, orca_condensed.json, ...

One ``submit`` runs bceg (ensembles) and then, ``afterany`` its jobs, a ``prepare`` array with one
element per molecule that splits the ensemble into ``conf_N/`` folders, applies the energy window and
submits that molecule's ORCA array. ``orca: off`` stops after bceg; the same command later with
``orca: on`` refines the finished ensembles. Only replicate 1 of an ensemble is refined.
"""
