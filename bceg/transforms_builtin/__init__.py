"""Transformation scripts shipped with bceg (see README.md in this directory).

They live inside the package so a config can name one as ``script: ni_x2.py``
without knowing where the repository is checked out -- and so they survive a
non-editable ``pip install``. Resolution order is in ``bceg.config._script_path``.
"""
