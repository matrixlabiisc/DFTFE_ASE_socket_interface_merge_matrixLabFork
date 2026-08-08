# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors.
#
# This file is part of the DFT-FE code.
#
# The DFT-FE code is free software; you can use it, redistribute it, and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 2.1 of the License,
# or (at your option) any later version.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""ASE interface for DFT-FE (socket backend).

Public API::

    from dftfe_ase import DFTFE, read_dftfe

`DFTFE` is an ASE ``Calculator`` that drives a persistent DFT-FE process over a
socket, avoiding per-call MPI startup. The transport lives behind a pluggable
``Backend`` (see :mod:`dftfe_ase.backends`); the wire format is defined once in
:mod:`dftfe_ase.protocol`.

There are two equivalent ways in. Describe the system in Python::

    atoms.calc = DFTFE(xc="GGA-PBE", polynomial_order=7, use_device=True)

or hand it an existing native DFT-FE deck (:mod:`dftfe_ase.io`)::

    atoms, calc = read_dftfe("parameterFileGPU32NodesMPS.prm")

Both land on the same calculation; ``tests/test_prm_roundtrip.py`` asserts it.
"""

from .calculator import DFTFE, DFTFEError, read_prm
from .io import read_dftfe, read_dftfe_atoms
from .protocol import PROTOCOL_VERSION

__all__ = ["DFTFE", "DFTFEError", "PROTOCOL_VERSION",
           "read_dftfe", "read_dftfe_atoms", "read_prm"]
__version__ = "1.0.0.dev0"
