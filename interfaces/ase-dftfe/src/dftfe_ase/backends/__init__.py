# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Pluggable compute backends for the DFT-FE ASE calculator.

`SocketBackend` is the primary (and currently only) backend. The `Backend` ABC
defines the seam so an in-process (pybind11) or i-PI-protocol backend can be
added later without changing the `DFTFE` calculator.
"""

from .base import Backend
from .socket import SocketBackend, DFTFEError

__all__ = ["Backend", "SocketBackend", "DFTFEError"]
