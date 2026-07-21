# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Backend abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """A compute transport for DFT-FE.

    Implementations translate a request dict (geometry + parameters, already in
    DFT-FE's atomic units) into a response dict (energy/forces/stress). The
    calculator owns all unit conversion; backends move data only.
    """

    @abstractmethod
    def compute(self, request: dict) -> dict:
        """Run one calculation and return the response object."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources (process, sockets, files). Idempotent."""

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
