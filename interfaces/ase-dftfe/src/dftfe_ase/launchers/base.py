# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Launcher abstract base class.

A Launcher knows *how* to start ``nproc`` MPI ranks of a given binary on this
machine (srun / mpirun / mpiexec / jsrun / direct). It builds an argv list; the
socket backend appends ``--socket host:port`` and spawns it (shell-free).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence


class Launcher(ABC):
    name = "base"

    def __init__(
        self,
        gpus_per_task: Optional[int] = None,
        extra_args: Sequence[str] = (),
    ):
        self.gpus_per_task = gpus_per_task
        self.extra_args = list(extra_args)

    @classmethod
    def detect(cls) -> bool:
        """True if this launcher's environment is active (for auto-selection)."""
        return False

    def default_nproc(self) -> Optional[int]:
        """Rank count inferred from the allocation, or None if unknown."""
        return None

    @abstractmethod
    def build_command(self, binary: str, nproc: int) -> List[str]:
        """Return the argv that launches ``nproc`` ranks of ``binary``."""
