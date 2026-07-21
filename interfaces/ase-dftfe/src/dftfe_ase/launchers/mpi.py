# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Generic MPI launchers: mpirun (OpenMPI/MPICH) and mpiexec (PBS/ALCF)."""

from __future__ import annotations

import os
from typing import List, Optional

from .base import Launcher


class MpirunLauncher(Launcher):
    name = "mpirun"

    def build_command(self, binary: str, nproc: int) -> List[str]:
        return ["mpirun", "-np", str(nproc)] + self.extra_args + [binary]


class MpiexecLauncher(Launcher):
    """mpiexec — the launcher inside a PBS job on ALCF Polaris/Aurora."""

    name = "mpiexec"

    @classmethod
    def detect(cls) -> bool:
        return "PBS_JOBID" in os.environ

    def default_nproc(self) -> Optional[int]:
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            with open(nodefile) as fh:
                return sum(1 for line in fh if line.strip())
        return None

    def build_command(self, binary: str, nproc: int) -> List[str]:
        return ["mpiexec", "-n", str(nproc)] + self.extra_args + [binary]
