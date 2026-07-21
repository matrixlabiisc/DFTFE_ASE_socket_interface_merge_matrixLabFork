# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Slurm launcher (srun), used inside a Slurm allocation (e.g. OLCF Frontier)."""

from __future__ import annotations

import os
from typing import List, Optional

from .base import Launcher


class SrunLauncher(Launcher):
    name = "srun"

    @classmethod
    def detect(cls) -> bool:
        return "SLURM_JOB_ID" in os.environ

    def default_nproc(self) -> Optional[int]:
        val = os.environ.get("SLURM_NTASKS")
        return int(val) if val else None

    def build_command(self, binary: str, nproc: int) -> List[str]:
        cmd = ["srun", "--ntasks", str(nproc)]
        if self.gpus_per_task is not None:
            cmd.append(f"--gpus-per-task={self.gpus_per_task}")
        cmd += self.extra_args
        cmd.append(binary)
        return cmd
