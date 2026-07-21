# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Local launcher: no scheduler. mpirun for nproc>1, direct exec for serial/dev."""

from __future__ import annotations

from typing import List, Optional

from .base import Launcher


class LocalLauncher(Launcher):
    name = "local"

    def __init__(self, use_mpirun: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_mpirun = use_mpirun

    @classmethod
    def detect(cls) -> bool:
        return True  # always valid as the final fallback

    def default_nproc(self) -> Optional[int]:
        return 1

    def build_command(self, binary: str, nproc: int) -> List[str]:
        if nproc and nproc > 1 and self.use_mpirun:
            return ["mpirun", "-np", str(nproc)] + self.extra_args + [binary]
        return [binary]
