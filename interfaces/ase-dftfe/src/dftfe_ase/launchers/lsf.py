# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""LSF launcher (jsrun), for IBM systems (e.g. OLCF Summit-class)."""

from __future__ import annotations

import os
from typing import List

from .base import Launcher


class JsrunLauncher(Launcher):
    name = "jsrun"

    @classmethod
    def detect(cls) -> bool:
        return "LSB_JOBID" in os.environ

    def build_command(self, binary: str, nproc: int) -> List[str]:
        # One resource set per rank with one GPU each is the common DFT-FE layout;
        # kept minimal here and overridable via extra_args.
        cmd = ["jsrun", "-n", str(nproc)]
        if self.gpus_per_task is not None:
            cmd += ["-g", str(self.gpus_per_task)]
        cmd += self.extra_args
        cmd.append(binary)
        return cmd
