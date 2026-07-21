# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Job launchers + registry/auto-detection."""

from __future__ import annotations

from .base import Launcher
from .slurm import SrunLauncher
from .mpi import MpirunLauncher, MpiexecLauncher
from .lsf import JsrunLauncher
from .local import LocalLauncher

_REGISTRY = {
    cls.name: cls
    for cls in (SrunLauncher, MpirunLauncher, MpiexecLauncher, JsrunLauncher, LocalLauncher)
}

# Scheduler-specific launchers, in auto-detection priority order.
_AUTODETECT = (SrunLauncher, MpiexecLauncher, JsrunLauncher)


def get_launcher(name: str, **kwargs) -> Launcher:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown launcher {name!r}; known: {sorted(_REGISTRY)}")
    return cls(**kwargs)


def detect_launcher(**kwargs) -> Launcher:
    """Pick a launcher from the environment: scheduler tool, else generic mpirun."""
    for cls in _AUTODETECT:
        if cls.detect():
            return cls(**kwargs)
    return MpirunLauncher(**kwargs)


__all__ = [
    "Launcher",
    "SrunLauncher",
    "MpirunLauncher",
    "MpiexecLauncher",
    "JsrunLauncher",
    "LocalLauncher",
    "get_launcher",
    "detect_launcher",
]
