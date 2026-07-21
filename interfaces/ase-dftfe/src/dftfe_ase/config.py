# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Layered configuration: binary paths, cluster profiles, scheduler detection.

Binary-location precedence (highest first):
    explicit kwargs  >  env vars  >  cluster profile  >  container canonical path  >  PATH

Cluster profiles capture the per-machine facts (scheduler, launcher, GPU backend,
default binary dir, PSP path, modules/env) for the production targets. The MATRIX
profile is fully populated from the site's working setup; the leadership-machine
profiles carry scheduler/launcher/backend and expect binary paths from env or a
site build (see github.com/dftfeDevelopers/install_DFTFE for the native builds).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, Optional

from .binary import DFTFEBinaries, DFTFEConfigError

# Canonical install path inside the container images (see containers/).
CONTAINER_PREFIX = "/opt/dftfe"


@dataclass
class ClusterProfile:
    name: str
    scheduler: str  # 'slurm' | 'pbs' | 'lsf' | 'local'
    launcher: str  # 'srun' | 'mpirun' | 'mpiexec' | 'jsrun' | 'local'
    gpu_backend: Optional[str]  # 'cuda' | 'hip' | 'sycl' | None
    bin_dir: Optional[str] = None  # holds release/{real,complex}/dftfe
    psp_path: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    modules: tuple = ()


# Threading env common to DFT-FE runs.
_DFTFE_THREADS = {
    "OMP_NUM_THREADS": "1",
    "DEAL_II_NUM_THREADS": "1",
    "DFTFE_NUM_THREADS": "1",
}

PROFILES: Dict[str, ClusterProfile] = {
    # IISc MATRIX — fully populated from the working site setup.
    "matrix": ClusterProfile(
        name="matrix",
        scheduler="slurm",
        launcher="mpirun",  # site launches mpirun inside the Slurm allocation
        gpu_backend="cuda",
        bin_dir="/home/pa01/Mehul/DFTFE/build_gpu",
        psp_path="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library",
        modules=(
            "spack",
            "openmpi/5.0.6-gcc-13.3.0-ytficip",
            "nccl/2.23.4-1-gcc-13.3.0-xyspmp2",
            "gdrcopy/2.4.1-gcc-13.3.0-dvwa323",
        ),
        env={
            **_DFTFE_THREADS,
            "LIBRARY_PATH": "/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib",
        },
    ),
    # Production targets — scheduler/launcher/backend known; binaries via env or
    # a native install_DFTFE build. bin_dir left None on purpose.
    "polaris": ClusterProfile("polaris", "pbs", "mpiexec", "cuda", env=dict(_DFTFE_THREADS)),
    "aurora": ClusterProfile("aurora", "pbs", "mpiexec", "sycl", env=dict(_DFTFE_THREADS)),
    "frontier": ClusterProfile("frontier", "slurm", "srun", "hip", env=dict(_DFTFE_THREADS)),
    "pravega": ClusterProfile("pravega", "slurm", "srun", "cuda", env=dict(_DFTFE_THREADS)),
    "siddhi": ClusterProfile("siddhi", "slurm", "srun", "cuda", env=dict(_DFTFE_THREADS)),
}


def get_profile(name: str) -> ClusterProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise DFTFEConfigError(
            f"unknown cluster profile {name!r}; known: {sorted(PROFILES)}"
        )


def detect_scheduler(env: Optional[dict] = None) -> str:
    env = os.environ if env is None else env
    if env.get("SLURM_JOB_ID"):
        return "slurm"
    if env.get("PBS_JOBID"):
        return "pbs"
    if env.get("LSB_JOBID"):
        return "lsf"
    return "local"


def resolve_binaries(
    real: Optional[str] = None,
    complex: Optional[str] = None,
    bin_dir: Optional[str] = None,
    cluster: Optional[str] = None,
    env: Optional[dict] = None,
) -> DFTFEBinaries:
    """Resolve real/complex binary paths using the documented precedence."""
    env = os.environ if env is None else env

    real = real or env.get("DFTFE_BIN_REAL")
    complex = complex or env.get("DFTFE_BIN_COMPLEX")
    bin_dir = bin_dir or env.get("DFTFE_BIN_DIR")

    if bin_dir is None and cluster is not None:
        bin_dir = get_profile(cluster).bin_dir

    if not (real and complex) and bin_dir is None and os.path.isdir(CONTAINER_PREFIX):
        bin_dir = CONTAINER_PREFIX

    if bin_dir is not None:
        real = real or os.path.join(bin_dir, "release", "real", "dftfe")
        complex = complex or os.path.join(bin_dir, "release", "complex", "dftfe")

    if not real and not complex:
        found = shutil.which("dftfe")
        if found:  # single unqualified binary; used for both kinds
            real = complex = found

    return DFTFEBinaries(real=real, complex=complex)
