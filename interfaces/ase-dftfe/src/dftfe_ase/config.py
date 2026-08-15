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

Cluster profiles capture the per-machine facts that are true for *every* user of
that machine: scheduler, launcher, GPU backend, site modules and site-wide
library paths. They deliberately carry **no binary or pseudopotential paths**,
because those live in an individual's directory and differ per user even on the
same cluster. Shipping one person's ``bin_dir`` in the package means every other
user of that cluster silently resolves to a directory they cannot read.

Per-user facts come from, in order of precedence: explicit kwargs, environment
variables (``DFTFE_BIN_REAL`` / ``DFTFE_BIN_COMPLEX`` / ``DFTFE_BIN_DIR``), and a
site file of your own (see :func:`load_site_profiles`). See
github.com/dftfeDevelopers/install_DFTFE for the native builds.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from .binary import DFTFEBinaries, DFTFEConfigError

# Canonical install path inside the container images (see containers/).
CONTAINER_PREFIX = "/opt/dftfe"

# Where a user's own profile overrides live. ``DFTFE_ASE_CONFIG`` wins if set.
SITE_CONFIG_ENV = "DFTFE_ASE_CONFIG"
SITE_CONFIG_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "dftfe_ase", "profiles.json",
)


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

# Only machines any DFT-FE user can plausibly be sitting on: the public
# leadership systems. Each carries scheduler, launcher and GPU backend and
# nothing else — no paths, no module names, no site library directories.
#
# Institution-local clusters are deliberately NOT here. Their module strings and
# dependency directories exist on exactly one site, rot on that site's next
# rebuild, and are unusable to everyone else, so they belong in a user's own
# profiles.json (see load_site_profiles) rather than in the shipped package.
PROFILES: Dict[str, ClusterProfile] = {
    "polaris": ClusterProfile("polaris", "pbs", "mpiexec", "cuda", env=dict(_DFTFE_THREADS)),
    "aurora": ClusterProfile("aurora", "pbs", "mpiexec", "sycl", env=dict(_DFTFE_THREADS)),
    "frontier": ClusterProfile("frontier", "slurm", "srun", "hip", env=dict(_DFTFE_THREADS)),
}

_SITE_CACHE: Optional[Dict[str, ClusterProfile]] = None


def _site_config_path(env: Optional[dict] = None) -> str:
    env = os.environ if env is None else env
    return env.get(SITE_CONFIG_ENV) or SITE_CONFIG_PATH


def load_site_profiles(path: Optional[str] = None) -> Dict[str, ClusterProfile]:
    """Your own machine facts, layered over the shipped profiles.

    JSON, ``{profile_name: {field: value}}``, read from ``$DFTFE_ASE_CONFIG`` or
    ``~/.config/dftfe_ase/profiles.json``. Fields override the built-in profile
    of the same name one at a time, and a name that is not built in defines a new
    profile (``scheduler`` and ``launcher`` are then required)::

        {"aurora":    {"bin_dir": "/lus/.../my_build", "psp_path": "/home/you/psp"},
         "mycluster": {"scheduler": "slurm", "launcher": "srun",
                       "gpu_backend": "cuda", "bin_dir": "/home/you/build"}}

    A malformed file raises rather than being skipped: a config that is silently
    ignored looks exactly like a config that had no effect, and the user then
    debugs the wrong thing.
    """
    path = path or _site_config_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise DFTFEConfigError(f"could not read site profiles from {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DFTFEConfigError(f"{path}: expected a JSON object of {{profile: {{field: value}}}}")

    allowed = {f for f in ClusterProfile.__dataclass_fields__ if f != "name"}
    out: Dict[str, ClusterProfile] = {}
    for pname, fields in raw.items():
        if not isinstance(fields, dict):
            raise DFTFEConfigError(f"{path}: profile {pname!r} must be an object, got {type(fields).__name__}")
        unknown = set(fields) - allowed
        if unknown:
            raise DFTFEConfigError(
                f"{path}: profile {pname!r} has unknown field(s) {sorted(unknown)}; "
                f"known: {sorted(allowed)}"
            )
        if "modules" in fields:
            fields = {**fields, "modules": tuple(fields["modules"])}
        base = PROFILES.get(pname)
        if base is None:
            missing = {"scheduler", "launcher"} - set(fields)
            if missing:
                raise DFTFEConfigError(
                    f"{path}: {pname!r} is not a built-in profile, so it must define "
                    f"{sorted(missing)}"
                )
            out[pname] = ClusterProfile(name=pname, gpu_backend=fields.pop("gpu_backend", None), **fields)
        else:
            out[pname] = replace(base, **fields)
    return out


def get_profile(name: str) -> ClusterProfile:
    global _SITE_CACHE
    if _SITE_CACHE is None:
        _SITE_CACHE = load_site_profiles()
    profile = _SITE_CACHE.get(name) or PROFILES.get(name)
    if profile is None:
        known = sorted(set(PROFILES) | set(_SITE_CACHE))
        raise DFTFEConfigError(f"unknown cluster profile {name!r}; known: {known}")
    return profile


def reload_site_profiles() -> None:
    """Drop the cached site file so the next :func:`get_profile` re-reads it."""
    global _SITE_CACHE
    _SITE_CACHE = None


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
