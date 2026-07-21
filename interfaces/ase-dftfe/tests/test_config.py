# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Unit tests for config resolution (binary paths, profiles, scheduler)."""

import pytest

from dftfe_ase.binary import DFTFEConfigError
from dftfe_ase.config import detect_scheduler, get_profile, resolve_binaries


def test_explicit_paths_beat_env():
    b = resolve_binaries(
        real="/x/real", complex="/x/complex", env={"DFTFE_BIN_REAL": "/env/real"}
    )
    assert b.real == "/x/real" and b.complex == "/x/complex"


def test_env_paths_used():
    b = resolve_binaries(env={"DFTFE_BIN_REAL": "/e/r", "DFTFE_BIN_COMPLEX": "/e/c"})
    assert b.real == "/e/r" and b.complex == "/e/c"


def test_bin_dir_expands_to_release_layout():
    b = resolve_binaries(bin_dir="/build", env={})
    assert b.real.endswith("/build/release/real/dftfe")
    assert b.complex.endswith("/build/release/complex/dftfe")


def test_cluster_profile_supplies_bin_dir():
    b = resolve_binaries(cluster="matrix", env={})
    assert b.real.endswith("build_gpu/release/real/dftfe")
    assert b.complex.endswith("build_gpu/release/complex/dftfe")


def test_unknown_cluster_raises():
    with pytest.raises(DFTFEConfigError):
        get_profile("nope")


def test_profile_fields():
    p = get_profile("frontier")
    assert p.scheduler == "slurm" and p.launcher == "srun" and p.gpu_backend == "hip"
    assert get_profile("aurora").gpu_backend == "sycl"
    assert get_profile("polaris").scheduler == "pbs"


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"SLURM_JOB_ID": "1"}, "slurm"),
        ({"PBS_JOBID": "1"}, "pbs"),
        ({"LSB_JOBID": "1"}, "lsf"),
        ({}, "local"),
    ],
)
def test_detect_scheduler(env, expected):
    assert detect_scheduler(env) == expected
