# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Unit tests for launcher command construction and auto-detection."""

import pytest

from dftfe_ase.launchers import (
    LocalLauncher,
    MpirunLauncher,
    SrunLauncher,
    detect_launcher,
    get_launcher,
)


def test_mpirun_build():
    assert get_launcher("mpirun").build_command("/bin/dftfe", 8) == [
        "mpirun", "-np", "8", "/bin/dftfe",
    ]


def test_srun_build_with_gpu():
    launcher = get_launcher("srun", gpus_per_task=1)
    assert launcher.build_command("/bin/dftfe", 4) == [
        "srun", "--ntasks", "4", "--gpus-per-task=1", "/bin/dftfe",
    ]


def test_mpiexec_build():
    assert get_launcher("mpiexec").build_command("/d", 2) == ["mpiexec", "-n", "2", "/d"]


def test_jsrun_build():
    assert get_launcher("jsrun").build_command("/d", 6)[:3] == ["jsrun", "-n", "6"]


def test_local_direct_when_serial():
    assert get_launcher("local").build_command("/d", 1) == ["/d"]


def test_local_uses_mpirun_when_parallel():
    assert get_launcher("local").build_command("/d", 4) == ["mpirun", "-np", "4", "/d"]


def test_unknown_launcher_raises():
    with pytest.raises(ValueError):
        get_launcher("nope")


def test_detect_prefers_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    assert isinstance(detect_launcher(), SrunLauncher)


def test_detect_falls_back_to_mpirun(monkeypatch):
    for var in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(detect_launcher(), MpirunLauncher)


def test_srun_default_nproc_from_env(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "16")
    assert SrunLauncher().default_nproc() == 16
