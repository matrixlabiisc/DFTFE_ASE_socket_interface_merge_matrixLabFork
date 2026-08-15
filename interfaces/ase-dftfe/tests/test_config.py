# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Unit tests for config resolution (binary paths, profiles, scheduler)."""

import pytest

from dftfe_ase.binary import DFTFEConfigError
from dftfe_ase.config import (PROFILES, detect_scheduler, get_profile,
                              load_site_profiles, reload_site_profiles,
                              resolve_binaries)


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


def test_no_builtin_profile_ships_a_binary_path():
    """Binary and PSP paths are per-user, so no shipped profile may carry one.

    A profile that hardcodes one person's ``bin_dir`` resolves, for every other
    user of that same cluster, to a directory they cannot read — and it does so
    silently, because the path only fails at launch.
    """
    for name, profile in PROFILES.items():
        assert profile.bin_dir is None, f"{name} ships a bin_dir: {profile.bin_dir}"
        assert profile.psp_path is None, f"{name} ships a psp_path: {profile.psp_path}"


def test_cluster_profile_without_bin_dir_resolves_nothing():
    b = resolve_binaries(cluster="frontier", env={})
    assert b.real is None and b.complex is None


def test_site_file_supplies_bin_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "profiles.json"
    cfg.write_text('{"frontier": {"bin_dir": "/home/someone/build_gpu"}}')
    monkeypatch.setenv("DFTFE_ASE_CONFIG", str(cfg))
    reload_site_profiles()
    try:
        assert get_profile("frontier").bin_dir == "/home/someone/build_gpu"
        # Site fields override one at a time; the site-wide facts survive.
        assert get_profile("frontier").launcher == "srun"
        assert get_profile("frontier").gpu_backend == "hip"
        b = resolve_binaries(cluster="frontier", env={})
        assert b.real.endswith("/home/someone/build_gpu/release/real/dftfe")
    finally:
        reload_site_profiles()


def test_site_file_can_define_a_new_cluster(tmp_path, monkeypatch):
    cfg = tmp_path / "profiles.json"
    cfg.write_text('{"mymachine": {"scheduler": "slurm", "launcher": "srun"}}')
    monkeypatch.setenv("DFTFE_ASE_CONFIG", str(cfg))
    reload_site_profiles()
    try:
        assert get_profile("mymachine").launcher == "srun"
    finally:
        reload_site_profiles()


def test_site_file_new_cluster_needs_scheduler_and_launcher(tmp_path):
    cfg = tmp_path / "profiles.json"
    cfg.write_text('{"mymachine": {"bin_dir": "/b"}}')
    with pytest.raises(DFTFEConfigError, match="must define"):
        load_site_profiles(str(cfg))


def test_site_file_rejects_unknown_field(tmp_path):
    cfg = tmp_path / "profiles.json"
    cfg.write_text('{"frontier": {"bindir": "/b"}}')
    with pytest.raises(DFTFEConfigError, match="unknown field"):
        load_site_profiles(str(cfg))


def test_malformed_site_file_raises_rather_than_being_skipped(tmp_path):
    cfg = tmp_path / "profiles.json"
    cfg.write_text("{not json")
    with pytest.raises(DFTFEConfigError, match="could not read site profiles"):
        load_site_profiles(str(cfg))


def test_absent_site_file_is_not_an_error(tmp_path):
    assert load_site_profiles(str(tmp_path / "nope.json")) == {}


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
