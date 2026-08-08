# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Auto-launch command construction in the calculator (no process spawned)."""

from dftfe_ase import DFTFE


def test_kpoints_select_complex_binary():
    calc = DFTFE(cluster="matrix", nproc=8, mp_grid=(2, 2, 2))
    argv = calc.build_launch_argv([True, True, True])
    assert argv[:3] == ["mpirun", "-np", "8"]
    assert argv[-1].endswith("build_gpu/release/complex/dftfe")


def test_gamma_selects_real_binary():
    calc = DFTFE(cluster="matrix", nproc=8)  # no mp_grid -> gamma-only
    argv = calc.build_launch_argv([True, True, True])
    assert argv[-1].endswith("build_gpu/release/real/dftfe")


def test_nonperiodic_forces_real_even_with_grid():
    calc = DFTFE(cluster="matrix", nproc=4, mp_grid=(2, 2, 2))
    argv = calc.build_launch_argv([False, False, False])
    assert argv[-1].endswith("release/real/dftfe")


def test_explicit_launcher_local_direct():
    calc = DFTFE(launcher="local", dftfe_real="/x/dftfe", nproc=1)
    assert calc.build_launch_argv([False, False, False]) == ["/x/dftfe"]


def test_launcher_args_sit_between_flags_and_binary():
    """Aurora binds one rank per GPU tile via a wrapper script, which must be
    argv[-2]: `mpiexec -n N --ppn 12 gpu_tile_compact.sh <binary>`."""
    calc = DFTFE(launcher="mpiexec", dftfe_real="/x/dftfe", nproc=384,
                 launcher_args=["--ppn", "12", "gpu_tile_compact.sh"])
    assert calc.build_launch_argv([True, True, True]) == [
        "mpiexec", "-n", "384", "--ppn", "12", "gpu_tile_compact.sh", "/x/dftfe"]


def test_launcher_args_default_to_nothing():
    calc = DFTFE(launcher="mpiexec", dftfe_real="/x/dftfe", nproc=4)
    assert calc.build_launch_argv([True, True, True]) == [
        "mpiexec", "-n", "4", "/x/dftfe"]
