# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Integration tests: the DFTFE calculator driving the mock DFT-FE over a socket.

No real binary or MPI needed. Verifies the round-trip, unit conversions, lazy
startup, lifecycle/cleanup, error surfacing, and the fixed `verbosity is None`
regression.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from dftfe_ase import DFTFE, DFTFEError


def _h2():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                 cell=[10, 10, 10], pbc=False)


def test_energy_roundtrip_and_units(mock_command):
    atoms = _h2()
    with DFTFE(command=mock_command, connect_timeout=30) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
    # mock: energy_Ha = -sum(Z) = -2 ; ASE wants eV
    assert energy == pytest.approx(-2.0 * Hartree)


def test_forces_shape_and_units(mock_command):
    atoms = _h2()
    with DFTFE(command=mock_command, connect_timeout=30) as calc:
        atoms.calc = calc
        forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    expected = np.array([[0.001, 0.002, -0.003],
                         [0.002, 0.004, -0.006]]) * (Hartree / Bohr)
    assert np.allclose(forces, expected)


def test_stress_is_voigt6_and_converted(mock_command):
    atoms = _h2()
    atoms.pbc = True
    with DFTFE(command=mock_command, compute_stress=True, connect_timeout=30) as calc:
        atoms.calc = calc
        stress = atoms.get_stress()
    assert stress.shape == (6,)
    # diag(0.01,0.02,0.03) Ha/Bohr^3 -> Voigt [xx,yy,zz,0,0,0], sign preserved.
    #
    # The sign is the point of this assertion, not an incidental detail. ASE
    # defines sigma = (1/V) dE/deps, and since 2026-08-16 the C++ side returns
    # exactly that, so the calculator passes it through
    # (calculator._WRAPPER_STRESS_SIGN = +1). Before that, getCellStress()
    # negated the tensor and this test asserted the flip instead.
    #
    # Getting this wrong is silent and expensive: a cell relaxation driven by the
    # opposite sign expands when it should contract.
    diag = np.array([0.01, 0.02, 0.03]) * (Hartree / Bohr**3)
    assert np.allclose(stress[:3], diag)
    assert np.allclose(stress[3:], 0.0)


def test_default_verbosity_none_does_not_crash(mock_command):
    """Regression: legacy `if self.verbosity > 0` crashed when unset (default)."""
    atoms = _h2()
    calc = DFTFE(command=mock_command, connect_timeout=30)  # verbosity defaults to None
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.close()


def test_error_frame_raises_dftfe_error(mock_command):
    atoms = _h2()
    with DFTFE(command=mock_command, env={"MOCK_DFTFE_ERROR": "1"},
               connect_timeout=30) as calc:
        atoms.calc = calc
        with pytest.raises(DFTFEError):
            atoms.get_potential_energy()


def test_process_that_never_connects_times_out(mock_command):
    atoms = _h2()
    calc = DFTFE(command=mock_command, env={"MOCK_DFTFE_NOCONNECT": "1"},
                 connect_timeout=2)
    atoms.calc = calc
    with pytest.raises(DFTFEError):
        atoms.get_potential_energy()
    calc.close()


def test_unknown_parameter_rejected(mock_command):
    with pytest.raises(TypeError):
        DFTFE(command=mock_command, not_a_real_param=5)
