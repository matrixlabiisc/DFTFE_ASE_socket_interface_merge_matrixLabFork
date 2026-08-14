# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""FileBackend: the native file-in/file-out arm of the parity comparison.

This backend exists so one ASE optimizer can drive both a socket run and a
classic write-inputs-launch-parse run, making the interface-overhead numbers a
like-for-like comparison. That only holds if it writes the coordinate file DFT-FE
actually expects, so the conventions are pinned here rather than assumed:

* any periodic axis  -> all three columns fractional
* fully non-periodic -> Cartesian Bohr from the **domain centre**

The second is the one worth pinning. dft.cc prints that branch under
"Cartesian coordinates of atoms (origin at center of domain)" and consumes
atomLocations verbatim -- convertToCellCenteredCartesianCoordinates() runs only
on the periodic path. Corner-origin coordinates are not rejected as malformed;
they are read as a different structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from dftfe_ase.backends.file import FileBackend
from dftfe_ase.backends.socket import DFTFEError

CELL = np.diag([40.0, 40.0, 40.0])          # Bohr
CENTRE = CELL.sum(axis=0) / 2.0             # (20, 20, 20)


@pytest.fixture
def backend(tmp_path):
    """A FileBackend over a minimal template; never launched."""
    (tmp_path / "coordinates.inp").write_text(
        "7 5 0.0 0.0 0.0\n"
        "7 5 0.0 0.0 0.0\n"
    )
    (tmp_path / "domainVectors.inp").write_text(
        "\n".join(" ".join(f"{v:.16f}" for v in row) for row in CELL) + "\n"
    )
    (tmp_path / "parameterFile.prm").write_text(
        "set SOLVER MODE=GS\n"
        "subsection Geometry\n  set NATOMS=2\nend\n"
    )
    return FileBackend(command="/bin/true", template_dir=str(tmp_path))


def _written(backend, tmp_path, coords, pbc):
    backend._write_coordinates(str(tmp_path), coords, CELL, pbc)
    raw = np.atleast_2d(np.loadtxt(tmp_path / "coordinates.inp"))
    return raw[:, 2:5]


# ── coordinate conventions ─────────────────────────────────────────────────


def test_periodic_writes_fractional(backend, tmp_path):
    coords = np.array([[4.0, 8.0, 12.0], [20.0, 20.0, 20.0]])
    got = _written(backend, tmp_path, coords, [True, True, True])
    np.testing.assert_allclose(got, coords @ np.linalg.inv(CELL), atol=1e-12)


def test_semiperiodic_also_writes_fractional(backend, tmp_path):
    """Any periodic axis -> fractional, matching dft.cc's `||` test."""
    coords = np.array([[4.0, 8.0, 12.0], [20.0, 20.0, 20.0]])
    got = _written(backend, tmp_path, coords, [True, True, False])
    np.testing.assert_allclose(got, coords @ np.linalg.inv(CELL), atol=1e-12)


def test_non_periodic_writes_cartesian_from_the_domain_centre(backend, tmp_path):
    """The bug this pins: corner-origin put every atom half a diagonal off."""
    coords = np.array([[20.0, 20.0, 20.0], [20.0, 20.0, 22.6]])
    got = _written(backend, tmp_path, coords, [False, False, False])

    # An atom at the centre of the box must be written as the origin.
    np.testing.assert_allclose(got[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(got, coords - CENTRE, atol=1e-12)


def test_non_periodic_round_trips_through_the_reader(backend, tmp_path):
    """What FileBackend writes, io.read_dftfe_atoms must read back unchanged."""
    coords = np.array([[18.0, 21.0, 19.5], [22.0, 19.0, 20.5]])
    got = _written(backend, tmp_path, coords, [False, False, False])
    # io.py's inverse: positions_bohr = raw + cell.sum(0)/2
    np.testing.assert_allclose(got + CENTRE, coords, atol=1e-12)


def test_bond_length_survives_either_convention(backend, tmp_path):
    """A frame error must never change the structure, only its placement."""
    coords = np.array([[20.0, 20.0, 18.7], [20.0, 20.0, 21.3]])
    for pbc in ([True, True, True], [False, False, False]):
        got = _written(backend, tmp_path, coords, pbc)
        if not any(pbc):
            assert np.linalg.norm(got[1] - got[0]) == pytest.approx(2.6, abs=1e-12)


def test_atomic_number_and_valence_columns_come_from_the_template(backend, tmp_path):
    backend._write_coordinates(str(tmp_path), np.zeros((2, 3)), CELL, [False] * 3)
    raw = np.atleast_2d(np.loadtxt(tmp_path / "coordinates.inp"))
    assert raw[:, 0].tolist() == [7.0, 7.0]
    assert raw[:, 1].tolist() == [5.0, 5.0]


# ── output parsing ─────────────────────────────────────────────────────────


FORCE_BLOCK = """
 Total free energy: -20.641688948796
 Ion forces (Hartree/Bohr)
 AtomId 0:  0.001 0.002 -0.003
 AtomId 1: -0.001 -0.002 0.003
 ---------------------------------
"""


def test_parses_energy_and_all_forces():
    e, f = FileBackend._parse(FORCE_BLOCK, natoms=2)
    assert e == pytest.approx(-20.641688948796)
    np.testing.assert_allclose(f, [[0.001, 0.002, -0.003], [-0.001, -0.002, 0.003]])


def test_last_energy_wins_when_scf_prints_several():
    text = ("Total free energy: -1.0\n"
            "Total free energy: -2.0\n"
            "Total free energy: -3.5\n")
    e, f = FileBackend._parse(text, natoms=1)
    assert e == pytest.approx(-3.5)
    assert f is None


def test_missing_energy_raises_rather_than_returning_none():
    with pytest.raises(DFTFEError, match="Total free energy"):
        FileBackend._parse("DFT-FE ran and said nothing useful\n", natoms=1)


def test_force_rows_beyond_natoms_are_truncated():
    text = FORCE_BLOCK.replace(
        " ---------------------------------",
        " AtomId 2:  9.9 9.9 9.9\n ---------------------------------")
    e, f = FileBackend._parse(text, natoms=2)
    assert len(f) == 2
    assert 9.9 not in np.asarray(f)
