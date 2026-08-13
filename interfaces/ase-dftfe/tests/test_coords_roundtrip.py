# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""``coordinates.inp`` must come back exactly as written -- no silent repair.

Reading a deck and placing atoms for the solver are deliberately separate jobs.
The reader is *faithful*: whatever the file says is what the ``Atoms`` object
gets, out-of-cell or not, because a reader that quietly moves atoms makes every
comparison against the native run meaningless. Placement happens later, once,
on the way to DFT-FE (``calculator._placed_coords``).

The convention being pinned here is DFT-FE's own, from ``dft.cc``:

* any periodic axis  -> all three columns are **fractional**
  (``dft.cc`` reads them into ``atomLocationsFractional``)
* fully non-periodic -> Cartesian Bohr, measured from the cell centre
  (``dftfeWrapper::reinit`` writes them shifted by ``-sum(cell[j][i])/2``)
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from dftfe_ase.geometry import needs_shift
from dftfe_ase.io import read_dftfe_atoms

CELL_BOHR = np.diag([10.0, 10.0, 40.0])


def _write_deck(tmp_path, frac_or_cart, pbc, numbers=(3, 3, 8)):
    (tmp_path / "domainVectors.inp").write_text(
        "\n".join(" ".join(f"{v:.16f}" for v in row) for row in CELL_BOHR) + "\n"
    )
    lines = []
    for z, xyz in zip(numbers, frac_or_cart):
        lines.append(f"{z} {z} " + " ".join(f"{v:.16f}" for v in xyz))
    (tmp_path / "coordinates.inp").write_text("\n".join(lines) + "\n")

    p1, p2, p3 = ("true" if b else "false" for b in pbc)
    (tmp_path / "in.prm").write_text(
        "subsection Geometry\n"
        f"  set NATOMS={len(numbers)}\n"
        "  set NATOM TYPES=2\n"
        "  set ATOMIC COORDINATES FILE=coordinates.inp\n"
        "  set DOMAIN VECTORS FILE=domainVectors.inp\n"
        "end\n"
        "subsection Boundary conditions\n"
        f"  set PERIODIC1={p1}\n  set PERIODIC2={p2}\n  set PERIODIC3={p3}\n"
        "end\n"
    )
    return tmp_path / "in.prm"


def test_semiperiodic_fractional_roundtrips_exactly(tmp_path):
    frac = np.array([[0.10, 0.20, 0.30],
                     [0.50, 0.50, 0.45],
                     [0.90, 0.80, 0.60]])
    atoms = read_dftfe_atoms(_write_deck(tmp_path, frac, [True, True, False]))

    np.testing.assert_allclose(atoms.get_scaled_positions(wrap=False), frac, atol=1e-12)
    np.testing.assert_allclose(atoms.get_cell()[:], CELL_BOHR * Bohr, atol=1e-12)
    assert list(atoms.get_pbc()) == [True, True, False]
    assert atoms.get_atomic_numbers().tolist() == [3, 3, 8]


def test_file_order_is_preserved(tmp_path):
    """Per-atom force comparison against a native run needs identical ordering."""
    frac = np.array([[0.7, 0.1, 0.2], [0.1, 0.9, 0.8], [0.4, 0.4, 0.5]])
    atoms = read_dftfe_atoms(_write_deck(tmp_path, frac, [True, True, False],
                                         numbers=(8, 3, 3)))
    assert atoms.get_atomic_numbers().tolist() == [8, 3, 3]
    np.testing.assert_allclose(atoms.get_scaled_positions(wrap=False), frac, atol=1e-12)


def test_out_of_cell_coordinates_are_read_verbatim_not_repaired(tmp_path):
    """The reader reports what the file says. Placement is a separate step."""
    frac = np.array([[0.5, 0.5, -0.15],
                     [0.5, 0.5, 0.00],
                     [0.5, 0.5, 0.15]])
    atoms = read_dftfe_atoms(_write_deck(tmp_path, frac, [True, True, False]))

    np.testing.assert_allclose(atoms.get_scaled_positions(wrap=False), frac, atol=1e-12)
    # Faithfully read *and* correctly diagnosed as needing placement.
    assert needs_shift(atoms.get_positions(), atoms.get_cell()[:], atoms.get_pbc()) == [2]


def test_fully_non_periodic_is_cartesian_from_the_cell_centre(tmp_path):
    """reinit() writes Cartesian shifted by -sum(cell)/2; the reader undoes it."""
    cart_centred = np.array([[-1.0, -1.0, -2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    atoms = read_dftfe_atoms(_write_deck(tmp_path, cart_centred, [False, False, False]))

    expected_bohr = cart_centred + CELL_BOHR.sum(axis=0) / 2.0
    np.testing.assert_allclose(atoms.get_positions(), expected_bohr * Bohr, atol=1e-12)
    # The middle atom sat at the cell centre, so it must land at cell/2.
    np.testing.assert_allclose(atoms.get_positions()[1],
                               CELL_BOHR.sum(axis=0) / 2.0 * Bohr, atol=1e-12)


def test_natoms_mismatch_warns_and_trusts_the_file(tmp_path, caplog):
    frac = np.array([[0.5, 0.5, 0.5]])
    prm = _write_deck(tmp_path, frac, [True, True, False], numbers=(3,))
    prm.write_text(prm.read_text().replace("set NATOMS=1", "set NATOMS=99"))

    with caplog.at_level("WARNING", logger="dftfe_ase.io"):
        atoms = read_dftfe_atoms(prm)

    assert len(atoms) == 1
    assert "NATOMS=99" in caplog.text
