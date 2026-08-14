# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Open-direction placement: the rules, and the physics that must survive them."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from dftfe_ase.geometry import (
    DFTFE_FRAC_TOL,
    OpenDirectionError,
    fractional,
    needs_shift,
    place_inside,
    rigid_shift,
)

CELL = np.diag([10.0, 10.0, 40.0])
SLAB_PBC = [True, True, False]


def _slab(z0, thickness=12.0, n=8):
    """n atoms strung along z from z0 to z0+thickness, x/y comfortably inside."""
    z = np.linspace(z0, z0 + thickness, n)
    return np.column_stack([np.full(n, 5.0), np.full(n, 5.0), z])


def test_in_cell_slab_is_left_exactly_alone():
    pos = _slab(14.0)
    shift = rigid_shift(pos, CELL, SLAB_PBC)
    assert not shift.any()
    placed, _ = place_inside(pos, CELL, SLAB_PBC)
    assert placed is pos  # not even copied


def test_slab_centred_on_origin_is_brought_inside():
    """z from -6 to +6 -- the classic abort. Fractional runs -0.15..0.15."""
    pos = _slab(-6.0)
    assert fractional(pos, CELL)[:, 2].min() < 0
    assert needs_shift(pos, CELL, SLAB_PBC) == [2]

    placed, shift = place_inside(pos, CELL, SLAB_PBC)
    f = fractional(placed, CELL)[:, 2]
    assert f.min() > DFTFE_FRAC_TOL and f.max() < 1.0 - DFTFE_FRAC_TOL
    # Only z moved.
    assert shift[0] == 0.0 and shift[1] == 0.0


def test_atom_exactly_on_the_face_is_repaired():
    """ase.build.surface() puts the bottom layer at z=0 -> fractional 0.0.

    DFT-FE's interval is open, so exactly-zero is a failure, not a pass.
    """
    pos = _slab(0.0)
    assert fractional(pos, CELL)[:, 2].min() == 0.0
    assert needs_shift(pos, CELL, SLAB_PBC) == [2]
    placed, _ = place_inside(pos, CELL, SLAB_PBC)
    assert not needs_shift(placed, CELL, SLAB_PBC)


def test_real_ase_slab_out_of_the_box():
    slab = fcc111("Al", size=(2, 2, 3), vacuum=8.0)
    slab.pbc = [True, True, False]
    placed, _ = place_inside(slab.get_positions(), slab.get_cell()[:], slab.get_pbc())
    assert not needs_shift(placed, slab.get_cell()[:], slab.get_pbc())


def test_shift_is_rigid_so_the_physics_is_untouched():
    """The whole point: distances are invariant, so energy and forces are too."""
    pos = _slab(-6.0)
    placed, _ = place_inside(pos, CELL, SLAB_PBC)

    before = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    after = np.linalg.norm(placed[:, None, :] - placed[None, :, :], axis=-1)
    np.testing.assert_allclose(before, after, atol=1e-12)

    # And it is one common translation, not a per-atom fold.
    deltas = placed - pos
    np.testing.assert_allclose(deltas, np.tile(deltas[0], (len(pos), 1)), atol=1e-12)


def test_periodic_axes_are_never_touched():
    """Folding those here would wreck optimizer state; DFT-FE does it instead."""
    pos = _slab(14.0)
    pos[:, 0] += 25.0  # way outside along periodic x
    shift = rigid_shift(pos, CELL, SLAB_PBC)
    assert shift[0] == 0.0
    assert needs_shift(pos, CELL, SLAB_PBC) == []


def test_fully_non_periodic_uses_all_three_axes():
    mol = Atoms("H2", positions=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                cell=np.diag([20.0, 20.0, 20.0]), pbc=False)
    placed, shift = place_inside(mol.get_positions(), mol.get_cell()[:], mol.get_pbc())
    assert (shift != 0).all()
    assert not needs_shift(placed, mol.get_cell()[:], mol.get_pbc())


def test_system_too_long_to_fit_is_an_error_not_a_silent_squeeze():
    pos = _slab(-5.0, thickness=45.0)  # 45 A of atoms in a 40 A box
    with pytest.raises(OpenDirectionError, match="No rigid shift can place them"):
        rigid_shift(pos, CELL, SLAB_PBC)


def test_non_orthogonal_cell():
    cell = np.array([[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 40.0]])
    pos = _slab(-6.0)
    pos[:, 0] += 3.0
    placed, _ = place_inside(pos, cell, SLAB_PBC)
    assert not needs_shift(placed, cell, SLAB_PBC)


def test_shift_lands_centred():
    """Centring is the placement with the most clearance from both faces."""
    pos = _slab(-6.0, thickness=12.0)
    placed, _ = place_inside(pos, CELL, SLAB_PBC)
    f = fractional(placed, CELL)[:, 2]
    assert f.min() == pytest.approx(1.0 - f.max(), abs=1e-12)


# ── as wired into the calculator ───────────────────────────────────────────


def _out_of_cell_slab():
    """Bottom layer at exactly z=0 -- what surface() gives with no vacuum call."""
    from ase.build import fcc111

    slab = fcc111("Al", size=(2, 2, 3), vacuum=None)
    cell = slab.get_cell()[:]
    cell[2, 2] = 30.0
    slab.set_cell(cell)
    slab.pbc = [True, True, False]
    return slab


def _calc_for(atoms):
    from dftfe_ase import DFTFE

    calc = DFTFE(command="/bin/true")  # lazy: never launched
    calc.atoms = atoms
    return calc


def test_calculator_sends_placed_coords_without_touching_the_atoms():
    slab = _out_of_cell_slab()
    original = slab.get_positions().copy()
    calc = _calc_for(slab)

    sent = calc._placed_coords()

    assert needs_shift(original, slab.get_cell()[:], slab.get_pbc()) == [2]
    assert needs_shift(sent, slab.get_cell()[:], slab.get_pbc()) == []
    # ASE keeps its own positions: the optimizer's state must stay continuous.
    np.testing.assert_allclose(slab.get_positions(), original, atol=1e-12)


def test_offset_is_frozen_across_steps():
    """Re-deriving it each step would drift the system through its own mesh."""
    slab = _out_of_cell_slab()
    calc = _calc_for(slab)

    first = calc._placed_coords()
    shift = calc._open_shift.copy()

    slab.positions[:, 2] += 0.05  # a relaxation step
    calc._placed_coords()
    np.testing.assert_allclose(calc._open_shift, shift, atol=1e-12)

    # And an unchanged geometry reproduces byte-identical coordinates.
    slab.positions[:, 2] -= 0.05
    np.testing.assert_allclose(calc._placed_coords(), first, atol=1e-12)


# ── degenerate input ───────────────────────────────────────────────────────


def test_zero_length_cell_vector_is_named_not_a_linalg_error():
    """Atoms(cell=[a, b, 0]) is legal ASE and is what you hold pre-center()."""
    with pytest.raises(OpenDirectionError, match="zero length"):
        needs_shift(np.array([[0.0, 0.0, 0.0]]),
                    [[3, 0, 0], [0, 3, 0], [0, 0, 0]], [True, True, False])


def test_near_zero_cell_vector_is_rejected_not_silently_used():
    """The dangerous one: inv() does not raise, it returns garbage."""
    with pytest.raises(OpenDirectionError):
        rigid_shift(np.array([[0.0, 0.0, 0.0]]),
                    [[3, 0, 0], [0, 3, 0], [0, 0, 1e-12]], [True, True, False])


def test_coplanar_cell_is_rejected():
    with pytest.raises(OpenDirectionError, match="singular"):
        needs_shift(np.array([[0.0, 0.0, 0.0]]),
                    [[1, 0, 0], [0, 1, 0], [1, 1, 0]], [True, True, False])


def test_empty_atoms_is_not_an_error():
    assert needs_shift(np.zeros((0, 3)), CELL, SLAB_PBC) == []
    assert not rigid_shift(np.zeros((0, 3)), CELL, SLAB_PBC).any()


def test_tight_but_placeable_system_is_placed_not_rejected():
    """Threshold must be DFT-FE's bound, not the clearance we would prefer.

    Span 0.999 of a 100 A cell centres to 5e-4 fractional margin -- 500x
    DFT-FE's 1e-6 requirement. Rejecting it told the user to enlarge a cell
    that was already big enough.
    """
    cell = np.diag([100.0, 100.0, 100.0])
    pos = np.array([[50.0, 50.0, -0.05], [50.0, 50.0, 99.85]])
    placed, shift = place_inside(pos, cell, [True, True, False])
    assert shift[2] != 0.0
    assert not needs_shift(placed, cell, [True, True, False])


# ── the frozen offset refuses to drift ─────────────────────────────────────


def test_cell_change_raises_because_dftfe_never_adopts_it():
    """socket_interface.cc omits cell deformation; the solver keeps reinit's cell."""
    from dftfe_ase.backends.socket import DFTFEError

    slab = _out_of_cell_slab()
    calc = _calc_for(slab)
    calc._placed_coords()

    cell = slab.get_cell()[:]
    cell[2, 2] += 0.3
    slab.set_cell(cell)
    with pytest.raises(DFTFEError, match="does not adopt a new cell"):
        calc._placed_coords()


def test_atom_leaving_the_cell_midrun_raises_rather_than_recentring():
    """Re-centring would move the system > break1 = 1.0 Bohr and force a remesh,
    putting this energy on a different mesh from every earlier one."""
    from dftfe_ase.backends.socket import DFTFEError

    slab = _out_of_cell_slab()
    calc = _calc_for(slab)
    calc._placed_coords()

    # Enough to clear the frozen +12.66 A offset and exit the bottom face.
    slab.positions[:, 2] -= 14.0
    with pytest.raises(DFTFEError, match="remesh"):
        calc._placed_coords()


def test_atom_count_change_raises_instead_of_reusing_the_offset():
    from dftfe_ase.backends.socket import DFTFEError

    slab = _out_of_cell_slab()
    calc = _calc_for(slab)
    calc._placed_coords()

    smaller = slab[:4]
    smaller.pbc = slab.get_pbc()
    calc.atoms = smaller
    with pytest.raises(DFTFEError, match="atom count changed"):
        calc._placed_coords()


def test_fully_periodic_system_is_passed_through_untouched():
    bulk = Atoms("Al", positions=[[0.0, 0.0, 0.0]],
                 cell=np.diag([4.0, 4.0, 4.0]), pbc=True)
    calc = _calc_for(bulk)
    np.testing.assert_allclose(calc._placed_coords(), bulk.get_positions(), atol=1e-12)
    assert calc._open_shift is None  # nothing computed, nothing cached
