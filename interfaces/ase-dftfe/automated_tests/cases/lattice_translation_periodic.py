# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""
Test case: translating one atom by exactly one lattice vector changes nothing.

This is the mathematical premise the minimum-image fold in
``dftfeWrapper::updateAtomPositions()`` rests on, tested directly rather than
argued. Under PBC, displacing an atom by a lattice vector maps the infinite
crystal onto itself, so the total energy and every force are invariant:

    E(x + n*a) == E(x)   for integer n

If that is false in this code, then subtracting a lattice vector from a
displacement -- which is what the fold does -- is not a symmetry, and every
number the fold has ever produced is suspect. Nothing else in the suite states
this. ``al_slab_semiperiodic`` tests a rigid translation of the WHOLE system
along an OPEN axis, which is a different claim (it is exactly true in exact
arithmetic for any cell, periodic or not); this tests a single atom moved along
a PERIODIC axis, which is true only because of periodicity.

Two arms, both computed here so a divergence fails in this case rather than
waiting for a reference diff:

    A   atom 0 inside the cell
    B   atom 0 at exactly +1 lattice vector a1 along x, i.e. fractional x > 1

DFT-FE folds B back into the cell at reinit, so the two runs describe the same
physical crystal and must agree. Note the fold under test is NOT exercised here
-- no ionic step is taken, so no displacement is ever reconstructed. That is
deliberate: this case isolates the premise from the mechanism.
``restart_across_boundary`` covers the mechanism.

Gamma-only (no ``mp_grid``), so this runs on the real binary. 2x2 in plane for
the same reason ``al_slab_semiperiodic`` needs it: a 1x1 cell is too small for
vselfBinsManager to fit its self-potential balls.
"""
import os

import numpy as np
from ase.build import fcc111
from ase.units import Bohr, Hartree

from dftfe_ase import DFTFE

# A lattice translation is exact in exact arithmetic. What is left is that
# (1 + f) - 1 != f in floating point -- the wrap loses the low bits of a
# fractional coordinate it has added 1 to, which perturbs the geometry by
# ~1e-16 of a lattice vector, some 2e-15 Bohr on this cell. That is 5 orders
# below the mesh's own translation sensitivity, so this reuses the suite's
# 1e-10 Ha gate rather than inventing a looser one.
TRANSLATION_TOL_HA = 1e-10
FORCE_TOL_HA_BOHR = 1e-8


def _slab(shift_atom0_by_a1: bool):
    """8-atom Al(111) slab, periodic in x/y, open along z, centred in z."""
    atoms = fcc111("Al", size=(2, 2, 2), vacuum=None)
    cell = atoms.get_cell()[:]
    cell[2, 2] = 24.0
    atoms.set_cell(cell)
    atoms.pbc = [True, True, False]
    atoms.center(axis=2)
    # Wrap first, deliberately. ase.build.fcc111 places atom 0 at fractional
    # x = -0.167, i.e. already outside the cell on the negative side, so
    # *adding* a1 to the raw slab would move it IN (to 0.833) and the two arms
    # would be the wrong way round -- the unshifted one would be the one testing
    # the boundary. Wrapping makes the baseline unambiguously inside, so the
    # shifted arm is unambiguously outside.
    atoms.wrap()
    if shift_atom0_by_a1:
        pos = atoms.get_positions()
        pos[0] += atoms.get_cell()[0]      # exactly one lattice vector, in x
        atoms.set_positions(pos)
    return atoms


def _make_calc(dftfe_bin: str, psp_library: str, np_tasks: int,
               use_device: bool, tag: str):
    return DFTFE(
        command=f"mpirun -np {np_tasks} {dftfe_bin}",
        psp_path={"Al": os.path.join(psp_library, "Al.upf")},
        mesh_size=1.6,
        polynomial_order=5,
        tolerance=1e-4,
        fermi_temp=500.0,
        xc="GGA-PBE",
        pseudopotential_calculation=True,
        compute_forces=True,
        compute_stress=False,
        use_device=use_device,
        verbosity=1,
        log_file=f"test_lattice_translation_{tag}.log",
    )


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8,
        use_device: bool = True):
    inside = _slab(shift_atom0_by_a1=False)
    shifted = _slab(shift_atom0_by_a1=True)

    # --- the case is only meaningful if arm B really is outside the cell ---
    frac_x = (shifted.get_positions()
              @ np.linalg.inv(shifted.get_cell()[:]))[:, 0]
    if frac_x[0] <= 1.0:
        raise AssertionError(
            f"atom 0 of arm B has fractional x = {frac_x[0]:.6f}, which is not "
            "outside the cell; this case is no longer testing a lattice "
            "translation"
        )
    # And the translation must be exactly a lattice vector, not merely close to
    # one: the whole claim is about an exact symmetry.
    # Not asserted as exactly zero: (y + nudge) + a1 minus (y + nudge) is not
    # bitwise a1 in floating point once any other arithmetic has touched the
    # atom. 1e-12 A is 4 orders below the mesh resolution and 12 below the
    # lattice vector, so it still pins "exactly one lattice vector" in the only
    # sense available.
    OFFSET_TOL_A = 1e-12
    delta = shifted.get_positions()[0] - inside.get_positions()[0]
    residual = float(np.max(np.abs(delta - shifted.get_cell()[0])))
    if residual > OFFSET_TOL_A:
        raise AssertionError(
            f"arm B's displacement is not one lattice vector: residual "
            f"{residual:.3e} A (tol {OFFSET_TOL_A:.0e})"
        )

    calc_a = _make_calc(dftfe_bin, psp_library, np_tasks, use_device, "A")
    with calc_a:
        inside.calc = calc_a
        energy_a_ha = inside.get_potential_energy() / Hartree
        forces_a = inside.get_forces() / (Hartree / Bohr)

    calc_b = _make_calc(dftfe_bin, psp_library, np_tasks, use_device, "B")
    with calc_b:
        shifted.calc = calc_b
        energy_b_ha = shifted.get_potential_energy() / Hartree
        forces_b = shifted.get_forces() / (Hartree / Bohr)

    d_energy = abs(energy_a_ha - energy_b_ha)
    d_force = float(np.max(np.abs(forces_a - forces_b)))
    print(f"  lattice translation: |dE| = {d_energy:.3e} Ha, "
          f"max|dF| = {d_force:.3e} Ha/Bohr")
    if d_energy >= TRANSLATION_TOL_HA or d_force >= FORCE_TOL_HA_BOHR:
        raise AssertionError(
            f"translating one atom by exactly one lattice vector changed the "
            f"calculation: |dE| = {d_energy:.3e} Ha (tol "
            f"{TRANSLATION_TOL_HA:.0e}), max|dF| = {d_force:.3e} Ha/Bohr (tol "
            f"{FORCE_TOL_HA_BOHR:.0e}). Periodicity is not being applied as the "
            "minimum-image fold in dftfeWrapper::updateAtomPositions() assumes."
        )

    return energy_a_ha, forces_a, None
