# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""
Test case: an ionic step taken while an atom sits outside the cell.

This is the case that killed job 8761021 on 32 nodes, reduced to 8 atoms. It is
the regression test for the minimum-image fold in
``dftfeWrapper::updateAtomPositions()``, and NOTHING in this suite constructed
it before -- which is why a coordinate-convention mismatch survived to take
down a 32-node run three ionic steps in.

WHAT GOES WRONG WITHOUT THE FOLD

The interface sends positions unwrapped along periodic axes on purpose
(``calculator.py:519-527``): wrapping would make a boundary crossing look like a
full-lattice-vector jump to the optimizer and destroy its state. DFT-FE wraps
its own copy at reinit. Both are correct in isolation, and they agree as long as
every atom starts inside the cell -- which is why a fresh run is always safe.
They stop agreeing the moment the FIRST frame has an atom outside, because
``socket_interface.cc:800`` reconstructs the step as
``new_coords - getAtomPositionsCart()``, subtracting the wrapped copy from the
unwrapped one:

    ASE holds        frac x = 1.0004  ->  24.019 Bohr
    DFT-FE holds     frac x = 0.0004  ->   0.011 Bohr
    reconstructed step = true step + one lattice vector

On the real system that read 24.02 Bohr against a 24.008 Bohr edge, every ionic
step, which is above ``moveAtoms.cc:258``'s hardcoded 0.5 Bohr threshold, so
DFT-FE rebuilt the vself bins from scratch (``createAtomBins``, 14.0 s) instead
of updating their boundary conditions (``updateBinsBc``, 6.8 s), and died in the
third rebuild.

On the real system the energies stayed right while this happened -- x + a is the
same point in a periodic crystal -- which is why it went unnoticed for a whole
campaign, and the first version of this case said an energy comparison therefore
could not detect it at all. **Measurement says otherwise.** Job 8768918 ran this
case against both binaries: with the fold the two arms agree to 1.03e-10 Ha,
without it they diverge to 3.27e-06 Ha, four and a half orders apart. The
physical geometry is indeed the same either way, but without the fold the
oversized displacement crosses ``moveAtoms.cc:258``'s 0.5 Bohr threshold, the
vself bins are rebuilt rather than updated, and the discretisation the forces are
computed on shifts -- which BFGS then compounds into the next step. So the energy
comparison does discriminate, given a tolerance placed between the two.

That is why this case asserts two independent things, either of which fails on a
binary without the fold:

    1. the two arms agree numerically, at 1e-8 Ha (the physics, plus the
       discretisation stability that the fold buys)
    2. the fold actually fired, by name, in arm A's log and NOT in arm B's (the
       mechanism)

Assertion 2 is the one that cannot be satisfied by accident: 8768918 recorded
2 folds in arm A and 0 in arm B with the fix, and 0 in both without it.

TWO ARMS

    A   atom 0 starts at exactly +1 lattice vector along x, i.e. outside the
        cell, so the deck is wrapped and the socket sends unwrapped -- the
        failing configuration
    B   the same geometry with atom 0 inside -- the configuration that always
        worked

Both take the same number of ionic steps from physically identical starting
points, so every energy along the way must agree.

Gamma-only, real binary, 2x2 in plane (a 1x1 cell is too small for
vselfBinsManager's self-potential balls).
"""
import os

import numpy as np
from ase.build import fcc111
from ase.optimize import BFGS
from ase.units import Bohr, Hartree

from dftfe_ase import DFTFE

# 1e-8 Ha, not the suite's usual 1e-10, and chosen from measurement rather than
# from principle. This case compares TRAJECTORIES, not a fixed configuration:
# the two arms' forces differ in their last bits, BFGS turns that into slightly
# different steps, and the difference compounds. Job 8768918 measured both arms
# on both binaries:
#
#            step 0        step 1        step 2
#   fold     2.98e-13      5.76e-12      1.03e-10      <- compounding, physics fine
#   no fold  2.98e-13      8.40e-08      3.27e-06      <- the defect
#
# 1e-8 sits with 100x margin above the fold's worst and 300x below the defect's.
# A tighter 1e-10 fails the correct binary on step 2 by a factor of 1.03, which
# is a flaky test, not a finding.
BOUNDARY_TOL_HA = 1e-8
NSTEPS = 2

# The fold announces itself at verbosity >= 1. Matching on the message rather
# than on a displacement number keeps this test independent of the cell.
FOLD_MARKER = "Minimum image applied to the displacement of"


def _slab(atom0_outside: bool):
    """8-atom Al(111) slab with atom 0 pushed off its site to create forces."""
    atoms = fcc111("Al", size=(2, 2, 2), vacuum=None)
    cell = atoms.get_cell()[:]
    cell[2, 2] = 24.0
    atoms.set_cell(cell)
    atoms.pbc = [True, True, False]
    atoms.center(axis=2)
    # See lattice_translation_periodic._slab: fcc111 starts atom 0 outside the
    # cell at fractional x = -0.167, so the baseline has to be wrapped or the
    # two arms come out reversed.
    atoms.wrap()

    # Break the symmetry so the optimizer has something to do. Without this the
    # ideal slab's forces are ~0, BFGS takes a null step, and no displacement is
    # ever reconstructed -- the case would pass while testing nothing.
    pos = atoms.get_positions()
    pos[0] += np.array([0.12, 0.05, 0.0])
    if atom0_outside:
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
        log_file=f"test_restart_across_boundary_{tag}.log",
    )


def _relax(atoms, calc, tag):
    """NSTEPS ionic steps, returning the energy at each evaluated geometry."""
    energies = []
    with calc:
        atoms.calc = calc
        opt = BFGS(atoms, logfile=f"test_restart_across_boundary_{tag}.opt")
        opt.attach(lambda: energies.append(
            atoms.get_potential_energy() / Hartree))
        opt.run(fmax=1e-6, steps=NSTEPS)
    return energies


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8,
        use_device: bool = True):
    outside = _slab(atom0_outside=True)
    inside = _slab(atom0_outside=False)

    # --- the case is only meaningful if arm A really starts outside ---
    frac_x = (outside.get_positions()
              @ np.linalg.inv(outside.get_cell()[:]))[:, 0]
    if frac_x[0] <= 1.0:
        raise AssertionError(
            f"arm A's atom 0 has fractional x = {frac_x[0]:.6f}, not outside "
            "the cell; the case is not exercising the fold"
        )
    # Not asserted as exactly zero: (y + nudge) + a1 minus (y + nudge) is not
    # bitwise a1 in floating point once any other arithmetic has touched the
    # atom. 1e-12 A is 4 orders below the mesh resolution and 12 below the
    # lattice vector, so it still pins "exactly one lattice vector" in the only
    # sense available.
    OFFSET_TOL_A = 1e-12
    delta = outside.get_positions()[0] - inside.get_positions()[0]
    residual = float(np.max(np.abs(delta - outside.get_cell()[0])))
    if residual > OFFSET_TOL_A:
        raise AssertionError(
            f"arm A is not offset from arm B by one lattice vector: residual "
            f"{residual:.3e} A (tol {OFFSET_TOL_A:.0e})"
        )

    energies_a = _relax(outside, _make_calc(
        dftfe_bin, psp_library, np_tasks, use_device, "A"), "A")
    energies_b = _relax(inside, _make_calc(
        dftfe_bin, psp_library, np_tasks, use_device, "B"), "B")

    if len(energies_a) != len(energies_b):
        raise AssertionError(
            f"the two arms took different numbers of steps: "
            f"{len(energies_a)} vs {len(energies_b)}. They start from the same "
            "physical geometry, so the optimizer must see the same forces."
        )
    if len(energies_a) < 2:
        raise AssertionError(
            f"only {len(energies_a)} evaluation(s) recorded; the case needs at "
            "least one ionic step for a displacement to be reconstructed at all"
        )

    worst = max(abs(a - b) for a, b in zip(energies_a, energies_b))
    print(f"  boundary restart: {len(energies_a)} evaluations, "
          f"worst |dE| = {worst:.3e} Ha")
    for k, (a, b) in enumerate(zip(energies_a, energies_b)):
        print(f"    step {k}: A {a:.12f}  B {b:.12f}  dE {a - b:+.3e} Ha")
    if worst >= BOUNDARY_TOL_HA:
        raise AssertionError(
            f"an ionic step taken with atom 0 outside the cell gave different "
            f"energies from the same step taken with it inside: worst |dE| = "
            f"{worst:.3e} Ha (tol {BOUNDARY_TOL_HA:.0e}). The reconstructed "
            "displacement is not being folded to its minimum image."
        )

    # --- and the mechanism, not just the number ---
    log_a = f"test_restart_across_boundary_A.log"
    log_b = f"test_restart_across_boundary_B.log"
    folds_a = folds_b = 0
    if os.path.exists(log_a):
        folds_a = open(log_a).read().count(FOLD_MARKER)
    if os.path.exists(log_b):
        folds_b = open(log_b).read().count(FOLD_MARKER)
    print(f"  fold reports: arm A {folds_a}, arm B {folds_b}")
    if folds_a == 0:
        raise AssertionError(
            "arm A's log never reported the minimum-image fold, so the "
            "displacement it applied was the raw one -- step plus a lattice "
            "vector. The energies matching does not rescue this: x + a is the "
            "same point in a periodic crystal, so they match either way. This "
            "is the assertion that fails on a binary without the fold."
        )
    if folds_b != 0:
        raise AssertionError(
            f"arm B folded {folds_b} time(s) with every atom inside the cell. "
            "Nothing should be folded there, and a fold on an ordinary step "
            "means the criterion is firing on steps it must leave alone."
        )

    return energies_a[0], None, None
