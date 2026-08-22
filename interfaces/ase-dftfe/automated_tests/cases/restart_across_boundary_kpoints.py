# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""
Test case: the same relation as ``restart_across_boundary``, on the COMPLEX
binary.

WHY THIS EXISTS, WHEN restart_across_boundary ALREADY TESTS THE FOLD

Because until this case there was no test in which the changed function ever
*ran* in a complex-arithmetic build. The fold and the post-move invariant are
compiled into libdftfeComplex.so -- checked, both diagnostic strings are present
-- but nothing had executed them there:

  * the suite's two complex cases (``graphene_periodic``, ``al_bulk_periodic``)
    are single points. They call ``get_forces()`` once, take no ionic step, and
    so never enter ``dftfeWrapper::updateAtomPositions()``.
  * DFT-FE's own complex ctest suite has exactly one test that moves atoms,
    ``complex/mdLGPSNVE_01``, and it moves them through DFT-FE's internal MD
    driver, not through the wrapper. ``nebH3`` -- the one ctest path that does
    reach the wrapper -- exists only in the real suite.

So "the complex binary passes everything" was true and did not mean the fix
worked there. This case closes that by construction: 3D periodic with a
Monkhorst-Pack grid forces the complex binary, and it takes real ionic steps
with one atom outside the cell.

WHAT IT ASSERTS

Identical to ``restart_across_boundary``: one arm starts with atom 0 pushed out
of the cell by exactly one lattice vector, the other with it inside, and the two
must walk the same trajectory. Comparing two arms rather than a stored number
means it needs no reference and cannot go stale (``RUN_ONLY``). The energy
agreement alone is not sufficient evidence -- ``x + a`` is the same point in a
periodic crystal -- so the fold count from each arm's log is asserted too.
"""
import os

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.units import Bohr, Hartree

from dftfe_ase import DFTFE

# Same bar and same reasoning as restart_across_boundary: this compares
# trajectories, not a fixed configuration, so the arms' last-bit force
# differences compound through BFGS. See that file for the measured separation
# between a folding and a non-folding binary (1.03e-10 against 3.27e-06).
BOUNDARY_TOL_HA = 1e-8
NSTEPS = 2
FOLD_MARKER = "Minimum image applied to the displacement of"


def _cell_and_atoms(atom0_outside: bool):
    """4-atom FCC Al cell, k-point sampled, atom 0 nudged off its site.

    Geometry is al_bulk_periodic's, which is already validated against a native
    pGD reference on this binary -- so if this case fails, the new thing is the
    ionic step, not the system.
    """
    cell_ang = np.diag(np.array([7.6, 7.6, 7.6])) * Bohr
    frac = np.array([[0.0, 0.0, 0.0],
                     [0.0, 0.5, 0.5],
                     [0.5, 0.0, 0.5],
                     [0.5, 0.5, 0.0]])
    atoms = Atoms(symbols=["Al"] * 4, positions=frac @ cell_ang,
                  cell=cell_ang, pbc=[True, True, True])

    # Break the symmetry, or the forces are ~0, BFGS takes a null step, and no
    # displacement is ever reconstructed -- the case would pass testing nothing.
    pos = atoms.get_positions()
    pos[0] += np.array([0.12, 0.05, 0.03])
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
        # The k-point grid is what makes this the complex binary rather than a
        # duplicate of restart_across_boundary.
        mp_grid=(2, 2, 2),
        mp_grid_shift=(1, 1, 1),
        use_time_reversal_symmetry=True,
        npkpt=2,
        compute_forces=True,
        compute_stress=False,
        use_device=use_device,
        verbosity=1,
        log_file=f"test_restart_across_boundary_kpoints_{tag}.log",
    )


def _relax(atoms, calc, tag):
    energies = []
    with calc:
        atoms.calc = calc
        opt = BFGS(atoms,
                   logfile=f"test_restart_across_boundary_kpoints_{tag}.opt")
        opt.attach(lambda: energies.append(
            atoms.get_potential_energy() / Hartree))
        opt.run(fmax=1e-6, steps=NSTEPS)
    return energies


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8,
        use_device: bool = True):
    outside = _cell_and_atoms(atom0_outside=True)
    inside = _cell_and_atoms(atom0_outside=False)

    frac_x = (outside.get_positions()
              @ np.linalg.inv(outside.get_cell()[:]))[:, 0]
    if frac_x[0] <= 1.0:
        raise AssertionError(
            f"arm A's atom 0 has fractional x = {frac_x[0]:.6f}, not outside "
            "the cell; the case is not exercising the fold"
        )

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
            f"{len(energies_a)} vs {len(energies_b)}"
        )
    if len(energies_a) < 2:
        raise AssertionError(
            f"only {len(energies_a)} evaluation(s); the case needs at least one "
            "ionic step for a displacement to be reconstructed at all"
        )

    worst = max(abs(a - b) for a, b in zip(energies_a, energies_b))
    print(f"  boundary restart (k-points, complex binary): "
          f"{len(energies_a)} evaluations, worst |dE| = {worst:.3e} Ha")
    for k, (a, b) in enumerate(zip(energies_a, energies_b)):
        print(f"    step {k}: A {a:.12f}  B {b:.12f}  dE {a - b:+.3e} Ha")
    if worst >= BOUNDARY_TOL_HA:
        raise AssertionError(
            f"an ionic step taken with atom 0 outside the cell gave different "
            f"energies from the same step with it inside: worst |dE| = "
            f"{worst:.3e} Ha (tol {BOUNDARY_TOL_HA:.0e})"
        )

    folds = {}
    for tag in ("A", "B"):
        path = f"test_restart_across_boundary_kpoints_{tag}.log"
        folds[tag] = open(path).read().count(FOLD_MARKER) if os.path.exists(path) else 0
    print(f"  fold reports: arm A {folds['A']}, arm B {folds['B']}")
    if folds["A"] == 0:
        raise AssertionError(
            "arm A's log never reported the fold, so the complex binary applied "
            "the raw displacement -- step plus a lattice vector. The energies "
            "matching does not rescue this: x + a is the same point in a "
            "periodic crystal, so they match either way. This is the assertion "
            "that fails on a binary without the fold, and the reason this case "
            "exists at all."
        )
    if folds["B"] != 0:
        raise AssertionError(
            f"arm B folded {folds['B']} time(s) with every atom inside the cell"
        )

    return energies_a[0], None, None
