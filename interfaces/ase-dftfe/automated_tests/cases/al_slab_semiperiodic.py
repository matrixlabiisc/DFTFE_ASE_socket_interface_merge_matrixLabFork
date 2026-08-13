# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Test case: Al(111) slab, periodic in x/y and open along z (real binary).

Regression cover for out-of-cell placement along a NON-periodic direction --
the class of bug fixed by ``dftfe_ase.geometry`` plus the C++ backstop in
``dftfeWrapper::reinit``. This is case A/B of
``runs/open_dir/check_open_direction.py``, folded into the suite.

``ase.build.fcc111(..., vacuum=None)`` puts the bottom layer at exactly z = 0,
i.e. fractional 0.0, which DFT-FE rejects along an open axis (it requires
strictly inside ``(1e-6, 1-1e-6)``; there is no periodic image to fold to).
The calculator repairs that with a single rigid translation of the whole
system, which leaves every interatomic distance -- and therefore the energy and
all forces -- unchanged.

Two things are asserted here, before any DFT is done:

* the raw slab really is out of the cell along z, so the case still exercises
  the placement path rather than silently degrading into an ordinary slab run;
* the same slab centred by hand (``atoms.center(axis=2)``) needs no shift.

The energy this returns is the *auto-placed* one. Pinning it is what makes the
invariant testable: it must equal the hand-centred energy, and it must not
drift when the placement code changes. Both A and B are computed and compared
in-case, so a divergence fails here rather than waiting for a reference diff.

2x2 in plane rather than 1x1 deliberately: at 1x1 the periodic cell is only
5.4 x 4.7 Bohr and ``vselfBinsManager`` cannot fit its self-potential balls
inside it, which kills the run *after* the coordinate check has passed.

Gamma-only (no ``mp_grid``), so this runs on the real binary.
"""
import os

import numpy as np
from ase.build import fcc111
from ase.units import Bohr, Hartree

from dftfe_ase import DFTFE
from dftfe_ase.geometry import fractional, needs_shift

# Energies of the auto-placed and the hand-centred slab must agree to at least
# this, in Hartree. A rigid translation is exact in exact arithmetic; what is
# left is the finite-element mesh not being translation-invariant, which is
# well below the 1e-10 Ha the suite pins energies to.
PLACEMENT_TOL_HA = 1e-10


def _slab(vacuum_centred: bool):
    """8-atom Al(111) slab, open along z. Bottom layer at z=0 unless centred."""
    atoms = fcc111("Al", size=(2, 2, 2), vacuum=None)
    cell = atoms.get_cell()[:]
    cell[2, 2] = 24.0
    atoms.set_cell(cell)
    atoms.pbc = [True, True, False]
    if vacuum_centred:
        atoms.center(axis=2)
    return atoms


def _make_calc(dftfe_bin: str, psp_library: str, np_tasks: int,
               use_device: bool, tag: str):
    # ATOM BALL RADIUS and SELF POTENTIAL RADIUS are deliberately left unset so
    # DFT-FE picks them adaptively -- which also exercises the "unset kwarg
    # reaches the solver at DFT-FE's own default" invariant.
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
        log_file=f"test_al_slab_semiperiodic_{tag}.log",
    )


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8,
        use_device: bool = True):
    raw = _slab(vacuum_centred=False)
    centred = _slab(vacuum_centred=True)

    # --- the case is only meaningful if the raw slab is genuinely outside ---
    frac_z = fractional(raw.get_positions(), raw.get_cell()[:])[:, 2]
    if frac_z.min() != 0.0:
        raise AssertionError(
            f"raw slab was supposed to sit on the z=0 face, but min fractional "
            f"z is {frac_z.min():.3e}; ase.build.fcc111 behaviour changed and "
            "this case no longer covers out-of-cell placement"
        )
    if needs_shift(raw.get_positions(), raw.get_cell()[:], raw.get_pbc()) != [2]:
        raise AssertionError(
            "raw slab no longer needs a shift along z; case is not exercising "
            "the open-direction placement path"
        )
    if needs_shift(centred.get_positions(), centred.get_cell()[:],
                   centred.get_pbc()):
        raise AssertionError(
            "hand-centred slab still reports needing a shift; "
            "geometry.needs_shift or ase.center(axis=2) changed"
        )

    # --- A: out-of-cell slab, placement active ---
    calc_a = _make_calc(dftfe_bin, psp_library, np_tasks, use_device, "A")
    with calc_a:
        raw.calc = calc_a
        energy_a_ha = raw.get_potential_energy() / Hartree
        forces_a = raw.get_forces() / (Hartree / Bohr)

    # --- B: same slab centred by hand; must be the same calculation ---
    calc_b = _make_calc(dftfe_bin, psp_library, np_tasks, use_device, "B")
    with calc_b:
        centred.calc = calc_b
        energy_b_ha = centred.get_potential_energy() / Hartree
        forces_b = centred.get_forces() / (Hartree / Bohr)

    d_energy = abs(energy_a_ha - energy_b_ha)
    d_force = float(np.max(np.abs(forces_a - forces_b)))
    if d_energy >= PLACEMENT_TOL_HA:
        raise AssertionError(
            f"auto-placed and hand-centred slab disagree: |dE| = {d_energy:.3e} "
            f"Ha (tol {PLACEMENT_TOL_HA:.0e}), max|dF| = {d_force:.3e} Ha/Bohr. "
            "The rigid translation is not energy-preserving as it must be."
        )

    return energy_a_ha, forces_a, None
