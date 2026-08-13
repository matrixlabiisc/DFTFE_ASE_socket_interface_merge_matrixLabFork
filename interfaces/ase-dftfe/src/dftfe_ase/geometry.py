# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Placing atoms inside the cell the way DFT-FE requires along open directions.

DFT-FE's contract, read off ``dft.cc`` rather than assumed
(``initImageChargesUpdateKPoints``, the sanity check on fractional coordinates):

* **periodic** axis -- fractional coordinate in ``[-1e-6, 1+1e-6]``. An atom
  outside is its own periodic image, so DFT-FE folds it back itself, both in
  ``reinit()`` and in ``updateAtomPositionsAndMoveMesh()``.
* **non-periodic** axis -- fractional coordinate strictly inside
  ``(1e-6, 1-1e-6)``. Nothing folds it, because there is no image to fold to:
  the cell is the entire universe along that axis, there is no mesh outside it,
  and moving an atom by one cell length would carry it through vacuum to a
  genuinely different structure.

So along an open direction the atom must simply *be* inside, and DFT-FE aborts
if it is not. That is easy to violate by accident -- ``ase.build.surface()``
puts the bottom layer at exactly ``z = 0``, which is fractional ``0.0``, which
is outside the open interval by DFT-FE's reckoning -- and the abort happens
seconds into a queued job.

There is exactly one repair that is not a lie about the physics: translate
**every atom by the same vector**. A rigid translation leaves every interatomic
distance unchanged, so the energy and all forces are unchanged; only the
system's placement inside its own vacuum moves. That is the operation this
module computes. What it deliberately does *not* do is fold atoms individually
along an open axis -- that would tear the slab in half.

The shift is chosen to centre the atoms, which is the placement with the most
clearance from both faces, and is where DFT-FE itself puts the system at the end
of setup (``internaldft::convertToCellCenteredCartesianCoordinates``).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("dftfe_ase.geometry")

__all__ = ["OpenDirectionError", "fractional", "needs_shift", "rigid_shift",
           "place_inside"]

# dft.cc uses 1e-6 for the fractional-coordinate sanity check. Match it exactly
# rather than picking our own: a looser bound here just moves the abort deeper
# into DFT-FE, where the message no longer says which atom or which axis.
DFTFE_FRAC_TOL = 1e-6

# Clearance we would LIKE after a shift, in fractional units -- enough that a
# later step wandering by a fraction of an Angstrom does not immediately re-trip
# the bound. This is an aspiration, not a requirement: centring already yields
# the largest margin the cell allows, and a system that cannot reach this still
# runs. Only DFTFE_FRAC_TOL decides what DFT-FE accepts.
TARGET_MARGIN = 1e-3


class OpenDirectionError(ValueError):
    """The atoms cannot be made to fit along a non-periodic direction.

    Raised only when no rigid translation can help -- the system is longer than
    the cell along an open axis -- because that is a cell that is genuinely too
    small, not a placement that can be repaired.
    """


def fractional(positions, cell):
    """Cartesian -> fractional. Rows of ``cell`` are the cell vectors."""
    cell = np.asarray(cell, float)
    # A cell row of zeros is legal in ASE and is what you hold after
    # Atoms(cell=[a, b, 0]) before center(vacuum=...). np.linalg.inv raises
    # LinAlgError on it, naming neither the axis nor the fix; and a merely tiny
    # row does not raise at all, it returns a garbage offset that would flow
    # straight to the solver. Reject both.
    lengths = np.linalg.norm(cell, axis=1)
    degenerate = np.flatnonzero(lengths < 1e-8)
    if degenerate.size:
        raise OpenDirectionError(
            f"cell vector(s) {[ 'abc'[i] for i in degenerate ]} have zero length "
            f"(lengths {np.array2string(lengths, precision=3)}). DFT-FE needs a "
            "finite cell along every axis, including non-periodic ones, because "
            "the mesh has to end somewhere. Set a box and place the atoms in it, "
            "e.g. atoms.center(vacuum=8.0)."
        )
    if abs(np.linalg.det(cell)) < 1e-8:
        raise OpenDirectionError(
            f"cell is singular (det = {np.linalg.det(cell):.3e}); its vectors are "
            "coplanar or collinear, so fractional coordinates are undefined."
        )
    return np.asarray(positions, float) @ np.linalg.inv(cell)


def needs_shift(positions, cell, pbc):
    """Axes where DFT-FE would abort: open, and an atom outside (0, 1)."""
    if len(np.asarray(positions, float)) == 0:
        return []
    frac = fractional(positions, cell)
    bad = []
    for axis in range(3):
        if pbc[axis]:
            continue  # DFT-FE folds these itself; leave them continuous.
        lo, hi = frac[:, axis].min(), frac[:, axis].max()
        if lo <= DFTFE_FRAC_TOL or hi >= 1.0 - DFTFE_FRAC_TOL:
            bad.append(axis)
    return bad


def rigid_shift(positions, cell, pbc, *, unit="A"):
    """Cartesian translation putting every atom strictly inside along open axes.

    Returns a ``(3,)`` vector to add to *every* position. It is exactly zero
    when nothing needs moving, and zero along periodic axes always.

    Raises :class:`OpenDirectionError` if the atoms span more than the cell
    along an open axis, since no translation fixes that.
    """
    positions = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    delta = np.zeros(3)
    axes = needs_shift(positions, cell, pbc)
    if not axes:
        return delta @ cell
    frac = fractional(positions, cell)

    for axis in axes:
        lo, hi = frac[:, axis].min(), frac[:, axis].max()
        span = hi - lo
        # "Cannot fit" is decided by what DFT-FE accepts (DFTFE_FRAC_TOL), not
        # by the clearance we would prefer. Testing against TARGET_MARGIN here
        # rejected systems a shift could place perfectly well -- e.g. span
        # 0.999 in a 100 A cell, which centres to margin 5e-4, a thousand times
        # DFT-FE's bound -- while telling the user to enlarge a cell that was
        # already big enough.
        if span >= 1.0 - 2.0 * DFTFE_FRAC_TOL:
            length = np.linalg.norm(cell[axis])
            raise OpenDirectionError(
                f"atoms span {span * length:.3f} {unit} along non-periodic axis "
                f"{'abc'[axis]} but the cell is only {length:.3f} {unit} long there. "
                "No rigid shift can place them inside, and DFT-FE has no mesh "
                "outside the cell along an open direction. Enlarge the cell / add "
                f"vacuum: at least {span * length:.3f} {unit} plus clearance."
            )
        # Centre: the placement with the largest clearance from both faces.
        delta[axis] = (1.0 - span) / 2.0 - lo
        if (1.0 - span) / 2.0 < TARGET_MARGIN:
            log.warning(
                "axis %s: centring leaves only %.2e fractional clearance "
                "(%.3f %s). Legal, but a later step moving outward will not be "
                "repairable -- consider more vacuum.",
                "abc"[axis], (1.0 - span) / 2.0,
                (1.0 - span) / 2.0 * np.linalg.norm(cell[axis]), unit,
            )

    return delta @ cell


def place_inside(positions, cell, pbc, *, context="", unit="A"):
    """``positions`` moved rigidly inside the cell along open axes.

    Convenience wrapper that logs what it did. Returns ``(positions, shift)``;
    ``positions`` is the input array untouched when no shift was needed.
    """
    shift = rigid_shift(positions, cell, pbc, unit=unit)
    if not shift.any():
        return positions, shift
    log.warning(
        "%satoms lay outside the cell along a non-periodic direction; translating "
        "all of them rigidly by (%.4f, %.4f, %.4f) %s to centre them. "
        "Interatomic distances, energy and forces are unchanged.",
        f"{context}: " if context else "", *shift, unit,
    )
    return positions + shift, shift
