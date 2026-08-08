#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Generate the BCC-Mo 6x6x6 single-vacancy inputs (431 atoms).

Reproduces ``coordinates.inp`` and ``domainVectors.inp`` from

    dftfe-benchmarks/performanceBenchmarks/DFTFEv1.0/Summit/
      GroundStateCalculations/MinWallTimes/BCCMoSuperCells/6x6x6Vac/

so the scaling study does not depend on that repo being checked out on Aurora.
Every value and the atom ordering match line for line; the only difference is
line endings (the benchmark files are CRLF, these are LF). Verified with::

    diff <(tr -d '\\r' < <benchmark>/coordinates.inp) coordinates.inp
The system is an ideal 6x6x6 BCC supercell (a = 35.7 Bohr for the full cell)
with the body-centre site at fractional (0.5, 0.5, 0.5) removed; nothing is
relaxed.

Generating both arms' geometry from one file matters for more than convenience:
force parity is compared component by component, so the native run and the ASE
run must list the atoms in the *same order*. Building the cell with
``ase.build.bulk`` would give the right physics in a different order and make
the force comparison meaningless.

Usage::

    python3 make_inputs.py [outdir]
"""

from __future__ import annotations

import os
import sys

SUPERCELL = 6
LATTICE_BOHR = 35.7          # full 6x6x6 cell edge
ATOMIC_NUMBER = 42           # Mo
VALENCE = 14                 # matches Mo_ONCV_PBE-1.0.upf
VACANCY = (0.5, 0.5, 0.5)    # fractional, body centre of the supercell
PSEUDO = "Mo_ONCV_PBE-1.0.upf"


def fractional_sites():
    """BCC sites in benchmark order: i, j, k outer loops; corner then body centre."""
    n = SUPERCELL
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for offset in (0.0, 0.5):
                    site = ((i + offset) / n, (j + offset) / n, (k + offset) / n)
                    if all(abs(a - b) < 1e-9 for a, b in zip(site, VACANCY)):
                        continue  # the vacancy
                    yield site


def write_inputs(outdir):
    os.makedirs(outdir, exist_ok=True)

    sites = list(fractional_sites())
    with open(os.path.join(outdir, "coordinates.inp"), "w") as fh:
        for x, y, z in sites:
            fh.write(f" {ATOMIC_NUMBER}   {VALENCE}  {x:.10f} {y:.10f} {z:.10f} \n")

    with open(os.path.join(outdir, "domainVectors.inp"), "w") as fh:
        for axis in range(3):
            row = [0.0, 0.0, 0.0]
            row[axis] = LATTICE_BOHR
            fh.write(" ".join(f"{v:.10f}" for v in row) + " \n")

    with open(os.path.join(outdir, "pseudo.inp"), "w") as fh:
        fh.write(f"{ATOMIC_NUMBER} {PSEUDO}\n")

    return len(sites)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    n = write_inputs(target)
    expected = 2 * SUPERCELL**3 - 1
    if n != expected:
        raise SystemExit(f"expected {expected} atoms, generated {n}")
    print(f"wrote coordinates.inp ({n} atoms), domainVectors.inp, pseudo.inp -> {target}")
