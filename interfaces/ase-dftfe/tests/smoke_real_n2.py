#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Real end-to-end smoke test: drive the actual DFT-FE binary via dftfe_ase.

Same N2 geometry/params as examples/n2_gs.py (known-good), but through the NEW
dftfe_ase package with zero-config auto-launch (cluster='matrix'). Proves the
redesigned package drives real DFT-FE over the socket and returns sane values.

NOTE: validates the CORE protocol against the installed binary; the build here
(build_gpu, dated 2026-05) predates the pGD merge, so this does not exercise the
merged code / new params — that needs a rebuild (P0.2).
"""

import os
import sys

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from dftfe_ase import DFTFE  # noqa: E402

PSP = "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library"

box_bohr = np.array([40.0, 40.0, 40.0])
cell = np.diag(box_bohr) * Bohr
center = box_bohr / 2.0
rel_bohr = np.array([[-1.3, 0.0, 0.0], [1.3, 0.0, 0.0]])
positions = (center + rel_bohr) * Bohr
atoms = Atoms("N2", positions=positions, cell=cell, pbc=[False, False, False])

with DFTFE(
    cluster="matrix",          # zero-config: resolves build_gpu binaries + mpirun launcher
    nproc=2,
    use_device=True,
    psp_path=PSP,
    env={"DFTFE_PSP_PATH": PSP},
    mesh_size=1.0,
    polynomial_order=6,
    atom_ball_radius=3.0,
    tolerance=5e-5,
    scf_mixing=0.5,
    fermi_temp=500.0,
    num_bands=15,
    xc="GGA-PBE",
    compute_forces=True,
    verbosity=2,
    log_file="smoke_n2.log",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

print(f"[smoke] energy    = {energy:.6f} eV  ({energy / Hartree:.6f} Ha)")
print(f"[smoke] max|force| = {np.abs(forces).max():.6e} eV/Ang")

assert np.isfinite(energy) and energy < 0.0, f"unphysical energy: {energy}"
assert forces.shape == (2, 3) and np.all(np.isfinite(forces)), "bad forces"
print("[smoke] PASS: dftfe_ase drove real DFT-FE and returned sane energy + forces")
