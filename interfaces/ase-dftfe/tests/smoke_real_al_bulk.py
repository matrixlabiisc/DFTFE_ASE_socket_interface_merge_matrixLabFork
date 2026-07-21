#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Real end-to-end smoke test for the PERIODIC / k-point path (COMPLEX binary).

FCC Al with a 2x2x2 Monkhorst-Pack mesh, driven through the NEW dftfe_ase
package with zero-config auto-launch. Because k-points are sampled, the package
must auto-select the *complex* binary. Exercises the half of the science surface
the N2 test does not: periodicity, k-point sampling + parallelization (npkpt),
and stress. Params mirror examples/bulk_fcc_al_gs.py (known-good).
"""

import os
import sys

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from dftfe_ase import DFTFE  # noqa: E402

PSP_DIR = "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library"
AL_UPF = os.path.join(PSP_DIR, "Al.upf")

box_bohr = np.array([7.6, 7.6, 7.6])
cell = np.diag(box_bohr) * Bohr
frac = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
positions = frac @ cell
atoms = Atoms("Al4", positions=positions, cell=cell, pbc=[True, True, True])

calc = DFTFE(
    cluster="matrix",
    nproc=4,
    use_device=True,
    psp_path=AL_UPF,
    env={"DFTFE_PSP_PATH": PSP_DIR},
    mesh_size=1.6,
    polynomial_order=5,
    tolerance=5e-5,
    fermi_temp=500.0,
    mp_grid=(2, 2, 2),
    mp_grid_shift=(1, 1, 1),
    use_time_reversal_symmetry=True,
    npkpt=2,
    xc="GGA-PBE",
    compute_stress=True,
    compute_forces=False,
    verbosity=2,
    log_file="smoke_al.log",
)

# k-points sampled -> the package must pick the complex binary.
argv = calc.build_launch_argv(atoms.get_pbc())
assert argv[-1].endswith("complex/dftfe"), f"expected complex binary, got {argv[-1]}"
print(f"[smoke] launch: {' '.join(argv)}")

with calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    stress = atoms.get_stress()

print(f"[smoke] energy = {energy:.6f} eV  ({energy / Hartree:.6f} Ha)")
print(f"[smoke] stress (Voigt, eV/Ang^3) = {np.array2string(stress, precision=6)}")

assert np.isfinite(energy) and energy < 0.0, f"unphysical energy: {energy}"
assert stress.shape == (6,) and np.all(np.isfinite(stress)), "bad stress"
print("[smoke] PASS: periodic k-point run via complex binary returned sane energy + stress")
