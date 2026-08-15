#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Quick check that the build supports meta-GGA MGGA-R2SCAN (used by Li2O).

Tiny N2 single point with xc=MGGA-R2SCAN + a high density quadrature (meta-GGA
needs it). If the build lacks the functional it errors at parameter parse
(fast); a finite energy means MGGA-R2SCAN is available and the socket passes it.
"""

import os
import sys

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
PSP_DIR = "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library"

box = np.array([40.0, 40.0, 40.0])
c = box / 2.0
atoms = Atoms("N2", positions=(np.array([[c[0] - 1.3, c[1], c[2]], [c[0] + 1.3, c[1], c[2]]])) * Bohr,
              cell=np.diag(box) * Bohr, pbc=[False, False, False])

with DFTFE(bin_dir=BUILD_MERGED, launcher="mpirun", nproc=2, use_device=True,
           psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
           xc="MGGA-R2SCAN", density_quadrature_rule=10,
           mesh_size=1.0, polynomial_order=6, atom_ball_radius=3.0,
           tolerance=5e-5, fermi_temp=500.0, num_bands=15,
           compute_forces=False,  # N.upf has NLCC; MGGA+ION FORCE+NLCC is unimplemented
           verbosity=2, log_file="mgga_check.log") as calc:
    atoms.calc = calc
    e = atoms.get_potential_energy() / Hartree

print(f"[mgga] MGGA-R2SCAN N2 energy = {e:.6f} Ha", flush=True)
assert np.isfinite(e) and e < 0.0
print("[mgga] PASS: build supports MGGA-R2SCAN and the socket passes it", flush=True)
