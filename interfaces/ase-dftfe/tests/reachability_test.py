#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Validate multi-node socket reachability.

Runs one single-point (Al bulk, complex) via **srun on 2 nodes**. DFT-FE rank 0
must connect back to the Python server, which now binds 0.0.0.0 and advertises
the node hostname (NOT 127.0.0.1). If an energy comes back, cross-node
connect-back works — the key prerequisite for Polaris/Frontier/Aurora, where the
launcher can place rank 0 on a different node than the Python driver.
"""

import os
import socket
import sys

import numpy as np
from ase.units import Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402
from parity_check import COMMON, DEMOS, PSP_DIR, atoms_ex2  # noqa: E402

BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
NPROC = int(os.environ.get("REACH_NPROC", "16"))

print(f"[reach] python host (advertised) = {socket.gethostname()}, nproc = {NPROC}, "
      f"launcher = srun", flush=True)

atoms = atoms_ex2()
params = {k: v for k, v in DEMOS["ex2"]["params"].items()
          if k not in ("compute_stress",)}

with DFTFE(bin_dir=BUILD_MERGED, launcher="srun", nproc=NPROC, use_device=True,
           psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
           verbosity=2, log_file="reach.log", **COMMON, **params) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy() / Hartree

print(f"[reach] energy = {energy:.8f} Ha", flush=True)
assert np.isfinite(energy) and energy < 0.0, f"unphysical energy {energy}"
print("[reach] PASS: DFT-FE rank 0 connected back across nodes via srun "
      "(bind 0.0.0.0 + advertised hostname)", flush=True)
