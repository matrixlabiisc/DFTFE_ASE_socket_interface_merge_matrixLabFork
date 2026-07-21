#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Overhead analysis: ASE socket interface vs native file-based DFT-FE.

Runs the SAME fixed sequence of single-point N2 geometries (a bond scan) both
ways. Using a predetermined geometry list — not a live ASE optimizer / MD
integrator — keeps the "math" identical between the two, so we measure the
*interface*, not differences between ASE's and DFT-FE's own opt/thermostat
algorithms. Reports:

  (1) per-call ASE/Python overhead: (calculate() wall - socket round-trip wall)
      / calculate() wall. The socket round-trip includes the DFT compute + the
      (tiny) serialization; the remainder is ASE-side Python (unit conversion,
      request build). Same methodology as the legacy debug_timing.
  (2) throughput: native re-inits MPI + DFT-FE (mesh, PSP) on *every* step; the
      persistent ASE server inits once, then does position updates. Total wall
      of N identical single-points shows the persistence benefit.

GPU mode. nproc from env (single-node = 8, multinode = 16).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

REPO = "/home/pa01/Mehul/DFTFE"
PSP_DIR = f"{REPO}/interfaces/ase-dftfe/psp_library"
REAL_BIN = f"{REPO}/build_gpu/release/real/dftfe"
NPROC = int(os.environ.get("OVH_NPROC", "8"))
TMP = os.environ.get("OVH_TMP", HERE)
LABEL = os.environ.get("OVH_LABEL", f"{NPROC} ranks")

HALF_SEP = [1.20, 1.25, 1.30, 1.35, 1.40]  # N half-separations, Bohr (bond scan)
BOX = 40.0  # Bohr
PARAMS = dict(
    xc="GGA-PBE", mesh_size=1.0, polynomial_order=6, atom_ball_radius=3.0,
    tolerance=5e-5, scf_mixing=0.5, max_scf_iterations=40, fermi_temp=500.0,
    num_bands=15,
)


def make_atoms(d):
    cell = np.diag([BOX, BOX, BOX]) * Bohr
    c = BOX / 2.0
    pos = np.array([[c - d, c, c], [c + d, c, c]]) * Bohr
    return Atoms("N2", positions=pos, cell=cell, pbc=[False, False, False])


def ase_run():
    totals, waits, energies = [], [], []
    calc = DFTFE(cluster="matrix", nproc=NPROC, use_device=True, compute_forces=True,
                 verbosity=1, log_file="ovh_ase.log", **PARAMS)
    for d in HALF_SEP:
        atoms = make_atoms(d)
        atoms.calc = calc
        t0 = time.perf_counter()
        e = atoms.get_potential_energy()
        totals.append(time.perf_counter() - t0)
        waits.append(calc.backend.last_wait_s)
        energies.append(e / Hartree)
    calc.close()
    return totals, waits, energies


def native_run():
    totals, energies = [], []
    demo = f"{REPO}/demo/ex1"
    for d in HALF_SEP:
        work = tempfile.mkdtemp(prefix="ovh_native_", dir=TMP)
        shutil.copy(f"{demo}/domainVectors.inp", work)
        shutil.copy(f"{demo}/pseudo.inp", work)
        shutil.copy(f"{PSP_DIR}/N.upf", work)
        with open(f"{work}/coordinates.inp", "w") as fh:
            fh.write(f"7 5 {-d:.8e} 0.0 0.0\n7 5 {d:.8e} 0.0 0.0\n")
        prm = open(f"{demo}/parameterFile_a.prm").read().replace(
            "set SOLVER MODE = GS", "set SOLVER MODE = GS\nset USE GPU = true")
        with open(f"{work}/parameterFile_a.prm", "w") as fh:
            fh.write(prm)
        env = dict(os.environ)
        env["DFTFE_PSP_PATH"] = PSP_DIR
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["mpirun", "-np", str(NPROC), REAL_BIN, "parameterFile_a.prm"],
            cwd=work, env=env, capture_output=True, text=True)
        totals.append(time.perf_counter() - t0)
        m = re.findall(r"Total free energy:\s*([-\d.eE+]+)", proc.stdout + proc.stderr)
        energies.append(float(m[-1]) if m else float("nan"))
    return totals, energies


def main():
    n = len(HALF_SEP)
    print(f"=== Overhead analysis: {LABEL} (nproc={NPROC}), {n} single-points ===")
    a_tot, a_wait, a_e = ase_run()
    n_tot, n_e = native_run()

    de = max(abs(ae - ne) for ae, ne in zip(a_e, n_e) if np.isfinite(ne))
    print(f"correctness cross-check: max|E_ase - E_native| over scan = {de:.2e} Ha")

    ovh = [(t - w) / t * 100.0 for t, w in zip(a_tot, a_wait)]
    print("\nASE per-call:   total_s   socket_wait_s   ASE_overhead_%")
    for i, (t, w, o) in enumerate(zip(a_tot, a_wait, ovh)):
        tag = " (first: server start + reinit)" if i == 0 else ""
        print(f"  call {i}:      {t:8.3f}   {w:11.3f}   {o:7.2f}{tag}")

    print(f"\nASE    total wall: {sum(a_tot):8.3f} s  (init once, then position updates)")
    print(f"native total wall: {sum(n_tot):8.3f} s  (full MPI+DFT-FE init every step)")
    print(f"throughput speedup (native/ASE): {sum(n_tot) / sum(a_tot):.2f}x")
    steady = ovh[1:] if len(ovh) > 1 else ovh
    print(f"mean steady-state ASE overhead (calls 2..N): {sum(steady) / len(steady):.2f}%")


if __name__ == "__main__":
    main()
