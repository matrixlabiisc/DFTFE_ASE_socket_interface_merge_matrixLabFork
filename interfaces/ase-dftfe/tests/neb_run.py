#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Li2O NEB via the ASE ecosystem (ase.mep.NEB) driven by DFT-FE forces.

Exercises the interface's NEB "ecosystem product": a climbing-image NEB run with
a SINGLE persistent socket DFT-FE process shared across all images
(allow_shared_calculator=True) -> serial image evaluation with density reuse, one
process on the full node (no GPU oversubscription). Starts from the benchmark's
converged MEP (output/FinalPath.txt) so the band re-relaxes to the *current*-build
minimum energy path in few steps. Reports the activation barrier vs native
DFT-FE NEB (241.03 meV, old build) / QE (244.9 meV).

Env: NEB_NPROC (ranks/GPUs), NEB_STEPS (optimizer step cap), NEB_CLIMB (1/0).
"""

import os
import sys

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.mep import NEB
from ase.optimize import FIRE
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

NEBDIR = "/home/pa01/Mehul/dftfe-benchmarks/accuracyBenchmarks/NEB/Li2O"
BENCH = f"{NEBDIR}/dftfe"
BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
NIMAGES, NAT = 7, 95
REF_BARRIER_MEV = 241.03
FMAX_HA_BOHR = 4e-4                                  # native PATH THRESHOLD
FMAX_EV_ANG = FMAX_HA_BOHR * Hartree / Bohr          # -> 0.02057 eV/Ang

cell = np.loadtxt(f"{BENCH}/domainVectors.inp") * Bohr
rows0 = [ln.split() for ln in open(f"{BENCH}/coordinates.inp") if len(ln.split()) >= 5][:NAT]
symbols = [chemical_symbols[int(float(r[0]))] for r in rows0]
fp = np.loadtxt(f"{BENCH}/output/FinalPath.txt")     # atom x (7 images * xyz)
images = [Atoms(symbols=symbols, scaled_positions=fp[:, 3 * k:3 * k + 3], cell=cell, pbc=True)
          for k in range(NIMAGES)]
print(f"[neb] {NIMAGES} images x {NAT} atoms; fmax target {FMAX_EV_ANG:.4f} eV/Ang "
      f"({FMAX_HA_BOHR:.0e} Ha/Bohr)", flush=True)

nproc = int(os.environ.get("NEB_NPROC", "8"))
steps = int(os.environ.get("NEB_STEPS", "12"))
climb = os.environ.get("NEB_CLIMB", "1") == "1"
calc = DFTFE(
    bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc, use_device=True,
    psp_path={"Li": f"{NEBDIR}/Li.upf", "O": f"{NEBDIR}/O.upf"},
    xc="GGA-PBE", polynomial_order=7, mesh_size=1.2, atom_ball_radius=10.0,
    tolerance=1e-7, mixing_scheme="ANDERSON", scf_mixing=0.2, fermi_temp=500.0,
    num_bands=220, compute_forces=True, verbosity=1, log_file="neb_run.log",
)

with calc:
    for img in images:
        img.calc = calc                              # one shared persistent process
    neb = NEB(images, climb=climb, k=0.1, allow_shared_calculator=True)
    opt = FIRE(neb, logfile="neb_opt.log", trajectory="neb.traj")

    def report():
        E = np.array([img.get_potential_energy() for img in images]) / Hartree
        imax = int(np.argmax(E))
        bar = (E[imax] - E[0]) * Hartree * 1000.0
        print(f"[neb] step barrier = {bar:.2f} meV (max img {imax}); "
              f"profile(meV)={[round((e-E[0])*Hartree*1000,1) for e in E]}", flush=True)
        return bar

    print(f"[neb] climb={climb}, step cap={steps}; starting FIRE...", flush=True)
    opt.run(fmax=FMAX_EV_ANG, steps=steps)
    barrier = report()
    converged = opt.converged()

d = abs(barrier - REF_BARRIER_MEV)
print(f"[neb] FINAL activation barrier = {barrier:.2f} meV  (converged={converged}, "
      f"fmax_target={FMAX_HA_BOHR:.0e} Ha/Bohr)", flush=True)
print(f"[neb] native DFT-FE NEB {REF_BARRIER_MEV} meV (old build) / QE 244.9 meV; |diff|={d:.2f} meV", flush=True)
