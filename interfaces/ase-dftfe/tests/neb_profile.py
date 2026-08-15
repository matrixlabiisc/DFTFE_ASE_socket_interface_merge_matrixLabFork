#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Li2O NEB energy profile from the benchmark's 7-image path.

Cheap first validation of the NEB benchmark: load the 7 provided images
(accuracyBenchmarks/NEB/Li2O) and compute each image energy with ONE persistent
socket DFT-FE process (sequential single-points -> density reuse across images,
the MD/relaxation hot loop). Reports the activation barrier max(E_i)-E_0 vs the
native NEB result (241.03 meV) / QE (244.9 meV).

If the provided path is the converged MEP, the barrier matches directly and a
full ase.mep.NEB optimization (neb_run.py) will converge in a few steps.
"""

import os
import sys

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

NEB = "/home/pa01/Mehul/dftfe-benchmarks/accuracyBenchmarks/NEB/Li2O"
BENCH = f"{NEB}/dftfe"
BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
NIMAGES = 7
NAT = 95
REF_BARRIER_MEV = 241.03  # native DFT-FE NEB activation energy

cell = np.loadtxt(f"{BENCH}/domainVectors.inp") * Bohr
rows = [ln.split() for ln in open(f"{BENCH}/coordinates.inp") if len(ln.split()) >= 5]
assert len(rows) == NIMAGES * NAT, f"expected {NIMAGES*NAT} atom lines, got {len(rows)}"
images_frac = [np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows[i*NAT:(i+1)*NAT]])
               for i in range(NIMAGES)]
symbols = [chemical_symbols[int(float(r[0]))] for r in rows[:NAT]]
print(f"[neb] {NIMAGES} images x {NAT} atoms ({symbols.count('Li')} Li, {symbols.count('O')} O)", flush=True)

nproc = int(os.environ.get("NEB_NPROC", "8"))
atoms = Atoms(symbols=symbols, scaled_positions=images_frac[0], cell=cell, pbc=True)
calc = DFTFE(
    bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc, use_device=True,
    psp_path={"Li": f"{NEB}/Li.upf", "O": f"{NEB}/O.upf"},
    xc="GGA-PBE", polynomial_order=7, mesh_size=1.2, atom_ball_radius=10.0,
    tolerance=1e-7, mixing_scheme="ANDERSON", scf_mixing=0.2, fermi_temp=500.0,
    num_bands=220, compute_forces=False, verbosity=2, log_file="neb_profile.log",
)
atoms.calc = calc

energies = []
with calc:
    for i, frac in enumerate(images_frac):
        atoms.set_scaled_positions(frac)   # same persistent process -> density reuse
        e = atoms.get_potential_energy() / Hartree
        energies.append(e)
        print(f"[neb] image {i}: E = {e:.8f} Ha", flush=True)

energies = np.array(energies)
imax = int(np.argmax(energies))
barrier_meV = (energies[imax] - energies[0]) * Hartree * 1000.0
print(f"[neb] profile (Ha, rel to img0, meV):", flush=True)
for i, e in enumerate(energies):
    print(f"    img{i}: {(e-energies[0])*Hartree*1000:8.2f} meV{'   <- max' if i == imax else ''}", flush=True)
d = abs(barrier_meV - REF_BARRIER_MEV)
print(f"[neb] activation barrier = {barrier_meV:.2f} meV  (native ref {REF_BARRIER_MEV} meV, d={d:.2f} meV)", flush=True)
print(f"[neb] {'PATH LOOKS CONVERGED' if d < 15 else 'PATH NOT CONVERGED -> run full ase.mep.NEB'}", flush=True)
