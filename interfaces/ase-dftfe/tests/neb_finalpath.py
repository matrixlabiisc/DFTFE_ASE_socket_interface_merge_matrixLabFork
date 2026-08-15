#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Li2O NEB accuracy: energy profile of the DFT-FE *converged* MEP (FinalPath.txt).

The benchmark's coordinates.inp is the initial NEB guess; the converged minimum
energy path is output/FinalPath.txt (95 atoms x 21 cols = 7 images x xyz). We
recompute each converged-image energy through the ASE interface (one persistent
socket process, density reuse) and compare the activation barrier max(E)-E_0 to
the native DFT-FE NEB result (241.03 meV) / QE (244.9 meV). This validates that
the interface reproduces the benchmark's NEB energetics.
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
NIMAGES, NAT = 7, 95
REF_BARRIER_MEV = 241.03

cell = np.loadtxt(f"{BENCH}/domainVectors.inp") * Bohr
# species from the first image block of coordinates.inp (FinalPath has no Z column)
rows0 = [ln.split() for ln in open(f"{BENCH}/coordinates.inp") if len(ln.split()) >= 5][:NAT]
symbols = [chemical_symbols[int(float(r[0]))] for r in rows0]
# FinalPath.txt: row = atom, cols = 7 images x (x,y,z)
fp = np.loadtxt(f"{BENCH}/output/FinalPath.txt")
assert fp.shape == (NAT, NIMAGES * 3), f"unexpected FinalPath shape {fp.shape}"
images_frac = [fp[:, 3 * k:3 * k + 3] for k in range(NIMAGES)]
print(f"[neb] converged MEP: {NIMAGES} images x {NAT} atoms "
      f"({symbols.count('Li')} Li, {symbols.count('O')} O)", flush=True)

nproc = int(os.environ.get("NEB_NPROC", "8"))
atoms = Atoms(symbols=symbols, scaled_positions=images_frac[0], cell=cell, pbc=True)
calc = DFTFE(
    bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc, use_device=True,
    psp_path={"Li": f"{NEB}/Li.upf", "O": f"{NEB}/O.upf"},
    xc="GGA-PBE", polynomial_order=7, mesh_size=1.2, atom_ball_radius=10.0,
    tolerance=1e-7, mixing_scheme="ANDERSON", scf_mixing=0.2, fermi_temp=500.0,
    num_bands=220, compute_forces=False, verbosity=2, log_file="neb_finalpath.log",
)
atoms.calc = calc

energies = []
with calc:
    for i, frac in enumerate(images_frac):
        atoms.set_scaled_positions(frac)
        e = atoms.get_potential_energy() / Hartree
        energies.append(e)
        print(f"[neb] image {i}: E = {e:.8f} Ha  ({(e-energies[0])*Hartree*1000:8.2f} meV)", flush=True)

energies = np.array(energies)
imax = int(np.argmax(energies))
barrier = (energies[imax] - energies[0]) * Hartree * 1000.0
d = abs(barrier - REF_BARRIER_MEV)
print(f"[neb] activation barrier = {barrier:.2f} meV  (max at image {imax})", flush=True)
print(f"[neb] native DFT-FE NEB  = {REF_BARRIER_MEV} meV   |diff| = {d:.2f} meV", flush=True)
print(f"[neb] {'PASS' if d < 10 else 'CHECK'} (interface reproduces converged NEB barrier)", flush=True)
sys.exit(0 if d < 10 else 1)
