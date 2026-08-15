#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""BCC Mo (127-atom mono-vacancy) ground-state accuracy vs dftfe-benchmarks.

Reproduces accuracyBenchmarks/BCCMo/dftfe/study1 via the ASE interface using
pure-Python kwargs (option B): GGA-PBE, Gamma-point (real binary auto-selected),
Kerker-preconditioned Anderson mixing, energy + forces + stress. Single-element
Mo ONCV psp via a psp dict.

DFT-FE native reference (CPU, Gamma):
  Total free energy            = -8.709816940500966666e+03 Ha
  Sum |force components|       =  3.629676462085286714e-01 Ha/Bohr
  Cell stress (diagonal)       = ~-4.284e-05 Ha/Bohr^3 (isotropic)
Reference is CPU; GGA-PBE is GPU/CPU-stable so agreement should be ~chemical
accuracy (README: 2.06e-5 Ha/atom vs QE). Bit-for-bit parity is asserted
separately (ASE-socket vs same-binary native), consistent with Li2O.
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

BENCH = "/home/pa01/Mehul/dftfe-benchmarks/accuracyBenchmarks/BCCMo/dftfe/study1"
BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
REF_E = -8.709816940500966666e+03            # Ha
REF_FSUM = 3.629676462085286714e-01          # Ha/Bohr (sum of abs force components)
REF_SDIAG = -4.284e-05                        # Ha/Bohr^3 (isotropic diagonal)
E_TOL_PER_ATOM = 1e-4                         # chemical accuracy (README target)

# ── parse benchmark geometry ────────────────────────────────────────────────
cell_bohr = np.loadtxt(f"{BENCH}/domainVectors.inp")
symbols, frac = [], []
for line in open(f"{BENCH}/coordinates.inp"):
    p = line.split()
    if len(p) >= 5:
        symbols.append(chemical_symbols[int(float(p[0]))])
        frac.append([float(p[2]), float(p[3]), float(p[4])])
atoms = Atoms(symbols=symbols, scaled_positions=np.array(frac),
              cell=cell_bohr * Bohr, pbc=[True, True, True])
nat = len(atoms)
print(f"[mo] {nat} atoms ({symbols.count('Mo')} Mo)", flush=True)

nproc = int(os.environ.get("MO_NPROC", "8"))
calc = DFTFE(
    bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc, use_device=True,
    psp_path={"Mo": f"{BENCH}/Mo_ONCV_PBE-1.0.upf"},
    xc="GGA-PBE", polynomial_order=7, mesh_size=2.0,
    mixing_scheme="ANDERSON_WITH_KERKER", scf_mixing=0.5,
    fermi_temp=500.0, tolerance=5e-5,
    compute_forces=True, compute_stress=True,
    verbosity=2, log_file="bcc_mo.log",
)
atoms.calc = calc
with calc:
    e = atoms.get_potential_energy() / Hartree
    f = atoms.get_forces()                              # eV/Ang
    s = atoms.get_stress(voigt=False)                   # eV/Ang^3, 3x3
fsum = float(np.abs(f).sum()) * Bohr / Hartree          # -> Ha/Bohr
sdiag = float(np.mean(np.diag(s))) * (Bohr**3) / Hartree  # -> Ha/Bohr^3

dE_atom = abs(e - REF_E) / nat
dF = abs(fsum - REF_FSUM)
dS = abs(sdiag - REF_SDIAG)
print(f"[mo] energy      = {e:.10e} Ha   (ref {REF_E:.10e})", flush=True)
print(f"[mo] dE/atom     = {dE_atom:.3e} Ha/atom   (tol {E_TOL_PER_ATOM:.0e})", flush=True)
print(f"[mo] sum|F|      = {fsum:.10e} Ha/Bohr   (ref {REF_FSUM:.6e}, d={dF:.2e})", flush=True)
print(f"[mo] stress diag = {sdiag:.6e} Ha/Bohr^3 (ref {REF_SDIAG:.3e}, d={dS:.2e})", flush=True)
ok = dE_atom < E_TOL_PER_ATOM
print(f"[mo] {'PASS' if ok else 'CHECK'} (energy within chemical accuracy)", flush=True)
sys.exit(0 if ok else 1)
