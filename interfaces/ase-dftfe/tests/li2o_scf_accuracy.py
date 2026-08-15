#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Li2O SCF accuracy vs the dftfe-benchmarks reference (MGGA-R2SCAN, k-points).

Reproduces accuracyBenchmarks/Li2O_fcc via the ASE interface using pure-Python
kwargs (option B): full param set (meta-GGA, density quadrature, single-prec
cheby, MP grid + shift + TRS), multi-element ONCV pseudos via a psp dict.
Compares the total free energy to the native reference.

  native reference (Li2O_scf_normal.out): -9.565888847486105533e+02 Ha

Energy-only (compute_forces=False): forces don't change the free energy, and it
sidesteps the MGGA+ION-FORCE+NLCC limitation regardless of the pseudo.
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

BENCH = "/home/pa01/Mehul/dftfe-benchmarks/accuracyBenchmarks/Li2O_fcc/dftfe"
BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
REF_HA = -9.565888847486105533e+02
E_TOL = 5e-5  # SCF tol is 1e-6; allow a loose window for run-to-run/GPU variance

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
print(f"[li2o] {len(atoms)} atoms ({symbols.count('Li')} Li, {symbols.count('O')} O)", flush=True)

nproc = int(os.environ.get("LI2O_NPROC", "8"))
do_forces = os.environ.get("LI2O_FORCES", "0") == "1"  # ONCV is NLCC-free -> MGGA forces OK
calc = DFTFE(
    bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc, use_device=True,
    psp_path={"Li": f"{BENCH}/Li_ONCV_PBE-1.2.upf", "O": f"{BENCH}/O_ONCV_PBE-1.2.upf"},
    xc="MGGA-R2SCAN", polynomial_order=7, density_quadrature_rule=10,
    mesh_size=1.2, atom_ball_radius=6.0, tolerance=1e-6,
    mixing_scheme="ANDERSON", scf_mixing=0.7, fermi_temp=500.0,
    mp_grid=(2, 2, 2), mp_grid_shift=(1, 1, 1), use_time_reversal_symmetry=True,
    use_single_prec_cheby=True, compute_forces=do_forces,
    verbosity=2, log_file="li2o_scf.log",
)
atoms.calc = calc
with calc:
    e = atoms.get_potential_energy() / Hartree
    if do_forces:
        f = atoms.get_forces()  # eV/Ang
        fmax = float(np.abs(f).max()) * Bohr / Hartree  # -> Ha/Bohr
        fsum = float(np.abs(f).sum()) * Bohr / Hartree
        print(f"[li2o] max |F|      = {fmax:.6e} Ha/Bohr", flush=True)
        print(f"[li2o] sum|F comps| = {fsum:.10e} Ha/Bohr   (same-binary native ref: 1.8210460746e-01)", flush=True)

d = abs(e - REF_HA)
print(f"[li2o] ASE energy   = {e:.10e} Ha", flush=True)
print(f"[li2o] native ref   = {REF_HA:.10e} Ha", flush=True)
print(f"[li2o] |dE|         = {d:.3e} Ha   (tol {E_TOL:.0e})   {'PASS' if d < E_TOL else 'CHECK'}", flush=True)
sys.exit(0 if d < E_TOL else 1)
