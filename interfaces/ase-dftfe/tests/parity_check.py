#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Parity check: native file-based DFT-FE vs the ASE socket interface.

Runs a pGD demo BOTH ways with the SAME binary, SAME MPI rank count, on CPU
(use_device=false -> deterministic), and asserts:
    * total free energy agrees to < 1e-10 Ha (all demos)
    * each force component of the reported max-force atom agrees to < 1e-4
      Ha/Bohr (only demos whose .prm computes ION FORCE, e.g. ex1)

CPU is used deliberately: with TOLERANCE=5e-5 the energy is only physically
converged to ~1e-6, so a 1e-10 comparison is only meaningful when both runs
follow an identical, deterministic numerical path. This isolates whether the
socket wrapper perturbs the computation and whether the ASE parameter mapping
reproduces the demo .prm exactly.

Demos:
    ex1  N2, non-periodic, real binary, forces
    ex2  FCC Al, periodic 3D, MP 2x2x2 shifted, complex binary
    ex3  graphene, periodic 2D, MP 4x4x1 shifted, complex binary
Pseudos are normalized to <Symbol>.upf on both sides (ex3's demo references
C_ONCV_PBE-1.0.upf, absent here, so C.upf is used on native AND ASE — a fair
socket-vs-file comparison, just not the published pseudo).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

REPO = "/home/pa01/Mehul/DFTFE"
PSP_DIR = f"{REPO}/interfaces/ase-dftfe/psp_library"
REAL_BIN = f"{REPO}/build_gpu/release/real/dftfe"
COMPLEX_BIN = f"{REPO}/build_gpu/release/complex/dftfe"
E_TOL = 1e-10
F_TOL = 1e-4
COMMON = dict(xc="GGA-PBE", fermi_temp=500.0, tolerance=5e-5)


# ── ASE geometry builders (match the demo coordinates.inp exactly) ──────────
def atoms_ex1():
    box = np.array([40.0, 40.0, 40.0])
    cell = np.diag(box) * Bohr
    c = box / 2.0
    pos = np.array([[c[0] - 1.3, c[1], c[2]], [c[0] + 1.3, c[1], c[2]]]) * Bohr
    return Atoms("N2", positions=pos, cell=cell, pbc=[False, False, False])


def atoms_ex2():
    cell = np.diag([7.6, 7.6, 7.6]) * Bohr
    frac = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    return Atoms("Al4", scaled_positions=frac, cell=cell, pbc=[True, True, True])


def atoms_ex3():
    cell = np.array([[4.654289, 0.0, 0.0],
                     [-2.3271445, 4.03073251, 0.0],
                     [0.0, 0.0, 50.0]]) * Bohr
    frac = [[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]]
    return Atoms("C2", scaled_positions=frac, cell=cell, pbc=[True, True, False])


DEMOS = {
    "ex1": dict(
        prm="parameterFile_a.prm", binary=REAL_BIN, nproc=4, atoms=atoms_ex1,
        compare_forces=True,
        params=dict(mesh_size=1.0, polynomial_order=6, atom_ball_radius=3.0,
                    scf_mixing=0.5, max_scf_iterations=40, num_bands=15,
                    compute_forces=True),
    ),
    "ex2": dict(
        prm="parameterFile_a.prm", binary=COMPLEX_BIN, nproc=4, atoms=atoms_ex2,
        compare_forces=False,
        params=dict(mesh_size=1.6, polynomial_order=5, mp_grid=(2, 2, 2),
                    mp_grid_shift=(1, 1, 1), use_time_reversal_symmetry=True,
                    npkpt=2, compute_forces=False, compute_stress=True),
    ),
    "ex3": dict(
        prm="parameterFile.prm", binary=COMPLEX_BIN, nproc=8, atoms=atoms_ex3,
        compare_forces=False,
        params=dict(mesh_size=0.5, polynomial_order=3, mp_grid=(4, 4, 1),
                    mp_grid_shift=(1, 1, 0), use_time_reversal_symmetry=True,
                    npkpt=8, compute_forces=False),
    ),
}


def parse_native(text):
    energy = None
    for m in re.finditer(r"Total free energy:\s*([-\d.eE+]+)", text):
        energy = float(m.group(1))
    fm = re.search(
        r"Maximum absolute force atom id:\s*(\d+),\s*Force vec:\s*"
        r"([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)", text)
    atom_id, force = None, None
    if fm:
        atom_id = int(fm.group(1))
        force = np.array([float(fm.group(2)), float(fm.group(3)), float(fm.group(4))])
    return energy, atom_id, force


def run_native(demo, cfg, nproc):
    demo_dir = f"{REPO}/demo/{demo}"
    tmproot = os.environ.get("PARITY_TMP", HERE)
    os.makedirs(tmproot, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"parity_{demo}_native_", dir=tmproot)
    for fn in os.listdir(demo_dir):
        if fn.endswith(".inp"):
            shutil.copy(f"{demo_dir}/{fn}", work)
    # Normalize pseudos to <Symbol>.upf on both sides.
    pseudo_lines = []
    for line in open(f"{demo_dir}/pseudo.inp"):
        parts = line.split()
        if len(parts) >= 2:
            z = int(parts[0])
            upf = f"{chemical_symbols[z]}.upf"
            shutil.copy(f"{PSP_DIR}/{upf}", work)
            pseudo_lines.append(f"{z} {upf}\n")
    with open(f"{work}/pseudo.inp", "w") as fh:
        fh.writelines(pseudo_lines)
    prm = open(f"{demo_dir}/{cfg['prm']}").read().replace(
        "set SOLVER MODE = GS",
        "set SOLVER MODE = GS\nset USE GPU = false\nset VERBOSITY = 4")
    with open(f"{work}/{cfg['prm']}", "w") as fh:
        fh.write(prm)
    env = dict(os.environ)
    env["DFTFE_PSP_PATH"] = PSP_DIR
    proc = subprocess.run(
        ["mpirun", "-np", str(nproc), cfg["binary"], cfg["prm"]],
        cwd=work, env=env, capture_output=True, text=True)
    text = proc.stdout + "\n" + proc.stderr
    with open(f"{work}/native.out", "w") as fh:
        fh.write(text)
    print(f"[parity] native workdir: {work} (rc={proc.returncode})")
    return parse_native(text), proc.returncode, text


def run_ase(cfg, nproc):
    atoms = cfg["atoms"]()
    calc = DFTFE(cluster="matrix", nproc=nproc, use_device=False,
                 psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
                 keep_scratch=True, verbosity=4, log_file="parity_ase.log",
                 **COMMON, **cfg["params"])
    with calc:
        atoms.calc = calc
        energy_ha = atoms.get_potential_energy() / Hartree
        forces_ha = None
        if cfg["params"].get("compute_forces"):
            forces_ha = atoms.get_forces() / (Hartree / Bohr)
    return energy_ha, forces_ha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=sorted(DEMOS), nargs="?", default="ex1")
    args = ap.parse_args()
    cfg = DEMOS[args.demo]
    nproc = int(os.environ.get("PARITY_NPROC", cfg["nproc"]))
    print(f"[parity] {args.demo}: nproc={nproc}, CPU, binary={os.path.basename(cfg['binary'])}")

    (e_nat, atom_id, f_nat), rc, native_text = run_native(args.demo, cfg, nproc)
    if e_nat is None:
        print(f"[parity] FAILED to parse native energy (rc={rc}). Tail:")
        print("\n".join(native_text.splitlines()[-40:]))
        sys.exit(2)
    e_ase, f_ase = run_ase(cfg, nproc)

    print(f"\n================ {args.demo} parity ================")
    print(f"  native  E = {e_nat:.15e} Ha")
    print(f"  ASE     E = {e_ase:.15e} Ha")
    dE = abs(e_ase - e_nat)
    e_ok = dE < E_TOL
    print(f"  |dE|      = {dE:.3e} Ha   (tol {E_TOL:.0e})   {'PASS' if e_ok else 'FAIL'}")

    ok = e_ok
    if cfg["compare_forces"] and f_nat is not None and f_ase is not None:
        dF = np.abs(f_ase[atom_id] - f_nat)
        f_ok = dF.max() < F_TOL
        print(f"  force atom {atom_id}: native={np.array2string(f_nat, precision=12)}")
        print(f"                  ASE   ={np.array2string(f_ase[atom_id], precision=12)}")
        print(f"  max|dF|   = {dF.max():.3e} Ha/Bohr   (tol {F_TOL:.0e})   {'PASS' if f_ok else 'FAIL'}")
        ok = ok and f_ok
    else:
        print("  (energy-only parity for this demo; forces not computed in its .prm)")
    print("=" * 45)
    print(f"{args.demo} PARITY {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
