#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Geometry-optimization head-to-head: native DFT-FE GEOOPT vs ASE relaxation.

Runs demo/ex1_b (N2 ionic relaxation, OPTIMIZATION MODE = ION, FORCE TOL = 1e-4,
relaxation flags "1 0 0" => relax only x) BOTH ways and compares the relaxed
result.

Controlling for the algorithm: ASE's optimizers and DFT-FE's GEOOPT solver are
DIFFERENT code, so trajectories cannot match to 1e-10. The fair comparison uses
the SAME force-convergence threshold (1e-4 Ha/Bohr) and the SAME constrained DOF
(x only), then checks that both reach the SAME minimum:
    * final energy agrees to < 1e-5 Ha
    * final N-N bond length agrees to < 1e-3 Bohr
For a 1-DOF relaxation the minimum is unique, so agreement is expected regardless
of optimizer; the residual is bounded by force_tol^2 / curvature.

GPU mode (relaxation doesn't need bit-determinism; the ~1e-5 tolerance absorbs
GPU non-determinism).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from ase import Atoms
from ase.constraints import FixCartesian
from ase.optimize import BFGS
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

REPO = "/home/pa01/Mehul/DFTFE"
PSP_DIR = f"{REPO}/interfaces/ase-dftfe/psp_library"
REAL_BIN = f"{REPO}/build_gpu/release/real/dftfe"
NPROC = int(os.environ.get("GEOOPT_NPROC", "4"))
FORCE_TOL_HA_BOHR = 1e-4
E_TOL = 1e-5
BOND_TOL_BOHR = 1e-3


def native_bond_length_and_energy(text):
    energy = None
    for m in re.finditer(r"Total free energy:\s*([-\d.eE+]+)", text):
        energy = float(m.group(1))
    # last block of cartesian atom coordinates (Bohr)
    positions = []
    lines = text.splitlines()
    last = None
    for i, ln in enumerate(lines):
        if "Cartesian coordinates of atoms" in ln:
            last = i
    if last is not None:
        for ln in lines[last + 1:]:
            m = re.search(r"AtomId\s*\d+\s*:?\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", ln)
            if m:
                positions.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            elif positions and ln.strip().startswith("---"):
                break
    bond = None
    if len(positions) >= 2:
        p = np.array(positions[:2])
        bond = np.linalg.norm(p[0] - p[1])  # Bohr
    return energy, bond


def run_native():
    demo = f"{REPO}/demo/ex1"
    work = tempfile.mkdtemp(prefix="geoopt_ex1b_native_", dir=os.environ.get("GEOOPT_TMP", HERE))
    for fn in ("coordinates.inp", "domainVectors.inp", "pseudo.inp", "relaxationFlags.inp"):
        shutil.copy(f"{demo}/{fn}", work)
    shutil.copy(f"{PSP_DIR}/N.upf", work)
    prm = open(f"{demo}/parameterFile_b.prm").read().replace(
        "set SOLVER MODE = GEOOPT",
        "set SOLVER MODE = GEOOPT\nset USE GPU = true\nset VERBOSITY = 4")
    with open(f"{work}/parameterFile_b.prm", "w") as fh:
        fh.write(prm)
    env = dict(os.environ)
    env["DFTFE_PSP_PATH"] = PSP_DIR
    proc = subprocess.run(
        ["mpirun", "-np", str(NPROC), REAL_BIN, "parameterFile_b.prm"],
        cwd=work, env=env, capture_output=True, text=True)
    text = proc.stdout + "\n" + proc.stderr
    with open(f"{work}/native.out", "w") as fh:
        fh.write(text)
    print(f"[geoopt] native workdir: {work} (rc={proc.returncode})")
    return native_bond_length_and_energy(text), proc.returncode, text


def run_ase():
    box = np.array([40.0, 40.0, 40.0])
    cell = np.diag(box) * Bohr
    c = box / 2.0
    pos = np.array([[c[0] - 1.3, c[1], c[2]], [c[0] + 1.3, c[1], c[2]]]) * Bohr
    atoms = Atoms("N2", positions=pos, cell=cell, pbc=[False, False, False])
    # relaxationFlags "1 0 0": relax x only -> fix y,z on both atoms.
    atoms.set_constraint(FixCartesian([0, 1], mask=[False, True, True]))
    calc = DFTFE(cluster="matrix", nproc=NPROC, use_device=True,
                 psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
                 xc="GGA-PBE", mesh_size=1.0, polynomial_order=6, atom_ball_radius=3.0,
                 tolerance=5e-5, scf_mixing=0.5, max_scf_iterations=40,
                 fermi_temp=500.0, num_bands=15, compute_forces=True,
                 verbosity=1, log_file="geoopt_ase.log")
    fmax_ev_ang = FORCE_TOL_HA_BOHR * (Hartree / Bohr)  # match DFT-FE FORCE TOL
    with calc:
        atoms.calc = calc
        opt = BFGS(atoms, logfile="geoopt_ase_opt.log")
        opt.run(fmax=fmax_ev_ang)
        energy_ha = atoms.get_potential_energy() / Hartree
        bond_bohr = atoms.get_distance(0, 1) / Bohr
        nsteps = opt.get_number_of_steps()
    return energy_ha, bond_bohr, nsteps


def main():
    print(f"[geoopt] ex1_b (N2 ion relax), nproc={NPROC}, GPU, force tol={FORCE_TOL_HA_BOHR} Ha/Bohr")
    (e_nat, bond_nat), rc, text = run_native()
    if e_nat is None:
        print(f"[geoopt] FAILED to parse native energy (rc={rc}). Tail:")
        print("\n".join(text.splitlines()[-40:]))
        sys.exit(2)
    e_ase, bond_ase, nsteps = run_ase()

    print("\n================ ex1_b geo-opt head-to-head ================")
    print(f"  native GEOOPT : E = {e_nat:.12e} Ha, bond = {bond_nat if bond_nat else float('nan'):.8f} Bohr")
    print(f"  ASE BFGS      : E = {e_ase:.12e} Ha, bond = {bond_ase:.8f} Bohr  ({nsteps} steps)")
    dE = abs(e_ase - e_nat)
    e_ok = dE < E_TOL
    print(f"  |dE| = {dE:.3e} Ha  (tol {E_TOL:.0e})  {'PASS' if e_ok else 'FAIL'}")
    ok = e_ok
    if bond_nat is not None:
        dB = abs(bond_ase - bond_nat)
        b_ok = dB < BOND_TOL_BOHR
        print(f"  |d(bond)| = {dB:.3e} Bohr  (tol {BOND_TOL_BOHR:.0e})  {'PASS' if b_ok else 'FAIL'}")
        ok = ok and b_ok
    else:
        print("  (could not parse native final geometry; energy-only)")
    print("=" * 60)
    print(f"GEOOPT HEAD-TO-HEAD {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
