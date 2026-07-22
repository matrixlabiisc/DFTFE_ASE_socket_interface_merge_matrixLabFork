#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Apples-to-apples geo-opt: SAME optimizer, forces via socket vs via file.

The plain geo-opt head-to-head (geoopt_parity.py) compares DFT-FE's own GEOOPT
solver against ASE BFGS -> different optimizers, so it can only check "same
minimum". Here we make the optimizer IDENTICAL: ASE BFGS drives BOTH runs; the
only difference is the force source:
    * socket : the dftfe_ase.DFTFE calculator (persistent server)
    * file   : native `dftfe parameterFile.prm` single-point per step (cold),
               wrapped as an ASE calculator
Same optimizer + same start + same tolerance + (proven) matching forces => the
two relaxation TRAJECTORIES should coincide, limited only by the SCF tolerance
(socket warm-starts the density; file cold-starts). This is the true head-to-head
that "swaps DFT-FE's optimizer for ASE's" on the native side.

ex1 only (N2): forces are reconstructed from the reliably-printed max-force atom
using Newton's third law (a 2-atom system's forces sum to zero) -> rigorous.
CPU, deterministic.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixCartesian
from ase.optimize import BFGS
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

REPO = "/home/pa01/Mehul/DFTFE"
PSP_DIR = f"{REPO}/interfaces/ase-dftfe/psp_library"
REAL_BIN = f"{REPO}/build_gpu/release/real/dftfe"
NPROC = int(os.environ.get("GEOID_NPROC", "4"))
FMAX = 1e-4 * (Hartree / Bohr)   # 1e-4 Ha/Bohr in eV/Ang
MAXSTEP = 25


class FileDFTFE(Calculator):
    """ASE calculator that runs native file-based DFT-FE (GS) once per call."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, workroot):
        super().__init__()
        self.demo = f"{REPO}/demo/ex1"
        self.work = tempfile.mkdtemp(prefix="geoid_file_", dir=workroot)
        for fn in ("domainVectors.inp", "pseudo.inp"):
            shutil.copy(f"{self.demo}/{fn}", self.work)
        shutil.copy(f"{PSP_DIR}/N.upf", self.work)
        prm = open(f"{self.demo}/parameterFile_a.prm").read().replace(
            "set SOLVER MODE = GS",
            "set SOLVER MODE = GS\nset USE GPU = false\nset VERBOSITY = 4")
        with open(f"{self.work}/parameterFile_a.prm", "w") as fh:
            fh.write(prm)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos_bohr = self.atoms.get_positions() / Bohr
        # coordinates.inp: "Z valence x y z" (Cartesian Bohr), N valence = 5
        with open(f"{self.work}/coordinates.inp", "w") as fh:
            for p in pos_bohr:
                fh.write(f"7 5 {p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n")
        env = dict(os.environ)
        env["DFTFE_PSP_PATH"] = PSP_DIR
        proc = subprocess.run(
            ["mpirun", "-np", str(NPROC), REAL_BIN, "parameterFile_a.prm"],
            cwd=self.work, env=env, capture_output=True, text=True)
        text = proc.stdout + "\n" + proc.stderr
        e = None
        for m in re.finditer(r"Total free energy:\s*([-\d.eE+]+)", text):
            e = float(m.group(1))
        fm = re.search(r"Maximum absolute force atom id:\s*(\d+),\s*Force vec:\s*"
                       r"([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)", text)
        if e is None or fm is None:
            raise RuntimeError(f"file DFT-FE parse failed (rc={proc.returncode})")
        f0 = np.array([float(fm.group(2)), float(fm.group(3)), float(fm.group(4))])
        # 2-atom system: forces sum to zero (Newton's third law).
        forces_ha = np.array([f0, -f0]) if int(fm.group(1)) == 0 else np.array([-f0, f0])
        self.results["energy"] = e * Hartree
        self.results["forces"] = forces_ha * (Hartree / Bohr)


def n2():
    box = np.array([40.0, 40.0, 40.0])
    cell = np.diag(box) * Bohr
    c = box / 2.0
    pos = np.array([[c[0] - 1.3, c[1], c[2]], [c[0] + 1.3, c[1], c[2]]]) * Bohr
    a = Atoms("N2", positions=pos, cell=cell, pbc=[False, False, False])
    a.set_constraint(FixCartesian([0, 1], mask=[False, True, True]))  # relax x only
    return a


def relax(calc, tag):
    import time
    atoms = n2()
    atoms.calc = calc
    traj, step_starts = [], []
    opt = BFGS(atoms, logfile=f"geoid_{tag}.log")

    def rec():
        traj.append(atoms.get_distance(0, 1) / Bohr)  # bond length, Bohr
        step_starts.append(time.perf_counter())

    opt.attach(rec, interval=1)
    t0 = time.perf_counter()
    opt.run(fmax=FMAX, steps=MAXSTEP)
    rec()
    wall = time.perf_counter() - t0
    return np.array(traj), atoms.get_potential_energy() / Hartree, wall


def main():
    print(f"[geoid] ex1_b apples-to-apples: SAME ASE BFGS, forces socket vs file; "
          f"nproc={NPROC}, CPU")
    tmproot = os.environ.get("GEOID_TMP", HERE)
    os.makedirs(tmproot, exist_ok=True)

    socket_calc = DFTFE(cluster="matrix", nproc=NPROC, use_device=False,
                        psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
                        xc="GGA-PBE", mesh_size=1.0, polynomial_order=6,
                        atom_ball_radius=3.0, tolerance=5e-5, scf_mixing=0.5,
                        max_scf_iterations=40, fermi_temp=500.0, num_bands=15,
                        compute_forces=True, verbosity=1, log_file="geoid_socket.log")
    with socket_calc:
        t_sock, e_sock, wall_sock = relax(socket_calc, "socket")
    t_file, e_file, wall_file = relax(FileDFTFE(tmproot), "file")

    n = min(len(t_sock), len(t_file))
    dtraj = np.abs(t_sock[:n] - t_file[:n])
    ns, nf = len(t_sock) - 1, len(t_file) - 1  # BFGS steps (traj has +1 for start)
    print("\n=========== ex1_b identical-optimizer relaxation OVERHEAD ===========")
    print("  (same ASE BFGS both sides; only the force transport differs)")
    print(f"  socket: {ns} steps, final bond = {t_sock[-1]:.8f} Bohr, E = {e_sock:.10e} Ha")
    print(f"  file  : {nf} steps, final bond = {t_file[-1]:.8f} Bohr, E = {e_file:.10e} Ha")
    print(f"  trajectory agreement: max per-step |d(bond)| = {dtraj.max():.3e} Bohr "
          f"(step counts {'match' if ns == nf else 'DIFFER'})")
    print("  --- wall time (the overhead result) ---")
    print(f"  socket : {wall_sock:8.2f} s total,  {wall_sock / max(ns,1):7.2f} s/step "
          "(persistent process, warm-started)")
    print(f"  file   : {wall_file:8.2f} s total,  {wall_file / max(nf,1):7.2f} s/step "
          "(cold restart every step)")
    print(f"  socket speedup over file-based multi-step workflow: {wall_file / wall_sock:.2f}x")
    print("=" * 68)
    ok = (ns == nf) and dtraj.max() < 1e-4
    print(f"IDENTICAL-OPTIMIZER RELAXATION {'consistent' if ok else 'CHECK'} "
          "(overhead numbers above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
