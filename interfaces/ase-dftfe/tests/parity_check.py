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
    * total free energy agrees to < 1e-10 Ha
    * each force component (of the reported max-force atom) agrees to < 1e-4 Ha/Bohr

CPU is used deliberately: with TOLERANCE=5e-5 the energy is only physically
converged to ~1e-6, so a 1e-10 comparison is only meaningful when both runs
follow an identical, deterministic numerical path. This isolates whether the
socket wrapper perturbs the computation and whether the ASE parameter mapping
reproduces the demo .prm exactly.

Currently implements ex1 (N2, non-periodic, real binary).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

REPO = "/home/pa01/Mehul/DFTFE"
PSP_DIR = f"{REPO}/interfaces/ase-dftfe/psp_library"
REAL_BIN = f"{REPO}/build_gpu/release/real/dftfe"
NPROC = int(os.environ.get("PARITY_NPROC", "4"))
E_TOL = 1e-10
F_TOL = 1e-4


def parse_native(text):
    """Return (energy_Ha, max_atom_id, max_force_vec_Ha_Bohr)."""
    energy = None
    for m in re.finditer(r"Total free energy:\s*([-\d.eE+]+)", text):
        energy = float(m.group(1))
    fm = re.search(
        r"Maximum absolute force atom id:\s*(\d+),\s*Force vec:\s*"
        r"([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)",
        text,
    )
    force = None
    atom_id = None
    if fm:
        atom_id = int(fm.group(1))
        force = np.array([float(fm.group(2)), float(fm.group(3)), float(fm.group(4))])
    return energy, atom_id, force


def run_native_ex1():
    demo = f"{REPO}/demo/ex1"
    tmproot = os.environ.get("PARITY_TMP", HERE)  # shared FS, inspectable
    os.makedirs(tmproot, exist_ok=True)
    work = tempfile.mkdtemp(prefix="parity_ex1_native_", dir=tmproot)
    for fn in ("coordinates.inp", "domainVectors.inp", "pseudo.inp"):
        shutil.copy(f"{demo}/{fn}", work)
    # pseudo.inp references "N.upf" by bare name -> make it resolvable in cwd.
    shutil.copy(f"{PSP_DIR}/N.upf", work)
    prm = open(f"{demo}/parameterFile_a.prm").read()
    # Force CPU + high verbosity (max-force printout) in the COPY only.
    prm = prm.replace(
        "set SOLVER MODE = GS",
        "set SOLVER MODE = GS\nset USE GPU = false\nset VERBOSITY = 4",
    )
    with open(f"{work}/parameterFile_a.prm", "w") as fh:
        fh.write(prm)
    env = dict(os.environ)
    env["DFTFE_PSP_PATH"] = PSP_DIR
    proc = subprocess.run(
        ["mpirun", "-np", str(NPROC), REAL_BIN, "parameterFile_a.prm"],
        cwd=work, env=env, capture_output=True, text=True,
    )
    text = proc.stdout + "\n" + proc.stderr
    with open(f"{work}/native.out", "w") as fh:
        fh.write(text)
    print(f"[parity] native workdir: {work} (rc={proc.returncode})")
    return parse_native(text), work, proc.returncode, text


def run_ase_ex1():
    box_bohr = np.array([40.0, 40.0, 40.0])
    cell = np.diag(box_bohr) * Bohr
    center = box_bohr / 2.0
    rel = np.array([[-1.3, 0.0, 0.0], [1.3, 0.0, 0.0]])
    atoms = Atoms("N2", positions=(center + rel) * Bohr, cell=cell,
                  pbc=[False, False, False])
    with DFTFE(
        cluster="matrix",
        nproc=NPROC,
        use_device=False,               # CPU: deterministic, matches native run
        psp_path=PSP_DIR,
        env={"DFTFE_PSP_PATH": PSP_DIR},
        xc="GGA-PBE",
        mesh_size=1.0,
        polynomial_order=6,
        atom_ball_radius=3.0,
        tolerance=5e-5,
        scf_mixing=0.5,
        max_scf_iterations=40,
        fermi_temp=500.0,
        num_bands=15,
        compute_forces=True,
        keep_scratch=True,              # keep generated .prm for diagnosis
        verbosity=4,
        log_file="parity_ase_ex1.log",
    ) as calc:
        atoms.calc = calc
        energy_ev = atoms.get_potential_energy()
        forces_ev = atoms.get_forces()
    energy_ha = energy_ev / Hartree
    forces_ha = forces_ev / (Hartree / Bohr)
    return energy_ha, forces_ha


def main():
    print(f"[parity] ex1 (N2), nproc={NPROC}, CPU, binary={REAL_BIN}")
    (e_nat, atom_id, f_nat), work, rc, native_text = run_native_ex1()
    if e_nat is None:
        print(f"[parity] FAILED to parse native energy (rc={rc}). Native output tail:")
        print("\n".join(native_text.splitlines()[-40:]))
        sys.exit(2)
    e_ase, f_ase = run_ase_ex1()

    print("\n================ ex1 parity ================")
    print(f"  native  E = {e_nat:.15e} Ha")
    print(f"  ASE     E = {e_ase:.15e} Ha")
    dE = abs(e_ase - e_nat)
    print(f"  |dE|      = {dE:.3e} Ha   (tol {E_TOL:.0e})   {'PASS' if dE < E_TOL else 'FAIL'}")

    ok = dE < E_TOL
    if f_nat is not None:
        f_ase_atom = f_ase[atom_id]
        dF = np.abs(f_ase_atom - f_nat)
        print(f"  force atom {atom_id}:")
        print(f"    native = {np.array2string(f_nat, precision=12)}")
        print(f"    ASE    = {np.array2string(f_ase_atom, precision=12)}")
        print(f"    max|dF| = {dF.max():.3e} Ha/Bohr   (tol {F_TOL:.0e})   {'PASS' if dF.max() < F_TOL else 'FAIL'}")
        ok = ok and dF.max() < F_TOL
    else:
        print("  [warn] could not parse native forces")
        ok = False
    print("============================================")
    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
