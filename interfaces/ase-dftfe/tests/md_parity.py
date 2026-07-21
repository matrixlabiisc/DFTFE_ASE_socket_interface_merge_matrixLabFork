#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""MD head-to-head: native DFT-FE BOMD vs ASE VelocityVerlet, IDENTICAL setup.

The whole point (per the design intent): DFT-FE and ASE integrators are
different code, so to compare fairly we reproduce DFT-FE's MD setup EXACTLY:
    * NVE  (TEMPERATURE CONTROLLER TYPE = NO_CONTROL)
    * velocity-Verlet
    * STARTING TEMPERATURE = 0  -> zero initial velocities (deterministic; avoids
      unmatched random Maxwell-Boltzmann draws)
    * identical atomic masses (written to DFT-FE via ATOMIC MASSES FILE, and set
      on the ASE Atoms)
    * identical timestep and number of steps
With identical forces (same DFT-FE), identical integrator, identical masses/dt
and identical (zero) initial velocities, the two trajectories must agree.

Metric (per the agreed criterion for dynamics): the SAME dynamics/result, not
bit-identical energies. We compare the atom0-atom1 distance after N steps
(< 1e-3 Bohr) and report ASE NVE energy drift (conservation sanity check).

CPU (deterministic). Runs ex1/ex2/ex3; ex1 (N2 off-equilibrium) is the dynamic
case, ex2/ex3 (ideal lattices, ~zero force) mainly test stability/conservation.

NOTE: native MD trajectory parsing is best-effort; the first real run may reveal
an output-format tweak (as the single-point parity harness did once).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.units import Bohr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # for parity_check
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402
from parity_check import (  # noqa: E402
    COMMON, DEMOS, PSP_DIR, REPO, atoms_ex1, atoms_ex2, atoms_ex3,
)
from ase.data import chemical_symbols  # noqa: E402

BUILDERS = {"ex1": atoms_ex1, "ex2": atoms_ex2, "ex3": atoms_ex3}
DT_FS = float(os.environ.get("MD_DT_FS", "0.5"))
NSTEPS = int(os.environ.get("MD_NSTEPS", "10"))
DIST_TOL_BOHR = 1e-3


def final_distance_native(text):
    lines = text.splitlines()
    last = None
    for i, ln in enumerate(lines):
        if "Cartesian coordinates of atoms" in ln:
            last = i
    if last is None:
        return None
    pos = []
    for ln in lines[last + 1:]:
        m = re.search(r"AtomId\s*\d+\s*:?\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", ln)
        if m:
            pos.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        elif pos and ln.strip().startswith("---"):
            break
    if len(pos) >= 2:
        p = np.array(pos)
        return float(np.linalg.norm(p[0] - p[1]))  # Bohr
    return None


def run_native(demo, cfg, atoms, nproc):
    demo_dir = f"{REPO}/demo/{demo}"
    work = tempfile.mkdtemp(prefix=f"md_{demo}_native_", dir=os.environ.get("MD_TMP", HERE))
    for fn in os.listdir(demo_dir):
        if fn.endswith(".inp"):
            shutil.copy(f"{demo_dir}/{fn}", work)
    # pseudos normalized to <Symbol>.upf
    pseudo_lines = []
    for line in open(f"{demo_dir}/pseudo.inp"):
        parts = line.split()
        if len(parts) >= 2:
            z = int(parts[0])
            upf = f"{chemical_symbols[z]}.upf"
            shutil.copy(f"{PSP_DIR}/{upf}", work)
            pseudo_lines.append(f"{z} {upf}\n")
    open(f"{work}/pseudo.inp", "w").writelines(pseudo_lines)
    # identical masses file (one row per atom type, ASE masses)
    zmass = {}
    for z, m in zip(atoms.get_atomic_numbers(), atoms.get_masses()):
        zmass[int(z)] = float(m)
    with open(f"{work}/masses.inp", "w") as fh:
        for z, m in sorted(zmass.items()):
            fh.write(f"{z} {m:.10f}\n")
    md_block = (
        "\nsubsection Molecular Dynamics\n"
        "  set BOMD = true\n"
        "  set ATOMIC MASSES FILE = masses.inp\n"
        "  set TEMPERATURE CONTROLLER TYPE = NO_CONTROL\n"
        "  set STARTING TEMPERATURE = 0.0\n"
        f"  set TIME STEP = {DT_FS}\n"
        f"  set NUMBER OF STEPS = {NSTEPS}\n"
        "  set EXTRAPOLATE DENSITY = 0\n"
        "end\n"
    )
    prm = open(f"{demo_dir}/{cfg['prm']}").read()
    prm = prm.replace("set SOLVER MODE = GS", "set SOLVER MODE = MD\nset USE GPU = false\nset VERBOSITY = 4")
    prm += md_block
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
    print(f"[md] native workdir: {work} (rc={proc.returncode})")
    return final_distance_native(text), proc.returncode, text


def run_ase(cfg, atoms, nproc):
    calc = DFTFE(cluster="matrix", nproc=nproc, use_device=False,
                 psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
                 verbosity=1, log_file="md_ase.log",
                 **COMMON, **{k: v for k, v in cfg["params"].items()
                              if k not in ("compute_stress",)})
    atoms = atoms.copy()
    atoms.set_velocities(np.zeros((len(atoms), 3)))  # v0 = 0, deterministic
    atoms.calc = calc
    e0 = atoms.get_total_energy()
    dyn = VelocityVerlet(atoms, timestep=DT_FS * units.fs)
    emax_drift = 0.0
    for _ in range(NSTEPS):
        dyn.run(1)
        emax_drift = max(emax_drift, abs(atoms.get_total_energy() - e0))
    dist = atoms.get_distance(0, 1) / Bohr
    calc.close()
    return dist, emax_drift


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=["ex1", "ex2", "ex3"], nargs="?", default="ex1")
    args = ap.parse_args()
    cfg = DEMOS[args.demo]
    atoms = BUILDERS[args.demo]()
    nproc = int(os.environ.get("MD_NPROC", cfg["nproc"]))
    print(f"[md] {args.demo}: NVE, {NSTEPS} steps x {DT_FS} fs, v0=0, nproc={nproc}, CPU")

    (d_nat, rc, text) = run_native(args.demo, cfg, atoms, nproc)
    if d_nat is None:
        print(f"[md] could not parse native final geometry (rc={rc}). Tail:")
        print("\n".join(text.splitlines()[-40:]))
        sys.exit(2)
    d_ase, drift = run_ase(cfg, atoms, nproc)

    print(f"\n================ {args.demo} MD head-to-head ================")
    print(f"  atom0-atom1 distance after {NSTEPS} steps:")
    print(f"    native = {d_nat:.8f} Bohr")
    print(f"    ASE    = {d_ase:.8f} Bohr")
    dd = abs(d_ase - d_nat)
    ok = dd < DIST_TOL_BOHR
    print(f"  |d(dist)| = {dd:.3e} Bohr  (tol {DIST_TOL_BOHR:.0e})  {'PASS' if ok else 'FAIL'}")
    print(f"  ASE NVE energy drift over run = {drift:.3e} eV (conservation check)")
    print("=" * 55)
    print(f"{args.demo} MD HEAD-TO-HEAD {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
