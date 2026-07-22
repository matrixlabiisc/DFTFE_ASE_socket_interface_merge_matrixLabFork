#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Verify parameters actually REACH DFT-FE (not silently defaulting).

Coverage ("param is in the list") != correctness ("the value we pass is what
DFT-FE uses"). A param could be dropped by parse_request / mis-sed'd and the run
still succeeds on the template default — fooling us. So: run a small system with
deliberately NON-DEFAULT values and `keep_scratch=True`, then inspect the
generated `parameterFile.prm` and assert each value we sent is present with the
right key. This exercises the full chain: calculator -> socket JSON ->
parse_request -> dftfeWrapper sed -> .prm.
"""

import glob
import os
import re
import sys

import numpy as np
from ase import Atoms
from ase.units import Bohr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
PSP_DIR = "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library"
WORK = os.path.join(os.environ.get("VERIFY_TMP", HERE), "verify_scratch")

# Deliberately NON-DEFAULT values. (key regex, expected value, numeric?)
CHECKS = {
    "polynomial_order":            (r"POLYNOMIAL ORDER", 5, True),
    "mesh_size":                   (r"MESH SIZE AROUND ATOM", 1.7, True),
    "atom_ball_radius":            (r"ATOM BALL RADIUS", 4.0, True),
    "fermi_temp":                  (r"\bTEMPERATURE\b", 321.0, True),
    "tolerance":                   (r"\bTOLERANCE\b", 3e-4, True),
    "scf_mixing":                  (r"MIXING PARAMETER", 0.31, True),
    "mixing_scheme":               (r"MIXING METHOD", "ANDERSON_WITH_KERKER", False),
    "num_bands":                   (r"NUMBER OF KOHN-SHAM WAVEFUNCTIONS", 26, True),
    "max_scf_iterations":          (r"MAXIMUM ITERATIONS", 33, True),
    "density_quadrature_rule":     (r"DENSITY QUADRATURE RULE", 10, True),
    "use_single_prec_cheby":       (r"USE SINGLE PREC CHEBY", "true", False),
    "use_time_reversal_symmetry":  (r"USE TIME REVERSAL SYMMETRY", "true", False),
    "mp_grid":                     (r"SAMPLING POINTS 1", 2, True),
    "mp_grid_shift":               (r"SAMPLING SHIFT 1", 1, True),
    "npkpt":                       (r"\bNPKPT\b", 2, True),
    "orthogonalization_type":      (r"ORTHOGONALIZATION TYPE", "CGS", False),
    "smeared_nuclear_charges":     (r"SMEARED NUCLEAR CHARGES", "true", False),
    "use_group_symmetry":          (r"USE GROUP SYMMETRY", "true", False),
    "mixing_history":              (r"LBFGS HISTORY", 8, True),
}
# Known gaps (deliberately NOT injected): dispersion_correction_type needs a
# Dispersion Correction subsection added; start_magnetization maps to a
# non-existent DFT-FE key (initial magnetization is the per-atom m column in
# coordinates.inp). Tracked for follow-up.

PARAMS = dict(
    xc="GGA-PBE", polynomial_order=5, mesh_size=1.7, atom_ball_radius=4.0,
    fermi_temp=321.0, tolerance=3e-4, scf_mixing=0.31,
    mixing_scheme="ANDERSON_WITH_KERKER", num_bands=26, max_scf_iterations=33,
    density_quadrature_rule=10, use_single_prec_cheby=True,
    use_time_reversal_symmetry=True, mp_grid=(2, 2, 2), mp_grid_shift=(1, 1, 1),
    npkpt=2, orthogonalization_type="CGS", smeared_nuclear_charges=1,
    use_group_symmetry=1, mixing_history=8,
)


def find_prm():
    hits = glob.glob(os.path.join(WORK, "dftfeScratch*", "parameterFile.prm"))
    return max(hits, key=os.path.getmtime) if hits else None


def check(prm_text):
    print("\n================ generated parameterFile.prm (relevant lines) ================")
    ok = True
    for pyname, (key, expected, numeric) in CHECKS.items():
        m = re.search(rf"set\s+{key}\s*=?\s*([^\n#]+)", prm_text, re.IGNORECASE)
        if not m:
            print(f"  [MISS] {pyname:28s} key /{key}/ NOT FOUND in .prm  <-- possibly dropped/defaulted")
            ok = False
            continue
        rhs = m.group(1).strip()
        if numeric:
            try:
                got = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", rhs)[0])
                good = abs(got - float(expected)) < 1e-9 or abs(got - float(expected)) / max(abs(float(expected)), 1e-30) < 1e-6
            except (ValueError, IndexError):
                good = False
            status = "OK  " if good else "FAIL"
            if not good:
                ok = False
            print(f"  [{status}] {pyname:28s} sent={expected}  .prm='{rhs}'")
        else:
            good = str(expected).lower() in rhs.lower()
            if not good:
                ok = False
            print(f"  [{'OK  ' if good else 'FAIL'}] {pyname:28s} sent='{expected}'  .prm='{rhs}'")
    print("=" * 78)
    return ok


def main():
    os.makedirs(WORK, exist_ok=True)
    cell = np.diag([7.6, 7.6, 7.6]) * Bohr
    frac = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    atoms = Atoms("Al4", scaled_positions=frac, cell=cell, pbc=[True, True, True])
    nproc = int(os.environ.get("VERIFY_NPROC", "4"))
    calc = DFTFE(bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc, use_device=True,
                 psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
                 keep_scratch=True, cwd=WORK, verbosity=2,
                 log_file="verify.log", compute_forces=False, **PARAMS)
    atoms.calc = calc
    try:
        atoms.get_potential_energy()   # triggers reinit -> writes the .prm
    except Exception as exc:
        # The .prm is written at reinit, BEFORE the SCF. So even if the solve
        # crashes (e.g. a param value that's physically incompatible with this
        # small test system), the generated .prm still lets us verify INJECTION.
        print(f"[verify] run did not finish ({type(exc).__name__}: {exc}); "
              f"inspecting the generated .prm anyway (injection is set at reinit).")
    finally:
        calc.close()

    prm = find_prm()
    if not prm:
        print(f"[verify] no generated parameterFile.prm under {WORK} (keep_scratch off? run failed?)")
        sys.exit(2)
    print(f"[verify] inspecting {prm}")
    ok = check(open(prm).read())
    print(f"PARAM-INJECTION VERIFY {'PASS' if ok else 'FAIL — some values did NOT reach the .prm'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
