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
    "mixing_history":              (r"MIXING HISTORY", 8, True),   # generic path; was mis-mapped to LBFGS HISTORY
    "chebyshev_polynomial_degree": (r"CHEBYSHEV POLYNOMIAL DEGREE", 18, True),  # generic path
    "base_mesh_size":              (r"BASE MESH SIZE", 2.5, True),             # generic path
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
    chebyshev_polynomial_degree=18, base_mesh_size=2.5,
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


# Second, DISTINCT value set (set B). Every checked value differs from PARAMS
# (set A) so we can confirm the .prm value TRACKS the input — catching a param
# that silently shows a fixed default regardless of what we send.
PARAMS_B = dict(
    xc="LDA-PW", polynomial_order=6, mesh_size=1.9, atom_ball_radius=5.0,
    fermi_temp=277.0, tolerance=7e-5, scf_mixing=0.44,
    mixing_scheme="ANDERSON", num_bands=30, max_scf_iterations=41,
    density_quadrature_rule=14, use_single_prec_cheby=False,
    use_time_reversal_symmetry=False, mp_grid=(3, 3, 3), mp_grid_shift=(0, 0, 0),
    npkpt=3, orthogonalization_type="GS", smeared_nuclear_charges=0,
    use_group_symmetry=0, mixing_history=12,
    chebyshev_polynomial_degree=22, base_mesh_size=3.0,
)
CHECKS_B = {  # expected value in set B (for the same keys as CHECKS)
    "polynomial_order": 6, "mesh_size": 1.9, "atom_ball_radius": 5.0,
    "fermi_temp": 277.0, "tolerance": 7e-5, "scf_mixing": 0.44,
    "mixing_scheme": "ANDERSON", "num_bands": 30, "max_scf_iterations": 41,
    "density_quadrature_rule": 14, "use_single_prec_cheby": "false",
    "use_time_reversal_symmetry": "false", "mp_grid": 3, "mp_grid_shift": 0,
    "npkpt": 3, "orthogonalization_type": "GS", "smeared_nuclear_charges": "false",
    "use_group_symmetry": "false", "mixing_history": 12,
    "chebyshev_polynomial_degree": 22, "base_mesh_size": 3.0,
}


def run_and_read_prm(params, tag, nproc):
    work = os.path.join(WORK, tag)
    os.makedirs(work, exist_ok=True)
    calc = DFTFE(bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc, use_device=True,
                 psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
                 keep_scratch=True, cwd=work, verbosity=2,
                 log_file="verify.log", compute_forces=False, **params)
    atoms = Atoms("Al4", scaled_positions=[[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
                  cell=np.diag([7.6, 7.6, 7.6]) * Bohr, pbc=[True, True, True])
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except Exception as exc:
        print(f"[verify:{tag}] run did not finish ({type(exc).__name__}); .prm still written at reinit.")
    finally:
        calc.close()
    hits = glob.glob(os.path.join(work, "dftfeScratch*", "parameterFile.prm"))
    return open(max(hits, key=os.path.getmtime)).read() if hits else None


def extract(prm_text, key, numeric):
    m = re.search(rf"set\s+{key}\s*=?\s*([^\n#]+)", prm_text, re.IGNORECASE)
    if not m:
        return None
    rhs = m.group(1).strip()
    if numeric:
        try:
            return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", rhs)[0])
        except (ValueError, IndexError):
            return None
    return rhs


def main():
    os.makedirs(WORK, exist_ok=True)
    nproc = int(os.environ.get("VERIFY_NPROC", "4"))
    prmA = run_and_read_prm(PARAMS, "A", nproc)
    prmB = run_and_read_prm(PARAMS_B, "B", nproc)
    if not prmA or not prmB:
        print("[verify] missing generated .prm (keep_scratch off / run failed early)")
        sys.exit(2)

    print("\n===== two-value tracking (value must change A->B, not stay at a default) =====")
    ok = True
    for pyname, (key, expA, numeric) in CHECKS.items():
        expB = CHECKS_B[pyname]
        gotA, gotB = extract(prmA, key, numeric), extract(prmB, key, numeric)

        def match(got, exp):
            if got is None:
                return False
            if numeric:
                return abs(got - float(exp)) < 1e-9 or abs(got - float(exp)) / max(abs(float(exp)), 1e-30) < 1e-6
            return str(exp).lower() in str(got).lower()

        tracks = match(gotA, expA) and match(gotB, expB) and str(expA).lower() != str(expB).lower()
        if not tracks:
            ok = False
        print(f"  [{'OK  ' if tracks else 'FAIL'}] {pyname:26s} A: sent={expA} got={gotA} | B: sent={expB} got={gotB}")
    print("=" * 78)
    print(f"TWO-VALUE PARAM VERIFY {'PASS — every value tracks its input' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
