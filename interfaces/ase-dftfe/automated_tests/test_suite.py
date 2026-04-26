#!/usr/bin/env python3
"""
ASE-DFTFE Automated Test Suite
================================
Runs a configurable set of test cases, compares results against
references.json, and reports PASS/FAIL.

Usage:
    python test_suite.py \\
        --dftfe-real    /path/to/build_gpu/release/real/dftfe \\
        --dftfe-complex /path/to/build_gpu/release/complex/dftfe \\
        --psp-library   /path/to/psp_library \\
        --np 8 \\
        [--tests co2_nonperiodic n2_nonperiodic ...]

First-run behaviour
-------------------
If a test's reference value in references.json is null, the computed
value is saved as the new reference (auto-seeding) and the test is
marked [SEEDED] rather than PASS/FAIL.
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime

import numpy as np
from ase.units import Bohr, Hartree, eV, Angstrom

# ── Constants ──────────────────────────────────────────────────────────────
ENERGY_TOL_HA = 1e-10   # |E - E_ref| < this (Hartree, 10 decimal places)
FORCE_TOL_HA_BOHR = 1e-8  # max |F_computed - F_ref| per component (Ha/Bohr, 8 decimal places)

# Map test name → which binary kind to use
BINARY_KIND = {
    "co2_nonperiodic":  "real",
    "n2_nonperiodic":   "real",
    "relax_o2":         "real",
    "graphene_periodic": "complex",
    "al_bulk_periodic":  "complex",
}

# Tests that only check successful completion — no energy/force comparison.
# (Relaxation final geometry depends on optimizer path, so exact values vary.)
RUN_ONLY = {"relax_o2"}

ALL_TESTS = list(BINARY_KIND.keys())

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")
REFS_FILE  = os.path.join(os.path.dirname(__file__), "references.json")


# ── Helpers ────────────────────────────────────────────────────────────────

def load_case(name: str):
    """Dynamically import cases/<name>.py and return its `run` function."""
    path = os.path.join(CASES_DIR, f"{name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Case file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def check_energy(computed: float, ref: float) -> tuple:
    """Returns (passed: bool, diff_ha: float). Tolerance: 1e-10 Ha."""
    diff = abs(computed - ref)
    return diff < ENERGY_TOL_HA, diff


def check_forces(computed: np.ndarray, ref: np.ndarray) -> tuple:
    """
    Pointwise per-component comparison.
    Returns (passed: bool, max_abs_diff_ha_bohr: float, worst_atom: int, worst_comp: int).
    Tolerance: 1e-8 Ha/Bohr per component.
    """
    diff = np.abs(computed - ref)          # shape (N, 3)
    max_diff = diff.max()
    idx = np.unravel_index(diff.argmax(), diff.shape)
    return max_diff < FORCE_TOL_HA_BOHR, max_diff, int(idx[0]), int(idx[1])


# ── Main ───────────────────────────────────────────────────────────────────

def run_suite(args):
    tests = args.tests or ALL_TESTS
    bins  = {"real": args.dftfe_real, "complex": args.dftfe_complex}

    # Validate binary paths
    for kind, path in bins.items():
        if not path or not os.path.isfile(path):
            print(f"[ERROR] {kind} binary not found: {path}", flush=True)
            sys.exit(1)

    # Load references
    with open(REFS_FILE) as f:
        refs = json.load(f)

    results   = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines = [f"ASE-DFTFE Test Suite  —  {timestamp}", "=" * 60]

    print(log_lines[0], flush=True)
    print(log_lines[1], flush=True)

    for name in tests:
        if name not in BINARY_KIND:
            print(f"[SKIP] Unknown test: {name}", flush=True)
            continue

        print(f"\nRunning: {name} ...", flush=True)
        kind    = BINARY_KIND[name]
        dftfe_b = bins[kind]
        ref     = refs.get(name, {})

        try:
            run_fn = load_case(name)
            energy_ha, forces_ha_per_bohr, stress = run_fn(
                dftfe_bin=dftfe_b,
                psp_library=args.psp_library,
                np_tasks=args.np,
            )
        except Exception:
            tb = traceback.format_exc()
            msg = f"[FAIL] {name}  →  EXCEPTION\n{tb}"
            print(msg, flush=True)
            results[name] = {"status": "FAIL", "reason": "exception", "traceback": tb}
            log_lines.append(msg)
            continue

        # ── RUN_ONLY: just check it didn't crash ──
        if name in RUN_ONLY:
            line = f"[{'PASS':6s}] {name}  (run-only: completed successfully)"
            print(line, flush=True)
            log_lines.append(line)
            results[name] = {"status": "PASS", "energy_note": "run-only", "force_note": ""}
            continue

        # ── Energy check ──
        ref_energy = ref.get("energy_ha")
        seeded = False

        if ref_energy is None:
            refs[name]["energy_ha"] = energy_ha
            seeded = True
            energy_status = "SEEDED"
            energy_note   = f"energy = {energy_ha:.10e} Ha  (saved as reference)"
        else:
            e_pass, e_diff = check_energy(energy_ha, ref_energy)
            energy_status  = "PASS" if e_pass else "FAIL"
            energy_note    = (
                f"energy = {energy_ha:.10e} Ha  |  ref = {ref_energy:.10e} Ha  "
                f"|  diff = {e_diff:.3e} Ha"
            )

        # ── Force check ──
        force_status = None
        force_note   = ""
        if forces_ha_per_bohr is not None:
            ref_forces_list = ref.get("forces_ha_per_bohr")
            if seeded or ref_forces_list is None:
                refs[name]["forces_ha_per_bohr"] = forces_ha_per_bohr.tolist()
                force_status = "SEEDED"
                force_note   = f"forces saved as reference ({forces_ha_per_bohr.shape[0]} atoms)"
            else:
                ref_forces = np.array(ref_forces_list)
                f_pass, max_diff, worst_atom, worst_comp = check_forces(forces_ha_per_bohr, ref_forces)
                comp_labels = ['x', 'y', 'z']
                force_status = "PASS" if f_pass else "FAIL"
                force_note   = (
                    f"max |F_computed - F_ref| = {max_diff:.3e} Ha/Bohr  "
                    f"(tol 1e-8, worst: atom {worst_atom} {comp_labels[worst_comp]})"
                )

        # ── Overall status ──
        if seeded:
            overall = "SEEDED"
        elif energy_status == "FAIL" or force_status == "FAIL":
            overall = "FAIL"
        else:
            overall = "PASS"

        results[name] = {
            "status":       overall,
            "energy_ha":    energy_ha,
            "energy_note":  energy_note,
            "force_note":   force_note,
        }

        line = f"[{overall:6s}] {name}\n  {energy_note}"
        if force_note:
            line += f"\n  {force_note}"
        print(line, flush=True)
        log_lines.append(line)

    # ── Save updated references if anything was seeded ──
    with open(REFS_FILE, "w") as f:
        json.dump(refs, f, indent=2)

    # ── Summary ──
    summary = "\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60
    counts  = {"PASS": 0, "FAIL": 0, "SEEDED": 0, "FAIL(exc)": 0}
    for name, r in results.items():
        s = r["status"]
        counts[s] = counts.get(s, 0) + 1
        summary  += f"\n  [{s:6s}] {name}"
    summary += f"\n{'-'*60}"
    summary += f"\nPASS={counts.get('PASS',0)}  FAIL={counts.get('FAIL',0)}  SEEDED={counts.get('SEEDED',0)}"
    print(summary, flush=True)
    log_lines.append(summary)

    # ── Write results file ──
    results_file = f"test_results_{timestamp}.txt"
    with open(results_file, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults written to: {results_file}", flush=True)

    # Exit 1 if any FAIL
    if counts.get("FAIL", 0) > 0:
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASE-DFTFE Automated Test Suite"
    )
    parser.add_argument(
        "--dftfe-real",
        required=True,
        help="Path to real DFT-FE binary (for gamma-only tests)",
    )
    parser.add_argument(
        "--dftfe-complex",
        required=True,
        help="Path to complex DFT-FE binary (for k-point tests)",
    )
    parser.add_argument(
        "--psp-library",
        required=True,
        help="Path to directory containing *.upf pseudopotential files",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=8,
        help="Number of MPI tasks (default: 8)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        metavar="TEST",
        help=f"Subset of tests to run. Available: {ALL_TESTS}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())
