#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
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
        [--no-device] \\
        [--tests co2_nonperiodic n2_nonperiodic ...]

Missing references
------------------
A reference of ``null`` (or an absent key) is a **FAILURE**, not a licence to
invent one. It means nothing is pinned for that quantity, so the test cannot
regress and reporting PASS would be a lie.

Seeding a reference is therefore an explicit, deliberate act::

    python test_suite.py ... --seed

and should only ever be done on a binary you trust, from a run you have
looked at. Without ``--seed`` this script never writes references.json.
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

# Stress, per component, in DFT-FE's own units. Set to the force bar's *relative*
# strictness rather than copying its absolute value: printed stresses here are
# ~1e-4 Ha/Bohr^3, four orders below the ~1e-2 Ha/Bohr forces, so 1e-8 absolute
# would be a 1e-4 relative check -- no check at all. 1e-10 is the same 1e-6
# relative bar the forces get. NOT yet measured on a real run: al_bulk's energy
# agrees with native pGD bitwise (job 8755816), so stress is expected to as well,
# but the first run that compares it is the one that will say. If it lands near
# this bar rather than far under it, revisit the number instead of loosening it
# reflexively -- and look at the unit conversion first.
#
# Placement caveat, from PROJECT.md 14.2: stress is the most placement-dependent
# quantity measured here (2.0e-04 relative for a 2 A rigid shift, against 3e-06
# for energy). This bar is only meaningful because both arms compute the same
# geometry -- make_references.py mirrors the socket path's rigid_shift. A stress
# reference must never be compared across placements.
STRESS_TOL_HA_BOHR3 = 1e-10

# Map test name → which binary kind to use
BINARY_KIND = {
    "co2_nonperiodic":  "real",
    "n2_nonperiodic":   "real",
    "relax_o2":         "real",
    "al_slab_semiperiodic": "real",   # gamma-only: periodic in x/y, open along z
    "graphene_periodic": "complex",
    "al_bulk_periodic":  "complex",
}

# Nothing is run-only: every case has at least one number gated at 1e-10 Ha.
# For relax_o2 that number is the single point at its INITIAL geometry, which is
# what the native reference holds. Its relaxed landing point is reported beside
# it and not gated -- once the references stopped coming from the interface
# itself and started coming from native pGD, "the relaxed energy" became one
# LBFGS implementation against another, and no fixed-configuration determinism
# argument closes that. See check_relaxed.
RUN_ONLY = set()

ALL_TESTS = list(BINARY_KIND.keys())

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")
REFS_FILE  = os.path.join(os.path.dirname(__file__), "references.json")
# One file per device, because a reference file holds exactly ONE configuration
# per case: a single energy, forces, stress and provenance. use_device is in
# CONFIG_KEYS, so storing a GPU number where the CPU one lives does not merely
# overwrite it -- it makes every CPU run report MISMATCH. Keeping the two sets in
# separate files lets each stay diffable against its own native_refs/ audit
# trail, and leaves the CPU set (np=64, 6/6 PASS, five of six bitwise identical
# to native pGD) exactly as validated.
GPU_REFS_FILE = os.path.join(os.path.dirname(__file__), "references.gpu.json")

_NO_REF_HINT = ("re-run with --seed on a trusted binary to pin it deliberately")


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


def provenance(args, kind: str) -> dict:
    """What must be identical for the 1e-10 Ha tolerance to be meaningful.

    1e-10 Ha is a determinism bar, not a physics bar, and DFT-FE is only that
    reproducible against *itself in the same configuration*. Measured on this
    project (PROJECT.md 5.7): a 4x rank change moves the energy by 6.4e-9 Ha and
    forces by ~5 digits -- 60x and 1000x outside the tolerances here, with
    nothing wrong. Those are MPI reduction-order differences, not regressions.

    So a reference is only comparable to a run that shares its binary, its rank
    count and its device. Recording that with the number is what lets the suite
    tell "the code changed" apart from "you ran it differently", instead of
    reporting the second as the first.
    """
    binary = {"real": args.dftfe_real, "complex": args.dftfe_complex}[kind]
    try:
        st = os.stat(binary)
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        stamp = "unknown"
    return {
        "binary": os.path.abspath(binary),
        "binary_stamp": stamp,
        "np": args.np,
        "use_device": bool(args.use_device),
    }


# What must match for a 1e-10 Ha comparison to mean anything. The binary is
# deliberately NOT here. References generated by make_references.py come from
# native pGD, a different executable from the one the ASE arm drives -- that is
# the whole point, since a reference produced by the interface can only detect
# change, never wrongness. What cannot differ is the numerical environment:
# a 4x rank change alone moves the energy 6.4e-9 Ha (PROJECT.md 5.7).
#
# The unavoidable assumption is that pGD and the branch are built equivalently
# (same flags, same deal.II, physics sources identical). Verified by hand on
# 2026-08-13; it is not machine-checkable from here, so it is stated rather than
# silently relied upon.
CONFIG_KEYS = ("np", "use_device")


def provenance_mismatch(ref: dict, now: dict):
    """Config fields that differ between the reference's run and this one."""
    old = ref.get("provenance")
    if not old:
        return None  # pre-provenance reference; treated as unverifiable
    return [k for k in CONFIG_KEYS if old.get(k) != now.get(k)] or []


def check_energy(computed: float, ref: float) -> tuple:
    """Returns (passed: bool, diff_ha: float). Tolerance: 1e-10 Ha."""
    diff = abs(computed - ref)
    return diff < ENERGY_TOL_HA, diff


def stress_to_ha_per_bohr3(stress) -> np.ndarray:
    """ASE stress (Voigt-6 or 3x3, eV/A^3) -> a 3x3 matrix in Ha/Bohr^3.

    The two arms report stress in different units and different shapes, so the
    comparison has to normalise one to the other. Native DFT-FE prints a 3x3 in
    Ha/Bohr^3 (``Cell stress (Hartree/Bohr^3)``); ASE's ``get_stress()`` returns
    Voigt-6 in eV/A^3. Converting the *computed* value into DFT-FE's units keeps
    references.json in the units the code that produced it printed, which is what
    makes a reference auditable against its own native.out.

    Voigt order is ASE's: (xx, yy, zz, yz, xz, xy).
    """
    s = np.asarray(stress, dtype=float)
    if s.shape == (6,):
        xx, yy, zz, yz, xz, xy = s
        s = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    elif s.shape != (3, 3):
        raise ValueError(f"stress has shape {s.shape}; expected (6,) or (3, 3)")
    return s / (Hartree / Bohr ** 3)


def check_stress(computed, ref_list) -> tuple:
    """Returns (passed, max_abs_diff, worst_i, worst_j, ref_asymmetry), Ha/Bohr^3.

    Both sides are symmetrised before diffing, and that is not a convenience:
    DFT-FE's printed 3x3 is only symmetric to ~5e-13 (al_bulk's [0][1] and [1][0]
    differ by 4.6e-13), while anything arriving through ASE is symmetric *by
    construction* -- ``full_3x3_to_voigt_6_stress`` averages each off-diagonal
    pair. Diffing raw against raw therefore charges the interface for the native
    print's own asymmetry, and puts a ~7.5e-13 floor under a comparison that has
    nothing to do with the interface. Averaging the reference the same way ASE
    averages is the matching operation.

    The reference's asymmetry is returned rather than discarded: it is the floor
    below which STRESS_TOL_HA_BOHR3 cannot be tightened, so it belongs in the log
    where someone tightening the bar will see it.
    """
    got = stress_to_ha_per_bohr3(computed)
    # SIGN CONVENTION: none needed. The reference is DFT-FE's printed cell stress,
    # which job 8756125 measured to be ASE convention already -- dE/dV came out at
    # -1.0958e-04 Ha/Bohr^3 against a printed -1.1035e-04, same sign, ratio 0.993.
    # The interface therefore undoes dftfeWrapper's negation on its own side
    # (calculator._WRAPPER_STRESS_SIGN), so both sides are in ASE convention here
    # and this comparison is direct.
    #
    # It was briefly the other way round: the first GPU run reported max|dS| =
    # 2.207e-04 at relative error 2.0 on a ~1.1e-04 stress -- exactly twice the
    # magnitude, a pure sign flip -- and the first fix negated the REFERENCE,
    # which made the two arms agree with each other while both disagreed with
    # ASE's definition. If that constant ever returns to +1 without this comment
    # changing, this check will fail at exactly 2x the stress magnitude again.
    ref = np.array(ref_list, dtype=float)
    asym = np.abs(ref - ref.T).max()
    diff = np.abs(got - (ref + ref.T) / 2.0)
    idx = np.unravel_index(diff.argmax(), diff.shape)
    return (diff.max() < STRESS_TOL_HA_BOHR3, diff.max(),
            int(idx[0]), int(idx[1]), asym)


def check_relaxed(extra: dict, ref: dict) -> str:
    """Report an ASE relaxation's landing point against native GEOOPT's.

    Deliberately NOT a gate, and the reason is in the reference itself: the
    "relaxed" block comes from DFT-FE's own LBFGS, while the case drives ASE
    LBFGS matched to it (memory=5, maxstep=0.5 Bohr). Same family, same two
    settings -- but line search, initial inverse-Hessian scaling and the
    convergence test are still each implementation's own, so the two stop at two
    different points on the same basin, both legitimately converged. The gap
    between them is a property of the pair of optimizers -- something to measure
    on the first run and only then, if it proves stable, to turn into a bound.
    Asserting a number here before anyone has seen one would be inventing a
    tolerance, which is the same mistake as auto-seeding a reference.

    What *is* checked is that both arms actually stopped where they said they
    would: each one's max|F| against the shared ``force_tol_ha_bohr``. That is a
    self-consistency check on each arm separately, not a cross-implementation
    assertion, so it cannot fail for a reason nobody can act on.

    Returns a human-readable note ("" if this case has no relaxed half).
    """
    got = (extra or {}).get("relaxed")
    ref_rel = (ref or {}).get("relaxed")
    if got is None and ref_rel is None:
        return ""
    if ref_rel is None:
        return ("relaxed: this run produced a relaxed point but references.json "
                "has no 'relaxed' block for it — regenerate with "
                "`make_references.py --relax`; nothing compared")
    if got is None:
        return ("relaxed: references.json carries a 'relaxed' block but the case "
                "reported none — nothing compared")

    e_ase, e_nat = got["energy_ha"], ref_rel["energy_ha"]
    d_e = abs(e_ase - e_nat)
    ftol = ref_rel.get("force_tol_ha_bohr")
    f_ase = np.abs(np.array(got["forces_ha_per_bohr"])).max()
    f_nat = np.abs(np.array(ref_rel["forces_ha_per_bohr"])).max()

    note = (f"relaxed (MEASURED, not gated): ASE {e_ase:.10e} Ha vs native "
            f"{e_nat:.10e} Ha  |  ΔE = {d_e:.3e} Ha ({d_e * 27211.386:.3f} meV)")
    if ftol:
        note += (f"\n    max|F|: ASE {f_ase:.3e}, native {f_nat:.3e} Ha/Bohr "
                 f"(FORCE TOL {ftol:.6e})")
        for arm, fmax in (("ASE", f_ase), ("native", f_nat)):
            if fmax > ftol:
                note += (f"\n    WARNING: {arm} arm's max|F| exceeds the force "
                         f"tolerance it was given — that arm did not converge "
                         f"where it claimed to")
    steps = got.get("steps")
    nat_steps = ref_rel.get("native_steps")
    note += (f"\n    steps: ASE LBFGS {steps}, native LBFGS "
             f"{nat_steps if nat_steps is not None else 'not parsed'}")
    note += ("\n    ASE LBFGS (matched) vs DFT-FE LBFGS: record this ΔE, do not "
             "treat it as a pass/fail")
    return note


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

def refs_file_for(args) -> str:
    """Which reference file this run compares against."""
    if args.refs:
        return args.refs
    return GPU_REFS_FILE if args.use_device else REFS_FILE


def run_suite(args):
    tests = args.tests or ALL_TESTS
    bins  = {"real": args.dftfe_real, "complex": args.dftfe_complex}
    refs_file = refs_file_for(args)

    # Validate binary paths
    for kind, path in bins.items():
        if not path or not os.path.isfile(path):
            print(f"[ERROR] {kind} binary not found: {path}", flush=True)
            sys.exit(1)

    # Load references
    if not os.path.isfile(refs_file):
        print(f"[ERROR] no reference file at {refs_file}\n"
              f"        {'GPU' if args.use_device else 'CPU'} references have not "
              f"been generated for this device. Generate them with\n"
              f"        make_references.py "
              f"{'--use-device ' if args.use_device else ''}--np <ranks> ...\n"
              f"        or point at an existing set with --refs.", flush=True)
        sys.exit(1)
    with open(refs_file) as f:
        refs = json.load(f)
    print(f"references: {os.path.basename(refs_file)}  "
          f"(use_device={bool(args.use_device)})", flush=True)

    results    = {}
    any_seeded = False
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines  = [f"ASE-DFTFE Test Suite  —  {timestamp}", "=" * 60]
    if args.seed:
        log_lines.append("--seed given: null references WILL be overwritten "
                         "with this run's values.")

    for line in log_lines:
        print(line, flush=True)

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
            out = run_fn(
                dftfe_bin=dftfe_b,
                psp_library=args.psp_library,
                np_tasks=args.np,
                use_device=args.use_device,
            )
            # Cases return (energy, forces, stress), and optionally a fourth
            # dict of quantities that are reported but not gated -- currently
            # only a relaxation's landing point. Single-point cases stay
            # three-element; nothing had to change for them.
            if len(out) == 4:
                energy_ha, forces_ha_per_bohr, stress, extra = out
            else:
                energy_ha, forces_ha_per_bohr, stress = out
                extra = {}
        except Exception:
            tb = traceback.format_exc()
            msg = f"[FAIL] {name}  →  EXCEPTION\n{tb}"
            print(msg, flush=True)
            results[name] = {"status": "FAIL", "reason": "exception", "traceback": tb}
            log_lines.append(msg)
            continue

        # ── provenance gate ──
        # Checked before the numbers, because comparing across configurations
        # produces a FAIL that says "regression" when it means "different rank
        # count". At 1e-10 Ha that is not a corner case, it is the common case.
        now_prov = provenance(args, kind)
        drift = provenance_mismatch(ref, now_prov)
        if args.seed:
            refs.setdefault(name, {})["provenance"] = now_prov
        elif drift:
            old = ref["provenance"]
            detail = "; ".join(f"{k}: ref {old.get(k)!r} vs now {now_prov.get(k)!r}"
                               for k in drift)
            line = (f"[MISMATCH] {name}\n  reference was produced under a different "
                    f"numerical configuration -- {detail}\n  Not compared: a 4x rank "
                    f"change alone moves the energy 6.4e-9 Ha, 60x this tolerance, "
                    f"with nothing wrong. Re-generate for this configuration, or run "
                    f"at the recorded one.")
            print(line, flush=True)
            log_lines.append(line)
            results[name] = {"status": "MISMATCH", "drift": drift}
            continue
        elif drift is None and ref.get("energy_ha") is not None:
            log_lines.append(f"  note: {name} reference predates provenance "
                             f"tracking; its configuration is unverifiable.")
        elif (ref.get("provenance") or {}).get("generator") == "native-pgd":
            # Worth stating in the log: a PASS here is the interface agreeing
            # with an independent implementation, not with its own past output.
            print(f"  reference: native pGD "
                  f"({os.path.basename(ref['provenance'].get('binary',''))}), "
                  f"np={ref['provenance'].get('np')}", flush=True)

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
            if args.seed:
                refs.setdefault(name, {})["energy_ha"] = energy_ha
                seeded = any_seeded = True
                energy_status = "SEEDED"
                energy_note   = f"energy = {energy_ha:.10e} Ha  (saved as reference)"
            else:
                energy_status = "FAIL"
                energy_note   = (
                    f"energy = {energy_ha:.10e} Ha  |  ref = null  "
                    f"|  NO ENERGY REFERENCE — nothing is pinned; {_NO_REF_HINT}"
                )
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
            if ref_forces_list is None or seeded:
                if args.seed:
                    refs.setdefault(name, {})["forces_ha_per_bohr"] = \
                        forces_ha_per_bohr.tolist()
                    any_seeded   = True
                    force_status = "SEEDED"
                    force_note   = f"forces saved as reference ({forces_ha_per_bohr.shape[0]} atoms)"
                else:
                    force_status = "FAIL"
                    force_note   = (
                        f"NO FORCE REFERENCE — 'forces_ha_per_bohr' is null or "
                        f"absent; {_NO_REF_HINT}"
                    )
            else:
                ref_forces = np.array(ref_forces_list)
                f_pass, max_diff, worst_atom, worst_comp = check_forces(forces_ha_per_bohr, ref_forces)
                comp_labels = ['x', 'y', 'z']
                force_status = "PASS" if f_pass else "FAIL"
                force_note   = (
                    f"max |F_computed - F_ref| = {max_diff:.3e} Ha/Bohr  "
                    f"(tol 1e-8, worst: atom {worst_atom} {comp_labels[worst_comp]})"
                )

        # ── Stress check ──
        # al_bulk_periodic is the case that exists for this: it sets
        # compute_forces=False, compute_stress=True and returns stress as its
        # headline quantity. The suite used to discard the third return value
        # entirely, so that case's only real output was unchecked while the case
        # reported PASS -- the same shape of hole as the force check that read a
        # key references.json never had (PROJECT.md 14.3).
        stress_status = None
        stress_note   = ""
        if stress is not None:
            ref_stress = ref.get("stress_ha_per_bohr3")
            if ref_stress is None or seeded:
                if args.seed:
                    refs.setdefault(name, {})["stress_ha_per_bohr3"] = \
                        stress_to_ha_per_bohr3(stress).tolist()
                    any_seeded    = True
                    stress_status = "SEEDED"
                    stress_note   = "stress saved as reference (3x3, Ha/Bohr^3)"
                else:
                    stress_status = "FAIL"
                    stress_note   = (
                        f"NO STRESS REFERENCE — this case computes stress but "
                        f"'stress_ha_per_bohr3' is null or absent; {_NO_REF_HINT}"
                    )
            else:
                s_pass, s_diff, wi, wj, asym = check_stress(stress, ref_stress)
                stress_status = "PASS" if s_pass else "FAIL"
                rel = s_diff / max(np.abs(np.array(ref_stress)).max(), 1e-300)
                stress_note = (
                    f"max |S_computed - S_ref| = {s_diff:.3e} Ha/Bohr^3  "
                    f"(tol {STRESS_TOL_HA_BOHR3:g}, rel {rel:.1e}, "
                    f"worst: [{wi}][{wj}]; reference's own asymmetry "
                    f"{asym:.1e}, the floor for this bar)"
                )
        elif ref.get("stress_ha_per_bohr3") is not None:
            # Deliberate asymmetry, not a failure: a reference may carry more
            # than the case asks for. Say so, because "no line printed" is
            # indistinguishable from "checked and fine".
            stress_note = ("note: reference carries a stress but this case does "
                           "not request one; not compared")
        if forces_ha_per_bohr is None and ref.get("forces_ha_per_bohr") is not None:
            force_note = ("note: reference carries forces but this case sets "
                          "compute_forces=False; not compared")

        # ── Relaxed landing point: reported, never gated (see check_relaxed) ──
        relaxed_note = check_relaxed(extra, ref)

        # ── Overall status ──
        # FAIL wins over SEEDED: a pinned energy that regressed is still a
        # regression even if this run happened to seed the forces beside it.
        if "FAIL" in (energy_status, force_status, stress_status):
            overall = "FAIL"
        elif seeded or "SEEDED" in (force_status, stress_status):
            overall = "SEEDED"
        else:
            overall = "PASS"

        results[name] = {
            "status":       overall,
            "energy_ha":    energy_ha,
            "energy_note":  energy_note,
            "force_note":   force_note,
            "stress_note":  stress_note,
            "relaxed_note": relaxed_note,
        }

        line = f"[{overall:6s}] {name}\n  {energy_note}"
        if force_note:
            line += f"\n  {force_note}"
        if stress_note:
            line += f"\n  {stress_note}"
        if relaxed_note:
            line += f"\n  {relaxed_note}"
        print(line, flush=True)
        log_lines.append(line)

    # ── Save updated references, but ONLY if seeding was asked for and used ──
    if any_seeded:
        with open(refs_file, "w") as f:
            json.dump(refs, f, indent=2)
        print(f"\nReferences updated: {refs_file}", flush=True)

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

def parse_args(argv=None):
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
        "--no-device",
        dest="use_device",
        action="store_false",
        default=True,
        help="Run on the host (USE GPU off). Required when --dftfe-real / "
             "--dftfe-complex point at CPU-only builds: those have no DEVICE "
             "branch, so use_device=True leaves DFT-FE with no solver at all.",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Write this run's values into references.json wherever the "
             "reference is null. Without it, a null reference is a FAILURE "
             "and references.json is never modified.",
    )
    parser.add_argument(
        "--refs",
        help="Reference file to compare against. Defaults to references.json on "
             "the host and references.gpu.json with GPUs, since one file holds "
             "one configuration per case.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        metavar="TEST",
        help=f"Subset of tests to run. Available: {ALL_TESTS}",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())
