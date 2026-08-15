#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Compare the native and ASE arms of the BCC-Mo 431-atom scaling study.

Pass criteria (per PI): energy agreement to >=10 significant digits, and max
absolute per-component force difference to >=8 digits. Published reference
values are irrelevant -- only native-vs-ASE agreement is being measured.

Both arms list atoms in the same order (both geometries come from the same
``coordinates.inp``), so forces are compared component by component without any
matching step.

    python3 compare.py --native native_32nodes.out --ase result_ase_32nodes.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

ENERGY_RE = re.compile(r"Total free energy:\s*([-\d.eE+]+)")
FORCE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$")


def parse_native(path):
    """Last 'Total free energy' and the ion-force block that follows it."""
    text = open(path, errors="replace").read()

    energies = ENERGY_RE.findall(text)
    if not energies:
        raise SystemExit(f"{path}: no 'Total free energy' line -- did the run finish?")
    energy = float(energies[-1])

    lines = text.splitlines()
    forces = None
    for i, line in enumerate(lines):
        if "Ion forces (Hartree/Bohr)" not in line:
            continue
        rows = []
        for follow in lines[i + 1:]:
            m = FORCE_ROW_RE.match(follow)
            if m:
                rows.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
            elif rows:
                break
        if rows:
            forces = rows  # keep the last block
    if forces is None:
        raise SystemExit(f"{path}: no ion-force block found")
    return energy, np.array(forces, float)


def digits(a, b):
    """Agreeing significant digits between two scalars."""
    if a == b:
        return float("inf")
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return float("inf")
    return -np.log10(abs(a - b) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, help="native DFT-FE stdout log")
    ap.add_argument("--ase", required=True, help="result_ase*.json from ase_bccmo431.py")
    ap.add_argument("--energy-digits", type=float, default=10.0)
    ap.add_argument("--force-digits", type=float, default=8.0)
    ap.add_argument("--out", default=None, help="write the comparison as JSON")
    args = ap.parse_args()

    e_native, f_native = parse_native(args.native)
    ase = json.load(open(args.ase))
    e_ase = ase["energy_ha"]
    f_ase = np.array(ase["forces_ha_bohr"], float)

    if f_native.shape != f_ase.shape:
        raise SystemExit(
            f"force shape mismatch: native {f_native.shape} vs ASE {f_ase.shape}")

    d_energy = abs(e_native - e_ase)
    e_digits = digits(e_native, e_ase)

    df = np.abs(f_native - f_ase)
    max_df = float(df.max())
    worst = int(np.unravel_index(df.argmax(), df.shape)[0])
    f_scale = float(np.abs(f_native).max())
    # Force "digits" follows the convention already used for the accepted BCC Mo
    # and Li2O results (PROJECT_Aurora.md 5.1): -log10 of the max absolute
    # component deviation in Ha/Bohr, so 3.96e-8 reads as ~7.4 digits. Force
    # components pass through zero, which makes a per-component relative measure
    # meaningless; the deviation relative to the largest force in the system is
    # reported alongside as context, not as the pass criterion.
    f_digits = float("inf") if max_df == 0 else -np.log10(max_df)
    f_digits_rel = (float("inf") if max_df == 0
                    else -np.log10(max_df / f_scale)) if f_scale else float("nan")

    energy_ok = bool(e_digits >= args.energy_digits)
    force_ok = bool(f_digits >= args.force_digits)

    print(f"nodes                 : {ase.get('nodes')} ({ase.get('ranks')} ranks)")
    print(f"atoms                 : {len(f_native)}")
    print()
    print(f"energy native         : {e_native:.15f} Ha")
    print(f"energy ASE            : {e_ase:.15f} Ha")
    print(f"|dE|                  : {d_energy:.3e} Ha  (~{e_digits:.1f} digits)"
          f"  {'PASS' if energy_ok else 'FAIL'}")
    print()
    print(f"max |dF| component    : {max_df:.3e} Ha/Bohr (atom {worst})")
    print(f"max |F| native        : {f_scale:.3e} Ha/Bohr")
    print(f"force agreement       : ~{f_digits:.1f} digits"
          f"  {'PASS' if force_ok else 'FAIL'}")
    print(f"  (relative to max |F|: ~{f_digits_rel:.1f} digits, context only)")

    timing = ase.get("timing") or {}
    if timing:
        print()
        print(f"ASE wall total        : {ase.get('wall_total_s', float('nan')):.2f} s")
        # streaming_overhead_s is the number to quote: it is the socket round
        # trip minus the DFT time the C++ driver reports, so it is unaffected by
        # OS scheduling noise (PROJECT_Aurora.md section 6.5).
        for key in ("total_s", "dft_compute_s", "streaming_overhead_s",
                    "ase_overhead_s", "interface_overhead_s"):
            if key in timing:
                print(f"  {key:<24}: {timing[key]:.6f} s")
        if "interface_overhead_pct" in timing:
            print(f"  {'interface_overhead_pct':<24}: "
                  f"{timing['interface_overhead_pct']:.6f} %")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "nodes": ase.get("nodes"), "ranks": ase.get("ranks"),
                "energy_native_ha": float(e_native), "energy_ase_ha": float(e_ase),
                "delta_energy_ha": float(d_energy), "energy_digits": float(e_digits),
                "max_force_delta_ha_bohr": max_df, "force_digits": float(f_digits),
                "force_digits_relative": float(f_digits_rel),
                "max_force_native_ha_bohr": f_scale,
                "energy_pass": energy_ok, "force_pass": force_ok,
                "timing": timing, "wall_total_s": ase.get("wall_total_s"),
            }, fh, indent=2)

    return 0 if (energy_ok and force_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
