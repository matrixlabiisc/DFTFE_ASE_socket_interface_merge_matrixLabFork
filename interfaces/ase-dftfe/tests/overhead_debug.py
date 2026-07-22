#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Proper interface-overhead measurement, via debug_timing over an MD workflow.

Runs a short ASE MD (ASE does everything: Maxwell-Boltzmann velocities +
VelocityVerlet) on a demo system, with debug_timing ON. Every step therefore
reports the full per-step breakdown:

    dft_compute | streaming (serialize+send positions, recv forces, MPI bcast)
              | ase_overhead (Python unit-conv / request build) | interface %

The point of using a multi-step MD (not a single point) is exactly the concern
that matters: the streaming + Python cost RECURS every step, so the true
workflow overhead = mean over steady-state steps of
    interface_overhead / total_step_time.

Uses the fixed build_merged binary (has the C++ compute_time field). nproc from
env (single-node vs multi-node). Demo (ex1/ex2/ex3) as arg.
"""

import os
import sys

import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402
from parity_check import COMMON, DEMOS, PSP_DIR, atoms_ex1, atoms_ex2, atoms_ex3  # noqa: E402

BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
BUILDERS = {"ex1": atoms_ex1, "ex2": atoms_ex2, "ex3": atoms_ex3}
NSTEPS = int(os.environ.get("OVH_NSTEPS", "8"))
DT_FS = float(os.environ.get("OVH_DT_FS", "1.0"))
T0 = float(os.environ.get("OVH_T0", "300.0"))


def main():
    demo = sys.argv[1] if len(sys.argv) > 1 else "ex1"
    cfg = DEMOS[demo]
    nproc = int(os.environ.get("OVH_NPROC", cfg["nproc"]))
    label = os.environ.get("OVH_LABEL", f"{nproc} ranks")
    print(f"=== interface overhead via debug_timing: {demo}, {label}, "
          f"{NSTEPS}-step MD, GPU ===", flush=True)

    atoms = BUILDERS[demo]()
    # MD needs forces; drop stress; keep mesh/k-point params from the demo.
    params = {k: v for k, v in cfg["params"].items()
              if k not in ("compute_forces", "compute_stress")}
    calc = DFTFE(bin_dir=BUILD_MERGED, launcher="mpirun", nproc=nproc,
                 use_device=True, psp_path=PSP_DIR, env={"DFTFE_PSP_PATH": PSP_DIR},
                 compute_forces=True, debug_timing=True, verbosity=1,
                 log_file="overhead_debug.log", **COMMON, **params)

    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0)
    dyn = VelocityVerlet(atoms, timestep=DT_FS * units.fs)

    records = []
    for _ in range(NSTEPS):
        dyn.run(1)
        if calc.timing:
            records.append(dict(calc.timing))
    startup = getattr(calc.backend, "startup_s", None)
    calc.close()

    def _sum(key):
        return sum(r[key] for r in records if r.get(key) is not None)

    # ── FULL-RUN clubbed accounting ─────────────────────────────────────────
    # Everything DFT-FE natively does, clubbed = startup (process launch + MPI
    # init + connect) + all per-step compute (step 0's compute already includes
    # reinit + file read + mesh build). Interface overhead = all streaming + ASE.
    sum_compute = _sum("dft_compute_s")
    sum_stream = _sum("streaming_overhead_s")
    sum_ase = _sum("ase_overhead_s")
    startup_s = startup or 0.0
    dftfe_work = startup_s + sum_compute
    interface = sum_stream + sum_ase
    grand = dftfe_work + interface
    print("\n================ FULL-RUN overhead (clubbed, honest total) ================")
    print(f"  steps run                : {len(records)}")
    print(f"  startup (launch+MPI init+connect): {startup_s:.3f} s  (one-time)")
    print(f"  DFT-FE native work total : {dftfe_work:.3f} s  (startup + reinit + mesh + all SCF+forces)")
    print(f"  interface overhead total : {interface * 1e3:.1f} ms "
          f"(streaming {sum_stream * 1e3:.1f} ms + ASE {sum_ase * 1e3:.1f} ms)")
    if grand > 0:
        print(f"  => INTERFACE OVERHEAD    : {100 * interface / grand:.4f} % of {grand:.2f} s total wall")
    print("=" * 74)

    # ── steady-state per-step (excludes the one-time startup/reinit) ─────────
    steady = records[1:] if len(records) > 1 else records

    def mean(key):
        vals = [r[key] for r in steady if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("--- steady-state per step (steps 2..N, startup amortized) ---")
    print(f"  mean DFT compute       : {mean('dft_compute_s'):.4f} s")
    print(f"  mean streaming         : {mean('streaming_overhead_s') * 1e3:.2f} ms")
    print(f"  mean ASE python        : {mean('ase_overhead_s') * 1e3:.2f} ms")
    print(f"  mean interface OVERHEAD: {mean('interface_overhead_pct'):.3f} %")
    print("=" * 74)


if __name__ == "__main__":
    main()
