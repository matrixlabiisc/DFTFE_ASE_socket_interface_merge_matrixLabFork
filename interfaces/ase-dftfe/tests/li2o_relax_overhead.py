#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Li2O ion-relaxation interface-overhead study (single- vs multi-node).

Drives the dftfe-benchmarks Li2O_ionRelax workflow (GEOOPT / ION, FORCE TOL 5e-4)
with an ASE optimizer over the persistent socket, and measures the *interface*
overhead per relaxation step. This is the regime where the socket earns its keep:
consecutive geometry steps are small displacements, so DFT-FE reuses density +
wavefunctions (no per-step cold start), and the only interface cost is streaming
positions/forces + a little Python.

Per the agreed accounting, the honest per-step DFT cost is CLUBBED: reinit /
mesh move + SCF + force assembly (reported by the C++ driver as compute_time).
The interface overhead is what sits ON TOP of that:
    interface_overhead = ase_overhead (Python: unit conv, request build)
                       + streaming   (serialize + send positions, recv forces)
                       = total_step_wall - dft_compute
The one-time server start (process launch + MPI init + first mesh build) is
reported separately as startup, since it amortizes over the whole relaxation.

Env: OVH_NPROC, OVH_LAUNCHER (mpirun|srun), OVH_LABEL.
"""

import os
import sys
import time

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.optimize import LBFGS
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

BENCH = "/home/pa01/Mehul/dftfe-benchmarks/accuracyBenchmarks/Li2O_fcc/dftfe"
BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
NPROC = int(os.environ.get("OVH_NPROC", "8"))
LAUNCHER = os.environ.get("OVH_LAUNCHER", "mpirun")
LABEL = os.environ.get("OVH_LABEL", f"{NPROC} ranks")
FMAX_HA_BOHR = 5e-4                                   # native FORCE TOL
FMAX_EV_ANG = FMAX_HA_BOHR * Hartree / Bohr           # -> 0.02571 eV/Ang
MAX_STEPS = int(os.environ.get("OVH_STEPS", "8"))     # enough to characterize per-step overhead
# Socket density reuse is displacement-gated: a large geometry step degrades the
# moved-mesh quality past DFT-FE's Jacobian guard (moveAtoms.cc) -> auto-remesh ->
# cold atomic density. Keeping the optimizer step small stays in the mesh-reuse
# ("hot loop") regime -> density reused each step (few SCF iters). This is the
# correct way to drive a socket relaxation and the regime the overhead % describes.
MAXSTEP_ANG = float(os.environ.get("OVH_MAXSTEP", "0.03"))  # LBFGS max step (Ang)

# ── geometry (95-atom Li2O vacancy supercell, from the benchmark) ────────────
cell = np.loadtxt(f"{BENCH}/domainVectors.inp") * Bohr
rows = [ln.split() for ln in open(f"{BENCH}/coordinates.inp") if len(ln.split()) >= 5]
symbols = [chemical_symbols[int(float(r[0]))] for r in rows]
frac = np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])
atoms = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)
print(f"=== Li2O relaxation overhead: {LABEL} (launcher={LAUNCHER}, nproc={NPROC}) ===", flush=True)
print(f"[relax] {len(atoms)} atoms; FORCE TOL {FMAX_HA_BOHR:.0e} Ha/Bohr "
      f"({FMAX_EV_ANG:.4f} eV/Ang); max {MAX_STEPS} steps; LBFGS maxstep {MAXSTEP_ANG} Ang "
      f"(mesh-reuse regime)", flush=True)

# NPKPT controls k-point parallelization. On multi-node the default pooling can
# leave too few ranks per pool for domain decomposition -> per-GPU OOM. npkpt=1
# forces all ranks into one pool sharing the domain -> lowest per-rank memory.
extra = {}
if os.environ.get("OVH_NPKPT"):
    extra["npkpt"] = int(os.environ["OVH_NPKPT"])
    print(f"[relax] NPKPT={extra['npkpt']} (k-point pooling control)", flush=True)
# srun needs explicit per-task GPU binding (--gpus-per-task); without it, node-local
# rank->GPU detection breaks under srun and ranks pile onto one GPU -> OOM. mpirun
# gets it right via OpenMPI local-rank env. Set OVH_GPUS_PER_TASK=1 for multi-node srun.
if os.environ.get("OVH_GPUS_PER_TASK"):
    extra["gpus_per_task"] = int(os.environ["OVH_GPUS_PER_TASK"])
    print(f"[relax] gpus_per_task={extra['gpus_per_task']} (srun GPU binding)", flush=True)
calc = DFTFE(
    bin_dir=BUILD_MERGED, launcher=LAUNCHER, nproc=NPROC, use_device=True,
    psp_path={"Li": f"{BENCH}/Li_ONCV_PBE-1.2.upf", "O": f"{BENCH}/O_ONCV_PBE-1.2.upf"},
    xc="MGGA-R2SCAN", polynomial_order=7, density_quadrature_rule=10,
    mesh_size=1.2, atom_ball_radius=6.0, tolerance=1e-6,
    mixing_scheme="ANDERSON", scf_mixing=0.7, fermi_temp=500.0,
    mp_grid=(2, 2, 2), mp_grid_shift=(1, 1, 1), use_time_reversal_symmetry=True,
    use_single_prec_cheby=True, compute_forces=True,
    debug_timing=True, verbosity=1, log_file="li2o_relax.log", **extra,
)
atoms.calc = calc

steps = []  # per force-eval timing dicts


def grab():
    if calc.timing:
        steps.append(dict(calc.timing))


with calc:
    opt = LBFGS(atoms, maxstep=MAXSTEP_ANG, logfile="li2o_relax_opt.log",
                trajectory="li2o_relax.traj")
    opt.attach(grab, interval=1)
    t0 = time.perf_counter()
    opt.run(fmax=FMAX_EV_ANG, steps=MAX_STEPS)
    wall = time.perf_counter() - t0
    forces_final = atoms.get_forces()
    fmax_final = float(np.abs(forces_final).max()) * Bohr / Hartree
    e_final = atoms.get_potential_energy() / Hartree
    startup = getattr(calc.backend, "startup_s", None)

# ── report ───────────────────────────────────────────────────────────────────
n = len(steps)
converged = fmax_final < FMAX_HA_BOHR  # ASE 3.26 opt.converged() needs a gradient arg; check directly
print(f"\n[relax] converged={converged} in {opt.get_number_of_steps()} steps; "
      f"final fmax={fmax_final:.3e} Ha/Bohr, E={e_final:.8f} Ha", flush=True)
print(f"[relax] one-time startup (launch+MPI init+first mesh build) = "
      f"{startup:.2f}s (amortized over {n} evals)", flush=True)

print("\n step |  total_s | DFT_compute_s | streaming_ms | ase_ms | iface_ms | iface_%", flush=True)
tot_all = dft_all = ifc_all = 0.0
for i, s in enumerate(steps):
    total = s["total_s"]; dft = s.get("dft_compute_s") or 0.0
    strm = (s.get("streaming_overhead_s") or 0.0) * 1e3
    ase_ms = (total - s["round_trip_s"]) * 1e3
    ifc = s["interface_overhead_s"] * 1e3
    pct = s["interface_overhead_pct"]
    tag = " <- first (incl. startup)" if i == 0 else ""
    print(f" {i:4d} | {total:8.3f} | {dft:13.3f} | {strm:12.3f} | {ase_ms:6.2f} | "
          f"{ifc:8.2f} | {pct:6.3f}{tag}", flush=True)
    tot_all += total; dft_all += dft; ifc_all += s["interface_overhead_s"]

# aggregate excluding the first (which folds in one-time startup) for the steady-state number
steady = steps[1:] if n > 1 else steps
ss_tot = sum(s["total_s"] for s in steady)
ss_ifc = sum(s["interface_overhead_s"] for s in steady)
print(f"\n[relax] TOTAL over {n} evals: wall(sum) {tot_all:.2f}s | DFT compute {dft_all:.2f}s | "
      f"interface {ifc_all*1e3:.1f} ms ({ifc_all/tot_all*100:.3f}%)", flush=True)
print(f"[relax] STEADY-STATE (evals 2..{n}): interface {ss_ifc*1e3:.1f} ms of {ss_tot:.1f}s "
      f"= {ss_ifc/ss_tot*100:.4f}% overhead; mean streaming "
      f"{np.mean([(s.get('streaming_overhead_s') or 0)*1e3 for s in steady]):.2f} ms/step", flush=True)
print(f"[relax] {LABEL}: interface overhead is {ss_ifc/ss_tot*100:.4f}% of steady-state relaxation wall", flush=True)
