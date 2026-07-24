#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Li2O NEB via ase.mep.NEB with ONE persistent DFT-FE process PER IMAGE.

This mirrors native DFT-FE NEB (`src/neb/nudgedElasticBandClass.cc`), which holds
a `std::vector<dftfeWrapper>` — one solver per image, each keeping its OWN
mesh/density/wavefunctions and only ever seeing that image's small optimizer-step
displacement. So each image reuses its own density across NEB steps (warm SCF),
unlike a single shared process teleporting across distant images (which decorrelates
the density and, with raw non-minimum-image displacements, produces spurious
cell-sized moves -> no reuse).

Each image gets its own DFTFE calculator => its own persistent socket process
(distinct auto-assigned port, own cwd/log). All processes share the full node's
GPUs (NPROC_PER_IMAGE ranks each) and are evaluated SERIALLY by ase.mep.NEB
(allow_shared_calculator=False) — the same all-GPU/sequential/per-image-memory
profile native already proves fits. Optionally pin each image to a GPU subset via
NEB_GPUS_PER_IMAGE (sets CUDA_VISIBLE_DEVICES per image) to cap resident memory.

Barrier is reported after EVERY FIRE step, so a partial run still yields the trend.

Env: NEB_NPROC_PER_IMAGE (ranks/image, default 8), NEB_GPUS_PER_IMAGE (0=share all),
NEB_STEPS (FIRE cap), NEB_CLIMB (1/0), NEB_NIMAGES_TEST (subset for a mechanics check).
"""

import contextlib
import os
import sys

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.mep import NEB
from ase.optimize import FIRE, LBFGS
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dftfe_ase import DFTFE  # noqa: E402

NEBDIR = "/home/pa01/Mehul/dftfe-benchmarks/accuracyBenchmarks/NEB/Li2O"
BENCH = f"{NEBDIR}/dftfe"
BUILD_MERGED = "/home/pa01/Mehul/DFTFE/build_merged"
WORK = os.path.join(HERE, "neb_perimage_work")
NAT = 95
REF_BARRIER_MEV = 241.03
FMAX_HA_BOHR = 4e-4
FMAX_EV_ANG = FMAX_HA_BOHR * Hartree / Bohr

NIMAGES = int(os.environ.get("NEB_NIMAGES_TEST", "7"))    # 7 = full; smaller = mechanics test
NPROC_PER_IMAGE = int(os.environ.get("NEB_NPROC_PER_IMAGE", "8"))
GPUS_PER_IMAGE = int(os.environ.get("NEB_GPUS_PER_IMAGE", "0"))  # 0 => share all GPUs
STEPS = int(os.environ.get("NEB_STEPS", "20"))
CLIMB = os.environ.get("NEB_CLIMB", "1") == "1"
# NEB_INIT: 'initial' (default) starts from coordinates.inp -- EXACTLY what native DFT-FE NEB
# reads -- so the optimizer actually steps and per-image density reuse is exercised; the barrier
# is only trustworthy from a re-optimized path. 'final' starts from the converged FinalPath.txt
# (0-step sanity check only -- overestimates the saddle under the current XC; do NOT trust it).
NEB_INIT = os.environ.get("NEB_INIT", "initial")
# NEB_OPT: 'lbfgs' (default) mirrors native's `NEB OPT SOLVER = LBFGS` (history 5, max step 0.5 Bohr);
# 'fire' is a more robust fallback for a rough initial guess.
NEB_OPT = os.environ.get("NEB_OPT", "lbfgs").lower()

cell = np.loadtxt(f"{BENCH}/domainVectors.inp") * Bohr
# coordinates.inp = initial guess: full*NAT rows, image-major, cols [Z, Zval, fx, fy, fz]
raw = np.loadtxt(f"{BENCH}/coordinates.inp")
full = raw.shape[0] // NAT
symbols = [chemical_symbols[int(z)] for z in raw[:NAT, 0]]
init_frac = raw[:, 2:5].reshape(full, NAT, 3)                          # (full, NAT, 3), image-major
fp = np.loadtxt(f"{BENCH}/output/FinalPath.txt")                       # converged MEP: NAT x (full*xyz)
final_frac = np.stack([fp[:, 3 * k:3 * k + 3] for k in range(full)])  # (full, NAT, 3)
src = final_frac if NEB_INIT == "final" else init_frac
# pick evenly spaced image indices if testing a subset (always keep both endpoints)
idx = list(range(full)) if NIMAGES >= full else \
    sorted(set([0] + list(np.linspace(0, full - 1, NIMAGES).round().astype(int)) + [full - 1]))
images = [Atoms(symbols=symbols, scaled_positions=src[k], cell=cell, pbc=True) for k in idx]
n = len(images)
print(f"[neb-pi] {n} images (idx {idx}), {NAT} atoms; init={NEB_INIT} opt={NEB_OPT}; "
      f"{NPROC_PER_IMAGE} ranks/image, GPUS_PER_IMAGE={GPUS_PER_IMAGE or 'share-all'}; "
      f"fmax {FMAX_HA_BOHR:.0e} Ha/Bohr", flush=True)


def make_calc(i):
    env = None
    if GPUS_PER_IMAGE > 0:  # pin this image to a disjoint GPU subset
        gpus = [str((i * GPUS_PER_IMAGE + g) % 8) for g in range(GPUS_PER_IMAGE)]
        env = {"CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    d = os.path.join(WORK, f"img{i}")
    os.makedirs(d, exist_ok=True)
    return DFTFE(
        bin_dir=BUILD_MERGED, launcher="mpirun", nproc=NPROC_PER_IMAGE, use_device=True,
        port=0, cwd=d, env=env, log_file=f"neb_img{i}.log",
        psp_path={"Li": f"{NEBDIR}/Li.upf", "O": f"{NEBDIR}/O.upf"},
        xc="GGA-PBE", polynomial_order=7, mesh_size=1.2, atom_ball_radius=10.0,
        tolerance=1e-7, mixing_scheme="ANDERSON", scf_mixing=0.2, fermi_temp=500.0,
        num_bands=220, compute_forces=True, verbosity=1,
    )


with contextlib.ExitStack() as stack:
    calcs = [stack.enter_context(make_calc(i)) for i in range(n)]
    for img, c in zip(images, calcs):
        img.calc = c
    neb = NEB(images, climb=CLIMB, k=0.1, allow_shared_calculator=False)
    if NEB_OPT == "fire":
        opt = FIRE(neb, logfile="neb_pi_opt.log", trajectory="neb_pi.traj")
    else:  # LBFGS mirrors native: LBFGS HISTORY=5, MAXIMUM ION UPDATE STEP=0.5 Bohr
        opt = LBFGS(neb, logfile="neb_pi_opt.log", trajectory="neb_pi.traj",
                    memory=5, maxstep=0.5 * Bohr)

    def report_barrier():
        E = np.array([im.get_potential_energy() for im in images]) / Hartree
        bar = (E.max() - E[0]) * Hartree * 1000.0
        prof = [round((e - E[0]) * Hartree * 1000, 1) for e in E]
        print(f"[neb-pi] step {opt.get_number_of_steps():2d}: barrier {bar:7.2f} meV  profile(meV)={prof}", flush=True)

    opt.attach(report_barrier, interval=1)
    print(f"[neb-pi] climb={CLIMB}, step cap={STEPS}; starting FIRE...", flush=True)
    opt.run(fmax=FMAX_EV_ANG, steps=STEPS)
    # ASE 3.26 opt.converged() needs a gradient arg; check the band fmax directly
    band_fmax = float(np.abs(neb.get_forces()).max())
    converged = band_fmax < FMAX_EV_ANG
    E = np.array([im.get_potential_energy() for im in images]) / Hartree

barrier = (E.max() - E[0]) * Hartree * 1000.0
d = abs(barrier - REF_BARRIER_MEV)
print(f"[neb-pi] FINAL barrier = {barrier:.2f} meV (converged={converged}); "
      f"native old-build NEB {REF_BARRIER_MEV} meV / QE 244.9 meV; |diff|={d:.2f} meV", flush=True)
