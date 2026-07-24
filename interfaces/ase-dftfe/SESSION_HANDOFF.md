# ASE-DFTFE — Session Handoff (resume context)

**Date:** 2026-07-23. **Owner:** Mehul Darak (mehuldarak@iisc.ac.in). **Machine:** MATRIX (IISc).
**Repo:** `/home/pa01/Mehul/DFTFE`, branch **`ase_redesign`**. Interface: `interfaces/ase-dftfe/`.
Also see auto-memory: `~/.claude/projects/-home-pa01/memory/` (ase-dftfe-architecture.md,
ase-dftfe-param-accuracy.md, ase-dftfe-production-targets.md, ase-dftfe-git-workflow.md) and
`interfaces/ase-dftfe/ONBOARDING.md`.

## Goal
Make the DFT-FE ASE interface production-ready + deployable on any DFT-FE cluster (ALCF Polaris/Aurora,
OLCF Frontier, NSM Pravega/Siddhi). Validate accuracy + overhead on MATRIX, then scale on Aurora.

---

## STATUS OF DELIVERABLES

| Item | Status | Result |
|---|---|---|
| Li2O SCF energy | ✅ done | bit-for-bit vs same-binary native: ΔE 3e-13 Ha |
| Li2O SCF forces | ✅ done | sum\|F\| Δ 4e-11 Ha/Bohr; max\|F\| 6.47e-3 Ha/Bohr |
| BCC Mo GS (E+F+stress) | ✅ done | ASE vs native-GPU bit-for-bit (E~1e-9, F~2.5e-9, stress mag~1e-9) |
| Li2O relax overhead 1-node | ✅ done | interface ~2 ms/step (0.00%); streaming 1.8 ms; reuse 19→3 iters |
| Li2O relax overhead 2-node | ✅ done | interface ~3.3 ms/step (0.00%); streaming 3.3 ms; DFT 328→190 s (1.7×) |
| **Li2O NEB** | ⏳ IN PROGRESS | native current-build NEB submitted (job 29712, see below) |
| Aurora scaling | ⏸ deferred | after MATRIX NEB closes (task #4) |

### CURRENTLY RUNNING (check first on resume)
- **Job 29712** = `li2o_neb_native` — faithful **native DFT-FE NEB, current build, single-node (8 GPU,
  mpirun)**, reproducing `dftfe-benchmarks/.../NEB/Li2O/dftfe/Input_NEB.prm` exactly (7 images, GGA-PBE,
  ball 10, tol 1e-7, path threshold 4e-4, LBFGS). Real binary (gamma). **RUNNING on cn2 at handoff** (past "NUMBER OF IMAGES = 7", still in mesh setup — verify it reaches `Total number of MPI tasks: 8` and does NOT hit CUDA OOM).
  - Work dir: `interfaces/ase-dftfe/tests/li2o_neb_native/`; output `neb_native.out`, slurm `slurm_nebnat_*.out`.
  - Check: `squeue -j 29712`; `grep -iE "activation|barrier|Total number of MPI tasks|out of memory|NEB.*iteration" li2o_neb_native/neb_native.out`
  - **Watch for**: (a) `Total number of MPI tasks: 8` (good) not `1`; (b) CUDA OOM (7 resident images on 8
    GPU may be too much — if OOM, native NEB may need >1 node, but user said SINGLE-NODE ONLY for now).
  - GOAL: get the **current-build activation barrier** to compare vs native-old-build **241.03 meV** / QE
    **244.9 meV**. Then set up the ASE-interface NEB to match it.

---

## USER DIRECTIVES (must follow)
- **NEB must faithfully mirror the benchmark** (same params/pseudos/initial path, proper convergence) to
  get a barrier close to 241.03 meV (DFT-FE) / 244.9 meV (QE). Do NOT trust earlier NEB numbers (see below).
- **SINGLE NODE ONLY** right now — other users are active on the 2-node debug partition.
- Plan: **native current-build NEB first** (job 29712) → get real numbers → **then set up the ASE run**.
- Never waste compute on knowingly-wrong setups (killed a bad shared-process NEB earlier for this reason).
- Workflow rules: work on `ase_redesign` only; never push to pGD; commit only interface files; `rm`/destructive
  git needs approval; verify via SLURM (`sbatch`), never login-node compute; commit trailer `Co-author: Claude Opus 4.8`.

---

## KEY CROSS-CUTTING FINDINGS (this session)

1. **Current DFT-FE ≠ published benchmarks by ~1e-4/atom** (both R2SCAN and GGA-PBE). Cause: the 2025→2026
   **XC-subsystem reimplementation** (`src/excManager`, +1953 lines incl. `excTauMGGAClass.cpp` +546).
   - Li2O SCF: current build −956.5888847 Ha vs published −956.5976489 (old commit `2a419ef`, DFT-FE 1.1.0-pre).
   - Verified via git log between commits `2a419ef` (publicGithubDevelop, 2025-08-08) and `faa31dd`
     (socket_interface_merge, 2026-05-22). Ruled out the libxc toggle (on/off agree to 1.5e-12 in current build).
   - **Correct validation = ASE-socket vs same-binary native (bit-for-bit). All pass.** Matching the published
     numbers exactly needs the OLD build. NEB barrier will likely differ from 241 by this XC effect.

2. **Stress sign**: `dftfeWrapper::getCellStress()` deliberately negates the internal tensor
   (`src/dftfeWrapper.cc:1294`). So ASE stress = −(native .out "Cell stress"). Magnitude exact (~1e-9).
   TODO: confirm the negated sign is ASE's σ=(1/V)∂E/∂ε convention (spot-check via a cell relax direction).

3. **MGGA-R2SCAN**: supported (N2 = −20.7514 Ha). LIMITATION: MGGA + ION FORCE + **NLCC** pseudos is
   unimplemented in DFT-FE (`AssertThrow` at `src/dft/dft.cc:969`). Li2O ONCV pseudos are NLCC-free
   (`core_correction="F"`) so forces work; my early N2 test used `N.upf` which HAS NLCC → that assertion.

4. **Multi-node MUST use mpirun, NOT srun.** This OpenMPI (5.0.6) build isn't SLURM-PMI-integrated, so
   `srun --ntasks N <dftfe>` launches N singleton `MPI_COMM_WORLD=1` jobs → "Total number of MPI tasks: 1",
   each holds the FULL problem → CUDA OOM (this wrecked the 2-node relax twice). `mpirun -np N` inside the
   SLURM allocation spans nodes correctly → "MPI tasks: N", domain decomposed. `SrunLauncher`
   (`src/dftfe_ase/launchers/slurm.py`) also only adds `--gpus-per-task` when set. **Launcher robustness
   TODO**: default multi-node to mpirun (or fix srun PMI) so it "just works".

5. **NEB density reuse — root cause of failure diagnosed** (see architecture memory). Native DFT-FE NEB
   (`src/neb/nudgedElasticBandClass.cc`) keeps ONE solver PER image (`std::vector<dftfeWrapper>`,
   `d_dftfeWrapper[image]`) — each reuses its own density across NEB steps. Our shared-process ASE NEB made
   one process teleport across images; the socket computes displacement WITHOUT minimum-image convention
   (`src/socket_interface.cc:796`), so a wrapped atom gave **Max disp = 17.41 Bohr = full cell** → useless
   density guess → ~24 cold-level iters/eval. Fix = per-image processes (+ minimum-image displacement).
   - `Reading initial guess for electron-density` is DFT-FE's split-density reuse (ρ_atomic + Δρ), NOT a cold
     restart — do NOT use it as a reuse indicator. Real reuse metric = **SCF iters/step**.

---

## NEB SPECIFICS (the active task)
- Benchmark dir: `/home/pa01/Mehul/dftfe-benchmarks/accuracyBenchmarks/NEB/Li2O/`
  - `dftfe/Input_NEB.prm` (7 img, GGA-PBE, poly7, mesh1.2, ball10, tol1e-7, ANDERSON0.2, 220 bands, LBFGS,
    spring 0.1, path threshold 4e-4), `dftfe/coordinates.inp` (= INITIAL guess, 7×95 atoms),
    `dftfe/output/FinalPath.txt` (= CONVERGED MEP: 95 atoms × 21 cols = 7 img × xyz), pseudos `Li.upf`(NLCC-free)
    `O.upf`(NLCC "T"). Gamma point → **real binary**.
  - Reference: **activation energy 241.03 meV (DFT-FE, old build) / 244.9 meV (QE)**.
- Earlier (UNTRUSTED) numbers: 264 meV was a single-point re-eval of FinalPath with the current XC (NOT a
  re-optimized NEB → overestimates saddle). 3-image per-image test converged in 0 steps (middle image sits
  at saddle). Neither is a faithful barrier.
- **Per-image ASE NEB driver EXISTS + architecture validated**: `tests/neb_run_perimage.py` (one persistent
  socket process per image = native's design; `allow_shared_calculator=False`; reports barrier every step;
  env NEB_NIMAGES_TEST/NEB_NPROC_PER_IMAGE/NEB_GPUS_PER_IMAGE/NEB_STEPS/NEB_CLIMB). Launched fine (3 procs,
  GPU-pinned, sockets connected, barrier computed). CAVEAT: per-image MEMORY for the ball-10 mesh may need
  ≥2 GPU/image → hard to fit 7 images on 1 node; may be why native NEB is the practical path for this system.
- **Next after 29712**: (a) read the current-build barrier; (b) set up the ASE-interface NEB
  (`neb_run_perimage.py`, full 7 images, from a NON-converged path so the optimizer actually steps and
  reuse is exercised) and show it matches native. Need to resolve per-image GPU-memory (check GPU mem;
  nvidia-smi not on login node — query via a compute node or DFT-FE device log).

---

## FILES CREATED/EDITED THIS SESSION (all under interfaces/ase-dftfe/)
Tests (new): `tests/mgga_check.py`, `li2o_scf_accuracy.py` (+ `run_li2o_scf.slurm`), `bcc_mo_accuracy.py`
(+ `run_bcc_mo.slurm`, `run_bcc_mo_native.slurm`, dir `bcc_mo_native/`), `neb_profile.py`, `neb_finalpath.py`
(+ `run_neb_profile.slurm`, `run_neb_finalpath.slurm`), `neb_run.py` (+ `run_neb_run.slurm` — the OLD shared-
process NEB, superseded), `neb_run_perimage.py` (+ `run_neb_perimage.slurm` — the CORRECT per-image driver),
`li2o_relax_overhead.py` (+ `run_relax_ovh_1node.slurm`, `run_relax_ovh_2node.slurm`), `run_neb_native.slurm`
(+ dir `li2o_neb_native/` — the ACTIVE native NEB).
Edits: `li2o_relax_overhead.py` (added OVH_MAXSTEP small-step reuse regime, OVH_NPKPT, OVH_GPUS_PER_TASK,
fixed ASE-3.26 `opt.converged()` bug); `neb_run_perimage.py` (same converged() fix); ONBOARDING.md + memory.
Note the ASE-3.26 gotcha: `opt.converged()` needs a gradient arg — check fmax directly instead.

## HOW TO RUN THINGS (MATRIX)
SLURM header: `--gres=gpu:8 --ntasks-per-node=8`; modules `spack` + `openmpi/5.0.6-gcc-13.3.0-ytficip
nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323`; `export
LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"`; `OMP_NUM_THREADS=1
DEAL_II_NUM_THREADS=1 DFTFE_NUM_THREADS=1`; `source ~/.venvs/ase-env/bin/activate`. Binaries:
`build_merged/release/{real,complex}/dftfe` (real=gamma, complex=k-points). Debug partition = cn1,cn2 (8 GPU,
48 core, 579 GB each), shared with other users. Multi-node → **mpirun**.

## PARAM SYSTEM (context)
`src/dftfe_ase/params.py` = single source of truth (kwarg ↔ .prm). Two paths: A = `prm_file=`/`extra_prm={}`
passthrough; B = pure-Python kwargs. Generic C++ injector (`dftfeWrapper.setSocketPrmOverrides` +
delete+append loop) ended the silent-drop bug class. 21 params two-value A/B verified
(`tests/verify_param_injection.py`). Regenerate mapping: `python -m dftfe_ase.params` → `dev_help/parameter_mapping.md`.
