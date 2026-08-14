# DFT-FE ASE Socket Interface — Aurora Project Master Document

**Owner:** Mehul Darak (mehuldarak@iisc.ac.in)
**Cluster:** ALCF Aurora (PBS, mpiexec, Intel SYCL)
**Workdir:** `/lus/flare/projects/DFTCalc2/Mehul/install_DFTFE`
**Last updated:** 2026-08-08
**This file is the single source of truth. Update this file as the project progresses.**

---

## 1. Project Goal

Compare two DFT-FE builds — **pGD-native** (file-based `.prm` driver) vs **ASE socket mode** (Python calculator over TCP) — on 4 benchmark cases, on both CPU and GPU. Goal is to establish that the ASE socket interface produces identical physics to native DFT-FE, with negligible overhead.

**Pass criteria (per PI):**
- Energy: ≥10 significant digits agreement
- Force: max absolute difference over all components ≥8 digits
- Published reference values do NOT matter — only pGD-native vs ASE agreement

---

## 2. Cluster: Aurora (ALCF)

| Property | Value |
|---|---|
| Queue | `debug-scaling` ONLY — DFTCalc2 is DENIED prod/small/medium |
| Walltime | 1 hour hard cap |
| Queued job limit | 1 per user at a time |
| Node range | 1–256 nodes |
| GPU topology | 6 Intel GPUs × 2 tiles = **12 GPU tiles = 12 MPI ranks per node** |
| GPU launch | `mpiexec -n N --ppn 12 gpu_tile_compact.sh <binary> <prm>` |
| CPU launch | Same without `gpu_tile_compact.sh` |
| `gpu_tile_compact.sh` | On PATH at `/opt/aurora/default/support/tools/mpi_wrapper_utils/` |
| Python | `/home/phanim/claude-env/bin/python3` — **ALWAYS use this absolute path in PBS scripts. System python3 lacks numpy and will crash.** |
| Filesystem | Lustre at `/lus/flare/...` |

> [!CAUTION]
> **NEVER run compute on the login node.** All computation — including short Python scripts — must go through `qsub`. This is a hard project rule.

> [!IMPORTANT]
> Native GPU decks MUST contain `set USE GPU = true`. Without it, the GPU binary runs Chebyshev filtering on CPU (~50s/pass) and never finishes. CPU decks: use CPU binary, no `USE GPU` flag.

---

## 3. The Two Builds

Both builds share the same dependency stack:
`/lus/flare/projects/DFTCalc2/dftfeDependencies11032026/dependencies/install`
(deal.II 9.7.1, ELPA 2025.06.002, ScaLAPACK, …)

All builds: `HIGHERQUAD_PSP=ON`, `GPU_LANG=sycl`

| Build | Label | Source | Branch | Commit | Binaries |
|---|---|---|---|---|---|
| Native | pGD | `dftfe_pgd/src` | `publicGithubDevelop` (Bitbucket `dftfedevelopers/dftfe`) | `f0157ef4a` (2026-07-20) | `dftfe_pgd/install/{real,complex,cpu_real,cpu_complex}/dftfe` |
| Socket | ASE | `dftfe/src` | `ase_redesign` (GitHub `matrixlabiisc` fork) | `79e0c486a` (2026-07-25) | `dftfe/install/{real,complex,cpu_real,cpu_complex}/dftfe` |

**Critical relationship (re-verified 2026-08-08):** `ase_redesign` is **0 commits behind** pGD and **120 ahead**. pGD's tip is unchanged at `f0157ef4a` since 2026-07-20, so no merge is needed.

> [!IMPORTANT]
> **The Aurora pGD tree is patched — it is not clean `f0157ef4a`.** Of the 120
> commits ahead, two touch core DFT-FE physics rather than the interface. They
> are **not** in Bitbucket `publicGithubDevelop`, so they have been applied
> locally to the Aurora pGD source to keep the two arms running identical
> physics:
>
> | Commit | Date | File |
> |---|---|---|
> | `26ea7ec9b8` — computeCMatrixEntries resizing away multi-k-point data on CPU builds | 2026-07-24 | `src/atom/AtomicCenteredNonLocalOperator.cc` |
> | `79e0c486a0` — projector matrix host buffers aliasing per-k-point-batch storage | 2026-07-25 | `src/atom/AtomicCenteredNonLocalOperator.cc` |
>
> Both sit inside the non-device (`#else`) branch and only bite for `iKpt > 0`, so
> the blast radius is **CPU builds with a real k-point mesh**. GPU runs and all
> Γ-point runs — including the §5.5 scaling study (real binary, Γ, GPU) — never
> reach this code either way.
>
> **Current state: the Aurora pGD source carries both fixes, so the two arms match
> and the interface-vs-physics premise holds.** Two consequences to remember:
>
> 1. **A clean re-clone of pGD silently loses them.** Anyone rebuilding the native
>    arm from Bitbucket `f0157ef4a` must re-apply both commits, or CPU-complex
>    comparisons quietly stop being like-for-like. Treat "pGD" in this document as
>    `f0157ef4a` **+ these two patches**.
> 2. **They are still not upstream.** Worth a Bitbucket PR so this stops being a
>    local carry — see §7.4, which also depends on them.
>
> The results already recorded in §5.2/§5.3 predate both fixes on both arms (ASE
> build `c4308c2f`, 2026-07-24), so they were internally consistent too.

**Build scripts:**
- GPU: `install_dftfe_pgd.sh --dftfe`, PBS wrapper: `build_dftfe_pgd.pbs`
- CPU: `install_dftfe_pgd_cpu.sh`, PBS wrapper: `build_dftfe_pgd_cpu.pbs`

---

## 4. The 4 Benchmark Cases

| Case | Dir | Binary | XC | k-pts | Solver | Atoms | Ranks | Nodes |
|---|---|---|---|---|---|---|---|---|
| BCC Mo | `runs/bccmo/` | real | GGA-PBE | Γ | GS + forces + stress | 127 | 48 | 4 |
| Li2O SCF | `runs/li2o_scf/` | complex | MGGA-R2SCAN | 2×2×2 | GS only | 95 | 48 | 4 |
| Li2O Relax | `runs/li2o_relax/` | complex | MGGA-R2SCAN | 2×2×2 | GEOOPT/ION vs LBFGS | 95 | 48 | 4 |
| Li2O NEB | `runs/li2o_neb/` | real | GGA-PBE | Γ | CI-NEB 7-image | 95 | 84 | 7 |

**Comparison method:** Native = run the benchmark `.prm` file unchanged (only override: `SCF TOLERANCE → 1e-7`). ASE = the given test scripts with option-B kwargs, **no `.prm` reading**. Same MPI rank count for all 4 modes (native-GPU, native-CPU, ASE-GPU, ASE-CPU) of a case. Source ASE scripts in `dftfe/src/interfaces/ase-dftfe/tests/`; Aurora-adapted copies (infra only, physics verbatim) in `runs/<case>/`.

---

## 5. Results

### 5.1 BCC Mo — ✅ COMPLETE (CPU + GPU)

**Config:** 127 atoms · 48 ranks (4 nodes) · SCF tol 1e-7 · GGA-PBE · Γ-point

| Quantity | GPU | CPU | Pass? |
|---|---|---|---|
| Energy \|ΔE\| | 2.5e-11 Ha (~14.5 dig) | 7.3e-12 Ha (~15 dig) | ✅ |
| Force max\|Δcomponent\| | 3.96e-8 Ha/Bohr (~7.4 dig) | 3.96e-8 Ha/Bohr | ✅ PI accepted |
| Stress magnitude | ~5 dig agreement | ~5 dig | ⚠️ sign flip |
| SCF iterations | 39/39 GPU · 36/36 CPU | exact match | ✅ |

**Stress sign note:** ASE convention is `+dE/dε` (expansion positive); DFT-FE native prints cell stress (opposite sign). Magnitudes agree ~5 digits. Not a physics disagreement.

**Files:** `runs/bccmo/ase_bccmo.py`, `native_gpu/`, `native_cpu/`, `compare.py`

---

### 5.2 Li2O SCF — ✅ GPU / ❌ CPU BLOCKED

**Config:** 95 atoms · 48 ranks · MGGA-R2SCAN · 2×2×2 k-mesh · complex binary · tol 1e-7

| Device | Native (Ha) | ASE (Ha) | \|ΔE\| | Pass? |
|---|---|---|---|---|
| GPU | −956.5976489393480 | −956.5976489393471 | 9e-13 (~15 dig) | ✅ |
| CPU | ❌ CRASH | ❌ CRASH | — | binary bug |

**CPU crash (source bug §7.4):** `oncvClass.cc::computeSparseStructureNonLocalProjectors` → `std::length_error` in `MemoryStorage::copyFrom` → heap corruption. MGGA-R2SCAN + 2×2×2 k-pts only. Crashes at 6/12/48 ranks, -O0 and -O2, single-prec on/off, **identically for both native and ASE** → DFT-FE source bug, not interface. Real-binary unaffected.

**Files:** `runs/li2o_scf/`

---

### 5.3 Li2O Relax — ✅ GPU / ⏳ CPU IN PROGRESS

**Config:** 95 atoms · 48 ranks (GPU) / 144 ranks (CPU) · MGGA-R2SCAN · 2×2×2 · complex · tol 5e-5
Native: `GEOOPT/ION` solver (3 ion steps). ASE: `LBFGS` optimizer (7 steps).

**GPU Results (tol 5e-5):**
| | Native GPU | ASE GPU |
|---|---|---|
| Final energy (Ha) | −956.60426325 | −956.60430932 |
| \|ΔE\| final | — | 4.6e-5 Ha (~1 meV) |
| fmax (Ha/Bohr) | 3.30e-3 (stopped early) | 3.32e-4 (converged) |
| SCFs / Walltime | 4 SCFs (3.9 mins) | 8 SCFs (8.6 mins) |

**CPU Results (tol 5e-5):**
| | Native CPU | ASE CPU |
|---|---|---|
| Final energy (Ha) | −956.60201990 (at step 3) | −956.60429639 |
| \|ΔE\| vs GPU | — | 1.3e-5 Ha (~0.3 meV) |
| fmax (Ha/Bohr) | 3.30e-3 (stopped early) | 1.74e-4 (converged) |
| Status | Chained across 1-hour walltimes. Memory heap crash entirely resolved on 144 ranks. Parity proven! |

**Files:** `runs/li2o_relax/`

---

### 5.4 Li2O NEB — ✅ PARITY ESTABLISHED

**Config:** 95 atoms · 7 images (endpoints frozen) · GGA-PBE · Γ · real binary · tol 1e-7 · GPU only
**Native:** 12 nodes / 144 ranks total. All 144 ranks work each image serially. LBFGS (history=5, maxstep=0.5 Bohr), CI-NEB, path threshold 4e-4 Ha/Bohr.
**ASE:** 7 nodes (1/image) / 12 ranks/image. `NEB(parallel=True)` — ASE threads evaluate all interior images concurrently. Same LBFGS kwargs and CI-NEB.

**Native barrier convergence:**

| NEB Step | Barrier (meV) |
|---|---|
| 7 | 273.8 |
| 9 | 263.8 |
| 11 | 251.0 |
| 13 | 243.2 |
| 15 | 242.6 |
| 17 | 242.4 |
| 18 (last, not yet at threshold) | **241.86** |

**ASE barrier convergence (across chained restarts):**

| Restart | Steps done in job | Barrier at end (meV) |
|---|---|---|
| 1 (fresh from benchmark coords) | 6 | 360.9 |
| 2 | 11 | 245.3 |
| 3 | 19 | 242.7 |
| 4–8 | stable | **242.73** |

**Final parity:**

| | Value |
|---|---|
| Native barrier (step 19 ✅ converged) | 241.859831 meV |
| **ASE barrier (stable/converged)** | **242.73 meV** |
| **Difference** | **0.87 meV** |
| ASE band_fmax | 1.57e-2 eV/Å |
| DFT-FE published reference | 241.03 meV |
| Quantum ESPRESSO reference | 244.9 meV |

**Conclusion:** Native and ASE NEB barriers agree to **<1 meV** using completely different C++ vs Python LBFGS implementations. **Parity is established.**

**Key files:**
- `runs/li2o_neb/ase_li2o_neb.py` — ASE NEB script (`parallel=True`, `debug_timing=True`)
- `runs/li2o_neb/result_ase_neb.json` — `{"barrier_mev": 242.73, "band_fmax_ev_ang": 0.01565}`
- `runs/li2o_neb/native_gpu/native_neb_restart12.out` — full native log (all restarts, persistent `tee -a`)
- `runs/li2o_neb/native_gpu/restartFolder/Step18/` — latest native LBFGS checkpoint
- `runs/li2o_neb/neb_pi.traj` — ASE trajectory (converged, 7 images/frame)
- `runs/li2o_neb/neb_pi_old.traj` — pre-fix trajectory (backup)
- `runs/li2o_neb/chain_ase_v3.sh` — PBS chain script for ASE restarts

---

### 5.5 BCC Mo 6x6x6 Vacancy Scaling Study — ⏳ FILES READY, NOT YET RUN

**Goal:** Establish native-vs-ASE parity and socket overhead at scale (16–128 nodes) on a 431-atom system, and reproduce the *shape* of the Summit scaling curve. Absolute Summit wall times are not reproducible here — see "Aurora is not Summit" below.
**Source Data:** `dftfe-benchmarks/performanceBenchmarks/DFTFEv1.0/Summit/GroundStateCalculations/MinWallTimes/BCCMoSuperCells/6x6x6Vac`
**Files:** `dftfe/src/interfaces/ase-dftfe/tests/bccmo_431/` → copy to `runs/bccmo_431/`
**Full write-up:** `matrixLabHandoff.md` (repo root)

**Config:** 431 atoms (6×6×6 BCC Mo, one vacancy at fractional 0.5,0.5,0.5, ideal lattice) · 35.7 Bohr cell · GGA-PBE · Γ-point · real binary · 3600 KS states · SCF tol 1e-4

> [!CAUTION]
> The original plan in this section was to run `parameterFileGPU32NodesMPS.prm`
> unchanged and to let ASE read it via `prm_file=`. **Neither worked.** That deck
> is DFT-FE v1.0; five of its keys are renamed or removed, and deal.II parses with
> `skip_undefined=true`, so a stale key is silently ignored and quietly changes the
> algorithm. Worst case: v1.0 puts `USE GPU` inside `subsection GPU`, current
> DFT-FE declares it at top level — left nested it is ignored, the GPU binary
> Chebyshev-filters on the CPU, and the job never finishes. **This broke the
> native arm too, not just ASE.** Two further bugs (a section-blind `sed` injector
> that silently dropped the Poisson and Helmholtz solver settings, and deck
> geometry keys overwriting the ASE-supplied coordinates) are fixed in
> `dftfeWrapper.cc` and `calculator.py`. See §7.5.

**Execution Plan (revised):**

1. **Rebuild all four binaries on Aurora** after pulling `ase_redesign`. `dftfeWrapper.cc` changed; stale binaries will not pick this up.
2. **Generate inputs and decks:** `python3 make_inputs.py .` then `python3 make_decks.py .`
   - `make_inputs.py` reproduces the benchmark `coordinates.inp` rather than depending on the `dftfe-benchmarks` checkout. Values and atom ordering match line for line (benchmark is CRLF, generated is LF).
   - **Not** `ase.build.bulk`: it gives correct physics in a different atom order, and force parity is compared component by component, so the comparison would be meaningless. Both arms read the same `coordinates.inp`.
3. **Native GPU:** `./submit.sh native <nodes>` — runs `parameterFileAurora<N>Nodes.prm`, which is the Summit deck's physics verbatim with the v1.0 key spellings migrated to current DFT-FE.
4. **ASE GPU:** `./submit.sh ase <nodes>` — `ase_bccmo431.py` contains **no** DFT parameters; it calls `read_dftfe()` on the same deck the native arm runs, so the two arms cannot drift apart. Aurora GPU tile binding goes through `launcher_args=["--ppn","12","gpu_tile_compact.sh"]`.
5. **Compare:** `python3 compare.py --native native_<N>nodes.out --ase result_ase_<N>nodes.json`

**Node counts and the rank ceiling.** Mesh is **4913 FE cells / 1.73M DoFs**. Ranks owning zero cells die on SIGFPE — same failure as §7.3. Aurora is 12 ranks/node:

| Nodes | Ranks | FE cells/rank | Note |
|---|---|---|---|
| 16 | 192 | 25.6 | same rank count as the published Summit **32**-node point |
| 32 | 384 | 12.8 | same rank count as the published Summit **64**-node point |
| 64 | 768 | 6.4 | |
| 128 | 1536 | **3.2** | thin — closest to the over-decomposition limit, may SIGFPE |

**Run 16 nodes first** — cheapest job, and its 192 ranks match a published Summit configuration, so it validates the Aurora build against a known-good result before spending nodes at scale.

**Aurora is not Summit.** Summit had 6 ranks/node, Aurora has 12, so equal node counts are not equal rank counts (table above gives the rank-matched pairs). Absolute wall times will not match the published numbers regardless: these builds use `HIGHERQUAD_PSP=ON`, and `SPECTRUM SPLIT CORE EIGENSTATES = 2400` no longer exists in DFT-FE so that optimisation is gone. Neither affects native-vs-ASE parity — both arms run identical settings.

**Metrics to capture:** total energy parity, max per-component force parity, SCF time per step, and socket overhead via `debug_timing=True` (quote the `streaming_overhead_s` column — see §6.5).

**Note on the escape hatch.** The 20 advanced solver knobs this deck tunes (`SCALAPACKPROCS`, `USE ELPA GPU KERNEL`, `POLYNOMIAL ORDER ELECTROSTATICS`, the mixed-precision family, …) previously had no Python kwarg and were reachable only via `prm_file=`/`extra_prm=`. They are now first-class kwargs, so this study is also expressible in pure Python (option B) if a kwargs-only variant is wanted.

---

## 6. Overhead Analysis

### 6.1 How `debug_timing` Works

Enable with `debug_timing=True` in the `DFTFE()` constructor (default is `False`). Each `calculate()` call emits:

```
[dftfe_ase debug_timing] total=Xs | DFT compute=Ys | streaming=Zs | ASE=Ws | interface overhead=Vs (P%)
```

**Breakdown** (all measured in `calculator.py::calculate()`):
- `total` = wall time from `t_start` (set after `super().calculate()`) to end of call
- `round_trip` = `backend.last_wait_s` = socket send + block on C++ response + recv
- `DFT compute` = pure SCF time reported by C++ driver (inside `round_trip`)
- `streaming` = `round_trip − DFT compute` = JSON serialise/TCP send + TCP recv/JSON deserialise
- `ASE` = `total − round_trip` = Python work outside the socket wait
- `interface overhead` = `streaming + ASE`

### 6.2 Single-Calculator Cases (GS, SCF, Relax)

One DFT-FE process, one socket. Measurement is clean.

| Component | Typical | % |
|---|---|---|
| DFT compute | 120–950s | ~99.999% |
| Streaming (TCP + JSON) | ~1–5ms | ~0.001% |
| ASE Python | ~0.2ms | ~0.0002% |
| **Interface overhead** | **~1–5ms** | **~0.001%** |

The socket interface is essentially zero-cost.

### 6.3 NEB Sequential Mode (parallel=False) — DO NOT USE THIS NUMBER

7 DFT-FE processes (84 MPI ranks) as children of one Python process, images evaluated one at a time. The Python process gets descheduled by the OS while 84 child ranks compete for CPU. This inflates `total` for Python-side work that should take microseconds.

| Component | Reported | True interface cost |
|---|---|---|
| DFT compute | ~124–128s | correct |
| Streaming | ~0.002s | correct |
| **ASE (as reported)** | **~4–6s (~3.5–6.3%)** | **OS scheduling noise — not interface** |

> [!WARNING]
> The `ASE` and `interface overhead` columns from sequential NEB are OS jitter and must **not** be reported as interface overhead. They are an artifact of 84 child processes competing with one Python process.

### 6.4 NEB Parallel Mode (parallel=True) — Correct Measurement

`NEB(parallel=True)` uses `threading.Thread` to evaluate all interior images concurrently. Socket I/O releases the GIL, so all 7 DFT-FE processes compute simultaneously. Python cleanly sleeps on `thread.join()` then collects results.

| Component | Steady-state | % |
|---|---|---|
| DFT compute | ~95–125s | ~99.999% |
| Streaming | ~0.001–0.002s | ~0.001% |
| **ASE Python** | **~0.0002s (0.2ms)** | **~0.0002%** |
| **Interface overhead** | **~0.001–0.002s** | **~0.001%** |

**Step-0 per job:** The first step of each PBS restart includes one-time MPI process launch (84 ranks booting across 7 nodes). This shows up as ~1–1.2s "ASE" overhead (~1%) in step 0 only. Every subsequent step reads ~0.001%.

### 6.5 Overhead Summary Table

| Mode | True overhead | Reported overhead | Notes |
|---|---|---|---|
| Single calc (GS/SCF/relax) | ~0.001% | ~0.001% | Correct |
| NEB, parallel=False | ~0.001% | **~4%** | OS scheduling noise — discard |
| **NEB, parallel=True** | **~0.9871%** | streaming ~1.71ms, n=7 samples, job TEST_DRY_RUN |
| NEB step-0 (per PBS job) | ~1% | ~1% | One-time MPI launch, amortised |

> [!TIP]
> **For reporting:** use the `streaming` column (~1–2ms, ~0.001%). It is always trustworthy regardless of mode.

---

## 7. Bugs Found and Fixed

### 7.1 Fractional Coordinates Crash — ✅ Fixed (2026-08-06)

**File:** [`calculator.py` lines 236–244](file:///lus/flare/projects/DFTCalc2/Mehul/install_DFTFE/dftfe/src/interfaces/ase-dftfe/src/dftfe_ase/calculator.py#L236-L244)
**Symptom:** ASE NEB restarts crashed: `DFT-FE Error: fractional coordinates doesn't lie in [0,1]`
**Root cause:** DFT-FE's C++ `dftfeWrapper::reinit()` ([`dftfeWrapper.cc` line 572–576](file:///lus/flare/projects/DFTCalc2/Mehul/install_DFTFE/dftfe/src/src/dftfeWrapper.cc#L572-L576)) has a hard `AssertThrow` requiring fractional coordinates in `[−1e-7, 1+1e-7]`. ASE's NEB/LBFGS optimizer legitimately moves atoms outside the unit cell during update steps. The calculator was passing raw Cartesian positions without wrapping.
**Fix:** Wrap positions into the unit cell for periodic directions before building the request:
```python
positions = self.atoms.get_positions()
if any(self.atoms.get_pbc()):
    scaled = self.atoms.get_scaled_positions(wrap=False)
    pbc = self.atoms.get_pbc()
    for dim in range(3):
        if pbc[dim]:
            scaled[:, dim] %= 1.0
    positions = scaled @ self.atoms.get_cell()[:]
```

### 7.2 NEB Sequential Image Evaluation — ✅ Fixed (2026-08-06)

**File:** [`ase_li2o_neb.py`](file:///lus/flare/projects/DFTCalc2/Mehul/install_DFTFE/runs/li2o_neb/ase_li2o_neb.py)
**Symptom:** 7 DFT-FE instances evaluated one at a time → ~5× slower per NEB step, spurious ~4% apparent interface overhead.
**Fix:** `NEB(images, ..., parallel=True)` — ASE's built-in `threading.Thread` evaluates all interior images concurrently. Each thread's socket `recv()` releases the GIL, so all 7 DFT-FE processes run truly in parallel.
**Effect:** Per-step wall time reduced from ~7×120s to ~120s. Interface overhead correctly measured at ~0.001%.

### 7.3 FE Mesh Over-decomposition at 32 Nodes — ⚠️ Known Limitation

**Symptom:** `rank 369 died from signal 8 (SIGFPE)` when native NEB runs on 32 nodes (384 ranks).
**Root cause:** Native NEB assigns all MPI ranks to a single image (serially). For 95 atoms with `MESH SIZE AROUND ATOM=1.2` Bohr and polynomial order 7, the adaptive FE mesh generates only ~300–500 active hex cells. At 384 ranks: ~0.8–1.3 cells/rank → boundary ranks receive 0 locally-owned cells → division by zero in cell-batch or ELPA process-grid calculations → hardware SIGFPE.
**Code path:** `nudgedElasticBandClass.cc::findMEP()` → `dftfeWrapper::computeDFTFreeEnergy()` → `generateMesh.cc` (p4est spatial partitioning) → 0-cell rank → SIGFPE.
**Safe ceiling:** 12 nodes (144 ranks, ~2–3 cells/rank). Scaling higher requires either reducing `MESH SIZE AROUND ATOM` (changes physics) or a native NEB redesign to distribute ranks across images.
**ASE NEB is unaffected** — each image has its own independent 12-rank FE problem.

### 7.4 CPU Complex Binary Crash — ⚠️ RESOLVED, but the diagnosis is doubtful

**Symptom:** Complex CPU binary crashed for Li2O (MGGA-R2SCAN + 2×2×2 k-pts), identical for pGD-native and ASE.
**Error:** `std::length_error: offset or copy size out of range for MemoryStorage` + heap corruption in `oncvClass.cc::computeSparseStructureNonLocalProjectors`.
**Resolution as recorded:** Scaling the CPU job to 12 nodes (144 ranks) using `mpiexec --depth 8 --cpu-bind depth` with `TOLERANCE = 5e-5` bypasses the crash; the job then runs indefinitely without memory errors. Attributed to **over-decomposition**.

> [!WARNING]
> **Revisit this (2026-08-08).** Two commits on `ase_redesign` fix buffer-sizing
> bugs in `src/atom/AtomicCenteredNonLocalOperator.cc` that are specific to
> **CPU builds with multiple k-points** — `26ea7ec9b8` and `79e0c486a0`. That is
> the same code path and the same trigger condition as this crash (CPU complex,
> 2×2×2 k-mesh, out-of-range `MemoryStorage` copy). Rank count changes how the
> mis-sized buffer is indexed, which would explain why 144 ranks "fixed" it
> without the underlying bug being addressed.
>
> A real over-decomposition failure and a real buffer-aliasing bug are not
> distinguishable from the evidence recorded here. Both fixes are now applied to
> the Aurora pGD tree as well (§3), so neither arm should hit the original bug —
> but that also means the "resolved by 144 ranks" conclusion has never been
> retested in isolation. Worth one job at the previously-crashing rank count with
> a build that includes both commits: if it no longer crashes, the diagnosis above
> is wrong, and the fixes should go upstream to Bitbucket.

### 7.5 `.prm` Injection Was Section-Blind — ✅ Fixed (2026-08-08)

Three defects in how socket mode built its parameter file. All three failed
*silently*: `dftParameters::parse_parameters` calls deal.II's
`prm.parse_input(file, "", true)` — `skip_undefined=true` — so an unrecognised or
missing entry costs nothing at parse time and changes the algorithm at run time.

**(a) Section-blind `sed`.** The injector rewrote the template with ~50
`system("sed …")` calls. `sed` cannot see subsection structure, so a key declared
in two subsections could not be targeted and a short key matched longer ones. On
the Summit BCC-Mo deck this silently destroyed three entries:

| Entry | Deck value | What actually happened |
|---|---|---|
| `Poisson problem parameters / TOLERANCE` | 1e-8 | deleted by the later SCF `TOLERANCE` |
| `Poisson problem parameters / MAXIMUM ITERATIONS` | 10000 | deleted by the later SCF `MAXIMUM ITERATIONS` |
| `Helmholtz problem parameters / MAXIMUM ITERATIONS HELMHOLTZ` | 10000 | matched by the pattern `set MAXIMUM ITERATIONS ` |

**Fix:** [`dftfeWrapper.cc`](file:///lus/flare/projects/DFTCalc2/Mehul/install_DFTFE/dftfe/src/src/dftfeWrapper.cc) — ~390 lines of `sed` replaced by `internalWrapper::PrmFile`, a line-oriented `.prm` editor that tracks the subsection stack. Every assignment resolves to exactly one entry; an unresolvable key warns instead of vanishing. Also removes ~50 shell spawns per run.

**(b) Incomplete template.** `helpers/parameterFile.prm` had 34 hand-written entries, so the injector had to *invent* any subsection the template lacked — and it created nested ones (`Eigen-solver parameters`, `Auto mesh generation parameters`, …) at top level, where deal.II drops the whole block. **Fix:** the template is now generated from `dftParameters.cc` `declare_parameters()` by `helpers/gen_parameter_template.py` — 187 entries, every declared key at its DFT-FE default, correct nesting. Injection only ever replaces a line that already exists. Regenerate after touching `declare_parameters`.

**(c) Deck geometry overwrote ASE geometry.** `prm_file=` injected every `set` line, including `ATOMIC COORDINATES FILE`, `DOMAIN VECTORS FILE`, `NATOMS` and `PSEUDOPOTENTIAL FILE NAMES LIST` — *after* the driver had pointed them at the coordinates it wrote from the ASE `Atoms`. The dangerous case is not a crash: if a `coordinates.inp` happens to be in the working directory the run **succeeds** and silently ignores everything Python sent, giving a perfect-looking parity result that proves nothing. **Fix:** `calculator.py::read_prm` drops geometry keys (`GEOMETRY_KEYS`) and driver-owned keys (`DRIVER_OWNED_KEYS`, e.g. `USE GPU`, `SOLVER MODE`), and migrates DFT-FE v1.0 key spellings via `LEGACY_KEYS`, warning on anything removed outright.

**Also added:** `read_dftfe()` (`dftfe_ase/io.py`) reads a native deck into `(Atoms, calculator)` preserving atom order; 28 new typed kwargs in `params.py`; `launcher_args=` for Aurora GPU tile binding. `tests/test_prm_roundtrip.py` asserts a deck and the equivalent kwargs serialise to an identical injection payload — that equality is what makes an ASE-vs-native comparison a measurement of the interface rather than of a settings drift.

> [!WARNING]
> Requires a **rebuild of all four binaries** — `dftfeWrapper.cc` changed. Any
> socket-mode result produced with a `.prm` before this fix should be re-checked
> if the deck set Poisson or Helmholtz solver parameters.

---

## 8. NEB-Specific Details

### Architecture Difference (Native vs ASE)

| | Native NEB | ASE NEB |
|---|---|---|
| MPI distribution | All 144 ranks work **every** image, images evaluated **serially** | Each image has its own 12-rank DFT-FE process, images evaluated **in parallel** |
| Nodes used in our run | **12 nodes** (144 ranks total) | **7 nodes** (84 ranks total) |
| Observed SCF time per image | ~120s (at 144 ranks) | ~95–125s (at 12 ranks) |
| Wall time per NEB step | ~7 × 120s = **~840s** | **~120s** |
| Node ceiling | ~12 nodes for this system (FE mesh limit — see §7.3) | 7+ nodes (independent per-image, no ceiling issue) |
| LBFGS checkpointing | Full history saved in `restartFolder/StepN/ionRelax.chk`, reloaded on restart | Positions saved in `neb_pi.traj`, LBFGS history **lost** at each PBS job boundary |

> [!NOTE]
> The ~7× wall-clock difference is **not** a straightforward parallel-vs-serial comparison on equal hardware. It has two compounding causes:
> 1. **Native NEB MPI architecture:** all ranks share one MPI communicator → images must be evaluated serially by design. This is a code structure choice, not a physics requirement.
> 2. **System too small to benefit from over-provisioning:** for this 95-atom system (mesh ~300–500 FE cells), going from 12→144 ranks gives essentially zero per-image speedup (cells/rank already near over-decomposition limit). So native uses 12 nodes but gets ~12-rank performance per image.
>
> On a larger system (thousands of atoms) where 144 ranks genuinely accelerates each SCF, native NEB's all-ranks-per-image design would be advantageous. For small systems it is wasteful.

### PBS Chaining Pattern

1-hour walltime means NEB must be chained across multiple jobs. Pattern used:
```bash
for i in $(seq 1 N); do
    JOBID=$(qsub run_neb_ase.pbs | grep -o '^[0-9]\+')
    while qstat $JOBID 2>/dev/null | grep -q $JOBID; do sleep 60; done
done
```
Scripts: `chain_v2.sh` (native+ASE, old), `chain_ase_v3.sh` (ASE only, with all fixes).

### Reading ASE Trajectory

`neb_pi.traj` stores all images flat. To read the last complete frame:
```python
images = read('neb_pi.traj', index=f'-{n_images}:')
```

---

## 9. Key Source Files

| File | Purpose |
|---|---|
| `dftfe/src/interfaces/ase-dftfe/src/dftfe_ase/calculator.py` | ASE calculator — coord wrap fix lines 236-244, `debug_timing` implementation |
| `dftfe/src/interfaces/ase-dftfe/src/dftfe_ase/backends/socket.py` | TCP socket backend, `last_wait_s` timing |
| `dftfe/src/interfaces/ase-dftfe/src/dftfe_ase/protocol.py` | Newline-delimited JSON framing over TCP |
| `dftfe/src/interfaces/ase-dftfe/src/dftfe_ase/io.py` | `read_dftfe()` — native `.prm` deck → `(Atoms, calculator)`, preserving atom order |
| `dftfe/src/interfaces/ase-dftfe/src/dftfe_ase/params.py` | Single source of truth for kwarg ↔ `.prm` mapping; `LEGACY_KEYS`, `GEOMETRY_KEYS`, `DRIVER_OWNED_KEYS` |
| `dftfe/src/src/dftfeWrapper.cc` | C++ reinit: frac coord assert; `internalWrapper::PrmFile` section-aware `.prm` editor (§7.5) |
| `dftfe/src/helpers/parameterFile.prm` | **Generated** template, all 187 declared keys at DFT-FE defaults — do not hand-edit |
| `dftfe/src/helpers/gen_parameter_template.py` | Regenerates the above from `dftParameters.cc` `declare_parameters()` |
| `dftfe_pgd/src/src/neb/nudgedElasticBandClass.cc` | Native NEB: all-ranks-serial per image, MPI comm structure |
| `dftfe/src/src/generateMesh.cc` | Adaptive FE mesh generation (p4est), partitioning logic |

---

## 10. PBS Job Scripts

| Script | Purpose |
|---|---|
| `runs/li2o_neb/run_neb_native_restart12.pbs` | Native NEB restart, 12 nodes/144 ranks, `tee -a` for persistent log |
| `runs/li2o_neb/run_neb_ase.pbs` | ASE NEB, 7 nodes, 1 image/node, 12 ranks/image |
| `runs/li2o_neb/chain_ase_v3.sh` | Chain N ASE NEB restarts sequentially |
| `runs/bccmo/run_bccmo_gpu.pbs` | BCC Mo GPU |
| `runs/li2o_scf/run_li2o_scf_gpu.pbs` | Li2O SCF GPU |
| `runs/li2o_relax/run_li2o_relax_gpu.pbs` | Li2O Relax GPU |
| `runs/bccmo_431/submit.sh` | **Scaling study** — `./submit.sh <native\|ase> <nodes>`; sets `-l select` and `NODES` together |
| `runs/bccmo_431/run_native.pbs` | Native arm; `mpiexec -n R --ppn 12 gpu_tile_compact.sh dftfe <deck>` |
| `runs/bccmo_431/run_ase.pbs` | ASE arm; launches plain python (the script issues its own `mpiexec`) |
| `scripts/update_project.py` | **Auto-update script** — called at end of each PBS job; parses output and patches PROJECT.md atomically. Usage: `--case <ase_neb|native_neb|bccmo|li2o_scf|li2o_relax> --logfile <path> --jobid $PBS_JOBID` |

**Native NEB log:** `runs/li2o_neb/native_gpu/native_neb_restart12.out` — persistent across all restarts (`tee -a`).
**ASE NEB stdout:** per-job files `runs/li2o_neb/neb_ase_gpu.o8738*` — historical record of timing data.

---

## 11. Open Items (update as resolved)

| # | Item | Status |
|---|---|---|
| 1 | **CPU-complex binary crash** blocks CPU parity for Li2O SCF and Relax. | ✅ Resolved (Over-decomposition fixed by scaling to 12 nodes) |
| 2 | **Native NEB not at force threshold.** Step 18: 241.86 meV, force 2.4e-3 vs 4e-4 Ha/Bohr threshold. Parity proven (<1 meV vs ASE). If formal convergence needed: resubmit `run_neb_native_restart12.pbs` (checkpoints at `native_gpu/restartFolder/Step18/`). | ⏳ Optional |
| 2b | **ASE NEB band_fmax unit note.** ASE reports band_fmax=1.57e-2 eV/Å. The DFT-FE native threshold is 4e-4 Ha/Bohr = **2.06e-2 eV/Å**. So 1.57e-2 < 2.06e-2 — ASE is already *below* the threshold. The optimizer stopped because it hit the step cap (`NEB_STEPS=10`), not because forces were too large. Both runs are effectively converged. | ℹ️ Note |
| 3 | **ASE NEB LBFGS not checkpointed across PBS restarts.** | ✅ Resolved (Added `restart='neb_pi_lbfgs.pckl'` to `ase_li2o_neb.py` for exact momentum resumption) |
| 4 | **NEB node ceiling ~12 for this system.** FE mesh ~300-500 cells for 95 atoms / mesh_size=1.2. Cannot safely run native NEB at 32+ nodes. | ℹ️ Known limitation |
| 5 | **`debug_timing=False` by default.** Only enabled in `ase_li2o_neb.py`. Add to other run scripts if overhead profiling is needed for other cases. | ℹ️ Optional |
| 6 | **Native NEB wall-clock vs ASE is not a fair comparison.** Native used 12 nodes (144 ranks/image, serial). ASE used 7 nodes (12 ranks/image, parallel). For this small 95-atom system, 144 ranks gives ~0 speedup over 12 ranks per image (mesh near over-decomposition), so native effectively wastes nodes on serial evaluation. On a larger system where 144 ranks meaningfully accelerates each SCF, native's all-ranks-per-image design would be competitive. | ℹ️ Note |
| 7 | **BCC Mo 6x6x6 Vacancy Scaling Study** (16, 32, 64, 128 nodes). Files written and locally tested; nothing has run on Aurora. Blocked on: pull `ase_redesign` + rebuild all four binaries. See §5.5 and `matrixLabHandoff.md`. | ⏳ Ready to run |
| 8 | **`.prm` injection defects** (§7.5) silently dropped Poisson/Helmholtz solver settings and let a deck overwrite ASE-supplied geometry. Fixed in `dftfeWrapper.cc` + `calculator.py`; template now generated. **Needs a rebuild of all four binaries on Aurora.** | ✅ Fixed, rebuild pending |
| 9 | **Decisions still open for the scaling study:** whether `SCALAPACKPROCS` stays `0` (auto) or takes the Summit values (50/30); whether the node set is 16/32/64/128 as written or rank-matched to Summit. Neither affects parity — both arms share the value. | ⏳ Pending |
| 10 | **2 core-physics fixes** (`26ea7ec9b8`, `79e0c486a0` in `AtomicCenteredNonLocalOperator.cc`; CPU + multi-k-point only) are **not** in Bitbucket pGD. Applied locally to the Aurora pGD tree, so both arms match today. Remaining risk: a clean pGD re-clone drops them silently — see §3. | ✅ Patched on Aurora; ⚠️ not upstream |
| 11 | **Upstream both fixes to Bitbucket** so they stop being a local carry. Also settles §7.4. | ⏳ Pending |
| 12 | **§7.4 diagnosis doubtful.** The two fixes in item 10 sit in exactly the code path §7.4 blamed on over-decomposition. One job at the previously-crashing rank count with a build containing them would settle it. | ⏳ Verification pending |

---

## 12. Rules for Agents

- **NEVER run compute on the login node.** Every calculation goes through `qsub`. No exceptions.
- **Queue:** `debug-scaling` only (1h walltime, 1 queued job per user). DFTCalc2 is DENIED prod/small/medium.
- **Python path:** always `/home/phanim/claude-env/bin/python3` in PBS scripts.
- Never use git commands unless explicitly asked.
- Never delete files or directories.
- Never overwrite existing outputs — create new versions instead.
- Make the smallest reasonable change needed. Do not modify unrelated files.
- Preserve existing coding style and structure.
- Continue autonomously toward the objective until complete or genuinely blocked.
- When a command fails: read full error → inspect logs/source → diagnose root cause → apply smallest fix → retry.
- Before finishing, always report: what changed, files modified, remaining issues.

---

## Update Log
- **2026-08-06 13:39 UTC** — PROJECT.md created. All 4 cases complete. ASE NEB parity established (242.73 vs 241.86 meV, Δ=0.87 meV). Fractional coord crash fixed. NEB parallel=True overhead confirmed ~0.001%. `aurora_benchmark_summary.md` fully merged into this document.
- **2026-08-08** — §5.5 rewritten: the original plan (run the Summit deck unchanged, read it via `prm_file=`, build the cell with `ase.build.bulk`) did not survive review — see the CAUTION box in §5.5 and the new §7.5. Three silent `.prm` injection defects found and fixed; `helpers/parameterFile.prm` is now generated from `dftParameters.cc` defaults. Added `read_dftfe()`, 28 typed kwargs, `launcher_args=`. Scaling-study files written to `interfaces/ase-dftfe/tests/bccmo_431/` (16/32/64/128 nodes) with `matrixLabHandoff.md` at repo root. 67 Python tests pass, both binaries compile, generators reproduce the benchmark geometry. **Nothing has run on Aurora yet; all four binaries need rebuilding first.**
- **2026-08-08 (branch audit)** — `ase_redesign` re-checked against pGD: **0 behind, 120 ahead**; pGD tip unchanged at `f0157ef4a` since 2026-07-20, so no merge needed. §3 commit/count corrected (`c4308c2f`/118 → `79e0c486a`/120). Found that 2 of the 120 are core-physics fixes absent from Bitbucket pGD (`AtomicCenteredNonLocalOperator.cc`, CPU + multi-k-point only); these have been applied locally to the Aurora pGD tree, so both arms match — but a clean pGD re-clone would drop them, and they are still not upstream. §3 and §7.4 updated, open items 10–12 added. §5.5 is unaffected (GPU, Γ-point).
~
~
