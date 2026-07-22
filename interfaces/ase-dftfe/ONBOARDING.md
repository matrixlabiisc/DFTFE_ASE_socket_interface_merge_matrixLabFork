# ASE-DFTFE Interface — Project Onboarding (shared across all my Claude sessions)

**Goal:** Make the DFT-FE ASE interface production-ready and deployable on any DFT-FE-capable
cluster — targets: **ALCF Polaris (CUDA) / Aurora (SYCL)**, **OLCF Frontier (HIP)**,
**NSM PARAM Pravega & Siddhi (CUDA)**. Owner: **Mehul Darak** (mehuldarak@iisc.ac.in).

## Where the code lives
- **MATRIX (IISc)**: `/home/pa01/Mehul/DFTFE` — the working DFT-FE checkout.
- **Branches**: all redesign work is on **`ase_redesign`** (branched from `socket_interface_merge`,
  the in-flight PR #701 → `publicGithubDevelop`). Never push to pGD.
- **Bitbucket** `dftfedevelopers/dftfe` (PRIVATE): has `ase_redesign`, `socket_interface_merge`.
- **PUBLIC mirror** of `ase_redesign` (clone here where Bitbucket auth isn't available, e.g. Aurora):
  `https://github.com/matrixlabiisc/DFTFE_ASE_socket_interface_merge_matrixLabFork.git` (branch `ase_redesign`).
- New Python package: `interfaces/ase-dftfe/src/dftfe_ase/` (protocol, backends, calculator, launchers,
  config, binary). Supersedes the old `interfaces/ase-dftfe/dftfe/` package. Plan: `interfaces/ase-dftfe/REFACTOR_PLAN.md`.

## Architecture (verified facts)
- Python is the TCP **server**; DFT-FE (launched via mpirun/srun/mpiexec with `--socket host:port`) is the
  **client**. One persistent DFT-FE process amortizes MPI/mesh init across all force calls.
- Wire protocol: newline-delimited JSON (`protocol.py`), mirrored by C++ `src/socket_interface.cc`.
- Pluggable backends (`backends/`), launchers (`launchers/`: srun/mpirun/mpiexec/jsrun/local),
  cluster profiles + binary auto-select real(Γ)/complex(k-points) (`config.py`, `binary.py`).

## What's DONE & verified (via SLURM, never login-node)
- **Single-point parity vs native file-based DFT-FE** (same binary, CPU, deterministic): ex1 N2 dE=9.8e-12,
  ex2 Al bulk (complex,k-pts) dE=2.08e-12, ex3 graphene (2D,complex) dE=8.76e-13 Ha. Forces 1e-10.
- **Geo-opt** same-minimum (native GEOOPT vs ASE BFGS): |dE|=3.4e-8 Ha.
- **Overhead** via `debug_timing` (separates DFT compute / socket streaming / ASE python): interface
  overhead ~**0.01%** per MD step (streaming 1.7–3.0 ms vs seconds of SCF), single + multi-node (16 GPU).
- **C++ fixes** (in `build_merged`): `git_info.h` include; segfault guard (only fetch forces/stress when
  computed); `compute_time` in response for overhead split. **Multi-node reachability**: bind `0.0.0.0` +
  advertise `gethostname()` (not 127.0.0.1) + getaddrinfo preflight.
- 47 pytest unit tests (mock DFT-FE, no binary needed). `build_merged/` = merged+fixed GPU binary
  (build_gpu = old pre-merge, kept as fallback — never overwrite it).
- **Param-injection verifier** (`tests/verify_param_injection.py`, keep_scratch → inspect generated .prm;
  inspects even if the SCF crashes, since the .prm is written at reinit): confirms passed values actually
  REACH DFT-FE, not silently default. Caught + FIXED **5 silent-drops** (replace-sed no-op when key absent
  from template): MIXING METHOD, ORTHOGONALIZATION TYPE, SMEARED NUCLEAR CHARGES, USE GROUP SYMMETRY,
  LBFGS HISTORY — all now delete+append under their subsection. **18/18 params verified.**
  KNOWN GAPS (not fixed; unused by the 3 target benchmarks): `dispersion_correction_type` needs a
  Dispersion Correction subsection added; `start_magnetization` maps to a **non-existent** DFT-FE key
  (initial magnetization = per-atom `m` column in coordinates.inp — a separate feature). Also: enabling
  smeared-charges/group-symmetry crashes tiny test systems (setup, not injection).
- **Generic .prm injector (A + B) — DELIVERED & verified.** `params.py` is the single source of truth
  (kwarg ↔ .prm key/subsection/type; generates `dev_help/parameter_mapping.md`). One mechanism, no
  per-param sed: C++ `dftfeWrapper.setSocketPrmOverrides()` + a generic delete+append loop in reinit
  (creates the subsection if missing); calculator routes **B** (full-Python "planned" kwargs) + **A**
  (`extra_prm={...}` dict and `prm_file="…"` full-.prm passthrough) into one `{section,key,value}` list.
  Fixed `mixing_history` mis-map (was LBFGS HISTORY → now MIXING HISTORY). 21 params two-value-tracked;
  53/53 unit tests. `dispersion_correction_type` now works too (injector auto-creates the subsection);
  only `start_magnetization` remains unsupported (per-atom `m` column, separate feature).
- **`FileBackend`** (`backends/file.py`): native file-based DFT-FE as a Backend for apples-to-apples
  (same ASE optimizer, socket vs file), full per-atom force parsing, any system.
- **Benchmark coverage** (dftfe-benchmarks accuracyBenchmarks: BCC Mo GS, Li2O ion-relax, Li2O NEB):
  all DFT-FE params map to the interface (density_quadrature_rule + use_single_prec_cheby, added this
  session, are used by Li2O). NEB subsection is ASE-side (ase.mep.NEB). Verify meta-GGA MGGA-R2SCAN
  support + multi-element psp dict before Li2O runs.

## Reviewer-grade gaps still open (before publishing overhead / porting)
- Overhead needs a **native-MD head-to-head** (compare total time + mean SCF iters/step; the socket path
  lacks density extrapolation → could need more SCF iters, invisible to the streaming% metric) and a
  **scaling study** (overhead vs N_atoms and vs node count) — do these on Polaris/Frontier/Aurora.
- Robustness TODO: launcher validation for srun (in progress) / mpiexec / jsrun; hang/timeout heartbeat;
  protocol-version handshake enforcement; **Aurora tile/GPU binding is not encoded in the auto-launcher**
  (use explicit `command="mpiexec ... <bind>"` for now).

## Workflow rules (STRICT)
- Work only on `ase_redesign`. **Never** push to pGD. **No** `rm`/`git reset --hard`/force-push/remote-ref
  deletion without explicit approval. Prefer additive + plain fast-forward pushes.
- **Verify via SLURM (`sbatch`), never login-node compute.**
- Commit only interface-related files; never commit run-generated artifacts.
- Commit trailers (two lines): `Author: Mehul Darak` then `Co-author: Claude Opus 4.8`. New source files
  carry `# @author Mehul Darak`.

## Cluster specifics
- **MATRIX**: build/run modules in `COMPILING_ON_MATRIX.md` (spack + openmpi/nccl/gdrcopy, `CUDA_PATH`,
  `LIBRARY_PATH`). Shared venv `~/.venvs/ase-env` (ase+numpy+pytest). Debug partition = cn1-2 (8 GPU each).
- **Aurora**: ✅ **BUILT + first run WORKS** (SYCL/oneAPI, real + complex) at
  `/lus/flare/projects/DFTCalc2/Mehul/install_DFTFE/dftfe/install/{real,complex}/dftfe`. First ASE↔DFT-FE
  single-point via PBS: N2 = -20.641600 Ha (matches MATRIX/CUDA to ~1e-4), exit 0, 42 s.
  Binding used: `mpiexec -n 4 --ppn 4 --cpu-bind=list:1-8:9-16:17-24:25-32 --gpu-bind=list:0.0:0.1:1.0:1.1`
  + `advertise_host=gethostname()`. NEXT: scaling/accuracy studies (BCC Mo GS, Li2O FCC relax, Li2O NEB)
  from dftfe-benchmarks — with a parameter-coverage audit FIRST (do all benchmark .prm params map to the
  interface?). Setup notes: needs ALCF proxy
  `http://proxy.alcf.anl.gov:3128` for internet; build via `install_DFTFE` `auroraInstall` branch
  (`module load cmake boost ninja; ./install_dftfe.sh --all`), but point its DFT-FE clone at the PUBLIC
  fork above on `ase_redesign` (Bitbucket is private). Conda env `claude-env`. SYCL/oneAPI build.
