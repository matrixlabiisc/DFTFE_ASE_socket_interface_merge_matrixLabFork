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
- **Aurora** (setup in progress at `/lus/flare/projects/DFTCalc2/Mehul`): needs ALCF proxy
  `http://proxy.alcf.anl.gov:3128` for internet; build via `install_DFTFE` `auroraInstall` branch
  (`module load cmake boost ninja; ./install_dftfe.sh --all`), but point its DFT-FE clone at the PUBLIC
  fork above on `ase_redesign` (Bitbucket is private). Conda env `claude-env`. SYCL/oneAPI build.
