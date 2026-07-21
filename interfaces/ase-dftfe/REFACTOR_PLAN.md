# ASE-DFTFE — Production-Readiness & Performance Refactor Plan

**Target:** `interfaces/ase-dftfe/` (Python package) **plus** the C++ socket layer
(`src/socket_interface.cc`, `src/socket_interface.h`, `src/main.cc`) and the socket-path
hooks in `src/dftfeWrapper.cc`.

**Goal:** A DFT-FE ASE calculator that is *seamless* (zero-config startup), *fast* (reuses
electronic state, minimal transport overhead), and *deployable anywhere in the world* —
on any cluster that can run DFT-FE, without users editing source or hardcoding node details.

**Production targets:** **ALCF Polaris** (PBS Pro / NVIDIA A100 · CUDA+NCCL), **ALCF Aurora** (PBS / Intel PVC · SYCL-oneAPI+oneCCL),
**OLCF Frontier** (Slurm / AMD MI250X · HIP+RCCL), **NSM PARAM Pravega** & **PARAM Siddhi-AI** (Slurm / NVIDIA A100 · CUDA, InfiniBand).
→ Launcher priority: **Slurm + PBS/LSF**; GPU-backend matrix: **CUDA / HIP / SYCL**.

**Build foundation — reuse, don't reinvent:** DFT-FE already ships per-cluster native build recipes at
`github.com/dftfeDevelopers/install_DFTFE` (branch per machine: `polarisScript`, `auroraInstall`,
`frontierDevelop`, `nsm_A100`, `perlmutterDevelop`, `cpe2407`(Cray PE), CPU variants). These compile the
full stack (deal.II, p4est, ELPA, ScaLAPACK, Kokkos, libxc, alglib, spglib, dftd4, RCCL/oneCCL …) → the
`real` + `complex` binaries. The redesign **must not own the DFT-FE build** — it consumes whatever build
`install_DFTFE` (or a container wrapping it) produced.

**Decisions locked in:**
- **Scope:** Python package **and** the C++ socket protocol are both in scope.
- **API:** Breaking redesign allowed — aim for a clean `1.0`. Existing example scripts get rewritten.
- **Branch/workflow:** the redesign happens on **`ase_redesign`** (branched from the synced
  `socket_interface_merge`). `socket_interface_merge` stays as the in-flight PR #701 → `publicGithubDevelop`;
  the redesign is kept separate from it. Commit only interface-related files; never commit run-generated
  artifacts; never push to pGD directly.
- **Scope boundary:** the redesign owns the ASE package (Python) + the socket layer (C++ `socketDriver`,
  `main.cc`, and socket-path hooks in `dftfeWrapper.cc`). It does **not** own the DFT-FE build/deploy —
  that stays with `install_DFTFE` + container recipes that reference it. The package's job is to be trivially
  installable and to auto-detect/configure against an existing build.

**Status as of this revision:** `socket_interface_merge` synced with pGD (97 upstream commits merged, PR #701
up to date) and pushed. Committed there since sync: `density_quadrature_rule` + `use_single_prec_cheby` params;
`git_info.h` include fix for the upstream header reorg. `ase_redesign` branched from that state; all further
work lands here. **A clean rebuild (real+complex, GPU) is still the outstanding acceptance test for the merge.**

> Execution plan for a coding agent. Work by priority band (P0→P3). Each task lists files touched,
> an **impact/effort** tag, and an **acceptance check**. Keep `pytest -m "not integration"` green after every task.

---

## 0. The strategic picture — three levers + one fork

The persistent-server architecture (Python = TCP server, MPI DFT-FE = client) is the **right
foundation** — it already eliminates per-call MPI spawn overhead and is correct for notebooks,
active-learning, multi-node, and driver-orchestration use cases. **Do not replace it.** The wins are:

### Lever 1 — Reuse electronic state across steps  ⭐ *impact: very high · effort: medium*
**Grounded finding:** the socket hot-loop drives its own step loop
(`socketDriver::run()` → `dftfeWrapper::updateAtomPositions()` → `updateAtomPositionsAndMoveMesh()`
at `src/dftfeWrapper.cc:1257` → `computeDFTFreeEnergy()`) and **bypasses**
`molecularDynamicsClass` / `geometryOptimizationClass`. All the state-reuse machinery lives *inside*
those native drivers and is **not wired into the socket path**:
- `extrapolateDensity` — 2nd-order density prediction from t, t−dt, t−2dt
  (`molecularDynamicsClass::DensityExtrapolation` / `DensitySplitExtrapolation`)
- `reuseWfcGeoOpt`, `reuseDensityGeoOpt` (`include/dftfe/dftParameters.h:137-162`)

So every socket step re-converges the SCF from a cold-ish guess. Wiring extrapolation + wavefunction
reuse into the socket loop is typically a **2–5× reduction in SCF iterations per MD/relaxation step** —
the single biggest throughput win, and mostly plumbing into existing, tested DFT-FE code.
**First: verify exactly what state `updateAtomPositionsAndMoveMesh` already preserves** (the density
may ride the moved mesh even today; the 2nd-order *predictor* is what's definitely missing) so the
win is sized before implementing.

### Lever 2 — Transport: binary framing + UNIX-domain sockets  *impact: high · effort: medium*
- Current wire format is ASCII JSON at 16 sig-figs parsed by a hand-rolled `std::string::find`
  parser (`parse_matrix` keys off `]]`, O(n) scans per field). For a 10k-atom system, forces are
  megabytes of text. Replace bulk arrays (coords/forces/stress) with **length-prefixed binary
  doubles** → negligible parse time + kills the fragile parser.
- Add **AF_UNIX (UNIX-domain) sockets** for same-node runs (the common case, incl. single-node
  multi-GPU): no TCP, no port allocation, no `gethostbyname`, no firewall/reachability problem,
  lower latency. Keep INET only for genuine multi-node. This is a *speed* **and** *deployability* win.

### Lever 3 — Zero-config startup + deployment  *impact: high · effort: medium-high*
Seamless = the calculator infers everything; "deployable anywhere" is won at the build, which we
**reuse from `install_DFTFE`** rather than reinvent. Two deployment modes, one config contract:

- **Two deployment modes** (both end at the same contract — canonical binary paths + a cluster profile):
  1. **Native build via `install_DFTFE`** — primary for Cray/Slingshot leadership machines
     (Aurora/Frontier/Polaris) where Cray PE + Slingshot make full containers painful. ASE package pip-installs on top.
  2. **Apptainer container** — primary for standard stacks (PARAM Pravega/Siddhi = OpenMPI+InfiniBand,
     generic CPU/GPU, CI/dev/laptops). Bundles the *whole* DFT-FE + ASE at fixed paths.
- **Container is a matrix, not one image:** `{cuda, hip, sycl}` (one per GPU backend — a CUDA image can't
  run DFT-FE's GPU path on AMD/Intel). Build recipes **wrap the matching `install_DFTFE` branch**
  (`hip`←frontierDevelop, `sycl`←auroraInstall, `cuda`←polarisScript/nsm_A100). Build Docker in CI, ship
  **Apptainer** `.sif` (HPC forbids Docker). On the fabric, use **MPICH-ABI / host-MPI injection**
  (bind-mount host MPI + libfabric + GPU runtime) for Slingshot/InfiniBand + GPU-aware MPI.
- **Auto-detect** (makes `DFTFE()` Just Work): scheduler (Slurm/PBS/LSF) from env; node/GPU counts from the
  allocation; **real-vs-complex binary from k-points** (Γ-only → real, any k-mesh → complex — a rule on the
  ASE inputs, decided in Python); **binary *location* from config** (kwarg > env `DFTFE_BIN_{REAL,COMPLEX}` >
  `~/.config/ase-dftfe/clusters.toml` profile > canonical container path > `PATH`); GPU backend validated
  against the build. Goal: `DFTFE()` inside an allocation; `DFTFE(cluster="frontier")` for a named profile;
  zero binary config inside the container.
- **Optional warm daemon**: attach to an already-running DFT-FE server so even the one-time `reinit`
  (mesh + PSP) is amortized across sessions/notebooks.

### The one architectural fork — pluggable backends *(design now, build later)*
Keep **socket** as the primary backend, but put it behind a `Backend` interface so we can add,
without touching the `DFTFE` calculator API:
- **pybind11 in-process** (call `dftfeWrapper` directly under mpi4py) — theoretical fastest (zero
  serialization) but forces `mpirun python …` and couples the Python env to the DFT-FE ABI. Good as a
  *fast path* for tight single-allocation MD; bad for the decoupled/notebook UX. **Defer; design the seam now.**
- **i-PI-protocol compatibility mode** — lets DFT-FE plug into the i-PI sampler ecosystem. Additive
  (i-PI can't carry the rich first-init params, so keep our init handshake, speak i-PI for the hot loop).

---

## 1. Current-state defects (grounded; fix during the relevant phase)

**Python (`dftfe/__init__.py`):**
- `verbosity is None` crash: `_start_server()`:158 & `_launch_dftfe()`:172 do `if self.verbosity > 0`.
- Fragile response framing: `calculate()`:278-284 breaks the read on the first `}` *or* `\n` — truncates
  large force arrays split across TCP chunks. (Superseded by Lever-2 binary framing.)
- Duplicate dict keys `calculate()`:217-225; overloaded `self.log_file` (path vs file object); bare `except:`
  in `close()`; `shell=True` f-string launch; unreliable `__del__` cleanup (no context manager);
  dead `_get_free_port()`; `host` default = `gethostname()` (not reachable on multi-node/multi-NIC).

**C++ (`src/socket_interface.cc`):**
- `exit(1)` on socket errors (no `MPI_Abort`); `gethostbyname` (IPv4-only, deprecated → fails on IPv6);
  `data += buffer` truncates at first NUL and drops post-`\n` bytes; connect/accept timeout mismatch
  (C++ 10 s vs Python 120 s); **no error channel** (SCF failure = closed socket, no diagnostic);
  ungated `std::cout` spam; `parse_scalar` duplicate `return`; `format_response` always emits stress.
- **Cell updates silently dropped** (`run()`:810-813 omit deformation) → variable-cell relax / NPT wrong.

**Packaging:** `pyproject.toml` `readme="README.md"` but no README exists → build broken; odd version;
PSPs bundled (size/licensing); import name `dftfe` may collide with future official bindings.

**Repo hygiene:** run-generated artifacts historically tracked (`.pyc`, `forces.txt`, scratch dirs,
`slurm_*.out`, `test_results_*.txt`); only `automated_tests/.gitignore` exists.

---

## 2. Target architecture

```
interfaces/ase-dftfe/
├── src/dftfe_ase/                 # importable pkg renamed dftfe -> dftfe_ase (§4.2)
│   ├── calculator.py              # DFTFE(Calculator): thin, ASE-facing, backend-agnostic
│   ├── backends/
│   │   ├── base.py                # Backend ABC (compute/close); pluggable
│   │   ├── socket.py              # primary: framing, timeouts, error frames, reconnect
│   │   ├── inprocess.py           # (later) pybind11 fast path
│   │   └── ipi.py                 # (later) i-PI protocol compat
│   ├── protocol.py                # wire schema + PROTOCOL_VERSION + binary framing (single source of truth)
│   ├── launchers/                 # slurm.py pbs.py lsf.py mpi.py local.py + base.py (detect/build_command/env)
│   ├── config.py                  # cluster profiles: kwargs > env > ~/.config/ase-dftfe/clusters.toml
│   ├── binary.py                  # real/complex selection + GPU/CPU capability probe
│   ├── params.py                  # typed param spec: validation, units, .prm key, sentinel (one source)
│   └── utils/                     # existing utils, typed + tested
├── containers/                    # Apptainer + Docker recipes
├── packaging/                     # spack/conda recipes
├── tests/                         # pytest unit (mock server) + integration (marked, needs binary)
├── examples/                      # rewritten, path-free
├── docs/                          # README + quickstart + per-scheduler recipes + porting checklist
└── pyproject.toml
```

C++ side keeps `socketDriver` but gains: binary framing, AF_UNIX support, versioned handshake,
error/status frames, gated logging, state-reuse hooks, and a cell-update path.

---

## 3. Phased execution plan (by priority band)

### P0 — Foundations & always-green *(do first)*
- **P0.1** `.gitignore` (scratch/`*.pyc`/`slurm_*`/`test_results_*`/build dirs); `git rm --cached` tracked junk. *(low/low)*
- **P0.2** Rebuild real+complex+GPU on the merged tree; run `automated_tests`; **record the numerical parity baseline.** *(the merge's acceptance test)* *(high/low)*
- **P0.3** Mock socket server in `tests/` so all later Python work is testable without a binary. *(high/med)*
- **Acceptance:** clean tree; merged tree builds & passes existing tests; mock server round-trips a request.

### P1 — Performance core (the levers) *(highest ROI)*
- **P1.1 Lever 1 — state reuse.** Verify what `updateAtomPositionsAndMoveMesh`/`computeDFTFreeEnergy`
  preserve; wire `extrapolateDensity` (2nd-order) + `reuseWfc/DensityGeoOpt` into `socketDriver::run()`;
  expose as calculator options (sensible defaults on). *(very high / med)*
  - **Acceptance:** an MD/relaxation run shows a measured drop in mean SCF iterations/step vs baseline,
    with identical converged energies/forces to tolerance.
- **P1.2 Lever 2 — protocol.** Define `protocol.py` (schema + `PROTOCOL_VERSION` + **binary length-prefixed
  framing** for bulk arrays); reimplement C++ `receive_data`/`send_data`/`format_response` to match; add a
  versioned handshake and **error/status frames** (SCF/setup failure → `DFTFEError` in Python, not a hang);
  fix the timeout mismatch. *(high / med)*
  - **Acceptance:** mock server round-trips framed request/response + an error frame; large-system force
    payload transfers with no truncation and lower wall-time than the JSON path.
- **P1.3 Lever 2 — UNIX sockets.** Add AF_UNIX on both ends; auto-select UDS for same-node, INET for multi-node. *(high / med)*
  - **Acceptance:** single-node run uses UDS with no TCP port opened; multi-node still connects via INET.
- **P1.4 Cell updates (correctness+perf).** Implement the omitted cell-deformation path in `run()`; choose
  deform-vs-reinit by threshold. *(med / med)*
  - **Acceptance:** variable-cell relaxation drives the cell & reduces stress; fixed-cell unchanged.

### P2 — Portability & seamless startup (deployable anywhere)
- **P2.1 Launchers.** `launchers/` for Slurm/PBS/LSF + mpi/local; env auto-detect; GPU-binding per scheduler;
  replace `shell=True` f-string with an argv list. *(high / med)*
- **P2.2 Reachability.** Bind `0.0.0.0`/named interface; compute & advertise a routable host to the job; preflight
  that fails fast on unresolvable host. (Mostly moot when UDS is used single-node.) *(high / med)*
- **P2.3 Zero-config (Lever 3).** `config.py` layered profiles (kwarg > env > `clusters.toml` > container path > `PATH`);
  `binary.py` real/complex auto-select from k-points + GPU/CPU validation; ship **cluster profiles for the 5 targets**
  distilled from the `install_DFTFE` `env2/env.sh` per-branch scripts (paths, modules, launcher, GPU backend).
  `DFTFE()` works inside an allocation; `DFTFE(cluster="polaris"|"aurora"|"frontier"|"pravega"|"siddhi")`. *(high / med)*
- **P2.4 Container matrix (Lever 3).** `containers/` with `{cuda,hip,sycl}` recipes that **wrap the matching
  `install_DFTFE` branch** for the DFT-FE build, then layer the ASE package at canonical paths. Docker in CI →
  convert to **Apptainer `.sif`**; document host-MPI/libfabric bind-mount per fabric. NOT one universal image. *(high / high)*
- **P2.5 Warm daemon (optional).** Attachable persistent server to amortize `reinit`. *(med / high)*
- **Acceptance:** the same example runs on a Slurm and a PBS allocation by only switching the detected launcher,
  editing nothing; a fresh clone builds & runs via the container on a machine with no hand-built toolchain.

### P3 — Hardening, API cleanup, tests, docs
- **P3.1 Calculator rewrite** on backends/launchers/config/params; fix all §1 Python defects; context-manager
  lifecycle; structured `logging`. *(high / med)*
- **P3.2 `params.py`** typed table (validation + generates JSON payload *and* the docs table — kills Python/C++
  sentinel drift). *(med / med)*
- **P3.3 Utils** type-annotated + unit-tested over fixture logs/CIFs; robust `parse_dftfe_log`. *(med / med)*
- **P3.4 Packaging** README, valid `1.0` metadata, `src/` layout, PSP policy (§4.1), entry point; `python -m build` clean. *(high / low)*
- **P3.5 Backend seam** finalized; stub `inprocess`/`ipi` behind the ABC (design complete, impl deferred). *(med / med)*
- **P3.6 CI** unit + lint + build on push; integration marked & documented. *(high / med)*
- **P3.7 Docs & examples** path-free examples; per-scheduler recipes; porting checklist; regenerate param reference;
  update `ase_dev_guide.md`. *(med / med)*
- **Acceptance:** `pytest -m "not integration"` green in CI with no binary; a new user installs + runs a
  single-point energy from README + one recipe, editing nothing but config/env.

---

## 4. Cross-cutting decisions (recommendations; confirm with maintainers)
- **4.1 PSPs:** do **not** bundle full `psp_library/`/`psp_spms/` in the wheel (size + licensing). Ship 1–2 small
  PSPs under `tests/data/` for CI; document `DFTFE_PSP_PATH` + the `psp_path` dict. Verify redistribution license first.
- **4.2 Package name:** rename import `dftfe` → `dftfe_ase` (dist `ase-dftfe`) to avoid colliding with future
  official DFT-FE Python bindings; keep an ASE entry point.
- **4.3 Protocol/build coupling:** the Python client and DFT-FE build must agree on `PROTOCOL_VERSION`; the handshake
  must reject a mismatch with a clear message. Document the coupling. The container pins a matched pair.

## 5. Sequencing & risk notes
- **P0 → P1** are prerequisites. Lock parity (P0.2) *before* touching the hot loop.
- **P1.1 (state reuse) is the top ROI item** — verify preserved state first so the win is real, not assumed.
- C++ work (P1.1-P1.4) needs rebuilds on real+complex+GPU — iterate against the mock server (P0.3) to avoid rebuild churn.
- Python P2/P3 largely parallelize once `protocol.py` (P1.2) is frozen.
- **Highest-risk items:** multi-node reachability (P2.2) — test on a real 2-node Slurm alloc early; cell updates
  (P1.4) — validate vs a known variable-cell reference; C++/Python framing parity (P1.2) — mock server is the guard.

## 6. Quick-win bugs (safe to batch anytime)
`verbosity is None` guard · read-until-`\n` (until binary framing lands) · dedupe dict keys · `except Exception:` ·
C++ `parse_scalar` dup return · gate C++ `std::cout` behind verbosity · add missing `README.md`.
*(Note: the `density_quadrature_rule`/`use_single_prec_cheby` params and the `git_info.h` include fix are already committed.)*
