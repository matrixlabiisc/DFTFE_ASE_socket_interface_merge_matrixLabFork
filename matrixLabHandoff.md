# MatrixLab Handoff — DFT-FE `.prm` handling rework + BCC-Mo 431-atom scaling study

**Branch:** `ase_redesign`
**Author:** Mehul Darak (mehuldarak@iisc.ac.in)
**Target machine:** ALCF Aurora, `debug-scaling` queue
**Status:** code complete and tested locally; nothing has run on Aurora yet

This document covers two things: a correctness rework of how DFT-FE parameter
files are read and injected, and the files for the BCC-Mo 6×6×6 vacancy scaling
study that motivated it.

---

## 1. Why this work happened

The scaling study was supposed to be simple: hand both arms the published Summit
deck and compare. It isn't, for three reasons — and two of them were silently
corrupting *any* run driven by a `.prm`, not just this study.

deal.II parses parameter files with `skip_undefined=true`. An unrecognised key
costs nothing at parse time and quietly changes the algorithm at run time. That
property turned three separate bugs into silent wrong answers.

### 1.1 The injector could not see subsections

Socket mode built its `.prm` by running `sed` over a template. `sed` has no idea
what a subsection is, so:

- `TOLERANCE` exists in **both** `SCF parameters` (1e-4) and `Poisson problem
  parameters` (1e-8). Writing one deleted the other.
- The pattern `set MAXIMUM ITERATIONS ` also matched
  `set MAXIMUM ITERATIONS HELMHOLTZ`.

Net effect on the Summit deck: **Poisson `TOLERANCE`, Poisson
`MAXIMUM ITERATIONS`, and `MAXIMUM ITERATIONS HELMHOLTZ` were all lost** and
fell back to built-in defaults. No warning. A native-vs-ASE comparison run this
way would have been measuring a solver-settings difference.

### 1.2 A deck could overwrite the geometry

`prm_file=` injected every `set` line, including `ATOMIC COORDINATES FILE`,
`DOMAIN VECTORS FILE`, `NATOMS` and `PSEUDOPOTENTIAL FILE NAMES LIST`. Those ran
*after* the driver had pointed the deck at the coordinates it wrote from the ASE
`Atoms`, so they clobbered it.

The dangerous case is not a crash. If a `coordinates.inp` happens to sit in the
working directory, the run **succeeds** and silently ignores everything Python
sent — a perfect-looking parity result that proves nothing.

### 1.3 Five v1.0 keys no longer exist

The Summit deck is DFT-FE v1.0. Against current `develop`:

| v1.0 key | now |
|---|---|
| `USE MIXED PREC CHEBY` | `USE SINGLE PREC CHEBY` |
| `USE MIXED PREC CGS O` | `USE MIXED PREC XTOX` |
| `USE MIXED PREC XTHX SPECTRUM SPLIT` | `USE MIXED PREC XTHX` |
| `SPECTRUM SPLIT CORE EIGENSTATES` | removed (feature gone) |
| `USE GPU` inside `subsection GPU` | top level, `utils/runParameters.cc` |

The last one **also breaks the native arm**. Left nested, it is ignored, the GPU
binary runs Chebyshev filtering on the CPU at ~50 s/pass, and the job never
finishes — exactly the failure `PROJECT_Aurora.md` §2 warns about. "Run the
published deck unchanged" was never going to work on either side.

---

## 2. Code changes

### 2.1 `helpers/parameterFile.prm` — now generated

Was 34 hand-written entries. The injector had to invent any subsection the
template lacked, and it created *nested* ones at top level, where deal.II
silently drops the whole block.

Now generated from `utils/dftParameters.cc` `declare_parameters()` by
**`helpers/gen_parameter_template.py`** (new): 187 entries, every declared key at
its DFT-FE default, correct nesting. Injection only ever replaces a line that is
already there.

> Regenerate after touching `declare_parameters`:
> `python3 helpers/gen_parameter_template.py`

The previous template is preserved as `helpers/parameterFile.prm.pre-autogen`.

### 2.2 `src/dftfeWrapper.cc` — section-aware injection

~390 lines of `sed`/`system()` replaced by `internalWrapper::PrmFile`, a small
line-oriented `.prm` editor that tracks the subsection stack. Every assignment
now resolves to exactly one entry, and an unresolvable key prints a warning
instead of vanishing.

Side benefits: ~50 shell spawns per run gone, and the file is written once.

Also removed one dead call: `START MAGNETIZATION` is not declared anywhere in
this branch, so that `sed` had always been a no-op. `TOTAL MAGNETIZATION` (which
is declared) still gets written.

### 2.3 `interfaces/ase-dftfe` — Python side

| File | Change |
|---|---|
| `params.py` | +28 `ParamSpec` rows (36 → 64). Adds `LEGACY_KEYS` (v1.0→current), `GEOMETRY_KEYS`, `DRIVER_OWNED_KEYS`. Fixed `to_prm_entries` emitting `None` vs `""` for top level, so kwargs and deck entries now compare equal. |
| `calculator.py` | `_parse_prm_file` → public `read_prm`: strips geometry and driver-owned keys, migrates legacy spellings, **warns** on removed ones, strips comments. New `launcher_args=` kwarg. |
| `io.py` *(new)* | `read_dftfe()` / `read_dftfe_atoms()` — read a native deck into `(Atoms, calculator)`. |
| `tests/test_prm_roundtrip.py` *(new)* | 12 tests pinning all of the above. |
| `tests/test_calculator_launch.py` | +2 tests for `launcher_args`. |
| `dev_help/parameter_mapping.md` | regenerated (64 params documented). |

The 20 performance knobs this study needs — `scalapack_procs`,
`use_elpa_gpu_kernel`, `poisson_tolerance`, `polynomial_order_electrostatics`,
`cheby_degree_scaling_first_scf`, the mixed-precision family, … — previously had
**no** Python kwarg and were reachable only through the escape hatch. They are
now first-class.

### 2.4 The two entry points

Both doors now land on the same calculation:

```python
# door 1 — an existing native deck
atoms, calc = read_dftfe("parameterFileAurora32Nodes.prm")

# door 2 — pure Python
atoms.calc = DFTFE(xc="GGA-PBE", polynomial_order=7, scalapack_procs=0, ...)
```

`tests/test_prm_roundtrip.py::test_deck_and_kwargs_produce_identical_overrides`
asserts the two serialise to an identical injection payload. That equality is
what makes an ASE-vs-native comparison a measurement of the interface rather
than of a settings drift.

`launcher_args` is what makes Aurora work:

```
mpiexec -n 384 --ppn 12 gpu_tile_compact.sh <binary>
```

Without the wrapper every rank on a node lands on GPU tile 0.

---

## 3. What Aurora needs before anything runs

> [!IMPORTANT]
> `PROJECT_Aurora.md` is the single source of truth for this project and takes
> precedence over this document wherever they disagree. Read §12 (Rules for
> Agents) before doing anything: **never run compute on the login node**, queue is
> `debug-scaling` only (1 h cap, one queued job per user), and PBS scripts must
> use `/home/phanim/claude-env/bin/python3` — the system `python3` has no numpy
> and will crash.

**Workdir:** `/lus/flare/projects/DFTCalc2/Mehul/install_DFTFE`

1. **Pull `ase_redesign`** into `dftfe/src`. It is 0 commits behind pGD and 120
   ahead as of 2026-08-08, so no merge is needed.
2. **Rebuild all four binaries** (`real`, `complex`, `cpu_real`, `cpu_complex`).
   `dftfeWrapper.cc` changed, so stale binaries will not pick this up — and every
   binary currently on Aurora still contains the old `sed` injector.
   - GPU: `install_dftfe_pgd.sh --dftfe`, PBS wrapper `build_dftfe_pgd.pbs`
   - CPU: `install_dftfe_pgd_cpu.sh`, PBS wrapper `build_dftfe_pgd_cpu.pbs`
   - Builds go through `qsub`, not the login node.
3. Confirm `helpers/parameterFile.prm` shipped with the build — the driver reads
   it from `DFTFE_PATH` at run time, and it is now **generated**. If it is stale
   or missing, injection silently falls back to defaults. Regenerate with
   `python3 helpers/gen_parameter_template.py`.
4. **Do not re-clone the pGD tree from Bitbucket.** The Aurora pGD source carries
   two local patches that are *not* upstream — `26ea7ec9b8` and `79e0c486a0` in
   `src/atom/AtomicCenteredNonLocalOperator.cc`, which fix buffer sizing for CPU
   builds with multiple k-points. A clean re-clone of `f0157ef4a` drops them
   silently and CPU-complex comparisons stop being like-for-like. See
   `PROJECT_Aurora.md` §3.

Nothing else in the build configuration changes. `HIGHERQUAD_PSP=ON`,
`GPU_LANG=sycl`, same dependency stack.

### 3.1 Deploying the study files

```bash
cd /lus/flare/projects/DFTCalc2/Mehul/install_DFTFE
mkdir -p runs/bccmo_431
cp dftfe/src/interfaces/ase-dftfe/tests/bccmo_431/* runs/bccmo_431/
cd runs/bccmo_431
```

The directory is self-contained: decks, `coordinates.inp`, `domainVectors.inp`,
`pseudo.inp` and `Mo_ONCV_PBE-1.0.upf` (md5 `e75ed155…`, identical to the
benchmark copy) all ship together. Nothing needs fetching from the
`dftfe-benchmarks` checkout.

Layout is flat on purpose: DFT-FE resolves `ATOMIC COORDINATES FILE` relative to
the **working directory**, so a deck only runs from a directory that also holds
the input files.

---

## 4. The scaling study

**Location:** `interfaces/ase-dftfe/tests/bccmo_431/`
Copy to `runs/bccmo_431/` on Aurora, per the existing convention.

**System:** BCC Mo, 6×6×6 supercell, one vacancy at fractional (0.5, 0.5, 0.5).
431 atoms, ideal lattice, nothing relaxed. 35.7 Bohr cell. GGA-PBE, Γ-point,
real binary. 3600 Kohn-Sham states. SCF tolerance 1e-4.

### 4.1 Files

| File | Purpose |
|---|---|
| `make_inputs.py` | Generates `coordinates.inp`, `domainVectors.inp`, `pseudo.inp` |
| `make_decks.py` | Generates `parameterFileAurora{16,32,64,128}Nodes.prm` |
| `ase_bccmo431.py` | ASE arm — contains **no** DFT parameters, reads the same deck |
| `run_native.pbs` / `run_ase.pbs` | PBS jobs |
| `submit.sh` | Sets `-l select` and `NODES` together so they cannot disagree |
| `compare.py` | Energy + per-component force parity, and the timing breakdown |

`make_inputs.py` reproduces the benchmark geometry rather than depending on the
`dftfe-benchmarks` checkout. Values and atom ordering match the published file
line for line; only line endings differ (benchmark is CRLF, generated is LF):

```
diff <(tr -d '\r' < <benchmark>/coordinates.inp) coordinates.inp   # empty
```

Ordering matters: force parity is compared component by component, so both arms
must list atoms identically. Building the cell with `ase.build.bulk` would give
correct physics in a different order and a meaningless force comparison. Both
arms read the same `coordinates.inp`, so this holds by construction.

### 4.2 Running

```bash
export DFTFE_INSTALL=/lus/flare/projects/DFTCalc2/Mehul/install_DFTFE/dftfe/install
export ASE_DFTFE_SRC=/lus/flare/.../dftfe/src/interfaces/ase-dftfe/src

python3 make_inputs.py .
python3 make_decks.py .

./submit.sh native 32     # one at a time: debug-scaling allows 1 queued job
./submit.sh ase    32

python3 compare.py --native native_32nodes.out \
                   --ase result_ase_32nodes.json \
                   --out parity_32nodes.json
```

### 4.3 Node counts and the rank ceiling

The mesh is **4913 FE cells / 1.73M DoFs** (from the Summit 32-node log). Ranks
that end up owning zero cells divide by zero in the cell-batch / ELPA grid setup
and die on SIGFPE — this is the same failure as `PROJECT_Aurora.md` §7.3, where
Li2O broke at ~0.8–1.3 cells/rank and was safe at ~2–3.

Aurora has 12 ranks/node (6 GPUs × 2 tiles):

| Nodes | Ranks | FE cells/rank | Note |
|---|---|---|---|
| 16 | 192 | 25.6 | same rank count as the published Summit 32-node point |
| 32 | 384 | 12.8 | same rank count as the published Summit 64-node point |
| 64 | 768 | 6.4 | |
| 128 | 1536 | **3.2** | thin — closest to the over-decomposition limit |

**Recommendation:** run 16 nodes first. It is the cheapest job and its 192 ranks
match a published Summit configuration, so it validates the Aurora build against
a known-good result before spending time at scale. Then 32 → 64 → 128.

If 128 nodes hits SIGFPE, that is the FE mesh limit for this system, not an
interface problem — report the curve up to 64 and say so.

### 4.4 Aurora is not Summit — read the scaling curve carefully

Summit had 6 ranks/node, Aurora has 12. Equal *node* counts are not equal *rank*
counts. The table above gives the rank-matched pairs. Absolute wall times will
not reproduce the published Summit numbers regardless, for two independent
reasons: the builds use `HIGHERQUAD_PSP=ON`, and `SPECTRUM SPLIT CORE
EIGENSTATES = 2400` no longer exists so that optimisation is simply gone.

Neither affects the native-vs-ASE comparison — both arms run the same settings.
`PROJECT_Aurora.md` §5.5 has been reworded accordingly: the goal is reproducing
the Summit scaling *shape*, not the absolute values.

### 4.5 `SCALAPACKPROCS`

The decks use `0` (DFT-FE's documented thumb rule). The Summit deck hard-codes
50 at 192 ranks and 30 at 384; those were tuned for Summit's topology, the grid
is capped by rank count anyway, and the parameter does not affect parity because
both arms use the same value. Change `SCALAPACK_PROCS` at the top of
`make_decks.py` and regenerate if you want the published values instead.

### 4.6 Pass criteria

Per PI: energy ≥10 significant digits, max absolute per-component force
difference ≥8 digits. `compare.py` defaults to those and exits non-zero on
failure.

Force "digits" follows the convention already used for the accepted results
(`PROJECT_Aurora.md` §5.1): `-log10` of the max absolute component deviation in
Ha/Bohr, so 3.96e-8 reads as ~7.4 digits. Note that BCC Mo landed at 7.4 and was
accepted by the PI despite the stated threshold being 8 — so a marginal force
result here is a conversation, not an automatic failure. `compare.py` also
reports the deviation relative to the largest force in the system as context.

---

## 5. What has and has not been verified

**Verified locally:**

- 67 Python tests pass (53 pre-existing, 14 new).
- `dftfeWrapper.cc` compiles clean for both `real` and `complex`
  (`build_merged/release/{real,complex}`).
- `PrmFile` unit-tested standalone against the real template: the Poisson/SCF
  collision, the `MAXIMUM ITERATIONS HELMHOLTZ` prefix hazard, and nested
  subsection targeting all behave.
- Full pipeline on the real Summit deck: 42/42 entries injected, 0 unresolved,
  all values in the correct subsections, geometry untouched, 5 legacy keys
  warned about.
- Generated Aurora decks: all 49 keys resolve against the current build; read
  with **zero** warnings; `read_dftfe_atoms` returns Mo431, 18.891626 Å cell,
  vacancy present.
- `compare.py` parses the real Summit native log (431 forces, correct energy).

**Not verified — needs Aurora:**

- Nothing has actually run. No DFT-FE execution, native or ASE.
- The PBS scripts are syntax-checked only; `#PBS -A DFTCalc2` and the
  `filesystems=flare:home` line should be sanity-checked against a working job.
- The 128-node point may hit the FE mesh limit (§4.3).
- ELPA GPU kernels are enabled in the decks; if the Aurora ELPA build does not
  support them, set `USE ELPA GPU KERNEL = false` (the Summit 64-node deck did
  exactly this).

---

## 6. Open items

Split by who has to do them.

### Needs a human decision (not an agent)

- [ ] `helpers/parameterFile.prm.pre-autogen` — byte-identical to git HEAD, so
      pure redundancy. Commit or drop?
- [ ] `SCALAPACKPROCS`: stays `0` (auto), or takes the Summit values (50/30)?
- [ ] Node set: 16/32/64/128 as written, or rank-matched to Summit?
- [ ] Should the two `AtomicCenteredNonLocalOperator.cc` fixes go upstream to
      Bitbucket rather than staying a local carry on Aurora? (§3 item 4)

### Mechanical — an agent on Aurora can do these

- [ ] Pull `ase_redesign`, rebuild all four binaries via `qsub` (§3)
- [ ] Deploy `runs/bccmo_431/` (§3.1)
- [ ] Sanity-check `#PBS -A DFTCalc2` and `filesystems=flare:home` against a
      known-good job before the first submit
- [ ] Run 16 nodes first, both arms, then `compare.py`; then 32 → 64 → 128
- [ ] If ELPA GPU kernels are unsupported by the Aurora ELPA build, set
      `USE ELPA GPU KERNEL = false` in `make_decks.py` and regenerate
- [ ] If 128 nodes SIGFPEs, report the curve up to 64 and record it as the FE
      mesh limit — not an interface failure (§4.3)
- [ ] Update `PROJECT_Aurora.md` §5.5 with the results as they land

### Already done (do not redo)

- `PROJECT_Aurora.md` §5.5 has been rewritten; §7.5 added for the injection
  defects; §3 corrected for the branch state and the local pGD patches; open
  items 10–12 added.
