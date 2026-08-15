#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Generate references.json from **native pGD**, not from the ASE interface.

A reference seeded from the interface's own output pins whatever the interface
currently does, bugs included -- it can only ever detect *change*, never
*wrongness*. Generating the same numbers with the native file-in/file-out driver
of upstream DFT-FE turns the suite into what ctest is: a comparison against a
trusted independent implementation of the same calculation.

For each case this script

  1. captures the exact ``Atoms`` and DFT kwargs the case would have used, by
     substituting a recorder for :class:`DFTFE` and letting the case run until
     its first ``calculate()`` -- so the geometry and parameters are the case's
     own, never a hand-copy that can drift out of sync;
  2. writes a native deck: ``helpers/parameterFile.prm`` (every declared key at
     its declared default) with those kwargs injected subsection-aware, plus
     ``coordinates.inp`` / ``domainVectors.inp`` / ``pseudo.inp``;
  3. runs native pGD on it and parses energy and forces;
  4. writes references.json, recording that pGD produced them.

Nothing here touches the ASE socket path. That is the point: the reference must
come from somewhere the interface cannot influence.

    python make_references.py --pgd-real .../dftfe_pgd/install/cpu_real/dftfe \
                              --pgd-complex .../dftfe_pgd/install/cpu_complex/dftfe \
                              --psp-library ../psp_library --np 8 [--dry-run]

**Build the pGD binaries from current source first.** They are compared at
1e-10 Ha, and a binary older than its own source is the documented way to
manufacture a fake discrepancy: cpu_real here was three weeks older than the
CPU projector-aliasing fix in its own tree until 2026-08-13.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from ase.units import Bohr, Hartree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # dftfe/src
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from dftfe_ase.backends.file import FileBackend          # noqa: E402
from dftfe_ase.geometry import rigid_shift               # noqa: E402
from dftfe_ase.params import to_prm_entries              # noqa: E402

TEMPLATE = os.path.join(ROOT, "helpers", "parameterFile.prm")
CASES_DIR = os.path.join(HERE, "cases")
REFS_FILE = os.path.join(HERE, "references.json")
# GPU references live in their own file: a reference file holds one configuration
# per case, and use_device is part of the provenance gate, so a GPU number stored
# where the CPU one lives would make every CPU run report MISMATCH. See
# test_suite.refs_file_for.
GPU_REFS_FILE = os.path.join(HERE, "references.gpu.json")

# Cases whose ASE arm is a multi-step relaxation. Native DFT-FE relaxes with its
# own LBFGS and the ASE arm with ASE's, matched on history and max step, so a
# relaxed geometry is still not comparable at 1e-10 Ha -- two implementations of
# one algorithm take different paths to different minima-adjacent points. The native
# reference for these is therefore the single point at the INITIAL geometry,
# which is exactly what the recorder captures anyway.
INITIAL_POINT_ONLY = {"relax_o2"}


class _Captured(Exception):
    """Raised once the recorder has what it needs, to stop the case early."""


def _recorder(store):
    class Recorder:
        def __init__(self, **kwargs):
            store["kwargs"] = kwargs
            self.results = {}

        def get_potential_energy(self, atoms=None, **_):
            store["atoms"] = atoms.copy() if atoms is not None else None
            raise _Captured

        # ASE reaches the calculator through several entry points depending on
        # whether the case asks for energy, forces, or drives an optimizer.
        get_forces = get_stress = get_property = get_potential_energy

        # Cases written as `with DFTFE(...) as calc:` need these explicitly --
        # the protocol is looked up on the type, so __getattr__ never sees it.
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __getattr__(self, name):          # tolerate anything else the case does
            return lambda *a, **k: None

    return Recorder


def capture(name):
    """The Atoms and DFT kwargs a case would have used, taken from the case itself."""
    path = os.path.join(CASES_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"case_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    store = {}
    mod.DFTFE = _recorder(store)
    try:
        mod.run(dftfe_bin="/nonexistent", psp_library=os.path.join(HERE, "..", "psp_library"),
                np_tasks=1, use_device=False)
    except _Captured:
        pass
    if "atoms" not in store or "kwargs" not in store:
        raise RuntimeError(f"{name}: could not capture geometry/params")
    return store["atoms"], store["kwargs"]


def z_valence(upf_path):
    """Valence electron count, read the way dftfeWrapper::reinit reads it."""
    with open(upf_path, errors="replace") as fh:
        for line in fh:
            if "z_valence=" in line:
                return int(round(float(line.split()[1].strip('"'))))
    raise ValueError(f"no z_valence in {upf_path}")


def write_prm(entries, dest):
    """Apply {section,key,value} to the template, matching innermost subsection.

    Same rule as internalWrapper::PrmFile on the C++ side: track the subsection
    stack, replace an existing entry, and report anything the template does not
    declare rather than inventing it.
    """
    lines = open(TEMPLATE).read().splitlines()
    pending = {(e["section"] or "", e["key"]): e["value"] for e in entries}
    stack, out = [], []
    for line in lines:
        s = line.strip()
        if s.startswith("subsection"):
            stack.append(s[len("subsection"):].strip())
        elif s == "end" and stack:
            stack.pop()
        elif s.startswith("set"):
            key = s[3:].split("=")[0].strip()
            sec = stack[-1] if stack else ""
            if (sec, key) in pending:
                val = pending.pop((sec, key))
                indent = line[:len(line) - len(line.lstrip())]
                line = f"{indent}set {key}={val}"
        out.append(line)
    open(dest, "w").write("\n".join(out) + "\n")
    return pending  # anything left over was not declared in this build


def _binary_id(path):
    """Identify a binary without leaking whose directory it sits in.

    ``/lus/flare/projects/.../Mehul/install_DFTFE/dftfe_pgd/install/cpu_real/dftfe``
    says which build was used, and also publishes one person's scratch directory
    into a file that ships with the package. The last two components keep the
    useful half (``cpu_real/dftfe``, ``complex/dftfe``) and drop the rest; the
    identity check that actually matters is ``binary_stamp`` (size:mtime), which
    is unchanged. Nothing gates on this field -- see CONFIG_KEYS.
    """
    parts = os.path.normpath(path).split(os.sep)
    return os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]


def build_deck(work, atoms, kwargs, psp_library, compute_forces=True, relax=False,
               force_tol=None, use_device=False):
    os.makedirs(work, exist_ok=True)
    cell_bohr = atoms.get_cell()[:] / Bohr
    pos_bohr = atoms.get_positions() / Bohr
    symbols = atoms.get_chemical_symbols()
    uniq = sorted(set(symbols), key=symbols.index)

    # pseudo.inp + the .upf files themselves
    psp = kwargs.get("psp_path") or {}
    with open(os.path.join(work, "pseudo.inp"), "w") as fh:
        for sym in uniq:
            src = psp.get(sym) or os.path.join(psp_library, f"{sym}.upf")
            shutil.copy(src, work)
            fh.write(f"{atoms[symbols.index(sym)].number} {os.path.basename(src)}\n")

    valence = {s: z_valence(os.path.join(work, os.path.basename(
        psp.get(s) or os.path.join(psp_library, f"{s}.upf")))) for s in uniq}

    np.savetxt(os.path.join(work, "domainVectors.inp"), cell_bohr, fmt="%.16f")

    # The native arm must compute at the geometry the SOCKET arm actually sends,
    # otherwise the two are solving different structures and the comparison is
    # meaningless. dftfeWrapper::reinit does two things to incoming positions,
    # and both have to be mirrored here:
    #
    #   * open axes    -- rigid placement inside the cell (geometry.rigid_shift);
    #                     zero unless a case is deliberately out of cell.
    #   * periodic axes -- fold into [0,1]. The native file path does NOT wrap;
    #                     it asserts (dft.cc:1101). al_slab_semiperiodic has a
    #                     fractional x of -0.1667 straight out of fcc111() and
    #                     aborted here until this was added.
    pbc = atoms.get_pbc()
    pos_bohr = pos_bohr + rigid_shift(pos_bohr, cell_bohr, pbc)
    if pbc.any():
        rows = pos_bohr @ np.linalg.inv(cell_bohr)
        for i in range(3):
            if pbc[i]:
                rows[:, i] = np.mod(rows[:, i], 1.0)
    else:
        rows = pos_bohr - cell_bohr.sum(axis=0) / 2.0   # origin at domain centre
    with open(os.path.join(work, "coordinates.inp"), "w") as fh:
        for sym, r in zip(symbols, rows):
            z = atoms[symbols.index(sym)].number
            fh.write(f"{z} {valence[sym]} {r[0]:.16f} {r[1]:.16f} {r[2]:.16f}\n")

    # DFT parameters: the case's own kwargs, plus the geometry/driver entries
    # the calculator would otherwise have supplied.
    entries = to_prm_entries({k: v for k, v in kwargs.items() if v is not None})
    entries += [
        {"section": "", "key": "SOLVER MODE", "value": "GEOOPT" if relax else "GS"},
        {"section": "", "key": "VERBOSITY", "value": kwargs.get("verbosity", 2)},
        {"section": "Geometry", "key": "NATOMS", "value": len(atoms)},
        {"section": "Geometry", "key": "NATOM TYPES", "value": len(uniq)},
        {"section": "Geometry", "key": "ATOMIC COORDINATES FILE", "value": "coordinates.inp"},
        {"section": "Geometry", "key": "DOMAIN VECTORS FILE", "value": "domainVectors.inp"},
        {"section": "DFT functional parameters",
         "key": "PSEUDOPOTENTIAL FILE NAMES LIST", "value": "pseudo.inp"},
        {"section": "Boundary conditions", "key": "PERIODIC1", "value": str(bool(pbc[0])).lower()},
        {"section": "Boundary conditions", "key": "PERIODIC2", "value": str(bool(pbc[1])).lower()},
        {"section": "Boundary conditions", "key": "PERIODIC3", "value": str(bool(pbc[2])).lower()},
        {"section": "Optimization", "key": "ION FORCE",
         "value": "true" if compute_forces else "false"},
        {"section": "Optimization", "key": "CELL STRESS",
         "value": "true" if kwargs.get("compute_stress") else "false"},
    ]
    if relax:
        # Match ASE's stopping criterion, or the two optimizers are not being
        # asked the same question. ASE LBFGS uses fmax in eV/Ang; DFT-FE FORCE TOL
        # is Ha/Bohr, and its 1e-4 default is NOT the same number as the 0.01
        # eV/Ang the cases use (= 1.9447e-4 Ha/Bohr).
        entries += [
            {"section": "Optimization", "key": "OPTIMIZATION MODE", "value": "ION"},
            {"section": "Optimization", "key": "FORCE TOL",
             "value": f"{force_tol:.12e}"},
        ]
    dest = os.path.join(work, "parameterFile.prm")
    undeclared = write_prm(entries, dest)

    # USE GPU is driver-owned: runParameters.cc declares it, dftParameters does
    # not, and helpers/parameterFile.prm is generated from dftParameters alone --
    # so it is not in the template and write_prm (which only replaces declared
    # entries) cannot inject it. Prepend it as a top-level line instead. Safe
    # because the native driver parses with skip_undefined=true
    # (runParameters.cc:170), which is also why real decks can carry both
    # parameter sets in one file.
    if use_device:
        body = open(dest).read()
        open(dest, "w").write("set USE GPU=true\n" + body)
    return undeclared


def native_cmd(args, binary):
    """The launch command for one native run.

    On GPUs the ranks have to be pinned one per tile with Aurora's
    gpu_tile_compact.sh -- without it every rank on the node lands on tile 0,
    which is the convention every GPU run in runs/ already follows. On the host
    this stays the plain mpirun the CPU references were generated with, so those
    numbers remain reproducible by the same command that made them.
    """
    if not args.use_device:
        return ["mpirun", "-np", str(args.np), binary, "parameterFile.prm"]
    return ["mpiexec", "-n", str(args.np), "--ppn", str(args.ppn),
            args.gpu_wrapper, binary, "parameterFile.prm"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgd-real", required=True)
    p.add_argument("--pgd-complex", required=True)
    p.add_argument("--psp-library", default=os.path.join(HERE, "..", "psp_library"))
    p.add_argument("--np", type=int, default=8)
    p.add_argument("--workroot", default=os.path.join(HERE, "native_refs"))
    p.add_argument("--tests", nargs="*")
    p.add_argument("--dry-run", action="store_true",
                   help="write the decks and stop; run nothing")
    p.add_argument("--case-timeout", type=int, default=900,
                   help="seconds per native run before skipping it (default 900)")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave cases that already carry a native-pgd reference")
    p.add_argument("--use-device", action="store_true",
                   help="generate GPU references: injects USE GPU=true, launches "
                        "one rank per GPU tile, and writes references.gpu.json")
    p.add_argument("--ppn", type=int, default=12,
                   help="ranks per node for GPU runs (default 12 = one per tile)")
    p.add_argument("--gpu-wrapper", default="gpu_tile_compact.sh",
                   help="rank-to-tile binding wrapper for GPU runs")
    p.add_argument("--parse-only", action="store_true",
                   help="re-parse the native.out already in --workroot instead of "
                        "running DFT-FE. Use to extract a quantity a completed "
                        "run already printed (e.g. CELL STRESS) without spending "
                        "node time or moving the reference to a different run.")
    p.add_argument("--relax", action="store_true",
                   help="for INITIAL_POINT_ONLY cases, ALSO run a native GEOOPT "
                        "and record the relaxed energy under 'relaxed'")
    p.add_argument("--fmax-ev-ang", type=float, default=0.01,
                   help="ASE fmax the relaxation cases use; converted to FORCE TOL")
    args = p.parse_args()

    sys.path.insert(0, HERE)
    from test_suite import BINARY_KIND  # noqa: E402

    names = args.tests or list(BINARY_KIND)
    bins = {"real": args.pgd_real, "complex": args.pgd_complex}
    refs_file = GPU_REFS_FILE if args.use_device else REFS_FILE
    refs = json.load(open(refs_file)) if os.path.exists(refs_file) else {}
    print(f"writing to {os.path.basename(refs_file)} "
          f"(use_device={args.use_device}, np={args.np}"
          + (f", ppn={args.ppn}" if args.use_device else "") + ")")

    for name in names:
        kind = BINARY_KIND[name]
        binary = bins[kind]
        work = os.path.join(args.workroot, name)
        print(f"\n=== {name} ({kind}) ===", flush=True)

        if args.skip_existing and (refs.get(name, {}).get("provenance") or {}
                                   ).get("generator") == "native-pgd":
            print("  already has a native pGD reference; skipping")
            continue

        atoms, kwargs = capture(name)
        print(f"  captured {len(atoms)} atoms, pbc={list(atoms.get_pbc())}, "
              f"{len(kwargs)} kwargs")
        if name in INITIAL_POINT_ONLY:
            print("  NOTE: multi-step case -- native reference is the single point "
                  "at the initial geometry (optimizer paths are not comparable)")

        undeclared = build_deck(work, atoms, kwargs, args.psp_library,
                                use_device=args.use_device)
        if undeclared:
            print(f"  WARNING: not declared in this build, dropped: "
                  f"{sorted(undeclared)}")
        print(f"  deck written to {work}")
        if args.dry_run:
            continue

        st = os.stat(binary)

        # Re-parse an output this script already produced, instead of running
        # DFT-FE again. Added to pin CELL STRESS: job 8753802's native.out
        # already contained the stress block (the deck asks for it whenever the
        # case does), it was simply never parsed. Extracting a quantity that is
        # already in a completed run costs no node time, and re-running would
        # have changed nothing except the rank-order noise floor -- the reference
        # would no longer be the run the rest of the file documents.
        if args.parse_only:
            out_path = os.path.join(work, "native.out")
            if not os.path.exists(out_path):
                print(f"  --parse-only: no {out_path}; skipped")
                continue
            text = open(out_path, errors="replace").read()
            elapsed = 0.0
            proc = None
        else:
            cmd = native_cmd(args, binary)
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=work, capture_output=True,
                                      text=True, timeout=args.case_timeout)
            except subprocess.TimeoutExpired as exc:
                # TimeoutExpired carries RAW BYTES even under text=True --
                # decoding happens after communicate() returns, which it never
                # did. Writing str + bytes here raised TypeError and killed the
                # whole script, losing the three cases queued behind this one
                # (job 8753495).
                def _txt(v):
                    return v.decode(errors="replace") if isinstance(v, bytes) else (v or "")
                open(os.path.join(work, "native.out"), "w").write(
                    _txt(exc.stdout) + "\n" + _txt(exc.stderr))
                print(f"  TIMEOUT after {args.case_timeout}s -- skipped. Raise "
                      f"--case-timeout, or check this case's mesh: it is the "
                      f"expensive knob (relax_o2 uses MESH SIZE 0.6 / order 7).")
                continue
            elapsed = time.time() - t0
            text = proc.stdout + "\n" + proc.stderr
            open(os.path.join(work, "native.out"), "w").write(text)

        try:
            energy, forces = FileBackend._parse(text, natoms=len(atoms))
        except Exception as exc:
            print(f"  FAILED to parse native output after {elapsed:.0f}s: {exc}")
            print(f"  see {work}/native.out")
            continue

        # Stress only exists when the case asked for it (build_deck sets CELL
        # STRESS from the case's own compute_stress kwarg). None is recorded as
        # None, and test_suite.py treats "case computes stress, reference has
        # none" as a failure rather than skipping the check in silence.
        stress = FileBackend._parse_stress(text)

        print(f"  native energy = {energy:.12f} Ha"
              + (f", {len(forces)} force rows" if forces else ", no forces parsed")
              + (", stress 3x3" if stress else "")
              + f"  [{elapsed:.0f}s]")
        refs[name] = {
            "energy_ha": energy,
            "forces_ha_per_bohr": forces,
            "stress_ha_per_bohr3": stress,
            "_source": "native pGD via make_references.py",
            "provenance": {
                "generator": "native-pgd",
                "binary": _binary_id(binary),
                "binary_stamp": f"{st.st_size}:{int(st.st_mtime)}",
                "np": args.np,
                "use_device": bool(args.use_device),
            },
        }
        # Relaxed reference for multi-step cases. The single point above pins
        # the interface at 1e-10; this pins where the relaxation lands, which is
        # a physics comparison between two DIFFERENT LBFGS implementations
        # (DFT-FE's and ASE's) and cannot be expected to hold at 1e-10. Its achievable
        # tolerance is measured, not assumed -- that is what this records.
        if args.relax and name in INITIAL_POINT_ONLY:
            ftol = args.fmax_ev_ang / (Hartree / Bohr)
            rwork = work + "_relax"
            if not args.parse_only:
                build_deck(rwork, atoms, kwargs, args.psp_library,
                           relax=True, force_tol=ftol,
                           use_device=args.use_device)
            print(f"  native GEOOPT (FORCE TOL {ftol:.6e} Ha/Bohr) ...", flush=True)
            t1 = time.time()
            try:
                if args.parse_only:
                    rtext = open(os.path.join(rwork, "native.out"),
                                 errors="replace").read()
                else:
                    rp = subprocess.run(native_cmd(args, binary),
                                        cwd=rwork, capture_output=True, text=True,
                                        timeout=args.case_timeout)
                    rtext = rp.stdout + "\n" + rp.stderr
                    open(os.path.join(rwork, "native.out"), "w").write(rtext)
                re_ha, rf = FileBackend._parse(rtext, natoms=len(atoms))
                nsteps = len(re.findall(r"Ion position updates", rtext)) or None
                refs[name]["relaxed"] = {
                    "energy_ha": re_ha, "forces_ha_per_bohr": rf,
                    "force_tol_ha_bohr": ftol, "native_steps": nsteps,
                    "_note": ("native DFT-FE LBFGS (HISTORY 5, MAX ION UPDATE "
                              "STEP 0.5 a.u.); the ASE arm runs LBFGS matched to "
                              "it (memory=5, maxstep=0.5*Bohr). Two LBFGS "
                              "implementations still differ in line search and "
                              "initial Hessian scaling, so this is NOT a 1e-10 "
                              "comparison."),
                }
                print(f"  native relaxed = {re_ha:.12f} Ha  "
                      f"[{time.time()-t1:.0f}s]")
            except subprocess.TimeoutExpired:
                print(f"  GEOOPT TIMEOUT after {args.case_timeout}s -- "
                      f"relaxed reference not recorded")
            except Exception as exc:
                print(f"  GEOOPT failed: {exc}")

        # Written after EVERY case, not at the end. The first run of this script
        # completed co2 and n2, then hit the job walltime on relax_o2 -- and
        # discarded both, because the only write was after the loop.
        json.dump(refs, open(refs_file, "w"), indent=2)
        print(f"  -> {name} saved to {os.path.basename(refs_file)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
