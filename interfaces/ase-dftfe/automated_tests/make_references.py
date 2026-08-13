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
from ase.units import Bohr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # dftfe/src
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from dftfe_ase.backends.file import FileBackend          # noqa: E402
from dftfe_ase.geometry import rigid_shift               # noqa: E402
from dftfe_ase.params import to_prm_entries              # noqa: E402

TEMPLATE = os.path.join(ROOT, "helpers", "parameterFile.prm")
CASES_DIR = os.path.join(HERE, "cases")
REFS_FILE = os.path.join(HERE, "references.json")

# Cases whose ASE arm is a multi-step relaxation. Native DFT-FE relaxes with its
# own optimizer, so a relaxed geometry is not comparable at 1e-10 Ha -- different
# optimizers take different paths to different minima-adjacent points. The native
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


def build_deck(work, atoms, kwargs, psp_library, compute_forces=True):
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
        {"section": "", "key": "SOLVER MODE", "value": "GS"},
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
    undeclared = write_prm(entries, os.path.join(work, "parameterFile.prm"))
    return undeclared


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
    args = p.parse_args()

    sys.path.insert(0, HERE)
    from test_suite import BINARY_KIND  # noqa: E402

    names = args.tests or list(BINARY_KIND)
    bins = {"real": args.pgd_real, "complex": args.pgd_complex}
    refs = json.load(open(REFS_FILE)) if os.path.exists(REFS_FILE) else {}

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

        undeclared = build_deck(work, atoms, kwargs, args.psp_library)
        if undeclared:
            print(f"  WARNING: not declared in this build, dropped: "
                  f"{sorted(undeclared)}")
        print(f"  deck written to {work}")
        if args.dry_run:
            continue

        st = os.stat(binary)
        cmd = ["mpirun", "-np", str(args.np), binary, "parameterFile.prm"]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                                  timeout=args.case_timeout)
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired carries RAW BYTES even under text=True -- decoding
            # happens after communicate() returns, which it never did. Writing
            # str + bytes here raised TypeError and killed the whole script,
            # losing the three cases queued behind this one (job 8753495).
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

        print(f"  native energy = {energy:.12f} Ha"
              + (f", {len(forces)} force rows" if forces else ", no forces parsed")
              + f"  [{elapsed:.0f}s]")
        refs[name] = {
            "energy_ha": energy,
            "forces_ha_per_bohr": forces,
            "_source": "native pGD via make_references.py",
            "provenance": {
                "generator": "native-pgd",
                "binary": os.path.abspath(binary),
                "binary_stamp": f"{st.st_size}:{int(st.st_mtime)}",
                "np": args.np,
                "use_device": False,
            },
        }
        # Written after EVERY case, not at the end. The first run of this script
        # completed co2 and n2, then hit the job walltime on relax_o2 -- and
        # discarded both, because the only write was after the loop.
        json.dump(refs, open(REFS_FILE, "w"), indent=2)
        print(f"  -> {name} saved to references.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
