#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""ASE arm of the BCC-Mo 431-atom scaling study.

Single-point ground state + ionic forces, driven through the ASE socket
calculator. The native arm runs the same deck through the file-based driver;
the two are compared by ``compare.py``.

The whole point is that this file contains **no** DFT parameters. It reads the
same ``.prm`` the native arm runs, so the two arms cannot drift apart -- there
is no second copy of the settings to keep in sync. ``read_dftfe`` also builds
the Atoms from the same ``coordinates.inp``, which keeps the atom ordering
identical; force parity is compared component by component and would be
meaningless otherwise.

Usage (inside a PBS job)::

    python3 ase_bccmo431.py --prm native/parameterFileAurora32Nodes.prm \\
                            --nodes 32 --bin-dir $INSTALL/real
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from ase.units import Bohr, Hartree

from dftfe_ase import read_dftfe

RANKS_PER_NODE = 12  # Aurora: 6 GPUs x 2 tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prm", required=True,
                    help="native deck; supplies every DFT parameter and the geometry")
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--ranks-per-node", type=int, default=RANKS_PER_NODE)
    ap.add_argument("--dftfe-real", default=os.environ.get("DFTFE_REAL"),
                    help="path to the real-valued dftfe binary")
    ap.add_argument("--out", default="result_ase.json")
    ap.add_argument("--connect-timeout", type=float, default=1800.0,
                    help="MPI launch of >1000 ranks off Lustre is not fast")
    args = ap.parse_args()

    nproc = args.nodes * args.ranks_per_node

    atoms, calc = read_dftfe(
        args.prm,
        cluster="aurora",
        nproc=nproc,
        # One rank per GPU tile. Without the wrapper every rank on a node lands
        # on tile 0 and the run is both wrong and slow.
        launcher_args=["--ppn", str(args.ranks_per_node), "gpu_tile_compact.sh"],
        dftfe_real=args.dftfe_real,
        use_device=True,
        debug_timing=True,
        connect_timeout=args.connect_timeout,
        log_file=f"dftfe_ase_{args.nodes}nodes.log",
    )
    atoms.calc = calc

    print(f"[ase] {len(atoms)} atoms, {nproc} ranks on {args.nodes} nodes", flush=True)
    print(f"[ase] cell (Ang):\n{np.array2string(atoms.get_cell()[:], precision=6)}",
          flush=True)

    t0 = time.perf_counter()
    energy_ev = atoms.get_potential_energy()
    forces_ev_ang = atoms.get_forces()
    wall = time.perf_counter() - t0

    # Report in DFT-FE's own units so the numbers can be diffed against the
    # native log without a conversion step in between.
    energy_ha = energy_ev / Hartree
    forces_ha_bohr = forces_ev_ang / (Hartree / Bohr)
    fmax = float(np.abs(forces_ha_bohr).max())

    result = {
        "nodes": args.nodes,
        "ranks": nproc,
        "natoms": len(atoms),
        "energy_ha": energy_ha,
        "max_force_ha_bohr": fmax,
        "forces_ha_bohr": forces_ha_bohr.tolist(),
        "wall_total_s": wall,
        "timing": calc.timing,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[ase] energy      = {energy_ha:.15f} Ha", flush=True)
    print(f"[ase] max |force| = {fmax:.6e} Ha/Bohr", flush=True)
    print(f"[ase] wall        = {wall:.2f} s", flush=True)
    if calc.timing:
        t = calc.timing
        print(f"[ase] timing      = {json.dumps(t)}", flush=True)
    print(f"[ase] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
