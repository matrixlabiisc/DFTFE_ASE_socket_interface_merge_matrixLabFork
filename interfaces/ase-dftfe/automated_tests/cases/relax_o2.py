# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""
Test case: O2 relaxation (non-periodic → real binary).
Mirrors exactly: examples/relax_o2.py
"""
import os
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.optimize import BFGS
from dftfe_ase import DFTFE


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8,
        use_device: bool = True):
    # --- geometry (exactly as in examples/relax_o2.py) ---
    box_dims_bohr = np.array([40.0, 42.0, 38.0])
    cell_ang = np.diag(box_dims_bohr) * Bohr
    center = box_dims_bohr / 2.0
    rel_pos_bohr = np.array([
        [-1.2, 0.0, 0.0],
        [ 1.2, 0.0, 0.0],
    ])
    pos_ang = (center + rel_pos_bohr) * Bohr

    atoms = Atoms(
        symbols=['O', 'O'],
        positions=pos_ang,
        cell=cell_ang,
        pbc=[False, False, False],
    )

    # --- calculator (exactly as in examples/relax_o2.py) ---
    psp_dict = {"O": os.path.join(psp_library, "O.upf")}
    run_cmd = f"mpirun -np {np_tasks} {dftfe_bin}"

    calc = DFTFE(
        command=run_cmd,
        psp_path=psp_dict,
        mesh_size=0.6,
        polynomial_order=7,
        atom_ball_radius=6.0,
        tolerance=1e-6,
        scf_mixing=0.5,
        fermi_temp=500.0,
        num_bands=16,
        xc='GGA-PBE',
        compute_forces=True,
        compute_stress=False,
        use_device=use_device,
        verbosity=1,
        log_file="test_relax_o2.log",
    )

    atoms.calc = calc
    opt = BFGS(atoms, logfile="test_relax_o2_opt.log")
    opt.run(fmax=0.01)

    # The relaxed energy is pinned like any other, not merely run-only. At fixed
    # binary, rank count and device -- which the suite's provenance gate now
    # enforces -- BFGS is deterministic: identical forces give an identical step
    # sequence and an identical final geometry. So this case pins the whole
    # multi-step path (persistent socket, density reuse across steps, optimizer
    # round-trip), which a single-point case cannot reach.
    #
    # nsteps is reported alongside because it is the most sensitive signal here.
    # A change that perturbs forces below the energy tolerance can still move the
    # step count, and that shows up before the energy does.
    energy_ha = atoms.get_potential_energy() / Hartree
    forces_ha_per_bohr = atoms.get_forces() / (Hartree / Bohr)
    print(f"[relax_o2] converged in {opt.get_number_of_steps()} BFGS steps",
          flush=True)

    return energy_ha, forces_ha_per_bohr, None
