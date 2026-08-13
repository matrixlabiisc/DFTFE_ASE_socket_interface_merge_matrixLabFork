# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""
Test case: FCC Al bulk (3D periodic → complex binary).
Mirrors exactly: examples/bulk_fcc_al_gs.py
"""
import os
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from dftfe_ase import DFTFE


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8,
        use_device: bool = True):
    # --- geometry (exactly as in examples/bulk_fcc_al_gs.py) ---
    box_dims_bohr = np.array([7.6, 7.6, 7.6])
    cell_ang = np.diag(box_dims_bohr) * Bohr

    frac_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    positions_ang = frac_positions @ cell_ang

    atoms = Atoms(
        symbols=['Al', 'Al', 'Al', 'Al'],
        positions=positions_ang,
        cell=cell_ang,
        pbc=[True, True, True],
    )

    # --- calculator (exactly as in examples/bulk_fcc_al_gs.py) ---
    psp_dict = {"Al": os.path.join(psp_library, "Al.upf")}
    run_cmd = f"mpirun -np {np_tasks} {dftfe_bin}"

    calc = DFTFE(
        command=run_cmd,
        psp_path=psp_dict,
        mesh_size=1.6,
        polynomial_order=5,
        tolerance=5e-5,
        fermi_temp=500.0,
        use_time_reversal_symmetry=True,
        mp_grid=(2, 2, 2),
        mp_grid_shift=(1, 1, 1),
        npkpt=2,
        xc='GGA-PBE',
        compute_forces=False,
        compute_stress=True,
        use_device=use_device,
        verbosity=1,
        log_file="test_al_bulk_periodic.log",
    )

    atoms.calc = calc
    energy_ha = atoms.get_potential_energy() / Hartree
    stress = atoms.get_stress()   # eV/Å³ Voigt

    return energy_ha, None, stress
