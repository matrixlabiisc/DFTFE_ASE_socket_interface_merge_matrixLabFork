"""
Test case: Graphene (2D periodic, k-points → complex binary).
Mirrors exactly: examples/graphene_gs.py
"""
import os
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from dftfe import DFTFE


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8):
    # --- geometry (exactly as in examples/graphene_gs.py) ---
    cell_bohr = np.array([
        [ 4.65428900,  0.00000000, 0.0],
        [-2.32714450,  4.03073251, 0.0],
        [ 0.0,         0.0,       50.0],
    ])
    cell_ang = cell_bohr * Bohr

    frac_positions = np.array([
        [0.0000000000, 0.0000000000, 0.5],
        [0.3333333333, 0.6666666667, 0.5],
    ])
    positions_ang = frac_positions @ cell_ang

    atoms = Atoms(
        symbols=['C', 'C'],
        positions=positions_ang,
        cell=cell_ang,
        pbc=[True, True, False],
    )

    # --- calculator (exactly as in examples/graphene_gs.py) ---
    psp_dict = {"C": os.path.join(psp_library, "C.upf")}
    run_cmd = f"mpirun -np {np_tasks} {dftfe_bin}"

    calc = DFTFE(
        command=run_cmd,
        psp_path=psp_dict,
        mesh_size=0.5,
        polynomial_order=3,
        tolerance=5e-5,
        fermi_temp=500.0,
        use_time_reversal_symmetry=True,
        mp_grid=(4, 4, 1),
        mp_grid_shift=(1, 1, 0),
        npkpt=8,
        xc='GGA-PBE',
        compute_forces=True,
        use_device=True,
        verbosity=1,
        log_file="test_graphene_periodic.log",
    )

    atoms.calc = calc
    energy_ha = atoms.get_potential_energy() / Hartree
    forces_ha_per_bohr = atoms.get_forces() / (Hartree / Bohr)

    return energy_ha, forces_ha_per_bohr, None
