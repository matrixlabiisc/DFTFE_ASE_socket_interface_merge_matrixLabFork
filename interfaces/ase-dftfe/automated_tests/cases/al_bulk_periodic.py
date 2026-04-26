"""
Test case: FCC Al bulk (3D periodic, k-points → complex binary).
Returns (energy_ha, forces=None, stress_voigt).
"""

from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8):
    """
    Parameters
    ----------
    dftfe_bin    : path to the complex DFT-FE binary
    psp_library  : path to directory containing Al.upf
    np_tasks     : number of MPI tasks

    Returns
    -------
    energy_ha   : float (Hartree)
    forces      : None (not computed for this test)
    stress      : np.ndarray shape (6,) in Voigt notation (eV/Å³)
    """
    import os

    # --- geometry: FCC Al conventional cell ---
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
        symbols=["Al", "Al", "Al", "Al"],
        positions=positions_ang,
        cell=cell_ang,
        pbc=[True, True, True],
    )

    # --- calculator ---
    psp_dict = {"Al": os.path.join(psp_library, "Al.upf")}
    run_cmd = f"mpirun -np {np_tasks} {dftfe_bin}"

    calc = DFTFE(
        command=run_cmd,
        psp_path=psp_dict,
        mesh_size=1.6,
        polynomial_order=5,
        tolerance=5e-5,
        fermi_temp=500.0,
        xc="GGA-PBE",
        use_time_reversal_symmetry=True,
        mp_grid=(2, 2, 2),
        mp_grid_shift=(1, 1, 1),
        npkpt=2,
        compute_forces=False,
        compute_stress=True,
        use_device=True,
        verbosity=1,
        log_file="test_al_bulk_periodic.log",
    )

    atoms.calc = calc
    energy_eV = atoms.get_potential_energy()
    stress = atoms.get_stress()

    energy_ha = energy_eV / Hartree

    return energy_ha, None, stress
