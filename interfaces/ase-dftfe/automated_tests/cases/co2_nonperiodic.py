"""
Test case: CO2 molecule (non-periodic, gamma-only → real binary).
Returns (energy_ha, forces_ha_per_bohr, stress=None).
"""

from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8):
    """
    Parameters
    ----------
    dftfe_bin    : path to the real DFT-FE binary
    psp_library  : path to directory containing C.upf and O.upf
    np_tasks     : number of MPI tasks

    Returns
    -------
    energy_ha              : float (Hartree)
    forces_ha_per_bohr     : np.ndarray shape (N,3)
    stress                 : None (not computed)
    """
    import os

    # --- geometry ---
    box_dims_bohr = np.array([40.0, 40.0, 40.0])
    cell_ang = np.diag(box_dims_bohr) * Bohr
    center = box_dims_bohr / 2.0
    bond_bohr = 2.19
    rel_pos_bohr = np.array([
        [0.0, 0.0, 0.0],
        [-bond_bohr, 0.0, 0.0],
        [ bond_bohr, 0.0, 0.0],
    ])
    pos_ang = (center + rel_pos_bohr) * Bohr

    atoms = Atoms(
        symbols=["C", "O", "O"],
        positions=pos_ang,
        cell=cell_ang,
        pbc=[False, False, False],
    )

    # --- calculator ---
    psp_dict = {
        "C": os.path.join(psp_library, "C.upf"),
        "O": os.path.join(psp_library, "O.upf"),
    }
    run_cmd = f"mpirun -np {np_tasks} {dftfe_bin}"

    calc = DFTFE(
        command=run_cmd,
        psp_path=psp_dict,
        mesh_size=1.0,
        polynomial_order=6,
        atom_ball_radius=3.0,
        tolerance=5e-5,
        scf_mixing=0.5,
        fermi_temp=500.0,
        num_bands=20,
        xc="GGA-PBE",
        compute_forces=True,
        compute_stress=False,
        use_device=True,
        verbosity=1,
        log_file="test_co2_nonperiodic.log",
    )

    atoms.calc = calc
    energy_eV = atoms.get_potential_energy()
    forces_eV_per_ang = atoms.get_forces()

    energy_ha = energy_eV / Hartree
    forces_ha_per_bohr = forces_eV_per_ang / (Hartree / Bohr)

    return energy_ha, forces_ha_per_bohr, None
