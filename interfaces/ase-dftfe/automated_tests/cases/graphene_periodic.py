"""
Test case: Graphene (2D periodic, k-points → complex binary).
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
    dftfe_bin    : path to the complex DFT-FE binary
    psp_library  : path to directory containing C.upf
    np_tasks     : number of MPI tasks

    Returns
    -------
    energy_ha              : float (Hartree)
    forces_ha_per_bohr     : np.ndarray shape (N,3)
    stress                 : None (not computed for this test)
    """
    import os

    # --- geometry: graphene primitive cell ---
    # Lattice constant a = 2.461 Angstrom
    a = 2.461
    cell = np.array([
        [ a,           0.0, 0.0],
        [-a / 2.0,  a * np.sqrt(3) / 2.0, 0.0],
        [ 0.0,          0.0, 26.458],  # ~50 Bohr vacuum
    ])
    basis = np.array([[0.0, 0.0, 0.5],
                      [1.0/3.0, 2.0/3.0, 0.5]])
    positions = basis @ cell

    atoms = Atoms(
        symbols=["C", "C"],
        positions=positions,
        cell=cell,
        pbc=[True, True, False],
    )

    # --- calculator ---
    psp_dict = {"C": os.path.join(psp_library, "C.upf")}
    run_cmd = f"mpirun -np {np_tasks} {dftfe_bin}"

    calc = DFTFE(
        command=run_cmd,
        psp_path=psp_dict,
        mesh_size=0.5,
        polynomial_order=3,
        tolerance=5e-5,
        fermi_temp=500.0,
        xc="GGA-PBE",
        mp_grid=(4, 4, 1),
        mp_grid_shift=(1, 1, 0),
        npkpt=8,
        compute_forces=True,
        compute_stress=False,
        use_device=True,
        verbosity=1,
        log_file="test_graphene_periodic.log",
    )

    atoms.calc = calc
    energy_eV = atoms.get_potential_energy()
    forces_eV_per_ang = atoms.get_forces()

    energy_ha = energy_eV / Hartree
    forces_ha_per_bohr = forces_eV_per_ang / (Hartree / Bohr)

    return energy_ha, forces_ha_per_bohr, None
