from ase import Atoms
from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt

from dftfe import DFTFE

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


# =========================
# System Setup (Cu FCC)
# =========================

structure = bulk("Cu", "fcc", a=3.6)
structure.pbc = True


# =========================
# DFT-FE Setup
# =========================

psp_path = "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_spms/29_Cu_19_1.7_1.9_pbe_n_v1.0.upf"
dftfe_bin = "/home/pa01/Mehul/DFTFE/build_gpu/release/real/dftfe"

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 16 {dftfe_bin}"
)

calc = DFTFE(
    command=run_cmd,
    host="localhost",
    port=0,

    # Mesh parameters
    mesh_size=1.0,
    polynomial_order=6,
    atom_ball_radius=3.0,

    # SCF parameters
    tolerance=1e-6,
    scf_mixing=0.3,
    fermi_temp=300.0,
    num_eigen_states=20,

    debug_timing=True,

    # XC functional
    xc="GGA-PBE",

    # GPU
    use_device=True,

    # Needed for phonons
    compute_forces=True,
    compute_stress=False,

    # Output
    keep_scratch=True,
    verbosity=1,
    log_file="phonon_cu.log",
)


# =========================
# Convert ASE → PhonopyAtoms
# =========================

phnpy_struct = PhonopyAtoms(
    symbols=structure.get_chemical_symbols(),
    positions=structure.get_positions(),
    cell=structure.get_cell(),
)


# =========================
# Phonopy Setup
# =========================

phonons = Phonopy(
    phnpy_struct,
    supercell_matrix=[
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
    ],
    primitive_matrix="auto",
)


# =========================
# Generate Displacements
# =========================

phonons.generate_displacements(distance=0.01)

supercells = phonons.supercells_with_displacements

print("Number of displaced supercells:", len(supercells))


# =========================
# Force Calculations
# =========================

sets_of_forces = []

for i, sc in enumerate(supercells):

    print(f"Running displacement {i+1}/{len(supercells)}")

    sc_ase = Atoms(
        symbols=sc.symbols,
        positions=sc.positions,
        cell=sc.cell,
        pbc=True,
    )

    sc_ase.calc = calc

    forces = sc_ase.get_forces()

    sets_of_forces.append(forces)


sets_of_forces = np.array(sets_of_forces)

phonons.forces = sets_of_forces


# =========================
# Compute Force Constants
# =========================

phonons.produce_force_constants()

phonons.save("phonopy_cu.yaml")


# =========================
# Compute Phonon Band Structure
# =========================

phonons.auto_band_structure()

fig = phonons.plot_band_structure()

fig.savefig("plot_phonopy_cu.png", dpi=300)

print("Phonon band structure saved as plot_phonopy_cu.png")