from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE

# Note that this is just a test to show that the ASE interface can handle multi-element configurations with exact file paths!
# The CO2 example can be more complex, but it's not necessary for this test.
# =========================
# System Setup (CO2 Molecule)
# =========================

# domainVectors: 40 40 40 Bohr
box_dims_bohr = np.array([40.0, 40.0, 40.0])
cell_ang = np.diag(box_dims_bohr) * Bohr

# coordinates for linear CO2 (approx C-O bond length 1.16 A -> ~2.19 Bohr)
center = box_dims_bohr / 2.0
bond_bohr = 2.19
rel_pos_bohr = np.array([
    [0.0, 0.0, 0.0],          # C
    [-bond_bohr, 0.0, 0.0],   # O
    [ bond_bohr, 0.0, 0.0]    # O
])

abs_pos_bohr = center + rel_pos_bohr
pos_ang = abs_pos_bohr * Bohr

atoms = Atoms(
    symbols=['C', 'O', 'O'],
    positions=pos_ang,
    cell=cell_ang,
    pbc=[False, False, False]
)

# =========================
# DFT-FE Setup
# =========================

# Using passing a dictionary of exact file paths for multiple elements!
multi_element_psp_dict = {
    "C": "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library/C.upf",
    "O": "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library/O.upf"
}

dftfe_bin = "/home/pa01/Mehul/DFTFE/build_gpu/release/real/dftfe"
run_cmd = f"mpirun -np 8 {dftfe_bin}"

calc = DFTFE(
    command=run_cmd,
    psp_path=multi_element_psp_dict,

    # Finite element mesh parameters
    mesh_size=1.0,          
    polynomial_order=6,     
    atom_ball_radius=3.0,   

    # SCF parameters
    tolerance=5e-5,         
    scf_mixing=0.5,         
    fermi_temp=500.0,       
    num_kohn_sham=20,       

    # Functional
    xc='GGA-PBE',

    # Geometry optimization flags
    compute_forces=True, 
    compute_stress=False, 

    # Hardware Output
    use_device=True,
    verbosity=2,
    log_file="co2_gs_socket_test.log",
)

atoms.calc = calc

# =========================
# Run calculation
# =========================

energy = atoms.get_potential_energy()
print(f"Energy (Ha): {energy / Hartree}")

forces = atoms.get_forces()
print(f"Forces (Ha/Bohr): {forces}")
