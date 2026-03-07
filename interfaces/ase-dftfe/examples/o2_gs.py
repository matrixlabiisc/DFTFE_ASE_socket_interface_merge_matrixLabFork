from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE

# =========================
# System Setup
# =========================

# This replaces the subsection Geometry in a normal DFT-FE run
box_dims_bohr = np.array([40.0, 42.0, 38.0])
cell_ang = np.diag(box_dims_bohr) * Bohr

center = box_dims_bohr / 2.0
rel_pos_bohr = np.array([
    [-1.2, 0.0, 0.0],
    [ 1.2, 0.0, 0.0]
])
abs_pos_bohr = center + rel_pos_bohr
pos_ang = abs_pos_bohr * Bohr

atoms = Atoms(
    symbols=['O', 'O'],
    positions=pos_ang,
    cell=cell_ang,
    pbc=[False, False, False]
)

# =========================
# DFT-FE Setup
# =========================

psp_path = "/home/pa01/Mehul/ase-dftfe-socket/psp_library"
dftfe_bin = "/home/pa01/Mehul/DFTFE/build_gpu/release/real/dftfe"

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 8 {dftfe_bin}"
)

calc = DFTFE(
    command=run_cmd,
    host="localhost",
    port=0,

    # Mesh parameters
    mesh_size=0.6, # MESH SIZE AROUND ATOM
    polynomial_order=7, # POLYNOMIAL ORDER
    atom_ball_radius=6.0, # ATOM BALL RADIUS

    # SCF
    tolerance=1e-6, # SCF TOLERANCE
    scf_mixing=0.5, # SCF MIXING PARAMETER
    fermi_temp=500.0, # SCF TEMPERATURE
    num_kohn_sham=16, # NUMBER OF KOHN-SHAM STATES
    debug_timing=True, # enables timing ASE and DFT-FE to check overhead

    # Functional
    xc='GGA-PBE',

    # GPU
    use_device=True,

    # Geometry Optimization
    compute_stress=False, # CELL STRESS in Geometry-Optimization
    compute_forces=True, # ION FORCES in Geometry-Optimization

    # Output
    keep_scratch=True, # keeps the scratch folder after the calculation
    verbosity=4, # verbosity level
    log_file="o2_gs_socket_timing_test.log" # log file
)

atoms.calc = calc

energy = atoms.get_potential_energy()
print(f"Energy (Ha): {energy / Hartree}")

forces = atoms.get_forces()
print(f"Forces (Ha/Bohr): {forces}")