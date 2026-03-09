from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE

# =========================
# System Setup (N2)
# =========================

# domainVectors: 40 40 40 Bohr
box_dims_bohr = np.array([40.0, 40.0, 40.0])
cell_ang = np.diag(box_dims_bohr) * Bohr

# coordinates: ±1.3 Bohr
center = box_dims_bohr / 2.0
rel_pos_bohr = np.array([
    [-1.3, 0.0, 0.0],
    [ 1.3, 0.0, 0.0]
])

abs_pos_bohr = center + rel_pos_bohr
pos_ang = abs_pos_bohr * Bohr

atoms = Atoms(
    symbols=['N', 'N'],
    positions=pos_ang,
    cell=cell_ang,
    pbc=[False, False, False]
)

# =========================
# DFT-FE Setup
# =========================

psp_path = "/DFTFE/interfaces/ase-dftfe/psp_library/" # if not given a .upf file but just a psp library path, we will automatically fetch the element.upf file from the library for you
dftfe_bin = "/DFTFE/build/release/real/dftfe"

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 8 {dftfe_bin}"
)

calc = DFTFE(
    command=run_cmd,
    host="localhost",
    port=0,

    # =========================
    # Mesh (Finite element mesh parameters)
    # =========================
    mesh_size=1.0,          # MESH SIZE AROUND ATOM
    polynomial_order=6,     # POLYNOMIAL ORDER
    atom_ball_radius=3.0,   # ATOM BALL RADIUS

    # =========================
    # SCF parameters
    # =========================
    tolerance=5e-5,         # SCF TOLERANCE
    scf_mixing=0.5,         # MIXING PARAMETER
    fermi_temp=500.0,       # TEMPERATURE
    num_eigen_states=15,       # NUMBER OF KS STATES
    debug_timing=True, # enables timing ASE and DFT-FE to check overhead

    # =========================
    # Functional
    # =========================
    xc='GGA-PBE',

    # =========================
    # GPU
    # =========================
    use_device=True,

    # =========================
    # Geometry optimization flags
    # =========================
    compute_forces=True, #ION FORCE
    compute_stress=False, #CELL STRESS

    # =========================
    # Output
    # =========================
    keep_scratch=True,
    verbosity=4,
    log_file="n2_gs_socket_test.log",
)

atoms.calc = calc

# =========================
# Run calculation
# =========================

energy = atoms.get_potential_energy()
print(f"Energy (Ha): {energy / Hartree}")

forces = atoms.get_forces()
print(f"Forces (Ha/Bohr): {forces}")