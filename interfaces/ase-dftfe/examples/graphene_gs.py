from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE

# =========================
# System Setup (Graphene)
# =========================

# domainVectors.inp (Bohr)
cell_bohr = np.array([
    [ 4.65428900,  0.00000000, 0.0],
    [-2.32714450,  4.03073251, 0.0],
    [ 0.0,         0.0,       50.0]
])

cell_ang = cell_bohr * Bohr

# fractional coordinates
frac_positions = np.array([
    [0.0000000000, 0.0000000000, 0.5],
    [0.3333333333, 0.6666666667, 0.5]
])

positions_ang = frac_positions @ cell_ang

atoms = Atoms(
    symbols=['C', 'C'],
    positions=positions_ang,
    cell=cell_ang,
    pbc=[True, True, False]
)

# =========================
# DFT-FE Setup
# =========================

psp_path = "/DFTFE/interfaces/ase-dftfe/psp_library/C.upf"
dftfe_bin = "/DFTFE/build/release/complex/dftfe"

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 8 {dftfe_bin}"
)

calc = DFTFE(
    command=run_cmd,
    host="localhost",
    port=0,

    # =========================
    # Mesh
    # =========================
    mesh_size=0.5,
    polynomial_order=3,

    # =========================
    # SCF
    # =========================
    tolerance=5e-5,
    fermi_temp=500.0,

    # =========================
    # Brillouin zone sampling
    # =========================
    use_time_reversal_symmetry=True,
    mp_grid=(4,4,1),
    mp_grid_shift=(1,1,0),

    # =========================
    # Parallelization
    # =========================
    npkpt=8,

    # =========================
    # Geometry Optimization
    # =========================
    # =========================
    # Functional
    # =========================
    xc="GGA-PBE",

    # =========================
    # GPU
    # =========================
    use_device=True,

    # =========================
    # Output
    # =========================
    debug_timing=True,
    keep_scratch=True,
    verbosity=4,
    log_file="graphene_socket_test.log"
)

atoms.calc = calc

energy = atoms.get_potential_energy()
print("Energy (Ha):", energy / Hartree)