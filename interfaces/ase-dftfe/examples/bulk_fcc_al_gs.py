from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe_ase import DFTFE

# =========================
# System Setup (FCC Al)
# =========================

# domainBoundingVectors.inp
box_dims_bohr = np.array([7.6, 7.6, 7.6])
cell_ang = np.diag(box_dims_bohr) * Bohr

# fractional coordinates
frac_positions = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]
])

positions_ang = frac_positions @ cell_ang

atoms = Atoms(
    symbols=['Al', 'Al', 'Al', 'Al'],
    positions=positions_ang,
    cell=cell_ang,
    pbc=[True, True, True]
)

# =========================
# DFT-FE Setup
# =========================

psp_path = "/DFTFE/interfaces/ase-dftfe/psp_library/Al.upf" #ensure full path is given
dftfe_bin = "/DFTFE/build/release/complex/dftfe"

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 8 {dftfe_bin}"
)

calc = DFTFE(
    command=run_cmd,
    bind_host="localhost",
    port=0,

    # =========================
    # Mesh
    # =========================
    mesh_size=1.6,        # MESH SIZE AROUND ATOM
    polynomial_order=5,   # POLYNOMIAL ORDER

    # =========================
    # SCF
    # =========================
    tolerance=5e-5,
    fermi_temp=500.0,

    # =========================
    # Brillioun Zone k point sampling options
    # =========================
    use_time_reversal_symmetry = True,
    # Monkhorst-Pack (MP) grid generation
    mp_grid = (2,2,2),
    mp_grid_shift = (1,1,1),

    # =========================
    # Parallelization
    # =========================
    npkpt = 2,

    # =========================
    # Functional
    # =========================
    xc="GGA-PBE",

    # =========================
    # Geometry optimization options
    # =========================
    compute_stress=True,
    compute_forces=False,

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
    log_file="al_bulk_socket_test.log"
)

atoms.calc = calc

energy = atoms.get_potential_energy()
print(f"Energy (Ha): {energy / Hartree}")

stress = atoms.get_stress()
print("Stress:", stress)