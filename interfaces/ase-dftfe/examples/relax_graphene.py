from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE
from ase.optimize import FIRE
import os

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

psp_path = "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_library"
dftfe_bin = "/home/pa01/Mehul/DFTFE/build_gpu/release/complex/dftfe"

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 16 {dftfe_bin}"
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
    # Use CPU
    # =========================
    use_device=False,

    # =========================
    # Output
    # =========================
    debug_timing=True,
    keep_scratch=True,
    verbosity=4,
    log_file="graphene_relax_dftfe.log"
)

atoms.calc = calc

energy = atoms.get_potential_energy()
print("Energy (Ha) unrelaxed:", energy / Hartree)
# ---------------- Relaxation ----------------

print("Starting ionic relaxation (Complex Build)...")

opt = FIRE(atoms, trajectory="relax_graphene.traj", logfile="relax_graphene_optimizer.log")

# Force criterion in eV/Å
opt.run(fmax=0.01)

atoms.write("graphene_relaxed.xyz")

energy = atoms.get_potential_energy()

print("Relaxation finished.")
print("Final energy relaxed(Hartree):", energy / Hartree)
