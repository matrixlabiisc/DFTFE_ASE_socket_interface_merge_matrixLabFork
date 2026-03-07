from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE
from ase.optimize import BFGS
import os

# =========================
# System Setup
# =========================

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

psp_path = "/DFTFE/interfaces/ase-dftfe/psp_library/O.upf"
dftfe_bin = "/DFTFE/build/release/complex/dftfe" # to check complex build relaxation as well

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 8 {dftfe_bin}"
)

calc = DFTFE(
    command=run_cmd,
    host="localhost",
    port=0,

    # Mesh parameters
    mesh_size=0.6,
    polynomial_order=7,
    atom_ball_radius=6.0,

    # SCF
    tolerance=1e-6,
    scf_mixing=0.5,
    fermi_temp=500.0,
    num_kohn_sham=16,
    wfc_block_size=16,
    cheby_wfc_block_size=16,
    debug_timing=True,

    # Functional
    xc='GGA-PBE',

    # GPU
    use_device=True,

    # Runtime
    compute_stress=False,
    keep_scratch=True,
    verbosity=4,
    log_file="o2_relax_complex.log"
)

atoms.calc = calc
# ---------------- Relaxation ----------------

print("Starting ionic relaxation (Complex Build)...")

opt = BFGS(atoms, trajectory="relax_complex.traj", logfile="relax_complex.log")

# Force criterion in eV/Å
opt.run(fmax=0.01)

atoms.write("o2_relaxed_complex.xyz")

energy = atoms.get_potential_energy()

print("Relaxation finished.")
print("Final energy (Hartree):", energy / Hartree)
