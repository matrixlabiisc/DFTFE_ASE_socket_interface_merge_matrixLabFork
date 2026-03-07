from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
import os
from dftfe import DFTFE

# Create a parameterFile.prm with custom values to see if they are overwritten
with open("parameterFile.prm", "w") as f:
    f.write("set POLYNOMIAL ORDER=5\n")
    f.write("set TOLERANCE=1e-3\n")
    f.write("set MAXIMUM ITERATIONS=12\n")

# N2 domainVectors: 40 40 40 Bohr
box_dims_bohr = np.array([40.0, 40.0, 40.0])
cell_ang = np.diag(box_dims_bohr) * Bohr
center = box_dims_bohr / 2.0
rel_pos_bohr = np.array([[-1.3, 0.0, 0.0], [ 1.3, 0.0, 0.0]])
abs_pos_bohr = center + rel_pos_bohr
pos_ang = abs_pos_bohr * Bohr

atoms = Atoms(
    symbols=['N', 'N'],
    positions=pos_ang,
    cell=cell_ang,
    pbc=[False, False, False]
)

psp_path = "/home/pa01/Mehul/ase-dftfe-socket/psp_library"
dftfe_bin = "/home/pa01/Mehul/DFTFE/build_gpu/release/real/dftfe"

run_cmd = (
    f"export DFTFE_PSP_PATH={psp_path} && "
    f"mpirun -np 1 {dftfe_bin}"
)

calc = DFTFE(
    command=run_cmd,
    host="localhost",
    port=0,
    mesh_size=1.0, 
    # omitted: atom_ball_radius, scf_mixing, fermi_temp, num_kohn_sham
    # cheby_wfc_block_size, wfc_block_size, polynomial_order, tolerance
    xc='GGA-PBE', 
    use_device=False, # Use CPU just for quick config test
    compute_forces=False, 
    verbosity=4, 
    keep_scratch=True,
    parameterFile="parameterFile.prm",  # Use our custom file
)

atoms.calc = calc

try:
    atoms.get_potential_energy()
except Exception as e:
    print(f"Error: {e}")

scratch_dir = calc.scratch_dir
print(f"Scratch Directory: {scratch_dir}")

if scratch_dir and os.path.exists(os.path.join(scratch_dir, "parameterFile.prm")):
    with open(os.path.join(scratch_dir, "parameterFile.prm"), "r") as f:
        content = f.read()
        
        # We expect our custom values for POLYNOMIAL ORDER, TOLERANCE, and MAXIMUM ITERATIONS
        # because the original parameterFile.prm provided by the user had them, and they are not provided by python,
        # so dftfeWrapper.cc shouldn't overwrite them! Wait...
        # ACTUALLY, dftfeWrapper.cc copies `helpers/parameterFile.prm` first, and THEN appends/replaces stuff.
        # It completely IGNORES the user's custom `parameterFile` unless we do something about it!?
        # Wait, DFTFE socket calculator in python accepts `parameterFile` but does it send it to C++?
        pass
        
    print("Content of parameterFile.prm in scratch:")
    print("----------------------------------------")
    for line in content.splitlines():
        if "ATOM BALL RADIUS" in line or "MIXING PARAMETER" in line or "POLYNOMIAL ORDER" in line:
            print(line.strip())
    print("----------------------------------------")
