# ASE–DFT-FE Socket Interface

This repository provides an **ASE (Atomic Simulation Environment)** calculator designed to interface natively with **DFT-FE** over a TCP socket connection. 

By operating via a socket, the DFT-FE client remains persistent throughout the entire calculation suite. This completely eliminates the startup and initialization overhead associated with launching large MPI parallel jobs for every calculation step, enabling exceptionally fast iterative workflows including active learning loops, geometry optimizations, molecular dynamics, and more.

---

## Table of Contents

- [Features and Submodules](#features--submodules)
- [Prerequisites and Installation](#prerequisites--installation)
- [Cluster Deployment (CPU and GPU)](#cluster-deployment-cpu--gpu)
- [Interactive Use (Jupyter Notebooks)](#interactive-use-jupyter-notebooks)
- [Getting Started](#getting-started)
- [Machine Learning Dataset Generation](#machine-learning-dataset-generation)
- [Debugging and Tips](#debugging-and-tips)
- [Resources and Contact](#resources--contact)

---

## Features and Submodules

The interface is engineered for both robust, large-scale HPC simulations and seamless Machine Learning pipeline integration:
- **`dftfe.py`**: The core calculator class handling live TCP data serialization and native parameter mapping.
- **`utils/`**: Utilities for deep ML integration. Automatically constructs ML-ready datasets (`.extxyz`) natively handling unit conversions (Hartree/Bohr $\rightarrow$ eV/Å) from live calculations or static output logs.
- **`examples/`**: Extensive repository of example configurations and SLURM execution scripts spanning single-element, multi-element, real-space, and CPU/GPU contexts.

---

## Prerequisites and Installation

### 1. Requirements

- **Python**: 3.8 or newer.
- **ASE**: (`pip install ase`)
- **DFT-FE**: Ensure you have pulled a branch supporting the generic socket-driver (e.g. `publicGithubDevelop`). You must retain the absolute built path to the resulting `dftfe` executable.

### 2. Package Installation

Navigate into the interface directory within your cloned DFT-FE repository and install the calculator package directly into your active Python environment:

```bash
cd DFTFE/interfaces/ase-dftfe
pip install -e .
```

You can now use the calculator anywhere comprehensively via `from dftfe import DFTFE`.

---

## Cluster Deployment (CPU and GPU)

When deploying this interface to HPC clusters, pay close attention to the following architectural configurations. 

1. **Compiling the Engine**: 
   Compile DFT-FE as standard for your target architecture (`cmake .. and make -j`). The socket listener is statically integrated and enabled automatically; there are no special compilation flags required. Note the absolute path of your compiled binary (e.g., `/path/to/build_gpu/release/real/dftfe`).

2. **Calculator Instantiation and Device Selection**:
   When writing your ASE python script, you must carefully configure the target device to stop DFT-FE from faulting on mismatched hardware:
   - **For CPU-Only Clusters**: Ensure you explicitly declare `use_device=False` inside the `DFTFE(...)` constructor. If this is left true, the solver will attempt to access missing CUDA architectures and crash instantly.
   - **For GPU Clusters**: Toggle `use_device=True`. Ensure you pass the GPU-specific compilation of DFT-FE.

3. **MPI Launching Strategy**:
   **Do not run your python scripts with `mpirun`.** The ASE calculator orchestrates the MPI subsystem natively. You will invoke standard `python your_script.py`, and the script will internally spawn the entire DFT-FE parallel task pool using the string you provide to the `command=` attribute.

### Example SLURM Configuration

```bash
#!/bin/bash
#SBATCH --job-name=ase_dftfe_sim
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00

# 1. Load Spack, OpenMPI, and specific architectural modules
module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip 
module load nccl/2.23.4-1-gcc-13.3.0-xyspmp2 
module load gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/path/to/linAlgLibs/install/lib:$LIBRARY_PATH"

export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

# 2. Activate Python environment (containing ASE)
source ~/.venvs/ase-env/bin/activate

# 3. Execute Natively (The script manages mpirun automatically)
python sim_gs.py
```

---

## Interactive Use (Jupyter Notebooks)

To use the ASE-DFTFE interface interactively in a Jupyter Notebook on an HPC cluster, you must request a compute node allocation and launch the notebook server from that node. This ensures the persistent socket backend has access to the required CPU/GPU resources.

### 1. Request an Interactive Allocation
Request a compute node using `salloc` (for SLURM) or your cluster's equivalent command.

```bash
# Example: Request 1 node with GPUs for 3 hours
salloc --nodes=1 --ntasks-per-node=16 --gres=gpu:8 --time=03:00:00
```

### 2. Configure the Environment
Once the allocation is granted and you are on the compute node, load your modules and set your library paths.

```bash
# 1. Load your required MPI and GPU modules
module load openmpi/5.0.6-gcc-13.3.0-ytficip 
# ... load other modules (cuda/nccl/etc)

# 2. Set library paths for linking (Replace with your paths)
export LIBRARY_PATH="/path/to/linAlgLibs/install/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/path/to/linAlgLibs/install/lib:$LD_LIBRARY_PATH"

# 3. Activate Python environment
source ~/.venvs/ase-env/bin/activate
```

### 3. Launch the Jupyter Server
Launch the notebook server directly on the compute node.

```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

### 4. Accessing the Notebook
Create an SSH tunnel from your local machine to the compute node to access the notebook via your browser.

```bash
# On your local laptop:
ssh -L 8888:<compute-node-name>:8888 <username>@<cluster-address>
```
Open `http://localhost:8888` in your browser.

### 5. In-Notebook Initialization
Inside your notebook cell, you can initialize the calculator as usual. The environment variables set in Step 2 will be inherited by the kernel if started correctly; otherwise, you can set them using `os.environ`.

```python
import os
from dftfe import DFTFE
from ase.build import bulk

# Optional: Add library paths if not inherited
os.environ["LIBRARY_PATH"] = "/path/to/lib:" + os.environ.get("LIBRARY_PATH", "")

calc = DFTFE(
    command="mpirun -np 8 /path/to/dftfe",
    psp_path="/path/to/psp_library/",
    use_device=True
)

atoms = bulk("Cu", "fcc", a=3.6)
atoms.calc = calc

# Computed energy will trigger the persistent backend
print(f"Energy: {atoms.get_potential_energy()}")
```

---

## Getting Started

To execute a calculation, construct your standard ASE `Atoms` object and assign the instantiated `DFTFE` calculator to it. 

### 1. Single Point Energy ($CO_2$ Example)

Below is an example demonstrating a fully iterative ground-state calculation using a Carbon Dioxide ($CO_2$) configuration.

> **Note on Pseudopotentials**: If your pseudopotentials exist in a single directory perfectly formulated as `<Element>.upf`, you can simply pass the path string (`psp_path="/path/to/library/"`). For advanced or multi-element designs, you can pass an explicit dictionary as demonstrated below.

```python
from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE

# 1. Define Standard Parameterizations
box_dims_bohr = np.array([40.0, 40.0, 40.0])
cell_ang = np.diag(box_dims_bohr) * Bohr

bond_bohr = 2.19  # ≈ 1.16 Å C–O bond length
center = box_dims_bohr / 2.0
rel_pos_bohr = np.array([
    [0.0, 0.0, 0.0],          # C
    [-bond_bohr, 0.0, 0.0],   # O
    [ bond_bohr, 0.0, 0.0]    # O
])

atoms = Atoms(
    symbols=['C', 'O', 'O'],
    positions=(center + rel_pos_bohr) * Bohr,
    cell=cell_ang,
    pbc=[False, False, False]
)

# 2. Setup the socket interface
calc = DFTFE(
    command="mpirun -np 8 /absolute/path/to/dftfe",
    
    # Pseudopotential Dictionary Explicit Mapping
    psp_path={
        "C": "/absolute/path/to/C.upf",
        "O": "/absolute/path/to/O.upf"
    },
    
    # Solver Options
    mesh_size=1.0,          
    polynomial_order=6,     
    atom_ball_radius=3.0,   
    tolerance=5e-5,         
    num_bands=20,       
    xc='GGA-PBE',

    # ASE / Socket Options
    use_device=True,   # False if compiling on CPU-only machines!
    keep_scratch=True, # Prevent deletion of parameter grids for post-run debugging
    debug_timing=True, # Display python-to-C++ TCP transmission overhead
    verbosity=2,
)

atoms.calc = calc
energy = atoms.get_potential_energy()
print(f"Computed Energy (Ha): {energy / Hartree}")
```

### 2. Geometry Optimization / Relaxation (Graphene Example)

Because the DFT-FE calculator seamlessly hooks into the ASE ecosystem, you can natively leverage ASE's built-in advanced optimizer loops like `FIRE` or `BFGS`. Note that the python script itself controls the tight ionic optimization loop while the socket interface securely hot-reloads the persistent DFT-FE runtime for each frame evaluation.

Below is an example (`relax_graphene.py`) of setting up a periodic Graphene cell and running an ionic relaxation using the ASE `FIRE` optimizer.

```python
from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE
from ase.optimize import FIRE

# 1. Define Graphene Cell (Periodic)
cell_bohr = np.array([
    [ 4.65428900,  0.00000000, 0.0],
    [-2.32714450,  4.03073251, 0.0],
    [ 0.0,         0.0,       50.0]
])

frac_positions = np.array([
    [0.0000000000, 0.0000000000, 0.5],
    [0.3333333333, 0.6666666667, 0.5]
])

atoms = Atoms(
    symbols=['C', 'C'],
    positions=frac_positions @ (cell_bohr * Bohr),
    cell=cell_bohr * Bohr,
    pbc=[True, True, False] # 2D periodic
)

# 2. Setup the Socket Interface
calc = DFTFE(
    command="mpirun -np 16 /absolute/path/to/dftfe",
    psp_path="/absolute/path/to/psp_directory",
    
    # Example Periodic parameters
    use_time_reversal_symmetry=True,
    mp_grid=(4,4,1),
    npkpt=8,
    
    # Standard Options
    mesh_size=0.5,
    polynomial_order=3,
    use_device=True
)

atoms.calc = calc

# 3. Fire the ASE Optimizer Loop
opt = FIRE(atoms, trajectory="relax_graphene.traj", logfile="relax_graphene_optimizer.log")

# Converge when max force drops below 0.01 eV/Angstrom
opt.run(fmax=0.01)

print(f"Final relaxed energy (Ha): {atoms.get_potential_energy() / Hartree}")
```

To run this workload on a cluster via SLURM (`run_relax_graphene.slurm`), configure a multi-node job and execute the python natively to trigger the underlying continuous C++ simulation cycle.

```bash
#!/bin/bash
#SBATCH --job-name=ase_relax_graphene
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

module load openmpi/5.0.6-gcc-13.3.0-ytficip 
export LIBRARY_PATH="/path/to/linAlgLibs/install/lib:$LIBRARY_PATH"

source ~/.venvs/ase-env/bin/activate

# Launch single python orchestrator, mpirun is handled intrinsically
python relax_graphene.py
```

### 3. Phonon Calculations (Copper Example)

The socket interface is exceptionally powerful for phonon calculations, which require multiple force evaluations on slightly displaced supercells. Because the DFT-FE process stays alive, the significant overhead of initialization and mesh generation is only performed once.

Below is an example (`phonon_cu.py`) using **Phonopy** to calculate the phonon band structure of Copper (Cu).

```python
from ase import Atoms
from ase.build import bulk
from dftfe import DFTFE
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np

# 1. Setup Structure (ASE bulk)
structure = bulk("Cu", "fcc", a=3.6)

# 2. Setup DFT-FE Calculator
calc = DFTFE(
    command="mpirun -np 16 /path/to/dftfe",
    psp_path="/path/to/Cu.upf",
    mesh_size=1.0,
    polynomial_order=6,
    tolerance=1e-6,
    num_bands=20,
    compute_forces=True, # Crucial for phonons
    use_device=True
)

# 3. Phonopy Supercell & Displacements
ph_atoms = PhonopyAtoms(symbols=structure.symbols, 
                        positions=structure.positions, 
                        cell=structure.cell)

phonons = Phonopy(ph_atoms, supercell_matrix=[[3,0,0],[0,3,0],[0,0,3]])
phonons.generate_displacements(distance=0.01)
supercells = phonons.supercells_with_displacements

# 4. Force Calculation Loop (Persistent Socket)
sets_of_forces = []
for sc in supercells:
    sc_ase = Atoms(symbols=sc.symbols, positions=sc.positions, 
                   cell=sc.cell, pbc=True)
    sc_ase.calc = calc
    sets_of_forces.append(sc_ase.get_forces())

# 5. Post-process with Phonopy
phonons.forces = np.array(sets_of_forces)
phonons.produce_force_constants()
phonons.auto_band_structure()
fig = phonons.plot_band_structure()
fig.savefig("copper_phonons.png")
```

To run this on a GPU cluster using SLURM (`run_phonon_cu.slurm`):

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00

module load openmpi/5.0.6-gcc-13.3.0-ytficip 
export LIBRARY_PATH="/path/to/linAlgLibs/install/lib:$LIBRARY_PATH"

source ~/.venvs/ase-env/bin/activate
python phonon_cu.py
```

> **Multi-Node Scaling Tip**: When distributing DFT-FE across multiple nodes using SLURM (e.g. `--nodes=2`), it is strictly required to explicitly specify `#SBATCH --ntasks-per-node=8`. This guarantees that the MPI execution daemon distributes ranks optimally across the hardware, allowing the DFT-FE internal GPU mapping to bind processes perfectly to individual physical GPUs on each node.

---

## Machine Learning Dataset Generation

The interface features a robust Machine-learning (ML) suite designed to extract, convert, and store calculations. The tools natively identify failed SCF convergence drops and flawlessly format valid runs into standard Extended XYZ (`.extxyz`) datasets ready for MACE or NequIP ingestion.

### 1. Online Active Learning
To record live simulation steps as they happen, initialize the `DatasetRecorder` and pipe your `Atoms` object to it.

```python
from dftfe.utils import DatasetRecorder

recorder = DatasetRecorder("training_data.extxyz", max_force_threshold_ev_ang=100.0)

atoms.calc = calc
atoms.get_potential_energy()

# Safely extract Energy, Forces, Stress (in eV/Å) and log them
recorder.record(atoms, step=1, metadata={"temperature": 300, "source": "dftfe-md"})
```

### 2. Offline Log Harvesting 
To recover vast directories of old execution logs generated outside of ASE, leverage the batch utility. It regex-parses legacy logs and reconstructs valid, scaled ASE frames efficiently. It natively supports searching for multiple log extensions and extracting specific properties.

> **Important Note:** In order to automatically resolve the element symbols and structure constraints correctly during extraction, the corresponding `coordinates.inp` file *must* be present in the exact same directory alongside each log file.

```python
from dftfe.utils import build_dataset

build_dataset(
    input_dir="/path/to/old/dftfe/runs/",
    output_file="compiled_offline_dataset.extxyz",
    log_extension="*.op, *.out, *.log",      # Supports comma-separated extensions or lists
    extract_properties=['energy', 'force'],  # Optional: Exclusively extracts these properties. If omitted, extracts whatever is found.
    append=False,                            # Optional: If True, appends to the existing extxyz file. Default is False (overwrites).
    model="MACE"                             # Optional: Targeting "MACE" natively renames energy/forces -> REF_energy/REF_forces.
)
```

---

## Debugging and Tips

If iterative solvers crash silently or fail to inherit a specified configuration, leverage the following toggles within your `DFTFE(...)` constructor arguments:

1. **`keep_scratch=True`**  
   The wrapper builds dynamic scratch spaces containing exact `parameterFile.prm` and `coordinates.inp` grids formatted for the parallel engine. Setting this parameter to true retains these artifacts post-execution, allowing you to explicitly verify that your ASE inputs transformed successfully.

2. **`debug_timing=True`**  
   If you suspect bottlenecking, this parameter measures and logs specifically how much wall-clock time was devoted purely to C++ computational processing versus Python TCP-Socket transmission overhead.

---

## Resources and Contact

### Parameter Mappings
The socket translates standard arguments into internal parameters dynamically. For an explicit map aligning Python flags to innate `parameterFile.prm` variables, review the **[Parameter Mapping Guide](user_help/parameter_mapping.md)**. (Note: Features like `compute_stress` cleanly auto-toggle if `pbc=False` to prevent backend solver panics.)

### Citation
If you utilize this workflow in architectural or academic workloads, ensure you reference the core frameworks:
* [DFT-FE](https://sites.google.com/umich.edu/dftfe/) / [dftfe Github](https://github.com/dftfeDevelopers/dftfe)
* [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/)

### License
This interface operates within the DFT-FE ecosystem and is licensed under the **GNU Lesser General Public License (LGPL) v2.1 or later**.

### Contact and Support
For bugs, optimizations, or feature integration requests, please open an issue in the public repository. Alternatively, contact:

**Mehul Darak**
10pipioff@gmail.com
