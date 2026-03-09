# ASE–DFT-FE Socket Interface

This repository provides an **ASE (Atomic Simulation Environment)** calculator designed to interface natively with **DFT-FE** over a TCP socket connection. 

By operating via a socket, the DFT-FE client remains persistent throughout the entire calculation suite. This completely eliminates the startup and initialization overhead associated with launching large MPI parallel jobs for every calculation step, enabling exceptionally fast iterative workflows including active learning loops, geometry optimizations, molecular dynamics, and more.

---

## Table of Contents

- [Features and Submodules](#features--submodules)
- [Prerequisites and Installation](#prerequisites--installation)
- [Cluster Deployment (CPU and GPU)](#cluster-deployment-cpu--gpu)
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

## Getting Started

To execute a calculation, construct your standard ASE `Atoms` object and assign the instantiated `DFTFE` calculator to it. Below is an example demonstrating a fully iterative calculation using a Carbon Dioxide ($CO_2$) configuration. 

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
    num_kohn_sham=20,       
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
To recover vast directories of old `dftfe.log` or `*.op` execution logs generated outside of ASE, leverage the batch utility. It regex-parses legacy logs and reconstructs valid, scaled ASE frames efficiently.

```python
from dftfe.utils import build_dataset

build_dataset(
    input_dir="/path/to/old/dftfe/runs/",
    output_file="compiled_offline_dataset.extxyz",
    log_extension="*.op"
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
