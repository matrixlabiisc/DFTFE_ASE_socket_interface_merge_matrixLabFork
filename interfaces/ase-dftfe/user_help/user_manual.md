# ASE–DFT-FE Socket Interface

This submodule provides an **ASE (Atomic Simulation Environment)** calculator that enables communication with **DFT-FE** through a TCP socket interface.

The socket-based workflow allows the **DFT-FE process to remain persistent**, eliminating repeated initialization costs and enabling efficient iterative workflows such as:

- Geometry optimization
- Molecular dynamics
- Active learning loops
- Structure relaxation pipelines

The interface is designed for **high-performance and large-scale simulations** while maintaining compatibility with the standard ASE calculator API.

---

# Features

- ASE-compatible calculator for **DFT-FE**
- **Socket-based communication** with persistent DFT-FE runtime
- Reduced initialization overhead for iterative calculations
- Compatible with ASE workflows including but not limited to:
  - Geometry optimization
  - Molecular dynamics
  - Structure relaxation
  - Automated pipelines

---

# Prerequisites

Before installing this package, ensure the following dependencies are available.

## Python
- Python **3.8 or later**

## ASE
Install ASE via pip:

```bash
pip install ase
````

## DFT-FE

DFT-FE must be compiled with the required `SocketDriver` implementation (which is enabled by default in recent versions with socket support).

After compilation, ensure that the `dftfe` executable is either:

* available in your `PATH`, or
* referenced explicitly when initializing the calculator. -> See examples for implementation.

---

# Installation

This Python package is bundled natively within the DFT-FE repository. To install it in your Python environment, simply navigate to this directory (where this file resides) and install it via pip in editable mode:

```bash
cd interfaces/ase-dftfe
pip install -e .
```

Installing in editable mode registers the calculator with your Python environment.

You can then import the calculator using:

```python
from dftfe import DFTFE
```

---

# Getting Started

To use the interface, you construct an ASE `Atoms` object and assign the `DFTFE` calculator to it. 

### 1. The Python Script (`co2_gs.py`)
Below is a complete example of setting up a multi-element molecule ($CO_2$) and running it via the socket interface. Notice how we use a **dictionary** for `psp_path` to map exact pseudopotential file paths to each distinct element.

```python
from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from dftfe import DFTFE

# 1. Define Geometry
box_dims_bohr = np.array([40.0, 40.0, 40.0])
cell_ang = np.diag(box_dims_bohr) * Bohr

center = box_dims_bohr / 2.0
bond_bohr = 2.19
rel_pos_bohr = np.array([
    [0.0, 0.0, 0.0],          # C
    [-bond_bohr, 0.0, 0.0],   # O
    [ bond_bohr, 0.0, 0.0]    # O
])
pos_ang = (center + rel_pos_bohr) * Bohr

atoms = Atoms(
    symbols=['C', 'O', 'O'],
    positions=pos_ang,
    cell=cell_ang,
    pbc=[False, False, False]
)

# 2. Define exactly where the UPF files are located for each element
multi_element_psp_dict = {
    "C": "/absolute/path/to/C.upf",
    "O": "/absolute/path/to/O.upf"
}

# 3. Setup Calculator
dftfe_bin = "/absolute/path/to/dftfe"
run_cmd = f"mpirun -np 8 {dftfe_bin}"

calc = DFTFE(
    command=run_cmd,
    psp_path=multi_element_psp_dict,
    
    # Mesh and Convergence parameters
    mesh_size=1.0,          
    polynomial_order=6,     
    atom_ball_radius=3.0,   
    tolerance=5e-5,         
    num_kohn_sham=20,       
    xc='GGA-PBE',

    # ASE debugging controls
    keep_scratch=True, # Leaves the DFT-FE scratch folder undeleted for inspection
    debug_timing=True, # Prints out the timing overhead between ASE and DFT-FE

    use_device=True, # Set to False if compiled only for CPU
    verbosity=2,
    log_file="co2_gs_socket.log",
)

atoms.calc = calc
energy = atoms.get_potential_energy()
print(f"Energy (Ha): {energy / Hartree}")
```

> **Note on `psp_path`**:
> If your pseudopotentials share a single folder and are named perfectly as `<ElementSymbol>.upf` (e.g. `C.upf`, `O.upf`), you can simply provide the absolute directory string instead of a dictionary: `psp_path="/path/to/psp_directory"`.

---

### 2. The SLURM Batch Script (`run_gs_co2.slurm`)

Because the Python script itself launches `mpirun`, you do **not** run the python script with `mpirun`. Instead, you just invoke python natively and let it spawn the DFT-FE MPI processes in the background.

```bash
#!/bin/bash
#SBATCH --job-name=ase_dftfe_co2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00

# Make sure all libraries and MPI modules are loaded
module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip 
module load nccl/2.23.4-1-gcc-13.3.0-xyspmp2 
module load gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/path/to/linAlgLibs/install/lib:$LIBRARY_PATH"

export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

# Activate Python environment containing ASE and DFTFE 
source ~/.venvs/ase-env/bin/activate

# Execute natively
python co2_gs.py
```

# Machine Learning Dataset Generation

The ASE-DFT-FE interface comes with a built-in suite of tools for extracting DFT-FE calculations and converting them into Machine Learning (ML) ready datasets (specifically Extended XYZ `.extxyz` format). These tools explicitly handle all internal unit conversions from Hartree/Bohr to AES standard units (eV/Å).

### 1. Online Active Learning
If you are running an MD loop, geometry optimization, or any iterative pipeline through ASE, you can use the `DatasetRecorder` to automatically log every computed frame seamlessly:

```python
from dftfe.utils import DatasetRecorder

# Instantiate the recorder (uses append mode)
recorder = DatasetRecorder("training_data.extxyz", max_force_threshold_ev_ang=100.0)

# Inside your loop:
atoms.calc = calc
atoms.get_potential_energy()

# Safely extract Energy, Forces, Stress and append to the dataset
recorder.record(atoms, step=1, metadata={"temperature": 300, "source": "dftfe-md"})
```

### 2. Offline Dataset Building
If you already have a directory filled with old DFT-FE output logs (e.g., `*.op` files), you can batch-parse them and compile them into a single ML dataset using `build_dataset`. The parser is extremely robust and will gracefully skip crashed or incomplete calculations.

```python
from dftfe.utils import build_dataset

build_dataset(
    input_dir="/path/to/old/dftfe/runs/",
    output_file="compiled_offline_dataset.extxyz",
    log_extension="*.op"
)
```

---

# Debugging and Tips

When running iterative loops (like MD or NEB), if a calculation fails or crashes silently, the `DFTFE` calculator provides two built-in tools to help you identify the problem:

1. **`keep_scratch=True`**
   By default, the ASE persistent socket creates a temporary `dftfeScratch_<port>` directory and cleans it up when the script exits. Passing `keep_scratch=True` leaves this directory intact so you can manually inspect `parameterFile.prm`, `pseudo.inp`, and `coordinates.inp` to verify that ASE correctly passed your parameters over the socket.

2. **`debug_timing=True`**
   If you believe the socket connection is lagging or the Python wrapping is adding too much overhead, `debug_timing=True` will print an explicit breakdown at the end of each `calculate()` loop showing exactly how many seconds were spent strictly inside the C++ execution versus how many seconds were spent in Python packing/unpacking the JSON serialization.

---

# Examples

A comprehensive suite of example scripts covering single-element, multi-element, real-space, complex-space, CPU, and GPU workflows are available in the repository at:

```
interfaces/ase-dftfe/examples/
```

---

# Supported Parameters

The Python interface translates arguments directly into DFT-FE parameters. For a comprehensive mapping of all supported ASE Python variables to their corresponding native DFT-FE parameters (as they appear in `parameterFile.prm`), please review the full **[Parameter Mapping Guide](user_help/parameter_mapping.md)**.

A few notable parameters managed safely by default include `compute_forces=True` (ION FORCE) and `compute_stress=False` (CELL STRESS), which dynamically disable themselves if Period Boundary Conditions (PBCs) are absent to prevent solver crashes.

---

# License

The interface communicates with and resides within **DFT-FE**, which is licensed under the:

```
GNU Lesser General Public License (LGPL) v2.1 or later
```

---

# Citation

If you use this interface in academic work, please cite the **ASE** and **DFT-FE** projects appropriately.

```
https://sites.google.com/umich.edu/dftfe/
https://github.com/dftfeDevelopers/dftfe
https://wiki.fysik.dtu.dk/ase/
```

---

# Contact

For questions, bug reports, or feature requests, please open an issue in this repository or send a mail to Mehul Darak at 10pipioff@gmail.com

