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

# Basic Usage

A minimal example using ASE with the DFT-FE socket interface:

```python
from ase import Atoms
from dftfe import DFTFE

atoms = Atoms("O2", positions=[[0,0,0],[1.2,0,0]])

calc = DFTFE(
    command="/path/to/dftfe",
    xc="GGA-PBE",
    mesh_size=0.6
)

atoms.calc = calc

energy = atoms.get_potential_energy()
print("Energy:", energy)
```

---

# Examples

Example scripts demonstrating typical workflows are available in:

```
examples/
```

Current examples include:

* Ground-state energy calculation of O2 molecule
* Relaxation of O2 molecule

---

# Supported Parameters

The calculator currently supports the parameters demonstrated in:

```
examples/o2_gs.py
```

For a comprehensive mapping of all supported ASE Python variables to their corresponding native DFT-FE parameters (as they appear in `parameterFile.prm`), please review the full **[Parameter Mapping Guide](user_help/parameter_mapping.md)**.

Additional parameters and configuration options will be added in future releases. We are rapidly developing this interface and will keep updating the documentation as we go.

---

# Project Status

This interface is under active development.

Planned improvements include:

* Expanded parameter support
* Improved documentation
* Additional examples
* Robust testing across different simulation workflows

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

