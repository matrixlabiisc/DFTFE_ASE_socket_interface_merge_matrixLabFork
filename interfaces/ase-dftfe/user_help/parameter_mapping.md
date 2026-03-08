# ASE Python to DFT-FE Parameter Mapping

This document provides a comprehensive mapping between the parameters (kwargs) accepted by the `dftfe.py` ASE Socket Calculator and the native configuration variables set within DFT-FE's `parameterFile.prm`.

| ASE Python Keyword Argument | Native DFT-FE Parameter | Section / Subsection Name | Implementation Notes |
| :-------------------------- | :---------------------- | :------------------------ | :------------------- |
| **`verbosity`** | `VERBOSITY` | Root | Controls how much output DFT-FE logs. |
| **`compute_forces`** | `ION FORCE` | `Geometry` > `Optimization` | Enables ionic force calculation. |
| **`compute_stress`** | `CELL STRESS` | `Geometry` > `Optimization` | Enables stress tensor calculation. |
| **`xc`** | `EXCHANGE CORRELATION TYPE` | `DFT functional parameters` | E.g., `GGA-PBE`. |
| **`psp_path`** | `PSEUDOPOTENTIAL FILE NAMES LIST` | `DFT functional parameters` | Path to `.upf` file or directory. Generates `pseudo.inp`. |
| **`polynomial_order`** | `POLYNOMIAL ORDER` | `Finite element mesh parameters` | Finite-element polynomial order (e.g., `3` to `6`). |
| **`atom_ball_radius`** | `ATOM BALL RADIUS` | `Auto mesh generation parameters` | Mesh resolution control. |
| **`mesh_size`** | `MESH SIZE AROUND ATOM` | `Auto mesh generation parameters` | Size of finite elements near atoms. |
| **`use_time_reversal_symmetry`** | `USE TIME REVERSAL SYMMETRY` | `Brillouin zone k point sampling options` | Optimizes calculation for non-magnetic systems. |
| **`mp_grid`** | `SAMPLING POINTS 1, 2, 3` | `Monkhorst-Pack (MP) grid generation` | Tuple `(k1, k2, k3)` for BZ grid. |
| **`mp_grid_shift`** | `SAMPLING SHIFT 1, 2, 3` | `Monkhorst-Pack (MP) grid generation` | Tuple `(s1, s2, s3)` for shifting BZ grid. |
| **`npkpt`** | `NPKPT` | `Parallelization` | Number of MPI pools for K-point parallelization. |
| **`fermi_temp`** | `TEMPERATURE` | `SCF parameters` | Smearing temperature (Kelvin). |
| **`tolerance`** | `TOLERANCE` | `SCF parameters` | Convergence target. |
| **`scf_mixing`** | `MIXING PARAMETER` | `SCF parameters` | Float value for density mixing. |
| **`max_iterations`** | `MAXIMUM ITERATIONS` | `SCF parameters` | Limit on SCF loops. |
| **`num_kohn_sham`** | `NUMBER OF KOHN-SHAM WAVEFUNCTIONS`| `Eigen-solver parameters` | Set to `0` for auto-determination. |

### Parameters Managed Automatically by ASE
The following DFT-FE parameters are structurally abstracted by ASE and require no manual specification using the Python interface:
- `SOLVER MODE` (ASE manages it based on the workflow).
- `NATOMS` and `NATOM TYPES` (Derived from the `Atoms` object).
- `ATOMIC COORDINATES FILE` and `DOMAIN VECTORS FILE` (Interpreted automatically via socket).
- `PERIODIC1, PERIODIC2, PERIODIC3` (Inherited dynamically via `atoms.get_pbc()`).
