# ASE-DFTFE Parameter Mapping (auto-generated from `src/dftfe_ase/params.py`)

Do not edit by hand — regenerate with `python -m dftfe_ase.params`.
Single source of truth for ASE kwarg ↔ DFT-FE `.prm` parameter, subsection, type, and status.
`wired` = injected and two-value-verified; `planned` = defined for full-Python control,
activated with the generic `.prm` injector. Any param not listed can still be passed via
`extra_prm={...}` or a full `.prm` (option A).

| ASE kwarg | DFT-FE `.prm` key | subsection | type | status | note |
|---|---|---|---|---|---|
| `xc` | `EXCHANGE CORRELATION TYPE` | DFT functional parameters | str | wired |  |
| `pseudopotential_calculation` | `PSEUDOPOTENTIAL CALCULATION` | DFT functional parameters | bool | wired |  |
| `polynomial_order` | `POLYNOMIAL ORDER` | Finite element mesh parameters | int | wired |  |
| `density_quadrature_rule` | `DENSITY QUADRATURE RULE` | Finite element mesh parameters | int | wired |  |
| `mesh_size` | `MESH SIZE AROUND ATOM` | Auto mesh generation parameters | float | wired |  |
| `atom_ball_radius` | `ATOM BALL RADIUS` | Auto mesh generation parameters | float | wired |  |
| `tolerance` | `TOLERANCE` | SCF parameters | float | wired |  |
| `fermi_temp` | `TEMPERATURE` | SCF parameters | float | wired |  |
| `scf_mixing` | `MIXING PARAMETER` | SCF parameters | float | wired |  |
| `mixing_scheme` | `MIXING METHOD` | SCF parameters | str | wired |  |
| `max_scf_iterations` | `MAXIMUM ITERATIONS` | SCF parameters | int | wired |  |
| `num_bands` | `NUMBER OF KOHN-SHAM WAVEFUNCTIONS` | Eigen-solver parameters | int | wired |  |
| `orthogonalization_type` | `ORTHOGONALIZATION TYPE` | Eigen-solver parameters | str | wired |  |
| `use_single_prec_cheby` | `USE SINGLE PREC CHEBY` | Eigen-solver parameters | bool | wired |  |
| `wfc_block_size` | `WFC BLOCK SIZE` | Eigen-solver parameters | int | wired |  |
| `cheby_wfc_block_size` | `CHEBY WFC BLOCK SIZE` | Eigen-solver parameters | int | wired |  |
| `mp_grid` | `SAMPLING POINTS` | Monkhorst-Pack (MP) grid generation | int3 | wired | -> SAMPLING POINTS 1/2/3 |
| `mp_grid_shift` | `SAMPLING SHIFT` | Monkhorst-Pack (MP) grid generation | int3 | wired | -> SAMPLING SHIFT 1/2/3 |
| `use_time_reversal_symmetry` | `USE TIME REVERSAL SYMMETRY` | Brillouin zone k point sampling options | bool | wired |  |
| `use_group_symmetry` | `USE GROUP SYMMETRY` | Brillouin zone k point sampling options | bool | wired |  |
| `npkpt` | `NPKPT` | Parallelization | int | wired |  |
| `smeared_nuclear_charges` | `SMEARED NUCLEAR CHARGES` | Boundary conditions | bool | wired |  |
| `spin_polarized` | `SPIN POLARIZATION` | DFT functional parameters | int | wired |  |
| `mixing_history` | `MIXING HISTORY` | SCF parameters | int | planned | was mis-mapped to LBFGS HISTORY; fix to MIXING HISTORY |
| `verbosity` | `VERBOSITY` | (top-level) | int | wired |  |
| `base_mesh_size` | `BASE MESH SIZE` | Auto mesh generation parameters | float | planned |  |
| `mesh_size_at_atom` | `MESH SIZE AT ATOM` | Auto mesh generation parameters | float | planned |  |
| `auto_adapt_base_mesh_size` | `AUTO ADAPT BASE MESH SIZE` | Auto mesh generation parameters | bool | planned |  |
| `chebyshev_polynomial_degree` | `CHEBYSHEV POLYNOMIAL DEGREE` | Eigen-solver parameters | int | planned |  |
| `chebyshev_filter_tolerance` | `CHEBYSHEV FILTER TOLERANCE` | Eigen-solver parameters | float | planned |  |
| `highest_state_cheby` | `HIGHEST STATE OF INTEREST FOR CHEBYSHEV FILTERING` | Eigen-solver parameters | int | planned |  |
| `total_magnetization` | `TOTAL MAGNETIZATION` | DFT functional parameters | float | planned |  |
| `spin_mixing_enhancement` | `SPIN MIXING ENHANCEMENT FACTOR` | SCF parameters | float | planned |  |
| `hubbard_parameters_file` | `HUBBARD PARAMETERS FILE` | Hubbard Parameters | str | planned | DFT+U |
| `kpoint_rule_file` | `kPOINT RULE FILE` | Brillouin zone k point sampling options | str | planned |  |
| `dispersion_correction_type` | `DISPERSION CORRECTION TYPE` | Dispersion Correction | int | planned | needs Dispersion Correction subsection added |

Also handled outside this table: geometry (coords/cell/numbers/pbc), `compute_forces` (`ION FORCE`), `compute_stress` (`CELL STRESS`), `use_device` (`USE GPU`), `keep_scratch`, `psp_path` (dict → per-element `pseudo.inp`). Optimizer/NEB/MD settings (FORCE TOL, OPTIMIZATION MODE, NUMBER OF IMAGES, spring constants, …) are **ASE-side** (ASE optimizers / `ase.mep.NEB`), not DFT-FE `.prm` params.

