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
| `auto_gpu_block_sizes` | `AUTO GPU BLOCK SIZES` | GPU | bool | planned | false = honour the explicit block sizes below |
| `fine_grained_gpu_timings` | `FINE GRAINED GPU TIMINGS` | GPU | bool | planned |  |
| `subspace_rot_full_cpu_mem` | `SUBSPACE ROT FULL CPU MEM` | GPU | bool | planned |  |
| `use_gpudirect_mpi_allreduce` | `USE GPUDIRECT MPI ALL REDUCE` | GPU | bool | planned |  |
| `use_dccl` | `USE DCCL` | GPU | bool | planned |  |
| `use_elpa_gpu_kernel` | `USE ELPA GPU KERNEL` | GPU | bool | planned |  |
| `self_potential_radius` | `SELF POTENTIAL RADIUS` | Boundary conditions | float | planned |  |
| `polynomial_order_electrostatics` | `POLYNOMIAL ORDER ELECTROSTATICS` | Finite element mesh parameters | int | planned |  |
| `poisson_tolerance` | `TOLERANCE` | Poisson problem parameters | float | planned | distinct from `tolerance` (SCF); same key, different subsection |
| `poisson_max_iterations` | `MAXIMUM ITERATIONS` | Poisson problem parameters | int | planned | distinct from `max_scf_iterations` (SCF) |
| `helmholtz_tolerance` | `ABSOLUTE TOLERANCE HELMHOLTZ` | Helmholtz problem parameters | float | planned |  |
| `helmholtz_max_iterations` | `MAXIMUM ITERATIONS HELMHOLTZ` | Helmholtz problem parameters | int | planned |  |
| `kerker_mixing_parameter` | `KERKER MIXING PARAMETER` | SCF parameters | float | planned |  |
| `compute_energy_each_iter` | `COMPUTE ENERGY EACH ITER` | SCF parameters | bool | planned |  |
| `cheby_degree_scaling_first_scf` | `CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF` | Eigen-solver parameters | float | planned |  |
| `subspace_rot_dofs_block_size` | `SUBSPACE ROT DOFS BLOCK SIZE` | Eigen-solver parameters | int | planned |  |
| `scalapack_procs` | `SCALAPACKPROCS` | Eigen-solver parameters | int | planned | 0 = auto; tuned per rank count in scaling decks |
| `scalapack_block_size` | `SCALAPACK BLOCK SIZE` | Eigen-solver parameters | int | planned |  |
| `reuse_lanczos_upper_bound` | `REUSE LANCZOS UPPER BOUND` | Eigen-solver parameters | bool | planned |  |
| `allow_multiple_passes_post_first_scf` | `ALLOW MULTIPLE PASSES POST FIRST SCF` | Eigen-solver parameters | bool | planned |  |
| `use_mixed_prec_cgs_sr` | `USE MIXED PREC CGS SR` | Eigen-solver parameters | bool | planned |  |
| `use_mixed_prec_xtox` | `USE MIXED PREC XTOX` | Eigen-solver parameters | bool | planned | v1.0 decks: USE MIXED PREC CGS O |
| `use_mixed_prec_xthx` | `USE MIXED PREC XTHX` | Eigen-solver parameters | bool | planned | v1.0 decks: USE MIXED PREC XTHX SPECTRUM SPLIT |
| `use_mixed_prec_rr_sr` | `USE MIXED PREC RR_SR` | Eigen-solver parameters | bool | planned |  |
| `use_mixed_prec_commun_only_xtox_xthx` | `USE MIXED PREC COMMUN ONLY XTOX XTHX` | Eigen-solver parameters | bool | planned |  |
| `num_core_eigenstates_mixed_prec_rr` | `NUMBER OF CORE EIGEN STATES FOR MIXED PREC RR` | Eigen-solver parameters | int | planned |  |
| `npband` | `NPBAND` | Parallelization | int | planned |  |
| `band_paral_opt` | `BAND PARAL OPT` | Parallelization | bool | planned |  |

Also handled outside this table: geometry (coords/cell/numbers/pbc), `compute_forces` (`ION FORCE`), `compute_stress` (`CELL STRESS`), `use_device` (`USE GPU`), `keep_scratch`, `psp_path` (dict → per-element `pseudo.inp`). Optimizer/NEB/MD settings (FORCE TOL, OPTIMIZATION MODE, NUMBER OF IMAGES, spring constants, …) are **ASE-side** (ASE optimizers / `ase.mep.NEB`), not DFT-FE `.prm` params.

