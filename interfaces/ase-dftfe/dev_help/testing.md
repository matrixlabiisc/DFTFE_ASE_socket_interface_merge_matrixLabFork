# ASE-DFTFE Automated Testing Guide

This guide explains how to use the automated test pipeline for the ASE-DFTFE interface. As a developer, **you must run this test suite to ensure that your changes have not broken the interface and that everything is working as accurately as expected.**

## Purpose of the Test Suite

The test pipeline acts as a strict regression suite. It compares the results of your current code against a set of known-good reference calculations. 

The suite checks:
1.  **Energy Accuracy**: The computed total potential energy must match the reference value up to **10 decimal places** ($< 1\times 10^{-10}$ Hartree).
2.  **Pointwise Force Accuracy**: Every component of the force vector for every atom must match the reference value up to **8 decimal places** ($< 1\times 10^{-8}$ Hartree/Bohr absolute error).
3.  **Basic Functionality**: It ensures features like geometry relaxation (e.g., ASE's BFGS optimizer interfacing with DFT-FE forces) can run successfully without crashing.

## Location & Structure

All testing files are located within `interfaces/ase-dftfe/automated_tests/`:

```
automated_tests/
├── run_tests.slurm          # Top-level SLURM submission script
├── test_suite.py            # Main runner orchestrator in Python
├── references.json          # Stores the known-good reference values (energies, forces)
└── cases/                   # Implementations of individual test cases
    ├── co2_nonperiodic.py   # Gamma-only (real binary), computes forces
    ├── n2_nonperiodic.py    # Gamma-only (real binary), computes forces
    ├── graphene_periodic.py # k-points (complex binary), computes forces
    ├── al_bulk_periodic.py  # k-points (complex binary), computes stress
    └── relax_o2.py          # Geometry relaxation check (run-only)
```

## How to Run the Tests

You can run the entire test suite on a cluster via the provided SLURM script:

```bash
cd interfaces/ase-dftfe/automated_tests
sbatch run_tests.slurm
```

### Viewing the Results
Once the job finishes, a summary file named `test_results_YYYYMMDD_HHMMSS.txt` will be created in the `automated_tests` directory, and the output will also be visible in the SLURM out file (`slurm_tests_*.out`). 

**Look for `[PASS]` statuses:**
*   `[PASS]`: Energy matched within 1e-10 Ha, and maximum pointwise force difference was within 1e-8 Ha/Bohr.
*   `[FAIL]`: A threshold was exceeded, or the case threw an exception. This indicates your changes have introduced a regression. The logs will tell you exactly which value missed the threshold.
*   `[SEEDED]`: A new reference was generated (see changing references below).

## How the Checks Work (`test_suite.py`)

The suite maps each case to either the **real** or **complex** DFT-FE binary based on whether it needs k-points. 
*   **Energy Check:** `|E_computed - E_ref| < 1e-10`
*   **Force Check:** `max(|F_computed - F_ref|) < 1e-8` for all pointwise $(x, y, z)$ force components over all atoms.
*   **Run-Only:** Some tests like `relax_o2` are marked as `RUN_ONLY` in `test_suite.py`. This is because geometry optimizers traverse different trajectories based on numerical variations, so comparing strict pointwise final states is brittle. The test only verifies that the ASE optimization loop can complete correctly.

## Updating or Adding Tests

### Adding a new case
1. Create a Python script in `cases/` mirroring standard usage. Expose a single `run(dftfe_bin, psp_library, np_tasks)` function.
2. Ensure it returns paths matching `(energy_ha, forces_ha_per_bohr, stress)`. (Return `None` if not computed).
3. Add the case name to `BINARY_KIND` dictionary in `test_suite.py` to specify if it uses the real or complex binary.
4. Add an empty reference dictionary in `references.json` with null values. 

### Seeding references
If `references.json` has `null` values for a test's energy or forces (for example, on the first run of a new test), the test orchestrator treats it as a seeding run. 
Instead of checking for a `PASS/FAIL`, it saves the newly computed values back into `references.json` and reports `[SEEDED]`. Next time you run the suite, it will perform the standard regression checks against those newly seeded values. 

**Note**: Do not blindly commit seeded values without verifying that the initial answers are correct!
