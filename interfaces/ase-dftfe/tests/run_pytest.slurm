#!/bin/bash
#SBATCH --job-name=ase_dftfe_pytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=slurm_pytest_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe

# Runs the dftfe_ase unit + mock-integration test suite on a compute node.
# These tests use a fake DFT-FE (tests/mock_dftfe.py) — no real binary, no MPI,
# no GPU required — so this is a lightweight job that validates package logic,
# protocol framing, and the calculator lifecycle.

echo "=========================================="
echo "dftfe_ase pytest suite"
echo "Job ID : $SLURM_JOB_ID"
echo "Host   : $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# ── Environment (matches the existing ASE example scripts) ─────────────────
module load spack 2>/dev/null || true
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

# ── Shared ASE virtualenv (contains ase, numpy, pytest) ────────────────────
source ~/.venvs/ase-env/bin/activate

echo "python : $(which python)  ($(python --version 2>&1))"
python -c "import ase, numpy, pytest; print('ase', ase.__version__, '| numpy', numpy.__version__, '| pytest', pytest.__version__)"
echo "------------------------------------------"

# ── Run the suite ──────────────────────────────────────────────────────────
python -m pytest tests/ -v -ra
EXIT_CODE=$?

echo "------------------------------------------"
echo "Finished : $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
