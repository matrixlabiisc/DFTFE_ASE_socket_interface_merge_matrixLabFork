#!/bin/bash
#SBATCH --job-name=ase_dftfe_tests
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=slurm_tests_%j.out

echo "=========================================="
echo "ASE-DFTFE Automated Test Suite"
echo "Job ID   : $SLURM_JOB_ID"
echo "Nodes    : $SLURM_JOB_NUM_NODES"
echo "Tasks    : $SLURM_NTASKS"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="

# ── Modules ────────────────────────────────────────────────────────────────
module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

# ── Paths (edit these if binaries move) ────────────────────────────────────
DFTFE_ROOT="/home/pa01/Mehul/DFTFE"
DFTFE_REAL="${DFTFE_ROOT}/build_gpu/release/real/dftfe"
DFTFE_COMPLEX="${DFTFE_ROOT}/build_gpu/release/complex/dftfe"
PSP_LIBRARY="${DFTFE_ROOT}/interfaces/ase-dftfe/psp_library"

# ── Python environment ─────────────────────────────────────────────────────
source ~/.venvs/ase-env/bin/activate

# ── Run the suite ─────────────────────────────────────────────────────────
cd "$(dirname "$0")"

python test_suite.py \
    --dftfe-real    "${DFTFE_REAL}" \
    --dftfe-complex "${DFTFE_COMPLEX}" \
    --psp-library   "${PSP_LIBRARY}" \
    --np            "${SLURM_NTASKS}"

EXIT_CODE=$?
echo ""
echo "Finished : $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
