#!/bin/bash
#SBATCH --job-name=ase_dftfe_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_smoke_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# Real end-to-end smoke test: the NEW dftfe_ase package driving the installed
# DFT-FE GPU binary over the socket (N2 ground state), via SLURM on a GPU node.

echo "=== dftfe_ase real smoke test ==="
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate

python smoke_real_n2.py
EXIT=$?
echo "smoke exit: $EXIT  ($(date))"
exit $EXIT
