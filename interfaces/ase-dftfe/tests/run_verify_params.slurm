#!/bin/bash
#SBATCH --job-name=ase_verify_params
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=slurm_verify_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# Verify non-default params actually reach the generated .prm (keep_scratch).
echo "=== param-injection verify ==="; echo "Job $SLURM_JOB_ID on $(hostname), $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate
export VERIFY_NPROC="${SLURM_NTASKS:-4}"
export VERIFY_TMP="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests"

python verify_param_injection.py
echo "exit: $?  ($(date))"
