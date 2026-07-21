#!/bin/bash
#SBATCH --job-name=ase_dftfe_parity
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=slurm_parity_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# Native (file-based) DFT-FE vs the ASE socket interface, same binary/params, CPU.
echo "=== dftfe_ase parity check ==="
echo "Job $SLURM_JOB_ID on $(hostname), $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323

# The binary is a GPU build; even in CPU mode it must resolve CUDA runtime libs.
export CUDA_PATH="/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg"
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate
export PARITY_NPROC="${SLURM_NTASKS:-4}"
export PARITY_TMP="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests/parity_scratch"
mkdir -p "$PARITY_TMP"

python parity_check.py
EXIT=$?
echo "parity exit: $EXIT  ($(date))"
exit $EXIT
