#!/bin/bash
#SBATCH --job-name=ase_geoid
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_geoid_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# Apples-to-apples geo-opt: SAME ASE BFGS, forces via socket vs native file. CPU.
echo "=== identical-optimizer geo-opt (ex1_b) ==="
echo "Job $SLURM_JOB_ID on $(hostname), $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export CUDA_PATH="/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg"
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate
export GEOID_NPROC="${SLURM_NTASKS:-4}"
export GEOID_TMP="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests/parity_scratch"
mkdir -p "$GEOID_TMP"

python geoopt_identical.py
echo "exit: $?  ($(date))"
