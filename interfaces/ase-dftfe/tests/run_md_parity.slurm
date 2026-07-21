#!/bin/bash
#SBATCH --job-name=ase_md_parity
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_md_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# MD head-to-head: native DFT-FE BOMD (NVE) vs ASE VelocityVerlet, identical
# setup, CPU (deterministic). Demo passed as $1 (ex1/ex2/ex3).
echo "=== MD parity ($1) ==="
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
export MD_TMP="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests/parity_scratch"
mkdir -p "$MD_TMP"

python md_parity.py "${1:-ex1}"
echo "exit: $?  ($(date))"
