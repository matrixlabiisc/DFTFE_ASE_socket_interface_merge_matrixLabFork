#!/bin/bash
#SBATCH --job-name=ase_ovh_2node
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:40:00
#SBATCH --output=slurm_ovh2_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

echo "=== overhead analysis: 2 nodes x 8 GPU = 16 ranks ==="
echo "Job $SLURM_JOB_ID on $(hostname), nodes: $SLURM_JOB_NODELIST, $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate
export OVH_NPROC=16
export OVH_LABEL="multinode 2x8 GPU"
export OVH_TMP="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests/parity_scratch"
mkdir -p "$OVH_TMP"

python overhead_analysis.py
echo "exit: $?  ($(date))"
