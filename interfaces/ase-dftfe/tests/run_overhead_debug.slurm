#!/bin/bash
#SBATCH --job-name=ase_ovh_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:40:00
#SBATCH --output=slurm_ovhdbg_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# Interface-overhead measurement via debug_timing over a short MD.
# Single node: sbatch run_overhead_debug.slurm ex2
# Multi node : sbatch --nodes=2 --ntasks=16 --gres=gpu:8 run_overhead_debug.slurm ex2
echo "=== overhead_debug $1 ==="
echo "Job $SLURM_JOB_ID on $(hostname), nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS, $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate
export OVH_NPROC="${SLURM_NTASKS:-8}"
export OVH_LABEL="${SLURM_JOB_NUM_NODES} node(s) / ${SLURM_NTASKS} GPU"

python overhead_debug.py "${1:-ex2}"
echo "exit: $?  ($(date))"
