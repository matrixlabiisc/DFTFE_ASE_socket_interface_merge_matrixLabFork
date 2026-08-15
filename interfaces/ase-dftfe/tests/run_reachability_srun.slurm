#!/bin/bash
#SBATCH --job-name=ase_reach_srun
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH --output=slurm_reach_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# Multi-node reachability check: srun launches DFT-FE across 2 nodes; rank 0 must
# connect back to the Python server (bound 0.0.0.0, advertising the node hostname).
echo "=== reachability (srun, 2 nodes) ==="
echo "Job $SLURM_JOB_ID batch host $(hostname), nodes: $SLURM_JOB_NODELIST, $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate
export REACH_NPROC="${SLURM_NTASKS:-16}"

python reachability_test.py
echo "exit: $?  ($(date))"
