#!/bin/bash
#SBATCH --job-name=ase_geoopt_parity
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_geoopt_%j.out
#SBATCH --chdir=/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests

# Geo-opt head-to-head: native DFT-FE GEOOPT vs ASE BFGS, matched force tol, GPU.
echo "=== geo-opt parity (ex1_b N2 ion relax) ==="
echo "Job $SLURM_JOB_ID on $(hostname), $(date)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"
export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

source ~/.venvs/ase-env/bin/activate
export GEOOPT_NPROC="${SLURM_NTASKS:-4}"
export GEOOPT_TMP="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/tests/parity_scratch"
mkdir -p "$GEOOPT_TMP"

python geoopt_parity.py
echo "exit: $?  ($(date))"
