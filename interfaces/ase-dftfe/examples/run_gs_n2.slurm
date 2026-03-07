#!/bin/bash
#SBATCH --job-name=ase_gs_multi_n2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --nodelist=cn2
#SBATCH --output=slurm_%j.out

echo "Nodes      : $SLURM_JOB_NUM_NODES"
echo "Tasks      : $SLURM_NTASKS"
echo "Host       : $(hostname)"

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"

export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export DFTFE_NUM_THREADS=1

# activate your ASE environment
source ~/.venvs/ase-env/bin/activate

# Run relaxation (single Python, MPI inside)
python n2_gs.py
