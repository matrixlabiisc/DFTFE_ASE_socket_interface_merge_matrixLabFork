# Compiling DFT-FE on the Matrix Lab Server

This document contains instructions to compile the **DFT-FE** backend (real and complex versions) with GPU support on the local Matrix Server.

## 1. Required Modules

First, load the required modules before compiling or running the software.
This ensures that the correct compilers (`mpicxx`, `gcc`) and MPI dependencies are in your environment:

```bash
module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip
module load nccl/2.23.4-1-gcc-13.3.0-xyspmp2
module load gdrcopy/2.4.1-gcc-13.3.0-dvwa323
```

## 2. Setting Environment Variables

Several dependencies (like BLIS, libflame, and CUDA math libraries) are installed manually. **Failure to set these variables will result in linker errors** (e.g., `/usr/bin/ld: cannot find -lblis` or `-lcudart_static`).

Execute these exports in your terminal before running the build scripts:

```bash
# Set CUDA path pointing to exactly the compiler suite used for the Spack build
export CUDA_PATH="/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg"

# Add Matrix Lab's linear algebra libraries and CUDA runtime libraries to linkers
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"

# For compiling on the login node or allocating threads during compilation
export OMP_NUM_THREADS=1 
```

## 3. Configuring and Compiling 

You can use the helper script `setupDevelopPetscMATRIX.sh` or `setupMatrixLab.sh` to configure CMake and trigger `make`.
The scripts define paths to all external packages required (like `dealii`, `alglib`, `libxc`, etc.), which are located in `/storage/dftfeDependenciesNoMKL/`.

```bash
cd /home/pa01/Mehul/DFTFE

# Option 1: Using the standard mpicxx compiler setup
./setupDevelopPetscMATRIX.sh

# Option 2: Using the Intel mpiicpx / Ninja setup
./setupMatrixLab.sh
```

### Manual Re-compilation

If you've modified a source file and only want to incrementally rebuild without wiping the CMake cache, you can CD directly into the build directories:

```bash
# For real build:
cd build_gpu/release/real
make -j 8

# For complex build:
cd build_gpu/release/complex
make -j 8
```

## 4. Running the ASE Interface

When running via SLURM or interactively using the ASE python interface (`dftfe.py`), ensure your batch/submit script also incorporates these library paths. 

Example snippet for a batch script:
```bash
module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"

source ~/.venvs/ase-env/bin/activate
python o2_gs.py
```
