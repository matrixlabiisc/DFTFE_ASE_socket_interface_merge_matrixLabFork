#!/bin/bash
module purge
module load cuda/12.6
module load openmpi/gcc13cuda126
module load modulefiles/compiler-rt/2025.0.4
module load modulefiles/tbb/2022.0
module load modulefiles/mkl/2025.0

echo "Using mpicxx: $(which mpicxx)"
mpicxx --version

export PKG_CONFIG_PATH=/storage/dftfeDependenciesNoMKL/elpa/install/lib/pkgconfig:$PKG_CONFIG_PATH
export LIBRARY_PATH=/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LD_LIBRARY_PATH

cd /home/pa01/Mehul/DFTFE
rm -rf build_slurmmpi
mkdir build_slurmmpi
cd build_slurmmpi

cmake .. \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DUSE_DEVICE=ON \
  -DWITH_GPU=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/storage/dftfeDependenciesNoMKL/dealii/installReal

make -j 8
