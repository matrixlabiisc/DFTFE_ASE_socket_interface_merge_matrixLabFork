#!/bin/bash
# script to setup and build DFT-FE with local dependencies.

set -e
set -o pipefail

# if [ -s CMakeLists.txt ]; then
#     echo "This script must be run from the build directory!"
#     exit 1
# fi

# Path to project source
SRC=$(cd $(dirname $0) && pwd) # location of source directory

# Dependencies Root
DEP_ROOT="/home/pa01/rudrapanch/install_DFTFE_syclv3/dependencies"

# Source Intel OneAPI environment
# source /apps/softwares/oneapi/2025/setvars.sh

#Paths for required external libraries
dealiiDir="$DEP_ROOT"
alglibDir="$DEP_ROOT"
libxcDir="$DEP_ROOT"
spglibDir="$DEP_ROOT"
xmlIncludeDir="$DEP_ROOT/include/libxml2"
xmlLibDir="$DEP_ROOT/lib"
ELPA_PATH="$DEP_ROOT" 

# Other paths
dftdpath=""
numdiffdir=""
mdiPath=""
torchDir=""

#Toggle GPU compilation (OFF for now unless requested, assuming CPU build for demo)
withGPU=OFF
gpuLang="cuda"
gpuVendor="nvidia"
withGPUAwareMPI=OFF

withDCCL=OFF
withMDI=OFF
withTorch=OFF
withCustomizedDealii=OFF

#Compiler options and flags
cxx_compiler="mpiicpx"
cxx_flags="-fPIC"
cxx_flagsRelease="-O2"
device_flags=""
device_architectures=""

withHigherQuadPSP=OFF
build_type=Release
testing=OFF
useInt64=OFF

out="build_local"

if [ -d "$out" ]; then
    echo "Directory $out exists."
else
    mkdir -p "$out"
fi

cd $out

# Export PKG_CONFIG_PATH for ELPA
export PKG_CONFIG_PATH=$DEP_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Building Complex executable in $build_type mode..."
mkdir -p complex && cd complex

cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler \
    -DCMAKE_CXX_FLAGS="$cxx_flags" \
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type \
    -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir \
    -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir \
    -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir \
    -DWITH_MDI=$withMDI \
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii \
    -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DEP_ROOT" \
    -DWITH_COMPLEX=ON \
    -DWITH_GPU=$withGPU \
    -DWITH_TESTING=$testing \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_64BIT_INT=$useInt64 \
    "$SRC"

make -j12
cd ..
echo "Build complete."
