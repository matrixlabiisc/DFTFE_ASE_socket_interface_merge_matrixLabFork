# Install script for directory: /home/pa01/Mehul/DFTFE

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/pa01/Mehul/DFTFE/build_local/release/real/tests/dft/pseudopotential/real/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/pa01/Mehul/DFTFE/build_local/release/real/tests/dft/allElectron/real/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/pa01/Mehul/DFTFE/build_local/release/real/tests/dft/pseudopotential/real/d3/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/pa01/Mehul/DFTFE/build_local/release/real/tests/dft/pseudopotential/real/d4/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeReal.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeReal.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeReal.so"
         RPATH "/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg/targets/x86_64-linux/lib:/storage/dftfeDependenciesNoMKL/dealii/installReal/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/nccl-2.23.4-1-xyspmp23glxb4slgne4xpemahjrkuyrj/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/zlib-ng-2.2.3-nqbp3zzmzdgtwk34m3miae24str5vtnj/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installReal/lib:/storage/dftfeDependenciesNoMKL/petsc/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/openmpi-5.0.6-ytficipzlncogz4ktqkzkwszqj7jw2ne/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/pa01/Mehul/DFTFE/build_local/release/real/libdftfeReal.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeReal.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeReal.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeReal.so"
         OLD_RPATH "/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg/targets/x86_64-linux/lib:/storage/dftfeDependenciesNoMKL/dealii/installReal/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/nccl-2.23.4-1-xyspmp23glxb4slgne4xpemahjrkuyrj/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/zlib-ng-2.2.3-nqbp3zzmzdgtwk34m3miae24str5vtnj/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installReal/lib:/storage/dftfeDependenciesNoMKL/petsc/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/openmpi-5.0.6-ytficipzlncogz4ktqkzkwszqj7jw2ne/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib:"
         NEW_RPATH "/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg/targets/x86_64-linux/lib:/storage/dftfeDependenciesNoMKL/dealii/installReal/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/nccl-2.23.4-1-xyspmp23glxb4slgne4xpemahjrkuyrj/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/zlib-ng-2.2.3-nqbp3zzmzdgtwk34m3miae24str5vtnj/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installReal/lib:/storage/dftfeDependenciesNoMKL/petsc/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/openmpi-5.0.6-ytficipzlncogz4ktqkzkwszqj7jw2ne/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeReal.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe"
         RPATH "/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg/targets/x86_64-linux/lib:/storage/dftfeDependenciesNoMKL/dealii/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/zlib-ng-2.2.3-nqbp3zzmzdgtwk34m3miae24str5vtnj/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installReal/lib:/storage/dftfeDependenciesNoMKL/petsc/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/openmpi-5.0.6-ytficipzlncogz4ktqkzkwszqj7jw2ne/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/nccl-2.23.4-1-xyspmp23glxb4slgne4xpemahjrkuyrj/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/pa01/Mehul/DFTFE/build_local/release/real/dftfe")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe"
         OLD_RPATH "/home/pa01/Mehul/DFTFE/build_local/release/real:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg/targets/x86_64-linux/lib:/storage/dftfeDependenciesNoMKL/dealii/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/zlib-ng-2.2.3-nqbp3zzmzdgtwk34m3miae24str5vtnj/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installReal/lib:/storage/dftfeDependenciesNoMKL/petsc/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/openmpi-5.0.6-ytficipzlncogz4ktqkzkwszqj7jw2ne/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/nccl-2.23.4-1-xyspmp23glxb4slgne4xpemahjrkuyrj/lib:"
         NEW_RPATH "/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg/targets/x86_64-linux/lib:/storage/dftfeDependenciesNoMKL/dealii/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/zlib-ng-2.2.3-nqbp3zzmzdgtwk34m3miae24str5vtnj/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installReal/lib:/storage/dftfeDependenciesNoMKL/petsc/installReal/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/openmpi-5.0.6-ytficipzlncogz4ktqkzkwszqj7jw2ne/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib:/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/nccl-2.23.4-1-xyspmp23glxb4slgne4xpemahjrkuyrj/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/pa01/Mehul/DFTFE/include/")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/pa01/Mehul/DFTFE/build_local/release/real/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
