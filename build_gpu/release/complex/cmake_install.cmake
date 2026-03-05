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
  include("/home/pa01/Mehul/DFTFE/build_gpu/release/complex/tests/dft/pseudopotential/complex/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/pa01/Mehul/DFTFE/build_gpu/release/complex/tests/dft/allElectron/complex/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/pa01/Mehul/DFTFE/build_gpu/release/complex/tests/dft/pseudopotential/complex/d3/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/pa01/Mehul/DFTFE/build_gpu/release/complex/tests/dft/pseudopotential/complex/d4/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeComplex.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeComplex.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeComplex.so"
         RPATH "/storage/dftfeDependenciesNoMKL/dealii/installComplex/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installComplex/lib:/storage/dftfeDependenciesNoMKL/petsc/installComplex/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/pa01/Mehul/DFTFE/build_gpu/release/complex/libdftfeComplex.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeComplex.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeComplex.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeComplex.so"
         OLD_RPATH "/storage/dftfeDependenciesNoMKL/dealii/installComplex/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installComplex/lib:/storage/dftfeDependenciesNoMKL/petsc/installComplex/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib:"
         NEW_RPATH "/storage/dftfeDependenciesNoMKL/dealii/installComplex/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installComplex/lib:/storage/dftfeDependenciesNoMKL/petsc/installComplex/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdftfeComplex.so")
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
         RPATH "/storage/dftfeDependenciesNoMKL/dealii/installComplex/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installComplex/lib:/storage/dftfeDependenciesNoMKL/petsc/installComplex/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/pa01/Mehul/DFTFE/build_gpu/release/complex/dftfe")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dftfe"
         OLD_RPATH "/home/pa01/Mehul/DFTFE/build_gpu/release/complex:/storage/dftfeDependenciesNoMKL/dealii/installComplex/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installComplex/lib:/storage/dftfeDependenciesNoMKL/petsc/installComplex/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib:"
         NEW_RPATH "/storage/dftfeDependenciesNoMKL/dealii/installComplex/lib:/storage/dftfeDependenciesNoMKL/boost/install/lib:/storage/dftfeDependenciesNoMKL/kokkos/install/lib:/storage/dftfeDependenciesNoMKL/p4est/install/lib:/storage/dftfeDependenciesNoMKL/slepc/installComplex/lib:/storage/dftfeDependenciesNoMKL/petsc/installComplex/lib:/storage/dftfeDependenciesNoMKL/alglib/install/lib:/storage/dftfeDependenciesNoMKL/libxc/install/lib:/storage/dftfeDependenciesNoMKL/spglib/install/lib:/storage/dftfeDependenciesNoMKL/dftd/install/lib:/storage/dftfeDependenciesNoMKL/elpa/install/lib")
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
file(WRITE "/home/pa01/Mehul/DFTFE/build_gpu/release/complex/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
