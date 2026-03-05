# CMake generated Testfile for 
# Source directory: /home/pa01/Mehul/DFTFE/tests/dft/pseudopotential/complex/d4
# Build directory: /home/pa01/Mehul/DFTFE/build_gpu/release/complex/tests/dft/pseudopotential/complex/d4
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(d4/d4_graphite.mpirun=24.release "/usr/bin/cmake" "-DTRGT=d4.d4_graphite.mpirun24.release.test" "-DTEST=d4/d4_graphite.mpirun=24.release" "-DEXPECT=PASSED" "-DBINARY_DIR=/home/pa01/Mehul/DFTFE/build_gpu/release/complex" "-P" "/storage/dftfeDependenciesNoMKL/dealii/installComplex/share/deal.II/scripts/run_test.cmake")
set_tests_properties(d4/d4_graphite.mpirun=24.release PROPERTIES  FIXTURES_REQUIRED "test_dependency/dftfe_exe.executable" LABEL "d4" PROCESSORS "24" TIMEOUT "5000" WORKING_DIRECTORY "/home/pa01/Mehul/DFTFE/build_gpu/release/complex/tests/dft/pseudopotential/complex/d4/d4_graphite.release/mpirun=24" _BACKTRACE_TRIPLES "/storage/dftfeDependenciesNoMKL/dealii/installComplex/share/deal.II/macros/macro_deal_ii_add_test.cmake;619;add_test;/storage/dftfeDependenciesNoMKL/dealii/installComplex/share/deal.II/macros/macro_deal_ii_pickup_tests.cmake;383;deal_ii_add_test;/home/pa01/Mehul/DFTFE/tests/dft/pseudopotential/complex/d4/CMakeLists.txt;3;DEAL_II_PICKUP_TESTS;/home/pa01/Mehul/DFTFE/tests/dft/pseudopotential/complex/d4/CMakeLists.txt;0;")
