# CMake generated Testfile for 
# Source directory: /home/pa01/Mehul/DFTFE/tests/dft/pseudopotential/real/d3
# Build directory: /home/pa01/Mehul/DFTFE/build_local/real/tests/dft/pseudopotential/real/d3
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(d3/d3_kryptondimer.mpirun=12.release "/usr/bin/cmake" "-DTRGT=d3.d3_kryptondimer.mpirun12.release.test" "-DTEST=d3/d3_kryptondimer.mpirun=12.release" "-DEXPECT=PASSED" "-DBINARY_DIR=/home/pa01/Mehul/DFTFE/build_local/real" "-P" "/storage/dftfeDependenciesNoMKL/dealii/installReal/share/deal.II/scripts/run_test.cmake")
set_tests_properties(d3/d3_kryptondimer.mpirun=12.release PROPERTIES  FIXTURES_REQUIRED "test_dependency/dftfe_exe.executable" LABEL "d3" PROCESSORS "12" TIMEOUT "5000" WORKING_DIRECTORY "/home/pa01/Mehul/DFTFE/build_local/real/tests/dft/pseudopotential/real/d3/d3_kryptondimer.release/mpirun=12" _BACKTRACE_TRIPLES "/storage/dftfeDependenciesNoMKL/dealii/installReal/share/deal.II/macros/macro_deal_ii_add_test.cmake;619;add_test;/storage/dftfeDependenciesNoMKL/dealii/installReal/share/deal.II/macros/macro_deal_ii_pickup_tests.cmake;383;deal_ii_add_test;/home/pa01/Mehul/DFTFE/tests/dft/pseudopotential/real/d3/CMakeLists.txt;3;DEAL_II_PICKUP_TESTS;/home/pa01/Mehul/DFTFE/tests/dft/pseudopotential/real/d3/CMakeLists.txt;0;")
