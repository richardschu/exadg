#!/bin/sh
#########################################################################
#
#                 #######               ######  #######
#                 ##                    ##   ## ##
#                 #####   ##  ## #####  ##   ## ## ####
#                 ##       ####  ## ##  ##   ## ##   ##
#                 ####### ##  ## ###### ######  #######
#
#  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
#
#  Copyright (C) 2021 by the ExaDG authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#########################################################################

rm -rf CMakeFiles/ CMakeCache.txt

DEAL=$WORKING_DIRECTORY/dealii
DEAL_INSTALL=$WORKING_DIRECTORY/dealii/install

# TPLs
P4EST=$WORKING_DIRECTORY/p4est
METIS=$WORKING_DIRECTORY/metis
TRILINOS=$WORKING_DIRECTORY/trilinos/Trilinos-trilinos-release-16-1-0/build
PETSC=$WORKING_DIRECTORY/petsc/petsc-3.14.5
HDF5=$WORKING_DIRECTORY/hdf5-1.12.2

# Note on compiler flags: Note that "-march=native" requires that the hardware
# on which you compile the code is consistent with the hardware on which you
# execute the code. If this is not the case, consider to specify the target
# hardware for compilation, e.g. "-march=haswell" or "-march=skylake-avx512"
# in case of Intel Hardware.
# For more details, see https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html.
#
# Furthermore, note that we tell deal.II to compile the tensor product
# evaluators with highly efficient code paths for higher degrees than it
# would normally do - here, we select degree 15. For degrees outside this
# range, we will fail.

# ccache (optional, default OFF): ccache keeps the compiled object files in a
# cache (default ~/.cache/ccache) and reuses them when a file is compiled again
# with the same source code, headers and flags, e.g. when switching between
# git branches back and forth. To use it:
#   1. install ccache, e.g. `sudo apt install ccache`,
#   2. raise the cache size (default 5 GB), e.g. `ccache -o max_size=30G`,
#   3. set CCACHE=ON below and run this script again in the build folder.
# `ccache -s` shows the cache hits and the size of the cache.
# Measured (DebugRelease, 8 cores): the library takes about 1.3 GB in the
# cache, each compiled test (debug + release) about 3 MB more. Rebuilding the
# whole library with unchanged sources (e.g. after switching git branches back
# and forth) takes 79 s instead of about 35 min, compiling a test in the
# cache 15 s instead of 33 s (the rest is linking). Files that were changed are
# compiled anyway. Switching CCACHE on or off rebuilds deal.II once (the
# compile commands change).
# The tests (`make setup_tests`) are separate CMake projects that do not see
# this setting: set the environment variable CMAKE_CXX_COMPILER_LAUNCHER=ccache
# when a test folder is set up for the first time, e.g.
# `CMAKE_CXX_COMPILER_LAUNCHER=ccache make setup_tests`, and run
# `cmake -D CMAKE_CXX_COMPILER_LAUNCHER=ccache .` in each folder in tests/ of
# the build folder that was set up before.
CCACHE=OFF

if [ "$CCACHE" = "ON" ]; then
  COMPILER_LAUNCHER="ccache"
else
  COMPILER_LAUNCHER=""
fi

cmake \
    -D CMAKE_BUILD_TYPE="DebugRelease" \
    -D CMAKE_C_FLAGS="-march=native -Wno-array-bounds" \
    -D CMAKE_CXX_FLAGS="-std=c++17 -march=native -Wno-array-bounds -Wno-literal-suffix -pthread -DFE_EVAL_FACTORY_DEGREE_MAX=15" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D DEAL_II_CXX_FLAGS_DEBUG="-Og" \
    -D DEAL_II_WITH_MPI:BOOL="ON" \
    -D DEAL_II_WITH_HDF5:BOOL="ON" \
    -D HDF5_DIR="$HDF5" \
    -D DEAL_II_LINKER_FLAGS="-lpthread" \
    -D DEAL_II_WITH_64BIT_INDICES="ON" \
    -D DEAL_II_WITH_P4EST:BOOL="ON" \
    -D CMAKE_INSTALL_PREFIX="$DEAL_INSTALL" \
    -D P4EST_DIR="$P4EST" \
    -D DEAL_II_WITH_METIS:BOOL="ON" \
    -D METIS_DIR:FILEPATH="$METIS" \
    -D DEAL_II_WITH_TRILINOS:BOOL="ON" \
    -D TRILINOS_DIR:FILEPATH="$TRILINOS" \
    -D DEAL_II_WITH_PETSC:BOOL="ON" \
    -D PETSC_DIR="$PETSC" \
    -D PETSC_ARCH="arch-linux-c-opt" \
    -D DEAL_II_COMPONENT_DOCUMENTATION="OFF" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    -D CMAKE_CXX_COMPILER_LAUNCHER="$COMPILER_LAUNCHER" \
    $DEAL
