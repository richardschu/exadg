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

rm -rf CMakeFiles/ CMakeCache.txt libexadg.so libexadg.a include/exadg/configuration/config.h

WORKING_DIRECTORY="/home/richardschu/dealii-candi"

# use the dealii/build directory
DEAL=$WORKING_DIRECTORY/dealii/build
# or, alternatively, the dealii/install directory
#DEAL=$WORKING_DIRECTORY/dealii/install

EXADG_BIO=$WORKING_DIRECTORY/exadg-bio
FFTW=$WORKING_DIRECTORY/fftw/install
LIKWID=$WORKING_DIRECTORY/likwid/install

# Precompiled headers (option in CMakeLists.txt, default ON): the deal.II
# headers that ExaDG includes in most files are compiled once into
# build/CMakeFiles/exadg.dir/cmake_pch.hxx.gch (about 1 GB) and reused by the
# library, the applications and the tests. OFF saves the disk space, but the
# compilation takes longer.
PRECOMPILED_HEADERS=ON

# ccache (optional, default OFF): ccache keeps the compiled object files in a
# cache (default ~/.cache/ccache) and reuses them when a file is compiled again
# with the same source code, headers and flags, e.g. after running this script
# again (it deletes CMakeFiles/ with the object files of the library), after
# `make clean` or when switching between git branches. To use it:
#   1. install ccache, e.g. `sudo apt install ccache`,
#   2. raise the cache size (default 5 GB) if deal.II and ExaDG-Bio use it as
#      well, e.g. `ccache -o max_size=30G`,
#   3. set CCACHE=ON below and run this script again in the build folder.
# `ccache -s` shows the cache hits and the size of the cache.
# Measured (Release, 8 cores): the library, applications and tests take about
# 0.2 GB in the cache (the 1 GB precompiled header is cached as well and
# compresses well). Running this script again + `make exadg`: 259 s without
# ccache, 9 s with a filled cache; the ctest run afterwards (it recompiles the
# applications and tests) 99 s instead of 125 s. The first build with an empty
# cache takes about 7% longer.
# The sloppiness setting is required for the precompiled headers.
CCACHE=OFF

if [ "$CCACHE" = "ON" ]; then
  COMPILER_LAUNCHER="ccache;sloppiness=pch_defines,time_macros"
else
  COMPILER_LAUNCHER=""
fi

cmake \
    -D DEAL_II_DIR="$DEAL" \
    -D EXADG_WITH_EXADG_BIO=ON \
    -D EXADG_BIO_INCLUDE="$EXADG_BIO/include" \
    -D EXADG_WITH_FFTW=OFF \
    -D FFTW_LIB="$FFTW/lib" \
    -D FFTW_INCLUDE="$FFTW/include" \
    -D EXADG_WITH_LIKWID=OFF \
    -D LIKWID_LIB="$LIKWID/lib" \
    -D LIKWID_INCLUDE="$LIKWID/include" \
    -D BUILD_SHARED_LIBS=ON \
    -D EXADG_WITH_PRECOMPILED_HEADERS=$PRECOMPILED_HEADERS \
    -D CMAKE_CXX_COMPILER_LAUNCHER="$COMPILER_LAUNCHER" \
    -D PICKUP_TESTS=ON \
    ../

