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

# Reuse the precompiled headers of ExaDG in the test executables of the current
# directory, see EXADG_WITH_PRECOMPILED_HEADERS and EXADG_PICKUP_EXE().
MACRO(EXADG_REUSE_PRECOMPILED_HEADERS_IN_TESTS)

  IF(TARGET exadg AND EXADG_WITH_PRECOMPILED_HEADERS)
    GET_PROPERTY(_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    FOREACH(_target ${_targets})
      GET_TARGET_PROPERTY(_type ${_target} TYPE)
      IF(_type STREQUAL "EXECUTABLE")
        TARGET_COMPILE_OPTIONS(${_target} PRIVATE -fPIC)
        TARGET_PRECOMPILE_HEADERS(${_target} REUSE_FROM exadg)
      ENDIF()
    ENDFOREACH()
  ENDIF()

ENDMACRO(EXADG_REUSE_PRECOMPILED_HEADERS_IN_TESTS)

MACRO(EXADG_PICKUP_TESTS TARGET_NAME)

  IF(${PICKUP_TESTS})
    DEAL_II_PICKUP_TESTS(TARGET_NAME)
    EXADG_REUSE_PRECOMPILED_HEADERS_IN_TESTS()
  ENDIF() 

ENDMACRO(EXADG_PICKUP_TESTS)

MACRO(EXADG_PICKUP_TESTS)

  IF(${PICKUP_TESTS})
    DEAL_II_PICKUP_TESTS()
    EXADG_REUSE_PRECOMPILED_HEADERS_IN_TESTS()
  ENDIF() 

ENDMACRO(EXADG_PICKUP_TESTS)
