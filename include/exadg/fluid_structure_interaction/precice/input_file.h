/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_INPUT_FILE_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_INPUT_FILE_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/fluid_structure_interaction/precice/precice_parameters.h>
#include <exadg/fluid_structure_interaction/user_interface/declare_get_application.h>
#include <exadg/utilities/check_input_file.h>
#include <exadg/utilities/general_parameters.h>

namespace ExaDG
{
namespace FSI
{
namespace preCICE
{
/**
 * Declares the parameters of all classes reading the input file of the preCICE
 * fluid-structure interaction solver. The parameter objects are destroyed on
 * return, i.e., `prm` can be printed but not parsed.
 */
template<int dim, typename Number>
void
add_parameters_input_file(dealii::ParameterHandler & prm,
                          std::string const &        input_file,
                          MPI_Comm const &           mpi_comm)
{
  GeneralParameters general;
  general.add_parameters(prm);

  ExaDG::preCICE::ConfigurationParameters precice_parameters;
  precice_parameters.add_parameters(prm);

  FSI::get_application<dim, Number>(input_file, mpi_comm)->add_parameters(prm);
}

/**
 * Throws an exception if the input file contains entries that are not declared
 * in add_parameters_input_file(), e.g., misspelled ones.
 */
template<int dim, typename Number>
void
check_input_file(std::string const & input_file, MPI_Comm const & mpi_comm)
{
  dealii::ParameterHandler prm;
  add_parameters_input_file<dim, Number>(prm, input_file, mpi_comm);
  ExaDG::check_input_file(input_file, prm);
}

} // namespace preCICE
} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_INPUT_FILE_H_ */
