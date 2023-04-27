/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/numbers.h>

#ifndef INCLUDE_EXADG_GRID_GRID_MOTION_INTERFACE_H_
#  define INCLUDE_EXADG_GRID_GRID_MOTION_INTERFACE_H_

namespace ExaDG
{
/**
 * Pure-virtual interface class for moving grid functionality.
 */
template<int dim, typename Number>
class GridMotionInterface
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Destructor.
   */
  virtual ~GridMotionInterface()
  {
  }

  /**
   * Updates the mapping, i.e., moves the grid.
   */
  virtual void
  update(double const time, bool const print_solver_info, types::time_step time_step_number) = 0;

  /**
   * Print the number of iterations for PDE type grid motion problems.
   */
  virtual void
  print_iterations() const
  {
    AssertThrow(false, dealii::ExcMessage("Has to be overwritten by derived classes."));
  }

  /**
   * Extract the grid coordinates of the current mesh configuration and fill a dof-vector given a
   * corresponding dealii::DoFHandler object.
   */
  virtual void
  fill_grid_coordinates_vector(VectorType &                    grid_coordinates,
                               dealii::DoFHandler<dim> const & dof_handler) const = 0;

  /**
   * Return a shared pointer to dealii::Mapping<dim>.
   */
  virtual std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const = 0;
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_GRID_MOTION_INTERFACE_H_ */
