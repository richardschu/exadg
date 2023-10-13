/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_GRID_LAMBDA_FUNCTIONS_AMR_H_
#define INCLUDE_EXADG_GRID_LAMBDA_FUNCTIONS_AMR_H_

// deal.II
#include <deal.II/base/exceptions.h>

namespace ExaDG
{
/**
 * Helper functions for adaptive mesh refinement to be overwritten in respective driver.
 */
template<int dim, typename Number>
class HelpersAMR
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  /**
   * Setup relevant parts similar to driver setup function.
   */
  std::function<void()> setup = []() {
    AssertThrow(false, dealii::ExcMessage("The function setup() has not been implemented."));
  };

  /**
   * Flag cells of the triangulation for refinement based on a solution vector
   * and return flag indicating any refinement/coarsening done.
   */
  std::function<bool(dealii::Triangulation<dim> &, VectorType const &)> set_refine_flags =
    [](dealii::Triangulation<dim> &, VectorType const &) {
      AssertThrow(false,
                  dealii::ExcMessage("The function set_refine_flags() has not been implemented."));

      return false;
    };
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_TIME_INTEGRATION_LAMBDA_FUNCTIONS_AMR_H_ */
