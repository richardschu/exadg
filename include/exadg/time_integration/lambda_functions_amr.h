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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_LAMBDA_FUNCTIONS_AMR_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_LAMBDA_FUNCTIONS_AMR_H_

// deal.II
#include <deal.II/base/exceptions.h>

namespace ExaDG
{
/**
 * TODO
 */
template<int dim, typename Number>
class HelpersAMR
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  /**
   * TODO
   */
  std::function<dealii::DoFHandler<dim> const *()> get_dof_handler = []() {
    AssertThrow(false,
                dealii::ExcMessage("The function get_dof_handler() has not been implemented."));

    return nullptr;
  };

  /**
   * TODO
   */
  std::function<dealii::Triangulation<dim> *()> get_grid = []() {
    AssertThrow(false, dealii::ExcMessage("The function get_grid() has not been implemented."));

    return nullptr;
  };

  /**
   * TODO
   */
  std::function<void()> setup = []() {
    AssertThrow(false, dealii::ExcMessage("The function setup() has not been implemented."));
  };

  /**
   * TODO
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
