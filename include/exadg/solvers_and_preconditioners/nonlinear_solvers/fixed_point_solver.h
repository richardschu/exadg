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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_FIXED_POINT_SOLVER_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_FIXED_POINT_SOLVER_H_

// ExaDG
#include <exadg/fluid_structure_interaction/acceleration_schemes/parameters.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace FixedPointSolver
{
/**
 * Class implementing a fixed-point iteration with various acceleration methods. The actual
 * (single!) fixed-point iteration is performed in the provided lambda function
 * `lambda_fixed_point_iteration` which is called in each iteration of the fixed-point scheme. The
 * class itself only handles the fixed-point scheme including acceleration methods, but does not
 * know anything about the actual problem to be solved.
 */
template<typename Number, typename VectorType>
class FixedPointSolver
{
public:
  FixedPointSolver(Parameters const &           parameters,
                   dealii::ConditionalOStream & pcout,
                   std::shared_ptr<TimerTree> & timer_tree);

  /**
   * Solve function which requires lambda functions to manipulate data likely stored in operators
   * `lambda_set_up_vector`         set up a zero vector identical to the iterate
   * `lambda_get_iterate`           copy the iterate vector data into the provided vector
   * `lambda_set_iterate`           copy the data of the provided vector into the iterate vector
   * `lambda_fixed_point_iteration` perform a single fixed point iteration
   * `lambda_check_convergence`     check convergence of the fixed point scheme
   */
  unsigned int
  solve(std::function<void(VectorType &)> const &                     lambda_set_up_vector,
        std::function<void(VectorType &, unsigned int const)> const & lambda_get_iterate,
        std::function<void(VectorType const &)> const &               lambda_set_iterate,
        std::function<void(VectorType &, VectorType const &, unsigned int const)> const &
                                                        lambda_fixed_point_iteration,
        std::function<bool(VectorType const &)> const & lambda_check_convergence);

private:
  void
  print_solver_info_header(unsigned int const iteration) const;

  void
  print_solver_info_converged(unsigned int const iteration) const;

  Parameters                 parameters;
  dealii::ConditionalOStream pcout;
  std::shared_ptr<TimerTree> timer_tree;

  // Persistent storage and initialization flag required for quasi-Newton methods.
  bool                                                  iqn_initial_call;
  std::vector<std::shared_ptr<std::vector<VectorType>>> D_history, R_history, Z_history;
};

} // namespace FixedPointSolver
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_FIXED_POINT_SOLVER_H_ */
