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

// ExaDG
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>

namespace ExaDG
{
void
SolverData::print(dealii::ConditionalOStream const & pcout) const
{
  // `SolverData` can also be used to control tolerances without specifying the solver type.
  if(linear_solver != LinearSolver::Undefined)
  {
    print_parameter(pcout, "Solver", linear_solver);
  }

  print_parameter(pcout, "Maximum number of iterations", max_iter);
  print_parameter(pcout, "Absolute solver tolerance", abs_tol);
  print_parameter(pcout, "Relative solver tolerance", rel_tol);
  print_parameter(pcout, "Relative tolerance reference", rel_tol_reference);

  // Print maximum Krylov space size for relevant methods or when using the default
  // `LinearSolver`.
  if(linear_solver == LinearSolver::FGMRES or linear_solver == LinearSolver::GMRES or
     linear_solver == LinearSolver::Undefined)
  {
    print_parameter(pcout, "Maximum size of Krylov space", max_krylov_size);
  }
}

} // namespace ExaDG
