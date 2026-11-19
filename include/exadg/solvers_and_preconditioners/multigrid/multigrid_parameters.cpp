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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>

namespace ExaDG
{
bool
MultigridData::involves_h_transfer() const
{
  if(type != MultigridType::pMG and type != MultigridType::cpMG and type != MultigridType::pcMG and
     type != MultigridType::cMG)
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool
MultigridData::involves_c_transfer() const
{
  if(type == MultigridType::hMG or type == MultigridType::pMG or type == MultigridType::hpMG or
     type == MultigridType::phMG)
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool
MultigridData::involves_p_transfer() const
{
  if(type != MultigridType::hMG and type != MultigridType::hcMG and type != MultigridType::chMG and
     type != MultigridType::cMG)
  {
    return true;
  }
  else
  {
    return false;
  }
}

void
AMGData::print(dealii::ConditionalOStream const & pcout) const
{
  print_parameter(pcout, "    AMG type", amg_type);

  if(amg_type == AMGType::ML)
  {
#ifdef DEAL_II_WITH_TRILINOS
    print_parameter(pcout, "    Smoother sweeps", ml_data.smoother_sweeps);
    print_parameter(pcout, "    Number of cycles", ml_data.n_cycles);
    print_parameter(pcout, "    Smoother type", ml_data.smoother_type);
#endif
  }
  else if(amg_type == AMGType::BoomerAMG)
  {
#ifdef DEAL_II_WITH_PETSC
    print_parameter(pcout, "    Smoother sweeps", boomer_data.n_sweeps_coarse);
    print_parameter(pcout, "    Number of cycles", boomer_data.max_iter);
    print_parameter(pcout, "    Smoother type down", boomer_data.relaxation_type_down);
    print_parameter(pcout, "    Smoother type up", boomer_data.relaxation_type_up);
    print_parameter(pcout, "    Smoother type coarse", boomer_data.relaxation_type_coarse);
#endif
  }
  else
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
}

void
SmootherData::print(dealii::ConditionalOStream const & pcout) const
{
  print_parameter(pcout, "Smoother", smoother);
  print_parameter(pcout, "Preconditioner smoother", preconditioner);
  print_parameter(pcout, "Iterations smoother", iterations);

  if(smoother == MultigridSmoother::Jacobi)
  {
    print_parameter(pcout, "Relaxation factor", relaxation_factor);
  }

  if(smoother == MultigridSmoother::Chebyshev)
  {
    print_parameter(pcout, "Smoothing range", smoothing_range);
    print_parameter(pcout, "Iterations eigenvalue estimation", iterations_eigenvalue_estimation);
  }
}

void
CoarseGridData::print(dealii::ConditionalOStream const & pcout) const
{
  print_parameter(pcout, "Coarse grid solver", solver);
  print_parameter(pcout, "Coarse grid preconditioner", preconditioner);

  solver_data.print(pcout);

  if(solver == MultigridCoarseGridSolver::AMG or
     preconditioner == MultigridCoarseGridPreconditioner::AMG)
  {
    amg_data.print(pcout);
  }
}

void
MultigridData::print(dealii::ConditionalOStream const & pcout) const
{
  print_parameter(pcout, "Multigrid type", type);

  if(involves_p_transfer())
  {
    print_parameter(pcout, "p-sequence", p_sequence);
  }

  print_parameter(pcout, "Min degree matrix-free", min_degree_matrix_free);

  smoother_data.print(pcout);

  coarse_problem.print(pcout);
}

} // namespace ExaDG
