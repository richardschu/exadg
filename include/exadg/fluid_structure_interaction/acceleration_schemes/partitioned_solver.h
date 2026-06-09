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

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_

// ExaDG
#include <exadg/fluid_structure_interaction/acceleration_schemes/parameters.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/fluid.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>
#include <exadg/solvers_and_preconditioners/nonlinear_solvers/fixed_point_solver.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
class PartitionedSolver
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  PartitionedSolver(Parameters const & parameters, MPI_Comm const & comm);

  void
  setup(std::shared_ptr<SolverFluid<dim, Number>>     fluid_in,
        std::shared_ptr<SolverStructure<dim, Number>> structure_in);

  void
  solve(std::function<void(VectorType &, VectorType const &, unsigned int)> const &
          apply_dirichlet_robin_scheme);

  void
  print_iterations(dealii::ConditionalOStream const & pcout) const;

  std::shared_ptr<TimerTree>
  get_timings() const;

  void
  get_structure_velocity(VectorType & velocity_structure, unsigned int const iteration) const;

private:
  bool
  check_convergence(VectorType const & residual) const;

  Parameters parameters;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  std::shared_ptr<SolverFluid<dim, Number>>     fluid;
  std::shared_ptr<SolverStructure<dim, Number>> structure;

  // Computation time (wall clock time).
  std::shared_ptr<TimerTree> timer_tree;

  // Fixed-point solver with persistent memory initialized at setup.
  std::shared_ptr<FixedPointSolver::FixedPointSolver<Number, VectorType>> fixed_point_solver;

  /*
   * The first number counts the number of time steps, the second number the total number
   * (accumulated over all time steps) of iterations of the partitioned FSI scheme.
   */
  std::pair<unsigned int, unsigned long long> partitioned_iterations;
};

} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_ */
