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

 // ExaDG
#include <exadg/fluid_structure_interaction/acceleration_schemes/partitioned_solver.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
PartitionedSolver<dim, Number>::PartitionedSolver(Parameters const & parameters,
                                                  MPI_Comm const &   comm)
  : parameters(parameters),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    partitioned_iterations({0, 0})
{
  timer_tree = std::make_shared<TimerTree>();
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::setup(std::shared_ptr<SolverFluid<dim, Number>>     fluid_in,
                                      std::shared_ptr<SolverStructure<dim, Number>> structure_in)
{
  fluid     = fluid_in;
  structure = structure_in;

  // Set up fixed-point solver with persistent memory copying parameters.
  FixedPointSolver::Parameters solver_parameters;
  solver_parameters.acceleration_method = parameters.acceleration_method;
  solver_parameters.abs_tol             = parameters.abs_tol;
  solver_parameters.rel_tol             = parameters.rel_tol;
  solver_parameters.omega_init          = parameters.omega_init;
  solver_parameters.reused_time_steps   = parameters.reused_time_steps;
  solver_parameters.max_iter            = parameters.partitioned_max_iter;
  solver_parameters.print_solver_info   = fluid->time_integrator->print_solver_info();

  fixed_point_solver =
    std::make_shared<FixedPointSolver::FixedPointSolver<Number, VectorType>>(solver_parameters,
                                                                             pcout,
                                                                             timer_tree);
}

template<int dim, typename Number>
bool
PartitionedSolver<dim, Number>::check_convergence(VectorType const & residual) const
{
  double const residual_norm = residual.l2_norm();
  double const ref_norm_abs  = std::sqrt(structure->pde_operator->get_number_of_dofs());
  double const ref_norm_rel  = structure->time_integrator->get_velocity_np().l2_norm() *
                              structure->time_integrator->get_time_step_size();

  bool const converged = (residual_norm < parameters.abs_tol * ref_norm_abs) or
                         (residual_norm < parameters.rel_tol * ref_norm_rel);

  return converged;
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::print_iterations(dealii::ConditionalOStream const & pcout) const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  names = {"Partitioned iterations"};
  iterations_avg.resize(1);
  iterations_avg[0] =
    (double)partitioned_iterations.second / std::max(1.0, (double)partitioned_iterations.first);

  print_list_of_iterations(pcout, names, iterations_avg);
}

template<int dim, typename Number>
std::shared_ptr<TimerTree>
PartitionedSolver<dim, Number>::get_timings() const
{
  return timer_tree;
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::get_structure_velocity(VectorType & velocity_structure,
                                                       unsigned int iteration) const
{
  if(iteration == 0)
  {
    if(parameters.initial_guess_coupling_scheme ==
       InitialGuessCouplingScheme::SolutionExtrapolatedToEndOfTimeStep)
    {
      structure->time_integrator->extrapolate_velocity_to_np(velocity_structure);
    }
    else if(parameters.initial_guess_coupling_scheme ==
            InitialGuessCouplingScheme::ConvergedSolutionOfPreviousTimeStep)
    {
      velocity_structure = structure->time_integrator->get_velocity_n();
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Behavior for this `InitialGuessCouplingScheme` is not defined."));
    }
  }
  else
  {
    velocity_structure = structure->time_integrator->get_velocity_np();
  }
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::solve(
  std::function<void(VectorType &, VectorType const &, unsigned int const)> const &
    apply_dirichlet_robin_scheme)
{
  // Define lambda functions for fixed-point iteration.
  auto const lambda_set_up_vector = [&](VectorType & vector) {
    structure->pde_operator->initialize_dof_vector(vector);
  };
  auto const lambda_get_iterate = [&](VectorType & vector, unsigned int const iteration_counter) {
    if(iteration_counter == 0)
    {
      if(this->parameters.initial_guess_coupling_scheme ==
         InitialGuessCouplingScheme::SolutionExtrapolatedToEndOfTimeStep)
      {
        structure->time_integrator->extrapolate_displacement_to_np(vector);
      }
      else if(this->parameters.initial_guess_coupling_scheme ==
              InitialGuessCouplingScheme::ConvergedSolutionOfPreviousTimeStep)
      {
        vector = structure->time_integrator->get_displacement_n();
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      "Behavior for this `InitialGuessCouplingScheme` is not defined."));
      }
    }
    else
    {
      vector = structure->time_integrator->get_displacement_np();
    }
  };
  auto const lambda_set_iterate = [&](VectorType const & vector) {
    structure->time_integrator->set_displacement(vector);
  };
  auto const lambda_fixed_point_iteration =
    [&](VectorType & dst, VectorType const & src, unsigned int const iteration_counter) {
      apply_dirichlet_robin_scheme(dst, src, iteration_counter);
    };
  auto const lambda_check_convergence = [&](VectorType const & residual) {
    return check_convergence(residual);
  };

  unsigned int const iteration_counter = fixed_point_solver->solve(lambda_set_up_vector,
                                                                   lambda_get_iterate,
                                                                   lambda_set_iterate,
                                                                   lambda_fixed_point_iteration,
                                                                   lambda_check_convergence);

  // Update counters to compute average FSI iterations over time steps.
  partitioned_iterations.first += 1;
  partitioned_iterations.second += iteration_counter;
}

template class PartitionedSolver<2, float>;
template class PartitionedSolver<2, double>;
template class PartitionedSolver<3, float>;
template class PartitionedSolver<3, double>;

}
}