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


#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_

// ExaDG
#include <exadg/fluid_structure_interaction/acceleration_schemes/linear_algebra.h>
#include <exadg/fluid_structure_interaction/acceleration_schemes/parameters.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/fluid.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>
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
  setup(std::shared_ptr<SolverFluid<dim, Number>>     fluid_,
        std::shared_ptr<SolverStructure<dim, Number>> structure_);

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

  void
  print_solver_info_header(unsigned int const iteration) const;

  void
  print_solver_info_converged(unsigned int const iteration) const;

  Parameters parameters;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  std::shared_ptr<SolverFluid<dim, Number>>     fluid;
  std::shared_ptr<SolverStructure<dim, Number>> structure;

  // required for quasi-Newton methods
  std::vector<std::shared_ptr<std::vector<VectorType>>> D_history, R_history, Z_history;

  // Computation time (wall clock time).
  std::shared_ptr<TimerTree> timer_tree;

  /*
   * The first number counts the number of time steps, the second number the total number
   * (accumulated over all time steps) of iterations of the partitioned FSI scheme.
   */
  std::pair<unsigned int, unsigned long long> partitioned_iterations;
};

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
PartitionedSolver<dim, Number>::setup(std::shared_ptr<SolverFluid<dim, Number>>     fluid_,
                                      std::shared_ptr<SolverStructure<dim, Number>> structure_)
{
  fluid     = fluid_;
  structure = structure_;
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
PartitionedSolver<dim, Number>::print_solver_info_header(unsigned int const iteration) const
{
  if(fluid->time_integrator->print_solver_info())
  {
    pcout << std::endl
          << "======================================================================" << std::endl
          << " Partitioned FSI: iteration counter = " << std::left << std::setw(8) << iteration
          << std::endl
          << "======================================================================" << std::endl;
  }
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::print_solver_info_converged(unsigned int const iteration) const
{
  if(fluid->time_integrator->print_solver_info())
  {
    pcout << std::endl
          << "Partitioned FSI iteration converged in " << iteration << " iterations." << std::endl;
  }
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
  // define lambda functions for fixed-point iteration
  auto const lambda_setup_vector = [&](VectorType & vector) {
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

  unsigned int const number_of_time_steps = partitioned_iterations.first;

  // fixed-point iteration with various acceleration methods
  unsigned int iteration_counter = 0;
  if(parameters.acceleration_method == AccelerationMethod::FixedRelaxation)
  {
    VectorType x;
    lambda_setup_vector(x);

    // coupling loop
    bool         converged = false;
    double const omega     = parameters.omega_init; // fixed relaxation parameter
    while(not(converged) and iteration_counter < parameters.partitioned_iter_max)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      VectorType x_tilde(x);
      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      VectorType residual = x_tilde;
      residual.add(-1.0, x);
      converged = check_convergence(residual);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        x.add(omega, residual);
        lambda_set_iterate(x);

        timer_tree->insert({"FixedRelaxation"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++iteration_counter;
    }
  }
  else if(parameters.acceleration_method == AccelerationMethod::Aitken)
  {
    VectorType residual_old, x;
    lambda_setup_vector(residual_old);
    lambda_setup_vector(x);

    // coupling loop
    bool   converged = false;
    double omega     = 1.0;
    while(not(converged) and iteration_counter < parameters.partitioned_iter_max)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      VectorType x_tilde(x);
      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      VectorType residual = x_tilde;
      residual.add(-1.0, x);
      converged = check_convergence(residual);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        if(iteration_counter == 0)
        {
          omega = parameters.omega_init;
        }
        else
        {
          VectorType delta_residual = residual;
          delta_residual.add(-1.0, residual_old);
          omega *= -(residual_old * delta_residual) / delta_residual.norm_sqr();
        }

        residual_old = residual;

        x.add(omega, residual);
        lambda_set_iterate(x);

        timer_tree->insert({"Aitken"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++iteration_counter;
    }
  }
  else if(parameters.acceleration_method == AccelerationMethod::IQN_ILS)
  {
    std::shared_ptr<std::vector<VectorType>> D, R;
    D = std::make_shared<std::vector<VectorType>>();
    R = std::make_shared<std::vector<VectorType>>();

    VectorType x, x_tilde, x_tilde_old, residual, residual_old;
    lambda_setup_vector(x);
    lambda_setup_vector(x_tilde);
    lambda_setup_vector(x_tilde_old);
    lambda_setup_vector(residual);
    lambda_setup_vector(residual_old);

    // coupling loop
    bool converged = false;
    while(not(converged) and iteration_counter < parameters.partitioned_iter_max)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      residual = x_tilde;
      residual.add(-1.0, x);
      converged = check_convergence(residual);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        if(iteration_counter == 0 and
           (parameters.reused_time_steps == 0 or number_of_time_steps == 0))
        {
          x.add(parameters.omega_init, residual);
        }
        else
        {
          if(iteration_counter >= 1)
          {
            // append D, R matrices
            VectorType delta_x_tilde = x_tilde;
            delta_x_tilde.add(-1.0, x_tilde_old);
            D->push_back(delta_x_tilde);

            VectorType delta_residual = residual;
            delta_residual.add(-1.0, residual_old);
            R->push_back(delta_residual);
          }

          // fill vectors (including reuse)
          std::vector<VectorType> Q = *R;
          for(auto R_q : R_history)
            for(auto delta_residual : *R_q)
              Q.push_back(delta_residual);
          std::vector<VectorType> D_all = *D;
          for(auto D_q : D_history)
            for(auto delta_x : *D_q)
              D_all.push_back(delta_x);

          AssertThrow(D_all.size() == Q.size(),
                      dealii::ExcMessage("D, Q vectors must have same size."));

          unsigned int const iteration_counter_all = Q.size();
          if(iteration_counter_all >= 1)
          {
            // compute QR-decomposition
            Matrix<Number> U(iteration_counter_all);
            compute_QR_decomposition(Q, U);

            std::vector<Number> rhs(iteration_counter_all, 0.0);
            for(unsigned int i = 0; i < iteration_counter_all; ++i)
              rhs[i] = -Number(Q[i] * residual);

            // alpha = U^{-1} rhs
            std::vector<Number> alpha(iteration_counter_all, 0.0);
            backward_substitution(U, alpha, rhs);

            // x_{k+1} = x_tilde_{k} + delta x_tilde
            x = x_tilde;
            for(unsigned int i = 0; i < iteration_counter_all; ++i)
              x.add(alpha[i], D_all[i]);
          }
          else // despite reuse, the vectors might be empty
          {
            x.add(parameters.omega_init, residual);
          }
        }

        x_tilde_old  = x_tilde;
        residual_old = residual;

        lambda_set_iterate(x);

        timer_tree->insert({"IQN-ILS"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++iteration_counter;
    }

    dealii::Timer timer;
    timer.restart();

    // Update history
    D_history.push_back(D);
    R_history.push_back(R);
    if(D_history.size() > parameters.reused_time_steps)
      D_history.erase(D_history.begin());
    if(R_history.size() > parameters.reused_time_steps)
      R_history.erase(R_history.begin());

    timer_tree->insert({"IQN-ILS"}, timer.wall_time());
  }
  else if(parameters.acceleration_method == AccelerationMethod::IQN_IMVLS)
  {
    std::shared_ptr<std::vector<VectorType>> D, R;
    D = std::make_shared<std::vector<VectorType>>();
    R = std::make_shared<std::vector<VectorType>>();

    std::vector<VectorType> B;

    VectorType x, x_tilde, x_tilde_old, residual, residual_old, b, b_old;
    lambda_setup_vector(x);
    lambda_setup_vector(x_tilde);
    lambda_setup_vector(x_tilde_old);
    lambda_setup_vector(residual);
    lambda_setup_vector(residual_old);
    lambda_setup_vector(b);
    lambda_setup_vector(b_old);

    std::shared_ptr<Matrix<Number>> U;
    std::vector<VectorType>         Q;

    // coupling loop
    bool converged = false;
    while(not(converged) and iteration_counter < parameters.partitioned_iter_max)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      residual = x_tilde;
      residual.add(-1.0, x);
      converged = check_convergence(residual);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        // compute b vector
        inv_jacobian_times_residual(b, D_history, R_history, Z_history, residual);

        if(iteration_counter == 0 and
           (parameters.reused_time_steps == 0 or number_of_time_steps == 0))
        {
          x.add(parameters.omega_init, residual);
        }
        else
        {
          x = x_tilde;
          x.add(-1.0, b);

          if(iteration_counter >= 1)
          {
            // append D, R, B matrices
            VectorType delta_x_tilde = x_tilde;
            delta_x_tilde.add(-1.0, x_tilde_old);
            D->push_back(delta_x_tilde);

            VectorType delta_residual = residual;
            delta_residual.add(-1.0, residual_old);
            R->push_back(delta_residual);

            VectorType delta_b = delta_x_tilde;
            delta_b.add(1.0, b_old);
            delta_b.add(-1.0, b);
            B.push_back(delta_b);

            // compute QR-decomposition
            U = std::make_shared<Matrix<Number>>(iteration_counter);
            Q = *R;
            compute_QR_decomposition(Q, *U);

            std::vector<Number> rhs(iteration_counter, 0.0);
            for(unsigned int i = 0; i < iteration_counter; ++i)
              rhs[i] = -Number(Q[i] * residual);

            // alpha = U^{-1} rhs
            std::vector<Number> alpha(iteration_counter, 0.0);
            backward_substitution(*U, alpha, rhs);

            for(unsigned int i = 0; i < iteration_counter; ++i)
              x.add(alpha[i], B[i]);
          }
        }

        x_tilde_old  = x_tilde;
        residual_old = residual;
        b_old        = b;

        lambda_set_iterate(x);

        timer_tree->insert({"IQN-IMVLS"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++iteration_counter;
    }

    dealii::Timer timer;
    timer.restart();

    // Update history
    D_history.push_back(D);
    R_history.push_back(R);
    if(D_history.size() > parameters.reused_time_steps)
      D_history.erase(D_history.begin());
    if(R_history.size() > parameters.reused_time_steps)
      R_history.erase(R_history.begin());

    // compute Z and add to Z_history
    std::shared_ptr<std::vector<VectorType>> Z;
    Z  = std::make_shared<std::vector<VectorType>>();
    *Z = Q; // make sure that Z has correct size
    backward_substitution_multiple_rhs(*U, *Z, Q);
    Z_history.push_back(Z);
    if(Z_history.size() > parameters.reused_time_steps)
      Z_history.erase(Z_history.begin());

    timer_tree->insert({"IQN-IMVLS"}, timer.wall_time());
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("This AccelerationMethod is not implemented."));
  }

  // Update counters to compute average FSI iterations over time steps.
  partitioned_iterations.first += 1;
  partitioned_iterations.second += iteration_counter;

  print_solver_info_converged(iteration_counter);
}

} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_ */
