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

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/fluid_structure_interaction/acceleration_schemes/linear_algebra.h>
#include <exadg/solvers_and_preconditioners/nonlinear_solvers/fixed_point_solver.h>

namespace ExaDG
{
namespace FixedPointSolver
{
template<typename Number, typename VectorType>
FixedPointSolver<Number, VectorType>::FixedPointSolver(Parameters const &           parameters,
                                                       dealii::ConditionalOStream & pcout,
                                                       std::shared_ptr<TimerTree> & timer_tree)
  : parameters(parameters), pcout(pcout), timer_tree(timer_tree), iqn_initial_call(true)
{
}

template<typename Number, typename VectorType>
unsigned int
FixedPointSolver<Number, VectorType>::solve(
  std::function<void(VectorType &)> const &                     lambda_set_up_vector,
  std::function<void(VectorType &, unsigned int const)> const & lambda_get_iterate,
  std::function<void(VectorType const &)> const &               lambda_set_iterate,
  std::function<void(VectorType &, VectorType const &, unsigned int const)> const &
                                                  lambda_fixed_point_iteration,
  std::function<bool(VectorType const &)> const & lambda_check_convergence)
{
  unsigned int iteration_counter = 0;
  if(parameters.acceleration_method == AccelerationMethod::FixedRelaxation)
  {
    VectorType x;
    lambda_set_up_vector(x);

    // Fixed point iteration with fixed relaxation parameter `omega_init`.
    bool         converged = false;
    double const omega     = parameters.omega_init;
    while(not(converged) and iteration_counter < parameters.max_iter)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      VectorType x_tilde(x);
      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      VectorType residual = x_tilde;
      residual.add(-1.0, x);
      converged = lambda_check_convergence(residual);

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
    lambda_set_up_vector(residual_old);
    lambda_set_up_vector(x);

    // Fixed point iteration with Aitken acceleration.
    bool   converged = false;
    double omega     = 1.0;
    while(not(converged) and iteration_counter < parameters.max_iter)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      VectorType x_tilde(x);
      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      VectorType residual = x_tilde;
      residual.add(-1.0, x);
      converged = lambda_check_convergence(residual);

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
    lambda_set_up_vector(x);
    lambda_set_up_vector(x_tilde);
    lambda_set_up_vector(x_tilde_old);
    lambda_set_up_vector(residual);
    lambda_set_up_vector(residual_old);

    // Fixed point iteration with IQN-ILS acceleration.
    bool converged = false;
    while(not(converged) and iteration_counter < parameters.max_iter)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      residual = x_tilde;
      residual.add(-1.0, x);
      converged = lambda_check_convergence(residual);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        if(iteration_counter == 0 and (parameters.reused_time_steps == 0 or iqn_initial_call))
        {
          x.add(parameters.omega_init, residual);

          // Update flag for future calls since this is reached only in iteration 0.
          iqn_initial_call = false;
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
            LinearAlgebra::Matrix<Number> U(iteration_counter_all);
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
    lambda_set_up_vector(x);
    lambda_set_up_vector(x_tilde);
    lambda_set_up_vector(x_tilde_old);
    lambda_set_up_vector(residual);
    lambda_set_up_vector(residual_old);
    lambda_set_up_vector(b);
    lambda_set_up_vector(b_old);

    std::shared_ptr<LinearAlgebra::Matrix<Number>> U;
    std::vector<VectorType>                        Q;

    // Fixed point iteration with IQN-IMVLS acceleration.
    bool converged = false;
    while(not(converged) and iteration_counter < parameters.max_iter)
    {
      print_solver_info_header(iteration_counter);

      lambda_get_iterate(x, iteration_counter);

      lambda_fixed_point_iteration(x_tilde, x, iteration_counter);

      // compute residual and check convergence
      residual = x_tilde;
      residual.add(-1.0, x);
      converged = lambda_check_convergence(residual);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        // compute b vector
        LinearAlgebra::inv_jacobian_times_residual(b, D_history, R_history, Z_history, residual);

        if(iteration_counter == 0 and (parameters.reused_time_steps == 0 or iqn_initial_call))
        {
          x.add(parameters.omega_init, residual);

          // Update flag for future calls since this is reached only in iteration 0.
          iqn_initial_call = false;
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
            U = std::make_shared<LinearAlgebra::Matrix<Number>>(iteration_counter);
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

  print_solver_info_converged(iteration_counter);

  return iteration_counter;
}

template<typename Number, typename VectorType>
void
FixedPointSolver<Number, VectorType>::print_solver_info_header(unsigned int const iteration) const
{
  if(parameters.print_solver_info)
  {
    pcout << std::endl
          << "======================================================================" << std::endl
          << " Fixed-point iteration: iteration counter = " << std::left << std::setw(8)
          << iteration << std::endl
          << "======================================================================" << std::endl;
  }
}

template<typename Number, typename VectorType>
void
FixedPointSolver<Number, VectorType>::print_solver_info_converged(
  unsigned int const iteration) const
{
  if(parameters.print_solver_info)
  {
    pcout << std::endl
          << "Fixed-point iteration converged in " << iteration << " iterations." << std::endl;
  }
}

template class FixedPointSolver<float, dealii::LinearAlgebra::distributed::Vector<float>>;
template class FixedPointSolver<double, dealii::LinearAlgebra::distributed::Vector<double>>;

} // namespace FixedPointSolver
} // namespace ExaDG
