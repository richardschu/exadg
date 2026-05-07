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

#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/structure/spatial_discretization/interface.h>
#include <exadg/structure/time_integration/driver_inverse_analysis.h>
#include <exadg/structure/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
DriverInverseAnalysis<dim, Number>::DriverInverseAnalysis(
  std::shared_ptr<Interface::Operator<Number>> operator_in,
  std::shared_ptr<PostProcessor<dim, Number>>  postprocessor_in,
  Parameters const &                           param_in,
  MPI_Comm const &                             mpi_comm_in,
  bool const                                   is_test_in)
  : pde_operator(operator_in),
    postprocessor(postprocessor_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    is_test(is_test_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
    last_load_increment(param.load_increment),
    use_extrapolation(param.use_extrapolation),
    step_number(1),
    inverse_analysis_solver_data(param.inverse_analysis_solver_data),
    timer_tree(new TimerTree()),
    iterations({0, {0, 0}})
{
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::setup()
{
  AssertThrow(param.large_deformation,
              dealii::ExcMessage("DriverInverseAnalysis makes only sense for nonlinear problems. "
                                 "For linear problems, use DriverSteady instead."));

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by interpolation of initial data
  initialize_solution();
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::solve()
{
  dealii::Timer timer;
  timer.restart();

  postprocessing(true /* errors_only */, false /* export_configuration */);

  do_solve();

  postprocessing(false /* errors_only */, true /* export_configuration */);

  timer_tree->insert({"DriverInverseAnalysis"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(param.large_deformation)
  {
    names = {"Nonlinear iterations",
             "Linear iterations (accumulated)",
             "Linear iterations (per nonlinear it.)"};

    iterations_avg.resize(3);
    iterations_avg[0] =
      (double)std::get<0>(iterations.second) / std::max(1., (double)iterations.first);
    iterations_avg[1] =
      (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
    if(iterations_avg[0] > std::numeric_limits<double>::min())
      iterations_avg[2] = iterations_avg[1] / iterations_avg[0];
    else
      iterations_avg[2] = iterations_avg[1];
  }
  else // linear
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  print_list_of_iterations(pcout, names, iterations_avg);
}

template<int dim, typename Number>
std::shared_ptr<TimerTree>
DriverInverseAnalysis<dim, Number>::get_timings() const
{
  return timer_tree;
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::do_solve()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Solving inverse problem ..." << std::endl << std::flush;

  // perform time loop
  double load_factor    = 0.0;
  double load_increment = param.load_increment;
  // Step 0 is a pre step with a smaller load factor in order to make solving step 1 easier.
  step_number = 0;
  // In the first load step, we can not extrapolate the solution, so we solve the problem for a much
  // smaller load factor and afterwards extrapolate the solution to the actual load factor in order
  // to solve the first load step.
  double const reduction_load_factor_step_0 = 0.01;
  if(step_number == 0)
    load_increment *= reduction_load_factor_step_0;

  // Once the full load is applied, iterate until the displacement increment is sufficiently small.
  bool converged = false;

  while((load_factor < 1.0 - eps_load_factor) or not converged)
  {
    std::tuple<unsigned int, unsigned int> iter;

    // store old solution
    VectorType old_solution = solution;

    bool const update_preconditioner =
      this->param.update_preconditioner and
      ((this->step_number - 1) % this->param.update_preconditioner_every_time_steps == 0);

    // compute displacement for new load factor

    // reduce load increment in factors of 2 until the current
    // step can be solved successfully
    bool         success        = false;
    unsigned int re_try_counter = 0;
    while(not(success) and re_try_counter < 10)
    {
      try
      {
        // extrapolate solution
        if(use_extrapolation)
        {
          solution.add(load_increment / last_load_increment, displacement_increment);
        }

        iter    = solve_step(load_factor + load_increment, update_preconditioner);
        success = true;
      }
      catch(std::exception & exc)
      {
        pcout << "  Exception thrown when solving the current load step:" << std::endl
              << std::endl
              << "  " << exc.what() << std::endl
              << std::flush;

        // undo changes in solution vector
        solution = old_solution;
        ++re_try_counter;

        // reduce load increment by factor of 2
        load_increment *= 0.5;
        pcout << std::endl
              << "  Could not solve non-linear problem. Reduce load increment to " << load_increment
              << "." << std::endl
              << std::flush;
      }
      catch(...)
      {
        AssertThrow(false,
                    dealii::ExcMessage("Unknown exception thrown within current load step solve."));
      }
    }

    AssertThrow(success,
                dealii::ExcMessage(
                  "Could not solve inverse problem even after reducing the load increment."));

    // calculate increment as new_solution - old_solution
    displacement_increment = solution;
    displacement_increment.add(-1.0, old_solution);

    // check convergence of inverse analysis
    converged = check_convergence(displacement_increment,
                                  solution,
                                  step_number,
                                  load_factor + load_increment);

    // Update the mapping to give *initial* reference configuration shifted by `-solution`.
    solution *= -1.0;
    pde_operator->shift_reference_configuration(solution);
    solution *= -1.0;

    iterations.first += 1;
    std::get<0>(iterations.second) += std::get<0>(iter);
    std::get<1>(iterations.second) += std::get<1>(iter);

    // Keep loading constant after initial ramp.
    if(load_factor + load_increment >= 1.0 - eps_load_factor)
    {
      load_factor    = 1.0;
      load_increment = 0.0;
    }
    else
    {
      // increment load factor
      last_load_increment = load_increment;
      load_factor += load_increment;

      // re-init increment for next load step
      if(step_number == 0)
        load_increment = param.load_increment - last_load_increment;
      else
        load_increment = param.load_increment;

      // make sure to hit maximum load exactly
      if(load_factor + load_increment >= 1.0)
        load_increment = 1.0 - load_factor;
    }

    // finally, increment step number
    ++step_number;
  }

  pcout << std::endl << "... done!" << std::endl;

  timer_tree->insert({"DriverInverseAnalysis", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
bool
DriverInverseAnalysis<dim, Number>::check_convergence(VectorType const & update,
                                                      VectorType const & iterate,
                                                      unsigned int const step_number,
                                                      double const       load_factor_plus_increment)
{
  // Compute relative and absolute errors once load is fully applied.
  bool const load_fully_applied = (load_factor_plus_increment >= 1.0 - eps_load_factor);
  if(load_fully_applied)
  {
    double const abs_error = update.l2_norm();
    double const rel_error = abs_error / iterate.l2_norm();

    pcout << "\n"
          << "Inverse analysis errors :\n"
          << std::scientific << std::setprecision(5)
          << "  ||d_k+1 - d_k||           = " << abs_error << "\n"
          << "  ||d_k+1 - d_k||/||d_k+1|| = " << rel_error << "\n";

    bool const converged = (abs_error < inverse_analysis_solver_data.abs_tol) or
                           (rel_error < inverse_analysis_solver_data.rel_tol);

    if(not(converged))
    {
      AssertThrow(step_number < inverse_analysis_solver_data.max_iter,
                  dealii::ExcMessage(
                    "Inverse analysis did not converge within the maximum number of iterations."
                    "Consider increasing the number of load steps or relaxing the tolerances."));
    }

    return converged;
  }
  else
  {
    return false;
  }
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::initialize_vectors()
{
  pde_operator->initialize_dof_vector(solution);

  pde_operator->initialize_dof_vector(displacement_increment);
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::initialize_solution()
{
  pde_operator->prescribe_initial_displacement(solution, 0.0 /* time */);
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::output_solver_info_header(double const load_factor)
{
  pcout << std::endl
        << print_horizontal_line() << std::endl
        << std::endl
        << " Solve non-linear problem for load factor = " << std::scientific << std::setprecision(4)
        << load_factor << std::endl
        << print_horizontal_line() << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
DriverInverseAnalysis<dim, Number>::solve_step(double const load_factor,
                                               bool const   update_preconditioner)
{
  dealii::Timer timer;
  timer.restart();

  output_solver_info_header(load_factor);

  VectorType const const_vector; // will not be used

  auto const iter = pde_operator->solve_nonlinear(solution,
                                                  const_vector,
                                                  0.0 /* no acceleration term */,
                                                  0.0 /* no damping term */,
                                                  load_factor /* = time */,
                                                  update_preconditioner);

  unsigned int const N_iter_nonlinear = std::get<0>(iter);
  unsigned int const N_iter_linear    = std::get<1>(iter);

  if(not(is_test))
    print_solver_info_nonlinear(pcout, N_iter_nonlinear, N_iter_linear, timer.wall_time());

  return iter;
}

template<int dim, typename Number>
void
DriverInverseAnalysis<dim, Number>::postprocessing(bool const errors_only,
                                                   bool const export_configuration) const
{
  dealii::Timer timer;
  timer.restart();

  // The solution postprocessed is the displacement vector describing the mapping from the initial
  // reference configuration to the stress-free configuration. Since the `solution` is the solution
  // to the forward elasticity problem from the iteratively updated current reference configuration,
  // the vector`s sign is inverted for the standard output such that one can get the final reference
  // configuration by mapping with the solution vector provided in the output.
  VectorType tmp(solution);
  tmp *= -1.0;
  postprocessor->do_postprocessing(tmp, errors_only);

  // For comparison, output the current reference configuration considered and the `solution`
  // vector. Mapping the current reference configuration with that vector will yield the initial
  // reference configuration up to the specified tolerance. The mapping is not immediately available
  // after setup, only after calling `NonLinearOperator::set_solution_linearization()`.
  if(export_configuration)
  {
    pde_operator->export_configuration(postprocessor->get_data().output_data.directory, solution);
  }

  timer_tree->insert({"DriverInverseAnalysis", "Postprocessing"}, timer.wall_time());
}

template class DriverInverseAnalysis<2, float>;
template class DriverInverseAnalysis<2, double>;

template class DriverInverseAnalysis<3, float>;
template class DriverInverseAnalysis<3, double>;

} // namespace Structure
} // namespace ExaDG
