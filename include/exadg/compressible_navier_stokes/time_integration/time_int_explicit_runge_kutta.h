/*
 * time_int_explicit_runge_kutta.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/explicit_runge_kutta.h>
#include <exadg/time_integration/ssp_runge_kutta.h>
#include <exadg/time_integration/time_int_explicit_runge_kutta_base.h>

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

// forward declarations
class InputParameters;

template<typename Number>
class PostProcessorInterface;

namespace Interface
{
template<typename Number>
class Operator;
}

template<typename Number>
class TimeIntExplRK : public TimeIntExplRKBase<Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number> Operator;

  TimeIntExplRK(std::shared_ptr<Operator>                       operator_in,
                InputParameters const &                         param_in,
                unsigned int const                              refine_steps_time_in,
                MPI_Comm const &                                mpi_comm_in,
                bool const                                      print_wall_times_in,
                std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  initialize_time_integrator();

  void
  initialize_vectors();

  void
  initialize_solution();

  void
  detect_instabilities() const;

  void
  postprocessing() const;

  void
  solve_timestep();

  bool
  print_solver_info() const;

  void
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

  void
  calculate_pressure();

  void
  calculate_velocity();

  void
  calculate_temperature();

  void
  calculate_vorticity();

  void
  calculate_divergence();

  std::shared_ptr<Operator> pde_operator;

  std::shared_ptr<ExplicitTimeIntegrator<Operator, VectorType>> rk_time_integrator;

  InputParameters const & param;

  unsigned int const refine_steps_time;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // monitor the L2-norm of the solution vector in order to detect instabilities
  mutable double l2_norm;

  // time step calculation
  double const cfl_number;
  double const diffusion_number;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ \
        */