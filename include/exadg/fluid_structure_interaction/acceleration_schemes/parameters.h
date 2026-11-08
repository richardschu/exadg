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

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/nonlinear_solvers/fixed_point_solver.h>
#include <exadg/utilities/enum_patterns.h>

namespace ExaDG
{
namespace FSI
{
enum class UpdateMethod
{
  Undefined,
  Implicit,
  GeometricExplicit,
  ImplicitPressureStructure,
  ImplicitVelocityStructure
};

// The Robin coupling condition
// sigma_s * n_s + alpha * v_s = sigma_f * n_s + alpha * g
// is applied on the structure side only, where g is the velocity given to the
// fluid as Dirichlet data in the current coupling iteration. The fluid sees a
// pure Dirichlet condition u_f = g. The alpha terms cancel at convergence, so
// alpha affects the convergence of the partitioned scheme, but the converged
// solution only up to the tolerance of the partitioned scheme.
enum class CouplingMethod
{
  Undefined,
  DirichletNeumann,
  DirichletRobinFixedParameter,
  DirichletRobinAdaptiveParameter
};

// Transfer of the fluid traction to the structure:
//  - PointEvaluation: the fluid traction sigma_f n_f is interpolated into the
//    fluid velocity space and evaluated in the quadrature points of the
//    structure, where it is applied as NeumannCached boundary condition.
//  - Residual: the fluid computes the force in its quadrature points of the
//    interface, i.e., the consistent (interior penalty) flux of the viscous
//    term and the pressure, integrated over the deformed fluid interface. The
//    structure receives the corresponding force vector via the transpose of
//    the evaluation of the structure velocity in these points, which makes the
//    transfer energy conserving and applicable to non-matching grids. The force
//    is a dead load in the structure within a coupling iteration. Since the
//    force is integrated over the deformed interface, it contains the change of
//    the interface area also for linear elasticity, whereas PointEvaluation
//    applies the traction on the reference interface in this case.
// The default is PointEvaluation.
enum class TractionTransfer
{
  PointEvaluation,
  Residual
};

// The initial guess used in the FSI coupling loop is computed using an extrapolation of suitable
// order (defined within the single-field time integrators) or the last iterate from the previous
// time step.
enum class InitialGuessCouplingScheme
{
  SolutionExtrapolatedToEndOfTimeStep,
  ConvergedSolutionOfPreviousTimeStep
};

struct Parameters
{
  Parameters()
    : acceleration_method(FixedPointSolver::AccelerationMethod::Undefined),
      update_method(UpdateMethod::Implicit),
      coupling_method(CouplingMethod::DirichletNeumann),
      robin_parameter_scale(0.0),
      abs_tol(1.e-12),
      rel_tol(1.e-3),
      omega_init(0.1),
      initial_guess_coupling_scheme(
        InitialGuessCouplingScheme::SolutionExtrapolatedToEndOfTimeStep),
      reused_time_steps(0),
      partitioned_max_iter(100),
      geometric_tolerance(1.e-10),
      traction_transfer(TractionTransfer::PointEvaluation)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "FSI")
  {
    prm.enter_subsection(subsection_name);
    {
      prm.add_parameter("AccelerationMethod",
                        acceleration_method,
                        "Acceleration method.",
                        Patterns::Enum<FixedPointSolver::AccelerationMethod>(),
                        true);
      prm.add_parameter(
        "UpdateMethod", update_method, "Update method.", Patterns::Enum<UpdateMethod>(), false);
      prm.add_parameter("CouplingMethod",
                        coupling_method,
                        "Coupling method.",
                        Patterns::Enum<CouplingMethod>(),
                        false);
      prm.add_parameter("RobinParameterScale",
                        robin_parameter_scale,
                        "Robin parameter alpha in kg/(m^2 s) for "
                        "DirichletRobinFixedParameter. For "
                        "DirichletRobinAdaptiveParameter, a length l in m "
                        "giving alpha = l * rho_f / dt, e.g., l = gamma * mu_max "
                        "with mu_max the largest eigenvalue of the added-mass "
                        "operator and gamma ~ 0.01, see Badia, Nobile and "
                        "Vergara, J. Comput. Phys. 227 (2008).",
                        dealii::Patterns::Double(),
                        false);
      prm.add_parameter(
        "AbsTol", abs_tol, "Absolute solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter(
        "RelTol", rel_tol, "Relative solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter("OmegaInit",
                        omega_init,
                        "Initial relaxation parameter.",
                        dealii::Patterns::Double(0.0, 1.0),
                        true);
      prm.add_parameter("InitialGuessCouplingScheme",
                        initial_guess_coupling_scheme,
                        "Scheme for initial guess for the FSI coupling loop at every time step.",
                        Patterns::Enum<InitialGuessCouplingScheme>(),
                        false);
      prm.add_parameter("ReusedTimeSteps",
                        reused_time_steps,
                        "Number of time steps reused for acceleration.",
                        dealii::Patterns::Integer(0, 100),
                        false);
      prm.add_parameter("PartitionedMaxIter",
                        partitioned_max_iter,
                        "Maximum number of fixed-point iterations.",
                        dealii::Patterns::Integer(1, 1000),
                        true);
      prm.add_parameter("GeometricTolerance",
                        geometric_tolerance,
                        "Tolerance used to locate points at FSI interface.",
                        dealii::Patterns::Double(0.0, 1.0),
                        false);
      prm.add_parameter("TractionTransfer",
                        traction_transfer,
                        "Transfer of the fluid traction to the structure.",
                        Patterns::Enum<TractionTransfer>(),
                        false);
    }
    prm.leave_subsection();
  }

  FixedPointSolver::AccelerationMethod acceleration_method;
  UpdateMethod                         update_method;
  CouplingMethod                       coupling_method;
  double                               robin_parameter_scale;
  double                               abs_tol;
  double                               rel_tol;
  double                               omega_init;
  InitialGuessCouplingScheme           initial_guess_coupling_scheme;
  unsigned int                         reused_time_steps;
  unsigned int                         partitioned_max_iter;

  // tolerance used to locate points at the fluid-structure interface
  double geometric_tolerance;

  // transfer of the fluid traction to the structure
  TractionTransfer traction_transfer;
};

} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_ */
