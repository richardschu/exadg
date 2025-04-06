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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

namespace ExaDG
{
namespace IncNS
{
template<int dim>
class LidVelocityRegularized : public dealii::Function<dim>
{
public:
  LidVelocityRegularized(double const & start_time,
                         double const & end_time,
                         double const & time_ramp_fraction,
                         double const & space_ramp_fraction,
                         double const & length_domain,
                         double const & velocity_scale)
    : dealii::Function<dim>(dim, 0.0),
      start_time(start_time),
      end_time(end_time),
      time_ramp_fraction(time_ramp_fraction),
      space_ramp_fraction(space_ramp_fraction),
      length_domain(length_domain),
      velocity_scale(velocity_scale)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    if(component != 0)
    {
      return 0.0;
    }

    double val = velocity_scale;

    // Apply the regularization in time.
    double const t         = this->get_time();
    double const time_ramp = start_time + std::abs(end_time - start_time) * time_ramp_fraction;
    if(t < start_time)
    {
      return 0.0;
    }
    else if(t < time_ramp)
    {
      val *= dealii::Utilities::fixed_power<2>(
        std::sin(dealii::numbers::PI * (t - start_time) / (2.0 * time_ramp)));
    }

    // The lid is on the upper y-boundary.
    double const y = p[1];
    AssertThrow(std::abs(y - length_domain) < 1e-12,
                dealii::ExcMessage("LidVelocityRegularized expects lid at y = L."));


    double const x     = p[0];
    double const l_reg = length_domain * space_ramp_fraction;
    if(x < l_reg)
    {
      val *=
        1.0 - dealii::Utilities::fixed_power<4>(std::cos(dealii::numbers::PI * x / (2.0 * l_reg)));
    }
    else if(x > length_domain - l_reg)
    {
      val *= 1.0 - dealii::Utilities::fixed_power<4>(
                     std::cos(dealii::numbers::PI * (x - length_domain) / (2.0 * l_reg)));
    }

    if constexpr(dim == 3)
    {
      // Apply the regularization in z-direction.
      double const z = p[2];

      if(z < l_reg)
      {
        val *= 1.0 -
               dealii::Utilities::fixed_power<4>(std::cos(dealii::numbers::PI * z / (2.0 * l_reg)));
      }
      else if(z > length_domain - l_reg)
      {
        val *= 1.0 - dealii::Utilities::fixed_power<4>(
                       std::cos(dealii::numbers::PI * (z - length_domain) / (2.0 * l_reg)));
      }
    }

    return val;
  }

private:
  double const start_time;
  double const end_time;
  double const time_ramp_fraction;
  double const space_ramp_fraction;
  double const length_domain;
  double const velocity_scale;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      // clang-format off
      prm.add_parameter("ProblemType",                         problem_type,                                      "Problem type considered.");
      prm.add_parameter("IncludeConvectiveTerm",               include_convective_term,                           "Include the nonlinear convective term.",                       dealii::Patterns::Bool());
      prm.add_parameter("StartTime",                           start_time,                                        "Simulation start time.",                                       dealii::Patterns::Double());
      prm.add_parameter("EndTime",                             end_time,                                          "Simulation end time.",                                         dealii::Patterns::Double());
      prm.add_parameter("CFL",                                 cfl,                                               "Target maximum CFL used.",                                     dealii::Patterns::Double());      
      prm.add_parameter("TimeStepSize",                        time_step_size,                                    "Time step size used.",                                         dealii::Patterns::Double());
      prm.add_parameter("LidVelocity",                         lid_velocity,                                      "Lid velocity enforced.",                                       dealii::Patterns::Double());
      prm.add_parameter("Density",                             density,                                           "Incompressible model: density.",                               dealii::Patterns::Double());
      prm.add_parameter("KinematicViscosity",                  kinematic_viscosity,                               "Newtonian model: kinematic_viscosity.",                        dealii::Patterns::Double());
      prm.add_parameter("UseGeneralizedNewtonianModel",        generalized_newtonian_model_data.is_active,        "Use generalized Newtonian model, else Newtonian one.",         dealii::Patterns::Bool());
      prm.add_parameter("TemporalDiscretization",              temporal_discretization,                           "Temporal discretization.");
      prm.add_parameter("TreatmentOfConvectiveTerm",           treatment_of_convective_term,                      "Treat convective term implicit, else explicit");
      prm.add_parameter("TreatmentOfVariableViscosity",        treatment_of_variable_viscosity,                   "Treat the variable viscosity implicit or extrapolate in time.");
      prm.add_parameter("PreconditionerMomentum",              preconditioner_momentum,                           "Preconditioner for the viscous/momentum step.");
      prm.add_parameter("GeneralizedNewtonianViscosityMargin", generalized_newtonian_model_data.viscosity_margin, "Generalized Newtonian models: viscosity margin.",              dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianKappa",           generalized_newtonian_model_data.kappa,            "Generalized Newtonian models: kappa.",                         dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianLambda",          generalized_newtonian_model_data.lambda,           "Generalized Newtonian models: lambda.",                        dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianA",               generalized_newtonian_model_data.a,                "Generalized Newtonian models: a.",                             dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianN",               generalized_newtonian_model_data.n,                "Generalized Newtonian models: n.",                             dealii::Patterns::Double());

      prm.add_parameter("AbsTolLin",                                   abs_tol_lin,                                     "Absolute tolerance of linear solver.",                                                          dealii::Patterns::Double());
      prm.add_parameter("RelTolLin",                                   rel_tol_lin,                                     "Relative tolerance of linear solver.",                                                          dealii::Patterns::Double());
      prm.add_parameter("AbsTolNewton",                                abs_tol_newton,                                  "Absolute tolerance of nonlinear solver.",                                                       dealii::Patterns::Double());
      prm.add_parameter("RelTolNewton",                                rel_tol_newton,                                  "Relative tolerance of nonlinear solver.",                                                       dealii::Patterns::Double());
      prm.add_parameter("AbsTolLinInNewton",                           abs_tol_lin_in_newton,                           "Absolute tolerance of linear solver within nonlinear solver",                                   dealii::Patterns::Double());
      prm.add_parameter("RelTolLinInNewton",                           rel_tol_lin_in_newton,                           "Relative tolerance of linear solver within nonlinear solver",                                   dealii::Patterns::Double());
      prm.add_parameter("SchurComplementPreconditioner",               schur_complement_preconditioner,                 "Schur complement approximation considered.");
      prm.add_parameter("IterativeSolveVelocityBlock",                 iterative_solve_velocity_block,                  "Use an iterative solver for the velocity block within block preconditioner?",                   dealii::Patterns::Bool());
      prm.add_parameter("IterativeSolvePressureBlock",                 iterative_solve_pressure_block,                  "Use an iterative solver for the pressure block within block preconditioner?",                   dealii::Patterns::Bool());
      prm.add_parameter("AbsTolLinBlockInPreconditioner",              abs_tol_lin_block_in_preconditioner,             "Absolute solver tolerance of linear solver of block in preconditioner",                         dealii::Patterns::Double());
      prm.add_parameter("RelTolLinBlockInPreconditioner",              rel_tol_lin_block_in_preconditioner,             "Relative solver tolerance of linear solver of block in preconditioner",                         dealii::Patterns::Double());
      prm.add_parameter("IterationsEigenValueEstimationVelocity",      iterations_eigenvalue_estimation_velocity,       "Iterations used in the Eigenvalue estimation in Multigrid, Velocity problem.",                  dealii::Patterns::Integer());
      prm.add_parameter("IterationsEigenValueEstimationPressureSchur", iterations_eigenvalue_estimation_pressure_schur, "Iterations used in the Eigenvalue estimation in Multigrid, Pressure Schur complement problem.", dealii::Patterns::Integer());
      // clang-format on
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type = problem_type;
    this->param.equation_type =
      include_convective_term ? EquationType::NavierStokes : EquationType::Stokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::DivergenceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.right_hand_side             = false;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = kinematic_viscosity;
    this->param.density    = density;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type =
      problem_type == ProblemType::Steady ? SolverType::Steady : SolverType::Unsteady;
    this->param.temporal_discretization         = temporal_discretization;
    this->param.treatment_of_convective_term    = treatment_of_convective_term;
    this->param.treatment_of_variable_viscosity = treatment_of_variable_viscosity;
    this->param.adaptive_time_stepping          = cfl > 0.0;
    this->param.calculation_of_time_step_size   = this->param.adaptive_time_stepping ?
                                                    TimeStepCalculation::CFL :
                                                    TimeStepCalculation::UserSpecified;
    this->param.max_velocity                    = lid_velocity;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.cfl                             = cfl;
    this->param.time_step_size                  = time_step_size;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;

    // pseudo-timestepping for steady-state problems
    this->param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
    this->param.abs_tol_steady = abs_tol_steady;
    this->param.rel_tol_steady = rel_tol_steady;

    // restart
    this->param.restart_data.write_restart       = false;
    this->param.restart_data.interval_time       = 5.0;
    this->param.restart_data.interval_wall_time  = 1.e6;
    this->param.restart_data.interval_time_steps = 1e8;
    this->param.restart_data.filename            = "output/cavity/cavity_restart";

    // output of solver information
    this->param.solver_info_data.interval_time_steps = 1e8;
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 10;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = 1; // this->param.degree_u;
    this->param.mapping_degree_coarse_grids = 1; // this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // gradient term
    this->param.gradp_integrated_by_parts = true;
    this->param.gradp_use_boundary_data   = true;
    this->param.gradp_formulation         = FormulationPressureGradientTerm::Weak;

    // divergence term
    this->param.divu_integrated_by_parts = true;
    this->param.divu_use_boundary_data   = true;
    this->param.divu_formulation         = FormulationVelocityDivergenceTerm::Weak;

    // pressure level is undefined
    this->param.adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = 1.0;
    this->param.apply_penalty_terms_in_postprocessing_step = true;
    this->param.continuity_penalty_use_boundary_data =
      this->param.apply_penalty_terms_in_postprocessing_step;
    this->param.type_penalty_parameter        = TypePenaltyParameter::ConvectiveTerm;
    this->param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;

    // TURBULENCE
    this->param.turbulence_model_data.is_active        = false;
    this->param.turbulence_model_data.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    this->param.turbulence_model_data.constant = 1.35;


    // GENERALIZED NEWTONIAN MODEL
    generalized_newtonian_model_data.generalized_newtonian_model =
      GeneralizedNewtonianViscosityModel::GeneralizedCarreauYasuda;
    this->param.generalized_newtonian_model_data = generalized_newtonian_model_data;


    // NUMERICAL PARAMETERS
    if(this->param.viscosity_is_variable())
    {
      this->param.use_cell_based_face_loops = false;
      this->param.quad_rule_linearization   = QuadratureRuleLinearization::Standard;
    }
    else
    {
      this->param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;
    }


    // PROJECTION METHODS

    // Newton solver
    this->param.newton_solver_data_momentum =
      Newton::SolverData(100, abs_tol_newton, rel_tol_newton);

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson    = SolverData(1000, abs_tol_lin, rel_tol_lin);
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // clang-format off
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.p_sequence = PSequenceType::DecreaseByOne;

    this->param.multigrid_data_pressure_poisson.smoother_data.iterations        = 5;
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother          = MultigridSmoother::Chebyshev; // MultigridSmoother::Jacobi
    this->param.multigrid_data_pressure_poisson.smoother_data.relaxation_factor = 0.8; // Jacobi,    default: 0.8
    this->param.multigrid_data_pressure_poisson.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
    this->param.multigrid_data_pressure_poisson.smoother_data.iterations_eigenvalue_estimation = 20; // Chebyshev, default: 20
    this->param.multigrid_data_pressure_poisson.smoother_data.preconditioner    =
    this->param.use_cell_based_face_loops ? 
      PreconditionerSmoother::BlockJacobi : PreconditionerSmoother::PointJacobi;


    this->param.multigrid_data_pressure_poisson.coarse_problem.solver            = MultigridCoarseGridSolver::AMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.amg_type = AMGType::ML;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver_data       = SolverData(1000, abs_tol_multigrid_coarse_level, rel_tol_multigrid_coarse_level, 30);
    
    this->param.multigrid_data_momentum.coarse_problem.preconditioner =
    this->param.use_cell_based_face_loops ? 
      MultigridCoarseGridPreconditioner::BlockJacobi
      : MultigridCoarseGridPreconditioner::PointJacobi;
#ifdef DEAL_II_WITH_TRILINOS
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.smoother_sweeps       = 2;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.n_cycles              = 1;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.w_cycle               = false;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.aggregation_threshold = 1e-4;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.smoother_type         = "Chebyshev";
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.coarse_type           = "Amesos-KLU";
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.output_details        = false;
#endif
    this->param.update_preconditioner_pressure_poisson = false;
    // clang-format on

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, abs_tol_lin, rel_tol_lin);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
    this->param.preconditioner_block_diagonal_projection =
      Elementwise::Preconditioner::InverseMassMatrix;
    this->param.update_preconditioner_projection = false;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      this->param.solver_momentum =
        treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ? SolverMomentum::CG :
                                                                              SolverMomentum::GMRES;
      this->param.solver_data_momentum    = SolverData(1000, abs_tol_lin, rel_tol_lin);
      this->param.preconditioner_momentum = preconditioner_momentum;
      this->param.update_preconditioner_momentum =
        this->param.viscosity_is_variable() or this->param.non_explicit_convective_problem();
      this->param.update_preconditioner_momentum_every_time_steps = 10;
    }

    // clang-format off
    this->param.multigrid_data_momentum.type       = MultigridType::cphMG;
    this->param.multigrid_data_momentum.p_sequence = PSequenceType::DecreaseByOne;

    this->param.multigrid_operator_type_momentum                        = this->param.non_explicit_convective_problem() ? 
      MultigridOperatorType::ReactionConvectionDiffusion : MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_momentum.smoother_data.iterations        = 5;
    this->param.multigrid_data_momentum.smoother_data.smoother          = MultigridSmoother::Chebyshev; // MultigridSmoother::Jacobi
    this->param.multigrid_data_momentum.smoother_data.relaxation_factor = 0.8; // Jacobi,    default: 0.8
    this->param.multigrid_data_momentum.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
    this->param.multigrid_data_momentum.smoother_data.iterations_eigenvalue_estimation = iterations_eigenvalue_estimation_velocity; // Chebyshev, default: 20
    this->param.multigrid_data_momentum.smoother_data.preconditioner    =
      this->param.use_cell_based_face_loops ? 
        PreconditionerSmoother::BlockJacobi : PreconditionerSmoother::PointJacobi;

    this->param.multigrid_data_momentum.coarse_problem.solver            = MultigridCoarseGridSolver::AMG;
    this->param.multigrid_data_momentum.coarse_problem.amg_data.amg_type = AMGType::ML;
    this->param.multigrid_data_momentum.coarse_problem.solver_data       = SolverData(1000, abs_tol_multigrid_coarse_level, rel_tol_multigrid_coarse_level, 30);

    this->param.multigrid_data_momentum.coarse_problem.preconditioner =
      this->param.use_cell_based_face_loops ? 
        MultigridCoarseGridPreconditioner::BlockJacobi
        : MultigridCoarseGridPreconditioner::PointJacobi;
#ifdef DEAL_II_WITH_TRILINOS
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.smoother_sweeps       = 2;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.n_cycles              = 1;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.w_cycle               = false;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.aggregation_threshold = 1e-4;
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.smoother_type         = "Chebyshev";
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.coarse_type           = "Amesos-KLU";
    this->param.multigrid_data_pressure_poisson.coarse_problem.amg_data.ml_data.output_details        = false;
#endif
    // clang-format on


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled =
      Newton::SolverData(100, abs_tol_newton, rel_tol_newton);

    // linear solver
    this->param.solver_coupled = iterative_solve_velocity_block or iterative_solve_pressure_block ?
                                   SolverCoupled::FGMRES :
                                   SolverCoupled::GMRES;
    if(this->param.viscosity_is_variable())
    {
      this->param.solver_data_coupled =
        SolverData(1500, abs_tol_lin_in_newton, rel_tol_lin_in_newton, 1000);
    }
    else
    {
      this->param.solver_data_coupled = SolverData(1500, abs_tol_lin, rel_tol_lin, 1000);
    }

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled =
      this->param.viscosity_is_variable() or this->param.non_explicit_convective_problem();
    this->param.exact_inversion_of_velocity_block   = iterative_solve_velocity_block;
    this->param.exact_inversion_of_laplace_operator = iterative_solve_pressure_block;
    this->param.solver_data_velocity_block          = SolverData(1000,
                                                        abs_tol_lin_block_in_preconditioner,
                                                        rel_tol_lin_block_in_preconditioner,
                                                        30);
    this->param.solver_data_pressure_block          = SolverData(1000,
                                                        abs_tol_lin_block_in_preconditioner,
                                                        rel_tol_lin_block_in_preconditioner,
                                                        30);

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block =
      this->param.non_explicit_convective_problem() ?
        MultigridOperatorType::ReactionConvectionDiffusion :
        MultigridOperatorType::ReactionDiffusion;

    // clang-format off
    this->param.multigrid_data_velocity_block.type       = MultigridType::cphMG;
    this->param.multigrid_data_velocity_block.p_sequence = PSequenceType::DecreaseByOne;

    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.smoother          = MultigridSmoother::Chebyshev; // MultigridSmoother::Jacobi
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.8; // Jacobi,    default: 0.8
    this->param.multigrid_data_velocity_block.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
    this->param.multigrid_data_velocity_block.smoother_data.iterations_eigenvalue_estimation = iterations_eigenvalue_estimation_velocity; // Chebyshev, default: 20
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner    =
      this->param.use_cell_based_face_loops ? 
        PreconditionerSmoother::BlockJacobi : PreconditionerSmoother::PointJacobi;

    this->param.multigrid_data_velocity_block.coarse_problem.solver            = MultigridCoarseGridSolver::AMG;
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.amg_type = AMGType::ML;
    this->param.multigrid_data_velocity_block.coarse_problem.solver_data       = SolverData(1000, abs_tol_multigrid_coarse_level, rel_tol_multigrid_coarse_level, 30);

    this->param.multigrid_data_velocity_block.coarse_problem.preconditioner =
      this->param.use_cell_based_face_loops ? 
        MultigridCoarseGridPreconditioner::BlockJacobi
        : MultigridCoarseGridPreconditioner::PointJacobi;
#ifdef DEAL_II_WITH_TRILINOS
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.ml_data.smoother_sweeps       = 2;
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.ml_data.n_cycles              = 1;
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.ml_data.w_cycle               = false;
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.ml_data.aggregation_threshold = 1e-4;
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.ml_data.smoother_type         = "Chebyshev";
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.ml_data.coarse_type           = "Amesos-KLU";
    this->param.multigrid_data_velocity_block.coarse_problem.amg_data.ml_data.output_details        = false;
#endif
    // clang-format on

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block = schur_complement_preconditioner;

    // clang-format off
    this->param.multigrid_data_pressure_block.type       = MultigridType::cphMG;
    this->param.multigrid_data_pressure_block.p_sequence = PSequenceType::DecreaseByOne;

    this->param.multigrid_data_pressure_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_pressure_block.smoother_data.smoother          = MultigridSmoother::Chebyshev; // MultigridSmoother::Jacobi
    this->param.multigrid_data_pressure_block.smoother_data.relaxation_factor = 0.8; // Jacobi,    default: 0.8
    this->param.multigrid_data_pressure_block.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
    this->param.multigrid_data_pressure_block.smoother_data.iterations_eigenvalue_estimation = iterations_eigenvalue_estimation_pressure_schur; // Chebyshev, default: 20
    this->param.multigrid_data_pressure_block.smoother_data.preconditioner    =
      this->param.use_cell_based_face_loops ? 
        PreconditionerSmoother::BlockJacobi : PreconditionerSmoother::PointJacobi;

    this->param.multigrid_data_pressure_block.coarse_problem.solver            = MultigridCoarseGridSolver::AMG;
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.amg_type = AMGType::ML;
    this->param.multigrid_data_pressure_block.coarse_problem.solver_data       = SolverData(1000, abs_tol_multigrid_coarse_level, rel_tol_multigrid_coarse_level, 30);

    this->param.multigrid_data_pressure_block.coarse_problem.preconditioner =
      this->param.use_cell_based_face_loops ? 
        MultigridCoarseGridPreconditioner::BlockJacobi
        : MultigridCoarseGridPreconditioner::PointJacobi;
#ifdef DEAL_II_WITH_TRILINOS
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.ml_data.smoother_sweeps       = 2;
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.ml_data.n_cycles              = 1;
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.ml_data.w_cycle               = false;
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.ml_data.aggregation_threshold = 1e-4;
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.ml_data.smoother_type         = "Chebyshev";
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.ml_data.coarse_type           = "Amesos-KLU";
    this->param.multigrid_data_pressure_block.coarse_problem.amg_data.ml_data.output_details        = false;
#endif
    // clang-format on    
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        double const left = 0.0, right = L;
        dealii::GridGenerator::hyper_cube(tria, left, right);

        // set boundary indicator
        for(auto cell : tria.cell_iterators())
        {
          for(auto const & face : cell->face_indices())
          {
            if((std::fabs(cell->face(face)->center()(1) - L) < 1e-12))
              cell->face(face)->set_boundary_id(1);
          }
        }

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    // all boundaries have ID = 0 by default -> Dirichlet boundaries

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1, new LidVelocityRegularized<dim>(start_time,
                                              end_time,
                                              0.1 /* time_ramp_fraction */,                                              
                                              0.1 /* space_ramp_fraction */,
                                              L /* length_domain */,
                                              lid_velocity /* velocity_scale */)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->neumann_bc.insert(1);
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 100.0;
    pp_data.output_data.directory            = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename             = this->output_parameters.filename;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.write_vorticity      = false;
    pp_data.output_data.write_viscosity      = true;
    pp_data.output_data.write_streamfunction = false;
    pp_data.output_data.write_processor_id   = false;
    pp_data.output_data.degree               = this->param.degree_u;

    // line plot data
    pp_data.line_plot_data.time_control_data.is_active                = false;
    pp_data.line_plot_data.time_control_data.start_time               = start_time;
    pp_data.line_plot_data.time_control_data.trigger_every_time_steps = 1;
    pp_data.line_plot_data.directory = this->output_parameters.directory;

    // which quantities
    std::shared_ptr<Quantity> quantity_u;
    quantity_u.reset(new Quantity());
    quantity_u->type = QuantityType::Velocity;
    std::shared_ptr<Quantity> quantity_p;
    quantity_p.reset(new Quantity());
    quantity_p->type = QuantityType::Pressure;

    // lines
    std::shared_ptr<Line<dim>> vert_line, hor_line;

    // vertical line
    vert_line.reset(new Line<dim>());
    vert_line->begin    = dim == 2 ? dealii::Point<dim>(L*0.5, 0.0) : dealii::Point<dim>(L*0.5, 0.0, L*0.5);
    vert_line->end      = dim == 2 ? dealii::Point<dim>(L*0.5, L)   : dealii::Point<dim>(L*0.5, L  , L*0.5);
    vert_line->name     = this->output_parameters.filename + "_vert_line";
    vert_line->n_points = 100001; // 2001;
    vert_line->quantities.push_back(quantity_u);
    vert_line->quantities.push_back(quantity_p);
    pp_data.line_plot_data.lines.push_back(vert_line);

    // horizontal line
    hor_line.reset(new Line<dim>());
    hor_line->begin    = dim == 2 ? dealii::Point<dim>(0.0, L*0.5) : dealii::Point<dim>(0.0, L*0.5, L*0.5);
    hor_line->end      = dim == 2 ? dealii::Point<dim>(L, L*0.5)   : dealii::Point<dim>(L, L*0.5, L*0.5);
    hor_line->name     = this->output_parameters.filename + "_hor_line";
    hor_line->n_points = 10001; // 2001;
    hor_line->quantities.push_back(quantity_u);
    hor_line->quantities.push_back(quantity_p);
    pp_data.line_plot_data.lines.push_back(hor_line);

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const L = 1.0;

  bool include_convective_term = true;
  double start_time = 0.0;
  double end_time   = 10.0;
  double cfl = 3.0;
  double time_step_size = 1e-3;
  double lid_velocity = 1.0;
  double density = 1.0e3;
  double kinematic_viscosity = 1.0e-6;
  
  ProblemType problem_type = ProblemType::Steady;
  TemporalDiscretization temporal_discretization  = TemporalDiscretization::Undefined;
  TreatmentOfConvectiveTerm    treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  TreatmentOfVariableViscosity treatment_of_variable_viscosity =
    TreatmentOfVariableViscosity::Implicit;

  MomentumPreconditioner preconditioner_momentum = MomentumPreconditioner::Multigrid;

  GeneralizedNewtonianModelData generalized_newtonian_model_data;

  double abs_tol_lin    = 1.0e-12;
  double rel_tol_lin    = 1.0e-6;
  double abs_tol_newton = 1.0e-12;
  double rel_tol_newton = 1.0e-6;
  double abs_tol_lin_in_newton          = 1.0e-12;
  double rel_tol_lin_in_newton          = 1.0e-12;

  bool iterative_solve_velocity_block = false;
  bool iterative_solve_pressure_block = false;
  double abs_tol_lin_block_in_preconditioner   = 1.0e-12;
  double rel_tol_lin_block_in_preconditioner   = 1.0e-6;

  unsigned int iterations_eigenvalue_estimation_velocity = 20;
  unsigned int iterations_eigenvalue_estimation_pressure_schur = 20;

  // currently inactive, since a single AMG cycle/direct solve is used on the coarse grid.
  double abs_tol_multigrid_coarse_level = 1.0e-60;
  double rel_tol_multigrid_coarse_level = 1.0e-60; 

  double abs_tol_steady = 1.0e-12;
  double rel_tol_steady = 1.0e-6;

  SchurComplementPreconditioner schur_complement_preconditioner = SchurComplementPreconditioner::CahouetChabard;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
