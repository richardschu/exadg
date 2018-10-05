/*
 * unstable_beltrami.h
 *
 *  Created on: July, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 8;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 2;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 0.0; //TODO //1.0/(2.0e3);

// output folders and names
std::string OUTPUT_FOLDER = "output/unstable_beltrami/div_normal_conti_penalty_CFL0-2_reltol_1e-8/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "nu_inf_l2_k87";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 20.0;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL;
  max_velocity = 1.0; //TODO //std::sqrt(2.0);
  cfl = 0.2; //TODO //0.25;
  cfl_exponent_fe_degree_velocity = 2.0; //TODO //1.5;
  time_step_size = 1.0e-3;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = false; // true; // false;


  // SPATIAL DISCRETIZATION

  // spatial discretization method
  spatial_discretization = SpatialDiscretization::DG;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true;

  // special case: pure DBC's
  pure_dirichlet_bc = true;
  adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalMeanValue;

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_use_boundary_data = false;
  continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  type_penalty_parameter = TypePenaltyParameter::ConvectiveTerm;
  continuity_penalty_factor = divergence_penalty_factor;


  // PROJECTION METHODS

  // pressure Poisson equation
  IP_factor_pressure = 1.0;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-8;

  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  abs_tol_projection = 1.e-12;
  rel_tol_projection = 1.e-12;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // convective step

  // nonlinear solver
  newton_solver_data_convective.abs_tol = 1.e-12;
  newton_solver_data_convective.rel_tol = 1.e-8;
  newton_solver_data_convective.max_iter = 100;
  // linear solver
  abs_tol_linear_convective = 1.e-12;
  rel_tol_linear_convective = 1.e-8;
  max_iter_linear_convective = 1e4;
  use_right_preconditioning_convective = true;
  max_n_tmp_vectors_convective = 100;

  // stability in the limit of small time steps and projection step
  small_time_steps_stability = false;

  // viscous step
  solver_viscous = SolverViscous::PCG;
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //GeometricMultigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-8;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-12;
  newton_solver_data_momentum.rel_tol = 1.e-8;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;  //VelocityDiffusion; //VelocityConvectionDiffusion;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_momentum_linear = 1.e-12;
  rel_tol_momentum_linear = 1.e-8;
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 0;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;
  scaling_factor_continuity = 1.0e0;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-12;
  newton_solver_data_coupled.rel_tol = 1.e-8;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-8;
  max_iter_linear = 1e3;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::InverseMassMatrix;
  exact_inversion_of_momentum_block = false;
  rel_tol_solver_momentum_preconditioner = 1.e-6;
  max_n_tmp_vectors_solver_momentum_preconditioner = 100;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::CahouetChabard;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  exact_inversion_of_laplace_operator = false;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING

  // print input parameters
  print_input_parameters = true;

  // write output for visualization of results
  output_data.write_output = false; //true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/100;
  output_data.write_divergence = false;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = true;
  error_data.calculate_relative_errors = false; // TODO
  error_data.calculate_H1_seminorm_velocity = true;
  error_data.error_calc_start_time = start_time;
  //error_data.error_calc_interval_time = (end_time-start_time);
  error_data.calculate_every_time_steps = 10;
  error_data.write_errors_to_file = true;
  error_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME + "_error";

  // kinetic energy
  kinetic_energy_data.calculate = true;
  kinetic_energy_data.calculate_every_time_steps = 10;
  kinetic_energy_data.viscosity = VISCOSITY;
  kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;

  // output of solver information
  output_solver_info_every_timesteps = 1e5;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Analytical solution velocity:
 *
 *  - This function is used to calculate the L2/H1 seminorm error
 *
 *  - This function is used to prescribe initial conditions for the velocity field
 */
template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~AnalyticalSolutionVelocity(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;

  virtual Tensor<1,dim,double> gradient (const Point<dim>    &p,
                                         const unsigned int  component = 0) const;
};

template<int dim>
double AnalyticalSolutionVelocity<dim>::value(const Point<dim>   &p,
                                              const unsigned int component) const
{
  double t = this->get_time();
  double result = 0.0;

  const double pi = numbers::PI;

  if (component == 0)
    result = std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
  else if (component == 1)
    result = std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
  else if (component == 2)
    result = std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);

  result *= std::exp(-8.0*pi*pi*VISCOSITY*t);

  // add perturbations
  // TODO
//  double const eps = 1.e-6;
//  result *= (1.0 + eps * std::cos(20*pi*p[2]));

  return result;
}

template<int dim>
Tensor<1,dim,double> AnalyticalSolutionVelocity<dim>::gradient(const Point<dim>   &p,
                                                               const unsigned int component) const
{
  double t = this->get_time();
  Tensor<1,dim,double> result;

  const double pi = numbers::PI;

  AssertThrow(dim==3, ExcMessage("not implemented."));

  if (component == 0)
  {
    result[0] = 2.0*pi*std::cos(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
    result[1] = 2.0*pi*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
    result[2] = 0.0;
  }
  else if (component == 1)
  {
    result[0] = -2.0*pi*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
    result[1] = -2.0*pi*std::cos(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
    result[2] = 0.0;
  }
  else if (component == 2)
  {
    result[0] = 2.0*pi*std::sqrt(2.0)*std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
    result[1] = -2.0*pi*std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
    result[2] = 0.0;
  }

  result *= std::exp(-8.0*pi*pi*VISCOSITY*t);

  return result;
}

/*
 *  Analytical solution pressure
 *
 *  - It is used to calculate the L2 error
 *
 *  - It is used to adjust the pressure level in case of pure Dirichlet BC's
 *    (where the pressure is only defined up to an additive constant)
 *
 *  - This function is used to prescribe initial conditions for the pressure field
 */
template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  virtual ~AnalyticalSolutionPressure(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double AnalyticalSolutionPressure<dim>::value(const Point<dim>    &p,
                                              const unsigned int  /* component */) const
{
  double t = this->get_time();
  const double pi = numbers::PI;

  double result = - 0.5*(+ std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[0])
                         + std::cos(2.0*pi*p[1])*std::cos(2.0*pi*p[1]) - 1.0);

  result *= std::exp(-16.0*pi*pi*VISCOSITY*t);

  return result;
}


/*
 *  PressureBC_dudt:
 *
 *  This functions is only used when applying the high-order dual splitting scheme and
 *  is evaluated on Dirichlet boundaries (where the velocity is prescribed).
 *  Hence, this is the function that is set in the dirichlet_bc map of boundary_descriptor_pressure.
 *
 *  Note:
 *    When using a couples solution approach we do not have to evaluate something like
 *    pressure Neumann BC's on Dirichlet boundaries (we only have p⁺ = p⁻ on Dirichlet boundaries,
 *    i.e., no boundary data used). So it doesn't matter when writing this function into the
 *    dirichlet_bc map of boundary_descriptor_pressure because this function will never be evaluated
 *    in case of a coupled solution approach.
 *
 */
template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  virtual ~PressureBC_dudt(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double PressureBC_dudt<dim>::value(const Point<dim>   &p,
                                   const unsigned int component) const
{
  double t = this->get_time();
  double result = 0.0;

  const double pi = numbers::PI;

  if (component == 0)
    result = std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
  else if (component == 1)
    result = std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
  else if (component == 2)
    result = std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);

  result *= -8.0*pi*pi*VISCOSITY*std::exp(-8.0*pi*pi*VISCOSITY*t);

  return result;
}


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);

  // periodicity in x-,y-, and z-direction

  // x-direction
  triangulation.begin()->face(0)->set_all_boundary_ids(0);
  triangulation.begin()->face(1)->set_all_boundary_ids(1);
  // y-direction
  triangulation.begin()->face(2)->set_all_boundary_ids(2);
  triangulation.begin()->face(3)->set_all_boundary_ids(3);
  // z-direction
  triangulation.begin()->face(4)->set_all_boundary_ids(4);
  triangulation.begin()->face(5)->set_all_boundary_ids(5);

  GridTools::collect_periodic_faces(triangulation, 0, 1, 0 /*x-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 2, 3, 1 /*y-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);

  triangulation.add_periodicity(periodic_faces);

  // global refinements
  triangulation.refine_global(n_refine_space);
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > analytical_solution_velocity;
  analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  std::shared_ptr<Function<dim> > analytical_solution_pressure;
  analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());

  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new ZeroFunction<dim>(dim));

  field_functions->initial_solution_velocity = analytical_solution_velocity;
  field_functions->initial_solution_pressure = analytical_solution_pressure;
  field_functions->analytical_solution_pressure = analytical_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new AnalyticalSolutionVelocity<dim>());
  analytical_solution->pressure.reset(new AnalyticalSolutionPressure<dim>());
}

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, Number> >
construct_postprocessor(InputParameters<dim> const &param)
{
  PostProcessorData<dim> pp_data;

  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;

  std::shared_ptr<PostProcessor<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,Number> > pp;
  pp.reset(new PostProcessor<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,Number>(pp_data));

  return pp;
}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_UNSTABLE_BELTRAMI_H_ */