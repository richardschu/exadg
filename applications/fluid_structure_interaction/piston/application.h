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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_FLUID_STRUCTURE_INTERACTION_PISTON_APPLICATION_H_
#define APPLICATIONS_FLUID_STRUCTURE_INTERACTION_PISTON_APPLICATION_H_

/*
 * Rectangular elastic piston driving an incompressible Newtonian fluid, an
 * analytical solution of the fluid-structure interaction problem, see
 * Serino, Banks, Henshaw and Schwendeman, J. Comput. Phys. 399 (2019) 108923.
 *
 * The structure occupies (0,L) x (-H,0) in its reference configuration, the
 * fluid (0,L) x (d(t),H) with the interface at y = d(t). All motion is vertical:
 * the linear elastic structure (plane strain) is a standing wave
 *   d_s(Y,t) = 2 alpha sin(omega t) sin(k (Y + H)),  k = omega / c_p,
 *   c_p = sqrt((lambda + 2 mu) / rho_s),
 * clamped at Y = -H, with the interface displacement d(t) = a sin(omega t),
 * a = 2 alpha sin(k H). The fluid velocity is uniform in space,
 * u = (0, d'(t)), such that the pressure is linear in y,
 *   p(y,t) = p_I(t) - rho_f d''(t) (y - d(t)),
 *   p_I(t) = -(lambda + 2 mu) d_s,Y(0,t)
 * from the balance of momentum of the fluid and the traction balance at the
 * interface. The fluid solution is in the discrete spaces (degree_u >= 2) for
 * an affine fluid mesh, so that the fluid errors are temporal errors and
 * errors induced by the structure.
 *
 * Boundary conditions: the structure is clamped at Y = -H and has a zero
 * horizontal displacement and a zero vertical traction at x = 0 and x = L. The
 * fluid has the exact velocity at y = H, and a zero normal derivative of the
 * velocity and the exact pressure at x = 0 and x = L. The fluid mesh is fixed
 * at y = H and slides along x = 0 and x = L.
 *
 * The first time step of the fluid (BDF1) excites the high-frequency modes of
 * the structure. These are not damped by the Newmark method (spectral radius
 * 1), and the fluid pressure follows the time derivative of their velocity at
 * the interface, such that the pressure error does not converge smoothly. A
 * spectral radius of 0 (GenAlpha) damps these modes and gives second order in
 * all fields.
 */

namespace ExaDG
{
namespace Piston
{
double const L = 1.0;
double const H = 0.5;

// boundary IDs of both the fluid and the structure
dealii::types::boundary_id const BOUNDARY_ID_FSI     = 0;
dealii::types::boundary_id const BOUNDARY_ID_LATERAL = 1;
// y = H for the fluid, Y = -H for the structure
dealii::types::boundary_id const BOUNDARY_ID_OUTER = 2;

// cells of the coarse grids, (0,L) x (0,H) and (0,L) x (-H,0)
unsigned int const N_CELLS_X = 2;
unsigned int const N_CELLS_Y = 1;

struct Data
{
  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
    {
      prm.add_parameter("EndTime",          end_time,          "End time.",                                              dealii::Patterns::Double(0.0));
      prm.add_parameter("TimeStepSize",     time_step_size,    "Time step size.",                                        dealii::Patterns::Double(0.0));
      prm.add_parameter("ErrorInterval",    error_interval,    "Time interval of the error calculation.",                dealii::Patterns::Double(0.0));
      prm.add_parameter("DensityFluid",     density_fluid,     "Density of the fluid.",                                  dealii::Patterns::Double(0.0));
      prm.add_parameter("Viscosity",        viscosity,         "Kinematic viscosity of the fluid.",                      dealii::Patterns::Double(0.0));
      prm.add_parameter("DensityStructure", density_structure, "Density of the structure.",                              dealii::Patterns::Double(0.0));
      prm.add_parameter("YoungsModulus",    youngs_modulus,    "Young's modulus of the structure.",                      dealii::Patterns::Double(0.0));
      prm.add_parameter("PoissonsRatio",    poissons_ratio,    "Poisson's ratio of the structure.",                      dealii::Patterns::Double(0.0, 0.5));
      prm.add_parameter("Amplitude",        amplitude,         "Amplitude a of the interface displacement.",             dealii::Patterns::Double(0.0));
      prm.add_parameter("Omega",            omega,             "Angular frequency omega of the interface displacement.", dealii::Patterns::Double(0.0));
      prm.add_parameter("SolverAbsTol",     abs_tol,           "Absolute tolerance of all linear and nonlinear solvers.", dealii::Patterns::Double(0.0));
      prm.add_parameter("SolverRelTol",     rel_tol,           "Relative tolerance of all linear and nonlinear solvers.", dealii::Patterns::Double(0.0));
    }
    prm.leave_subsection();
    // clang-format on
  }

  // lambda + 2 mu of plane strain
  double
  p_wave_modulus() const
  {
    return youngs_modulus * (1.0 - poissons_ratio) /
           ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio));
  }

  double
  wave_number() const
  {
    return omega / std::sqrt(p_wave_modulus() / density_structure);
  }

  double
  alpha() const
  {
    return amplitude / (2.0 * std::sin(wave_number() * H));
  }

  // vertical displacement of the structure in the reference configuration
  double
  displacement_structure(double const Y, double const t) const
  {
    return 2.0 * alpha() * std::sin(omega * t) * std::sin(wave_number() * (Y + H));
  }

  double
  velocity_structure(double const Y, double const t) const
  {
    return 2.0 * alpha() * omega * std::cos(omega * t) * std::sin(wave_number() * (Y + H));
  }

  double
  displacement_interface(double const t) const
  {
    return amplitude * std::sin(omega * t);
  }

  double
  velocity_fluid(double const t) const
  {
    return amplitude * omega * std::cos(omega * t);
  }

  double
  acceleration_fluid(double const t) const
  {
    return -amplitude * omega * omega * std::sin(omega * t);
  }

  // pressure (not kinematic) of the fluid in the current configuration
  double
  pressure_fluid(double const y, double const t) const
  {
    double const k = wave_number();

    double const pressure_interface =
      -p_wave_modulus() * 2.0 * alpha() * k * std::sin(omega * t) * std::cos(k * H);

    return pressure_interface -
           density_fluid * acceleration_fluid(t) * (y - displacement_interface(t));
  }

  double end_time       = 0.5;
  double time_step_size = 0.0125;
  double error_interval = 0.05;

  // parameters of Serino et al. (2019)
  double density_fluid     = 1.0;
  double viscosity         = 0.1;
  double density_structure = 100.0;
  double youngs_modulus    = 5.0e3;
  double poissons_ratio    = 0.3;
  double amplitude         = 0.005;
  double omega             = dealii::numbers::PI;

  double abs_tol = 1.e-11;
  double rel_tol = 1.e-10;
};

template<int dim>
class VelocityFluid : public dealii::Function<dim>
{
public:
  VelocityFluid(Data const & data) : dealii::Function<dim>(dim, 0.0), data(data)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    (void)p;
    return component == 1 ? data.velocity_fluid(this->get_time()) : 0.0;
  }

private:
  Data const data;
};

template<int dim>
class KinematicPressureFluid : public dealii::Function<dim>
{
public:
  KinematicPressureFluid(Data const & data) : dealii::Function<dim>(1, 0.0), data(data)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    (void)component;
    return data.pressure_fluid(p[1], this->get_time()) / data.density_fluid;
  }

private:
  Data const data;
};

template<int dim>
class DisplacementStructure : public dealii::Function<dim>
{
public:
  DisplacementStructure(Data const & data) : dealii::Function<dim>(dim, 0.0), data(data)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    return component == 1 ? data.displacement_structure(p[1], this->get_time()) : 0.0;
  }

private:
  Data const data;
};

template<int dim>
class VelocityStructure : public dealii::Function<dim>
{
public:
  VelocityStructure(Data const & data) : dealii::Function<dim>(dim, 0.0), data(data)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    return component == 1 ? data.velocity_structure(p[1], this->get_time()) : 0.0;
  }

private:
  Data const data;
};

template<int dim>
void
create_rectangle(dealii::Triangulation<dim> & tria,
                 double const                 y_min,
                 double const                 y_max,
                 unsigned int const           global_refinements)
{
  dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                    {N_CELLS_X, N_CELLS_Y},
                                                    dealii::Point<dim>(0.0, y_min),
                                                    dealii::Point<dim>(L, y_max));

  double const tol = 1.e-10 * L;
  for(auto cell : tria.cell_iterators())
  {
    for(auto const & f : cell->face_indices())
    {
      if(cell->face(f)->at_boundary())
      {
        dealii::Point<dim> const center = cell->face(f)->center();
        if(std::abs(center[0]) < tol or std::abs(center[0] - L) < tol)
          cell->face(f)->set_boundary_id(BOUNDARY_ID_LATERAL);
        else if(std::abs(center[1]) < tol)
          cell->face(f)->set_boundary_id(BOUNDARY_ID_FSI);
        else
          cell->face(f)->set_boundary_id(BOUNDARY_ID_OUTER);
      }
    }
  }

  tria.refine_global(global_refinements);
}

} // namespace Piston

namespace FluidFSI
{
template<int dim, typename Number>
class Application : public FluidFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : FluidFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    data.add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Fluid");
    {
      prm.add_parameter("TemporalDiscretization",    temporal_discretization,      "Temporal discretization.");
      prm.add_parameter("TreatmentOfConvectiveTerm", treatment_of_convective_term, "Treatment of the convective term.");
      prm.add_parameter("OrderTimeIntegrator",       order_time_integrator,        "Order of the BDF time integrator.",      dealii::Patterns::Integer(1, 3));
      prm.add_parameter("MeshMovementType",          mesh_movement_type,           "Mesh motion: Poisson or Elasticity.");
      prm.add_parameter("ApplyLerayProjection",      apply_leray_projection,       "Consistent splitting: apply the Leray projection.", dealii::Patterns::Bool());
    }
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  set_parameters() final
  {
    using namespace IncNS;

    Parameters & param = this->param;

    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    param.use_outflow_bc_convective_term = true;
    param.right_hand_side                = false;

    // ALE
    param.ale_formulation                     = true;
    param.mesh_movement_type                  = mesh_movement_type;
    param.neumann_with_variable_normal_vector = false;

    // PHYSICAL QUANTITIES
    param.start_time = 0.0;
    param.end_time   = data.end_time;
    param.viscosity  = data.viscosity;
    param.density    = data.density_fluid;

    // TEMPORAL DISCRETIZATION
    param.solver_type                   = SolverType::Unsteady;
    param.temporal_discretization       = temporal_discretization;
    param.treatment_of_convective_term  = treatment_of_convective_term;
    param.order_time_integrator         = order_time_integrator;
    param.start_with_low_order          = true;
    param.adaptive_time_stepping        = false;
    param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    param.time_step_size                = data.time_step_size;

    // output of solver information
    param.solver_info_data.interval_time = data.end_time / 10.0;

    // SPATIAL DISCRETIZATION
    param.grid.triangulation_type     = TriangulationType::Distributed;
    param.mapping_degree              = 1;
    param.mapping_degree_coarse_grids = param.mapping_degree;
    param.degree_p                    = DegreePressure::MixedOrder;

    // convective term
    param.upwind_factor = 1.0;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    param.gradp_integrated_by_parts = true;
    param.gradp_use_boundary_data   = true;
    param.gradp_formulation         = FormulationPressureGradientTerm::Weak;
    param.divu_integrated_by_parts  = true;
    param.divu_use_boundary_data    = true;
    param.divu_formulation          = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_data_pressure_poisson =
      SolverData(1000, data.abs_tol, data.rel_tol, LinearSolver::CG, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;

    // projection step
    param.solver_data_projection = SolverData(1000, data.abs_tol, data.rel_tol, LinearSolver::CG);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // CONSISTENT SPLITTING SCHEME
    param.order_extrapolation_pressure_rhs = param.order_extrapolation_pressure_nbc;
    param.apply_leray_projection           = apply_leray_projection;

    // viscous step of the dual splitting scheme, momentum step of the
    // consistent splitting and the pressure-correction schemes
    param.newton_solver_data_momentum = Newton::SolverData(100, data.abs_tol, data.rel_tol);
    param.solver_data_momentum =
      SolverData(1e4, data.abs_tol, data.rel_tol, LinearSolver::FGMRES, 100);
    param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME
    param.order_pressure_extrapolation = param.order_time_integrator - 1;
    param.rotational_formulation       = true;

    // COUPLED NAVIER-STOKES SOLVER
    param.newton_solver_data_coupled = Newton::SolverData(100, data.abs_tol, data.rel_tol);
    param.solver_data_coupled =
      SolverData(1e4, data.abs_tol, data.rel_tol, LinearSolver::FGMRES, 100);
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;
    param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
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

        Piston::create_rectangle(tria, 0.0, Piston::H, global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

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
    using namespace Piston;

    std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor = this->boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // velocity
    boundary_descriptor->velocity->dirichlet_cached_bc.insert(BOUNDARY_ID_FSI);
    boundary_descriptor->velocity->neumann_bc.insert(
      pair(BOUNDARY_ID_LATERAL, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTER, new VelocityFluid<dim>(data)));

    // pressure
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_FSI);
    boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(BOUNDARY_ID_LATERAL, new KinematicPressureFluid<dim>(data)));
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_OUTER);
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions = this->field_functions;

    field_functions->initial_solution_velocity.reset(new Piston::VelocityFluid<dim>(data));
    field_functions->initial_solution_pressure.reset(new Piston::KinematicPressureFluid<dim>(data));
    field_functions->analytical_solution_pressure.reset(
      new Piston::KinematicPressureFluid<dim>(data));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    IncNS::PostProcessorData<dim> pp_data;

    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = 0.0;
    pp_data.output_data.time_control_data.trigger_interval = data.error_interval;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_fluid";
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    // errors in the absolute L2 norm, since the solution vanishes at t = 0
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = 0.0;
    pp_data.error_data_u.time_control_data.trigger_interval = data.error_interval;
    pp_data.error_data_u.analytical_solution.reset(new Piston::VelocityFluid<dim>(data));
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.name                      = "fluid velocity";

    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = 0.0;
    pp_data.error_data_p.time_control_data.trigger_interval = data.error_interval;
    pp_data.error_data_p.analytical_solution.reset(new Piston::KinematicPressureFluid<dim>(data));
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.name                      = "fluid pressure";

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  void
  set_parameters_ale_poisson() final
  {
    using namespace Poisson;

    Parameters & param = this->ale_poisson_param;

    param.right_hand_side        = false;
    param.spatial_discretization = SpatialDiscretization::CG;
    param.degree                 = this->param.mapping_degree;

    param.solver_data         = SolverData(1e4, data.abs_tol, data.rel_tol, LinearSolver::CG, 100);
    param.preconditioner      = Preconditioner::Multigrid;
    param.multigrid_data.type = MultigridType::phMG;
    param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data.coarse_problem.solver  = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
  }

  void
  set_boundary_descriptor_ale_poisson() final
  {
    using namespace Piston;

    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor =
      this->ale_poisson_boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    boundary_descriptor->dirichlet_cached_bc.insert(BOUNDARY_ID_FSI);

    // The displacement is linear in y for a uniform interface displacement.
    boundary_descriptor->neumann_bc.insert(
      pair(BOUNDARY_ID_LATERAL, new dealii::Functions::ZeroFunction<dim>(dim)));

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_OUTER, dealii::ComponentMask()));
  }

  void
  set_field_functions_ale_poisson() final
  {
    std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions =
      this->ale_poisson_field_functions;

    field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  void
  set_parameters_ale_elasticity() final
  {
    using namespace Structure;

    Parameters & param = this->ale_elasticity_param;

    param.problem_type         = ProblemType::Steady;
    param.body_force           = false;
    param.pull_back_body_force = false;
    param.large_deformation    = false;
    param.pull_back_traction   = false;

    param.degree = this->param.mapping_degree;

    param.newton_solver_data  = Newton::SolverData(1e4, data.abs_tol, data.rel_tol);
    param.solver_data         = SolverData(1e4, data.abs_tol, data.rel_tol, LinearSolver::CG, 100);
    param.preconditioner      = Preconditioner::Multigrid;
    param.multigrid_data.type = MultigridType::phMG;
    param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
  }

  void
  set_boundary_descriptor_ale_elasticity() final
  {
    using namespace Piston;

    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor =
      this->ale_elasticity_boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    boundary_descriptor->dirichlet_cached_bc.insert(BOUNDARY_ID_FSI);

    // The lateral boundaries slide in y-direction, such that the displacement
    // is linear in y for a uniform interface displacement.
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_LATERAL, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_LATERAL, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_LATERAL, dealii::ComponentMask(std::vector<bool>{true, false})));

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_OUTER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_OUTER, dealii::ComponentMask()));
  }

  void
  set_material_descriptor_ale_elasticity() final
  {
    using namespace Structure;

    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    this->ale_elasticity_material_descriptor->insert(
      Pair(0,
           new StVenantKirchhoffData<dim>(MaterialType::StVenantKirchhoff,
                                          1.0 /* youngs_modulus */,
                                          0.3 /* poissons_ratio */,
                                          Type2D::PlaneStrain)));
  }

  void
  set_field_functions_ale_elasticity() final
  {
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions =
      this->ale_elasticity_field_functions;

    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_displacement.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  Piston::Data data;

  IncNS::TemporalDiscretization temporal_discretization =
    IncNS::TemporalDiscretization::BDFCoupledSolution;
  IncNS::TreatmentOfConvectiveTerm treatment_of_convective_term =
    IncNS::TreatmentOfConvectiveTerm::Explicit;
  unsigned int            order_time_integrator  = 2;
  IncNS::MeshMovementType mesh_movement_type     = IncNS::MeshMovementType::Poisson;
  bool                    apply_leray_projection = true;
};
} // namespace FluidFSI

namespace StructureFSI
{
template<int dim, typename Number>
class Application : public StructureFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : StructureFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    data.add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Structure");
    {
      prm.add_parameter("GenAlphaType",   gen_alpha_type,  "Type of the generalized-alpha method.");
      prm.add_parameter("SpectralRadius", spectral_radius, "Spectral radius of the generalized-alpha method.", dealii::Patterns::Double(0.0, 1.0));
    }
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  set_parameters() final
  {
    using namespace Structure;

    Parameters & param = this->param;

    param.problem_type         = ProblemType::Unsteady;
    param.body_force           = false;
    param.pull_back_body_force = false;
    // The traction balance of the analytical solution is the one of linear
    // elasticity, the interface does not change its length.
    param.large_deformation  = false;
    param.pull_back_traction = false;

    param.density = data.density_structure;

    param.start_time                     = 0.0;
    param.end_time                       = data.end_time;
    param.time_step_size                 = data.time_step_size;
    param.gen_alpha_type                 = gen_alpha_type;
    param.spectral_radius                = spectral_radius;
    param.solver_info_data.interval_time = data.end_time / 10.0;

    param.grid.triangulation_type     = TriangulationType::Distributed;
    param.mapping_degree              = 1;
    param.mapping_degree_coarse_grids = param.mapping_degree;

    param.newton_solver_data  = Newton::SolverData(1e4, data.abs_tol, data.rel_tol);
    param.solver_data         = SolverData(1e4, data.abs_tol, data.rel_tol, LinearSolver::CG, 100);
    param.preconditioner      = Preconditioner::Multigrid;
    param.multigrid_data.type = MultigridType::phMG;
    param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
    param.update_preconditioner = false;
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

        Piston::create_rectangle(tria, -Piston::H, 0.0, global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

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
    using namespace Piston;

    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor =
      this->boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    // clamped
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_OUTER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_OUTER, dealii::ComponentMask()));

    // zero horizontal displacement and zero vertical traction
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_LATERAL, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_LATERAL, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_LATERAL, dealii::ComponentMask(std::vector<bool>{true, false})));

    // fluid-structure interface
    boundary_descriptor->neumann_cached_bc.insert(BOUNDARY_ID_FSI);
  }

  void
  set_material_descriptor() final
  {
    using namespace Structure;

    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    this->material_descriptor->insert(
      Pair(0,
           new StVenantKirchhoffData<dim>(MaterialType::StVenantKirchhoff,
                                          data.youngs_modulus,
                                          data.poissons_ratio,
                                          Type2D::PlaneStrain)));
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions = this->field_functions;

    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_displacement.reset(new Piston::DisplacementStructure<dim>(data));
    field_functions->initial_velocity.reset(new Piston::VelocityStructure<dim>(data));
  }

  std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    using namespace Structure;

    PostProcessorData<dim> pp_data;

    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = 0.0;
    pp_data.output_data.time_control_data.trigger_interval = data.error_interval;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_structure";
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    pp_data.error_data.time_control_data.is_active        = true;
    pp_data.error_data.time_control_data.start_time       = 0.0;
    pp_data.error_data.time_control_data.trigger_interval = data.error_interval;
    pp_data.error_data.analytical_solution.reset(new Piston::DisplacementStructure<dim>(data));
    pp_data.error_data.calculate_relative_errors = false;
    pp_data.error_data.name                      = "structure displacement";

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  Piston::Data data;

  GenAlphaType gen_alpha_type  = GenAlphaType::Newmark;
  double       spectral_radius = 1.0;
};

} // namespace StructureFSI

namespace FSI
{
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
  {
    AssertThrow(dim == 2, dealii::ExcMessage("The piston is implemented in 2D only."));

    this->structure = std::make_shared<StructureFSI::Application<dim, Number>>(input_file, comm);
    this->fluid     = std::make_shared<FluidFSI::Application<dim, Number>>(input_file, comm);
  }
};
} // namespace FSI

} // namespace ExaDG

#include <exadg/fluid_structure_interaction/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_FLUID_STRUCTURE_INTERACTION_PISTON_APPLICATION_H_ */
