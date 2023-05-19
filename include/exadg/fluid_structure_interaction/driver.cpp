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

// ExaDG
#include <exadg/fluid_structure_interaction/driver.h>
#include <exadg/grid/marked_vertices.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(std::string const &                           input_file,
                            MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    is_test(is_test),
    application(app)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);

  dealii::ParameterHandler prm;
  parameters.add_parameters(prm);
  prm.parse_input(input_file, "", true, true);
  parameters.parse_parameters(prm);

  structure = std::make_shared<SolverStructure<dim, Number>>();
  fluid     = std::make_shared<SolverFluid<dim, Number>>();

  partitioned_solver = std::make_shared<PartitionedSolver<dim, Number>>(parameters, mpi_comm);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up fluid-structure interaction solver:" << std::endl;

  // setup application
  {
    dealii::Timer timer_local;

    application->setup();

    timer_tree.insert({"FSI", "Setup", "Application"}, timer_local.wall_time());
  }

  // setup fluid
  {
    dealii::Timer timer_local;

    fluid->setup(application->fluid, mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
  }

  // setup structure
  {
    dealii::Timer timer_local;

    structure->setup(application->structure, compute_robin_parameter(), mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
  }

  setup_interface_coupling();

  partitioned_solver->setup(fluid, structure);

  timer_tree.insert({"FSI", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_interface_coupling()
{
  // structure to ALE
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> ALE ..." << std::endl;

    auto const & tria         = structure->pde_operator->get_dof_handler().get_triangulation();
    auto const   boundary_ids = extract_set_of_keys_from_map(
      application->structure->get_boundary_descriptor()->neumann_cached_bc);
    auto const marked_vertices_structure = get_marked_vertices_via_boundary_ids(tria, boundary_ids);

    if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<1, dim, Number>>();
      structure_to_ale->setup(fluid->ale_poisson_operator->get_container_interface_data(),
                              structure->pde_operator->get_dof_handler(),
                              *application->structure->get_grid()->mapping,
                              marked_vertices_structure,
                              parameters.geometric_tolerance);
    }
    else if(application->fluid->get_parameters().mesh_movement_type ==
            IncNS::MeshMovementType::Elasticity)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<1, dim, Number>>();
      structure_to_ale->setup(
        fluid->ale_elasticity_operator->get_container_interface_data_dirichlet(),
        structure->pde_operator->get_dof_handler(),
        *application->structure->get_grid()->mapping,
        marked_vertices_structure,
        parameters.geometric_tolerance);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> ALE"}, timer_local.wall_time());
  }

  // structure to fluid
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> fluid ..." << std::endl;

    auto const & tria         = structure->pde_operator->get_dof_handler().get_triangulation();
    auto const   boundary_ids = extract_set_of_keys_from_map(
      application->structure->get_boundary_descriptor()->neumann_cached_bc);
    auto const marked_vertices_structure = get_marked_vertices_via_boundary_ids(tria, boundary_ids);

    structure_to_fluid = std::make_shared<InterfaceCoupling<1, dim, Number>>();
    structure_to_fluid->setup(fluid->pde_operator->get_container_interface_data(),
                              structure->pde_operator->get_dof_handler(),
                              *application->structure->get_grid()->mapping,
                              marked_vertices_structure,
                              parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> fluid"}, timer_local.wall_time());
  }

  // fluid to structure
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling fluid -> structure ..." << std::endl;

    std::shared_ptr<dealii::Mapping<dim> const> mapping_fluid =
      get_dynamic_mapping<dim, Number>(application->fluid->get_grid(), fluid->ale_grid_motion);

    auto const & tria         = fluid->pde_operator->get_dof_handler_u().get_triangulation();
    auto const   boundary_ids = extract_set_of_keys_from_map(
      application->fluid->get_boundary_descriptor()->velocity->dirichlet_cached_bc);
    auto const marked_vertices_fluid = get_marked_vertices_via_boundary_ids(tria, boundary_ids);

    fluid_to_structure = std::make_shared<InterfaceCoupling<1, dim, Number>>();
    fluid_to_structure->setup(structure->pde_operator->get_container_interface_data_neumann(),
                              fluid->pde_operator->get_dof_handler_u(),
                              *mapping_fluid,
                              marked_vertices_fluid,
                              parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling fluid -> structure"}, timer_local.wall_time());
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  // The fluid domain is the master that dictates the start time
  structure->time_integrator->reset_time(fluid->time_integrator->get_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  // The fluid domain is the master that dictates the time step size
  structure->time_integrator->set_current_time_step_size(
    fluid->time_integrator->get_time_step_size());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_ale(VectorType const & displacement_structure) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  structure_to_ale->update_data(displacement_structure);
  timer_tree.insert({"FSI", "Coupling structure -> ALE"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_fluid(unsigned int const iteration,
		                                         bool const         extrapolate) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType velocity_structure;
  structure->pde_operator->initialize_dof_vector(velocity_structure);
  if(iteration == 0)
  {
	if(extrapolate)
    {
	  structure->time_integrator->extrapolate_velocity_to_np(velocity_structure);
    }
	else
	{
	  velocity_structure = structure->time_integrator->get_velocity_n();
	}
  }
  else
  {
    velocity_structure = structure->time_integrator->get_velocity_np();
  }

  structure_to_fluid->update_data(velocity_structure);

  timer_tree.insert({"FSI", "Coupling structure -> fluid"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_fluid_to_structure(bool const end_of_time_step) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType stress_fluid;
  fluid->pde_operator->initialize_vector_velocity(stress_fluid);
  // calculate fluid stress at fluid-structure interface
  if(end_of_time_step)
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity_np(),
                                               fluid->time_integrator->get_pressure_np());
  }
  else
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity(),
                                               fluid->time_integrator->get_pressure());
  }

  stress_fluid *= -1.0;
  fluid_to_structure->update_data(stress_fluid);

  timer_tree.insert({"FSI", "Coupling fluid -> structure"}, sub_timer.wall_time());
}

template<int dim, typename Number>
double
Driver<dim, Number>::compute_robin_parameter() const
{
  if(parameters.coupling_scheme == CouplingScheme::DirichletNeumann)
  {
    return 0.0;
  }
  else if(parameters.coupling_scheme == CouplingScheme::DirichletRobinFixedParam)
  {
    return parameters.robin_parameter_scale;
  }
  else if(parameters.coupling_scheme == CouplingScheme::DirichletRobinAdaptiveParam)
  {
    return parameters.robin_parameter_scale * application->fluid->get_parameters().density /
           fluid->time_integrator->get_time_step_size();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("This CouplingScheme is not implemented."));
  }

  return 0.0;
}

template<int dim, typename Number>
void
Driver<dim, Number>::update_robin_parameters(double const & robin_parameter_in) const
{
  // update fluid operator's Robin parameter
  fluid->pde_operator->set_robin_parameter(robin_parameter_in);

  // update structure operator's Robin parameter
  AssertThrow(application->structure->get_boundary_descriptor()->neumann_cached_bc.size() > 0,
              dealii::ExcMessage(
                "FSI boundary id on structure side expected as cached Neumann BC."));

  std::map<dealii::types::boundary_id, std::pair<std::array<bool, 2>, std::array<double, 3>>>
    robin_k_c_p_param_fsi;
  for(auto const & entry : application->structure->get_boundary_descriptor()->neumann_cached_bc)
  {
    robin_k_c_p_param_fsi.insert(std::make_pair(
      entry.first /* boundary_id */,
      std::make_pair(std::array<bool, 2>{{false /* normal_projection_displacement */,
                                          false /* normal_projection_velocity */}},
                     std::array<double, 3>{{0.0 /* coefficient_displacement */,
                                            robin_parameter_in /* coefficient_velocity */,
                                            0.0 /* exterior_pressure */}})));
  }

  structure->pde_operator->set_combine_robin_param(robin_k_c_p_param_fsi);
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_subproblem_mesh(VectorType const & d, unsigned int const iteration) const
{
  if(parameters.field_update_variant == FieldUpdateVariant::Implicit or
     (iteration == 0 and
      parameters.field_update_variant == FieldUpdateVariant::GeometricExplicit) or
     (iteration == 0 and
      parameters.field_update_variant == FieldUpdateVariant::ImplicitPressureStructure))
  {
    // update structure to ALE data
    coupling_structure_to_ale(d);

    // move the fluid mesh and update dependent data structures
    fluid->solve_ale(application->fluid, is_test);
  }
  else if((iteration != 0 and
           parameters.field_update_variant == FieldUpdateVariant::GeometricExplicit) or
          (iteration != 0 and
           parameters.field_update_variant == FieldUpdateVariant::ImplicitPressureStructure))
  {
    // do not perform a mesh update
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("FieldUpdateVariant not implemented."));
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_subproblem_fluid(unsigned int const iteration) const
{
  // update velocity boundary condition for fluid
  coupling_structure_to_fluid(iteration, parameters.use_extrapolation);

  if(parameters.field_update_variant == FieldUpdateVariant::Implicit or
     parameters.field_update_variant == FieldUpdateVariant::GeometricExplicit or
     (iteration == 0 and
      parameters.field_update_variant == FieldUpdateVariant::ImplicitPressureStructure))
  {
    // perform all substeps of the fluid subproblem solver
    fluid->time_integrator->advance_one_timestep_partitioned_solve(iteration == 0 /* use_extrapolation */,
                                                                   true /* update_velocity */,
                                                                   true /* update_pressure */);
  }
  else if(iteration != 0 and
          parameters.field_update_variant == FieldUpdateVariant::ImplicitPressureStructure)
  {
    // perform only the pressure update of the subproblem solver
    fluid->time_integrator->advance_one_timestep_partitioned_solve(false /* use_extrapolation */,
                                                                   false /* update_velocity */,
                                                                   true /* update_pressure */);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("FieldUpdateVariant not implemented."));
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_subproblem_structure(VectorType &       d_tilde,
                                                unsigned int const iteration) const
{
  if(parameters.field_update_variant == FieldUpdateVariant::Implicit or
     parameters.field_update_variant == FieldUpdateVariant::GeometricExplicit or
     parameters.field_update_variant == FieldUpdateVariant::ImplicitPressureStructure)
  {
    // update stress boundary condition for solid
    coupling_fluid_to_structure(/* end_of_time_step = */ true);

    // solve structural problem
    structure->time_integrator->advance_one_timestep_partitioned_solve(iteration == 0);

    // update displacement iterate in partitioned scheme
    d_tilde = structure->time_integrator->get_displacement_np();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("FieldUpdateVariant not implemented."));
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::apply_dirichlet_robin_scheme(VectorType &       d_tilde,
                                                  VectorType const & d,
                                                  unsigned int const iteration) const
{
  // update Robin parameter dependent on CouplingScheme
  update_robin_parameters(compute_robin_parameter());

  solve_subproblem_mesh(d, iteration);

  solve_subproblem_fluid(iteration);

  solve_subproblem_structure(d_tilde, iteration);
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // compute initial acceleration for structural problem
  {
    // update stress boundary condition for solid at time t_n (not t_{n+1})
    update_robin_parameters(0.0 /* Dirichlet-Neumann scheme */);
    coupling_fluid_to_structure(false /* end_of_time_step */);
    structure->time_integrator->compute_initial_acceleration(
      application->structure->get_parameters().restarted_simulation);
  }

  // The fluid domain is the master that dictates when the time loop is finished
  while(not fluid->time_integrator->finished())
  {
    // pre-solve
    fluid->time_integrator->advance_one_timestep_pre_solve(true);
    structure->time_integrator->advance_one_timestep_pre_solve(false);

    // solve (using strongly-coupled partitioned scheme)
    auto const lambda_dirichlet_robin =
      [&](VectorType & d_tilde, VectorType const & d, unsigned int const k) {
        apply_dirichlet_robin_scheme(d_tilde, d, k);
      };
    partitioned_solver->solve(lambda_dirichlet_robin);

    // post-solve
    fluid->time_integrator->advance_one_timestep_post_solve();
    structure->time_integrator->advance_one_timestep_post_solve();

    if(application->fluid->get_parameters().adaptive_time_stepping)
      synchronize_time_step_size();
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

  // iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;

  pcout << std::endl << "FSI:" << std::endl;
  partitioned_solver->print_iterations(pcout);

  pcout << std::endl << "Fluid:" << std::endl;
  fluid->time_integrator->print_iterations();

  pcout << std::endl << "ALE:" << std::endl;
  fluid->ale_grid_motion->print_iterations();

  pcout << std::endl << "Structure:" << std::endl;
  structure->time_integrator->print_iterations();

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"FSI"}, total_time);

  timer_tree.insert({"FSI"}, fluid->time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"FSI"}, fluid->get_timings_ale());
  timer_tree.insert({"FSI"}, structure->time_integrator->get_timings(), "Structure");
  timer_tree.insert({"FSI"}, partitioned_solver->get_timings());

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index DoFs =
    fluid->pde_operator->get_number_of_dofs() + structure->pde_operator->get_number_of_dofs();

  if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
  {
    DoFs += fluid->pde_operator->get_number_of_dofs();
  }
  else if(application->fluid->get_parameters().mesh_movement_type ==
          IncNS::MeshMovementType::Elasticity)
  {
    DoFs += fluid->ale_elasticity_operator->get_number_of_dofs();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  unsigned int N_time_steps = fluid->time_integrator->get_number_of_time_steps();

  print_throughput_unsteady(pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FSI
} // namespace ExaDG
