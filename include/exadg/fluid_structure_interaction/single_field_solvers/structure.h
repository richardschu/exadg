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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SINGLE_FIELD_SOLVERS_STRUCTURE_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SINGLE_FIELD_SOLVERS_STRUCTURE_H_

// Structure
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
class SolverStructure
{
public:
  void
  setup(std::shared_ptr<StructureFSI::ApplicationBase<dim, Number>> application,
        Number const &                                              robin_parameter_in,
        MPI_Comm const                                              mpi_comm,
        bool const                                                  is_test);

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  // spatial discretization
  std::shared_ptr<Structure::Operator<dim, Number>> pde_operator;

  // temporal discretization
  std::shared_ptr<Structure::TimeIntGenAlpha<dim, Number>> time_integrator;

  // postprocessor
  std::shared_ptr<Structure::PostProcessor<dim, Number>> postprocessor;
};

template<int dim, typename Number>
void
SolverStructure<dim, Number>::setup(
  std::shared_ptr<StructureFSI::ApplicationBase<dim, Number>> application,
  Number const &                                              robin_parameter_in,
  MPI_Comm const                                              mpi_comm,
  bool const                                                  is_test)
{
  // setup spatial operator
  pde_operator =
    std::make_shared<Structure::Operator<dim, Number>>(application->get_grid(),
                                                       application->get_boundary_descriptor(),
                                                       application->get_field_functions(),
                                                       application->get_material_descriptor(),
                                                       application->get_parameters(),
                                                       "elasticity",
                                                       mpi_comm);

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
  matrix_free->reinit(*application->get_grid()->mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  pde_operator->setup(matrix_free, matrix_free_data);

  // initialize postprocessor
  postprocessor = application->create_postprocessor();
  postprocessor->setup(pde_operator->get_dof_handler(), *application->get_grid()->mapping);

  // initialize time integrator
  time_integrator = std::make_shared<Structure::TimeIntGenAlpha<dim, Number>>(
    pde_operator, postprocessor, application->get_parameters(), mpi_comm, is_test);

  time_integrator->setup(application->get_parameters().restarted_simulation);

  // Robin parameter needs to be set *before* solver setup
  AssertThrow(application->get_boundary_descriptor()->neumann_cached_bc.size() > 0,
              dealii::ExcMessage(
                "FSI boundary id on structure side expected as cached Neumann BC."));

  std::map<dealii::types::boundary_id, std::pair<std::array<bool, 2>, std::array<double, 3>>>
    robin_k_c_p_param_fsi;
  for(auto const & entry : application->get_boundary_descriptor()->neumann_cached_bc)
  {
    robin_k_c_p_param_fsi.insert(std::make_pair(
      entry.first /* boundary_id */,
      std::make_pair(std::array<bool, 2>{{false /* normal_projection_displacement */,
                                          false /* normal_projection_velocity */}},
                     std::array<double, 3>{{0.0 /* coefficient_displacement */,
                                            robin_parameter_in /* coefficient_velocity */,
                                            0.0 /* exterior_pressure */}})));
  }
  pde_operator->set_combine_robin_param(robin_k_c_p_param_fsi);

  pde_operator->setup_solver(time_integrator->get_scaling_factor_mass() + application->get_parameters().mass_damping_coefficient*time_integrator->get_scaling_factor_mass_velocity(),
                             time_integrator->get_scaling_factor_mass_velocity());
}

} // namespace FSI
} // namespace ExaDG

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SINGLE_FIELD_SOLVERS_STRUCTURE_H_ */
