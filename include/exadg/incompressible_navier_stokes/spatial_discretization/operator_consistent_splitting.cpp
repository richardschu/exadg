/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_consistent_splitting.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

#include <exadg/time_integration/bdf_constants.h>
#include <exadg/time_integration/extrapolation_constants.h>
#include <exadg/time_integration/time_int_multistep_base.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
OperatorConsistentSplitting<dim, Number>::OperatorConsistentSplitting(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  Parameters const &                                    parameters_in,
  std::string const &                                   field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : ProjectionBase(grid_in,
                   mapping_in,
                   multigrid_mappings_in,
                   boundary_descriptor_in,
                   field_functions_in,
                   parameters_in,
                   field_in,
                   mpi_comm_in)
{
}

template<int dim, typename Number>
OperatorConsistentSplitting<dim, Number>::~OperatorConsistentSplitting()
{
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::apply_velocity_divergence_term(
  VectorType &       dst,
  VectorType const & src) const
{
  this->divergence_operator.apply(dst, src);
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::apply_convective_divergence_term(
  VectorType &       dst,
  VectorType const & src) const
{
  // The inner faces also read the grid velocity of the neighbor.
  if(this->param.ale_formulation)
    this->convective_kernel->update_ghost_values_grid_velocity();

  this->get_matrix_free().loop(&This::local_rhs_ppe_div_term_convective_cell,
                               &This::local_rhs_ppe_div_term_convective_inner_face,
                               &This::local_rhs_ppe_div_term_convective_boundary_face,
                               this,
                               dst,
                               src,
                               true /* zero_dst_vector */);

  if(this->param.ale_formulation)
    this->convective_kernel->zero_out_ghost_values_grid_velocity();
}



template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_div_term_convective_cell(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  unsigned int dof_index_pressure         = this->get_dof_index_pressure();
  unsigned int dof_index_velocity         = this->get_dof_index_velocity();
  unsigned int quad_index_overintegration = this->get_quad_index_velocity_overintegration();

  CellIntegrator<dim, 1, Number>   pressure(matrix_free,
                                          dof_index_pressure,
                                          quad_index_overintegration);
  CellIntegrator<dim, dim, Number> velocity(matrix_free,
                                            dof_index_velocity,
                                            quad_index_overintegration);
  CellIntegrator<dim, dim, Number> grid_velocity(matrix_free,
                                                 dof_index_velocity,
                                                 quad_index_overintegration);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    pressure.reinit(cell);
    velocity.reinit(cell);

    velocity.gather_evaluate(src,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    if(use_ale_form_convective_term())
    {
      grid_velocity.reinit(cell);
      grid_velocity.gather_evaluate(this->convective_kernel->get_grid_velocity(),
                                    dealii::EvaluationFlags::values);
    }

    for(const unsigned int q : pressure.quadrature_point_indices())
    {
      // see use_ale_form_convective_term()
      vector transport_velocity = velocity.get_value(q);
      if(use_ale_form_convective_term())
        transport_velocity -= grid_velocity.get_value(q);

      vector const convective_flux = -velocity.get_gradient(q) * transport_velocity;
      pressure.submit_gradient(convective_flux, q);
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
  }
}



template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_div_term_convective_inner_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int dof_index_pressure         = this->get_dof_index_pressure();
  unsigned int dof_index_velocity         = this->get_dof_index_velocity();
  unsigned int quad_index_overintegration = this->get_quad_index_velocity_overintegration();

  FaceIntegratorP pressure_minus(matrix_free, true, dof_index_pressure, quad_index_overintegration);
  FaceIntegratorP pressure_plus(matrix_free, false, dof_index_pressure, quad_index_overintegration);
  FaceIntegratorU velocity_minus(matrix_free, true, dof_index_velocity, quad_index_overintegration);
  FaceIntegratorU velocity_plus(matrix_free, false, dof_index_velocity, quad_index_overintegration);
  FaceIntegratorU grid_velocity_minus(matrix_free,
                                      true,
                                      dof_index_velocity,
                                      quad_index_overintegration);
  FaceIntegratorU grid_velocity_plus(matrix_free,
                                     false,
                                     dof_index_velocity,
                                     quad_index_overintegration);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure_minus.reinit(face);
    pressure_plus.reinit(face);
    velocity_minus.reinit(face);
    velocity_plus.reinit(face);

    velocity_minus.gather_evaluate(src,
                                   dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients);
    velocity_plus.gather_evaluate(src,
                                  dealii::EvaluationFlags::values |
                                    dealii::EvaluationFlags::gradients);

    if(use_ale_form_convective_term())
    {
      grid_velocity_minus.reinit(face);
      grid_velocity_minus.gather_evaluate(this->convective_kernel->get_grid_velocity(),
                                          dealii::EvaluationFlags::values);
      grid_velocity_plus.reinit(face);
      grid_velocity_plus.gather_evaluate(this->convective_kernel->get_grid_velocity(),
                                         dealii::EvaluationFlags::values);
    }

    for(const unsigned int q : pressure_minus.quadrature_point_indices())
    {
      vector const normal = pressure_minus.normal_vector(q);

      // see use_ale_form_convective_term()
      vector transport_velocity_minus = velocity_minus.get_value(q);
      vector transport_velocity_plus  = velocity_plus.get_value(q);
      if(use_ale_form_convective_term())
      {
        transport_velocity_minus -= grid_velocity_minus.get_value(q);
        transport_velocity_plus -= grid_velocity_plus.get_value(q);
      }

      vector const gradu_u_minus   = velocity_minus.get_gradient(q) * transport_velocity_minus;
      vector const gradu_u_plus    = velocity_plus.get_gradient(q) * transport_velocity_plus;
      scalar const convective_flux = Number(0.5) * (gradu_u_minus + gradu_u_plus) * normal;

      pressure_minus.submit_value(convective_flux, q);
      pressure_plus.submit_value(-convective_flux, q);
    }

    pressure_minus.integrate_scatter(dealii::EvaluationFlags::values, dst);
    pressure_plus.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}


template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_div_term_convective_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int dof_index_velocity         = this->get_dof_index_velocity();
  unsigned int dof_index_pressure         = this->get_dof_index_pressure();
  unsigned int quad_index_overintegration = this->get_quad_index_velocity_overintegration();

  FaceIntegratorP pressure_minus(matrix_free, true, dof_index_pressure, quad_index_overintegration);
  FaceIntegratorU velocity_minus(matrix_free, true, dof_index_velocity, quad_index_overintegration);
  FaceIntegratorU grid_velocity(matrix_free, true, dof_index_velocity, quad_index_overintegration);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    dealii::types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);

    BoundaryTypeU const boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(boundary_id);



    if(boundary_type == BoundaryTypeU::Dirichlet or boundary_type == BoundaryTypeU::DirichletCached)
    {
      // The convective term cancels with the consistent boundary condition.
      // In the Eulerian form on a moving mesh, the time derivative at fixed
      // points in space in the boundary condition is the time derivative at
      // fixed mesh points minus grad(u) * w, where the former is evaluated
      // from the Dirichlet data, see
      // rhs_ppe_nbc_numerical_time_derivative_add().
      if(this->param.ale_formulation and not use_ale_form_convective_term())
      {
        pressure_minus.reinit(face);
        velocity_minus.reinit(face);
        velocity_minus.gather_evaluate(src, dealii::EvaluationFlags::gradients);
        grid_velocity.reinit(face);
        grid_velocity.gather_evaluate(this->convective_kernel->get_grid_velocity(),
                                      dealii::EvaluationFlags::values);

        for(const unsigned int q : pressure_minus.quadrature_point_indices())
        {
          vector const normal = pressure_minus.normal_vector(q);
          scalar const flux =
            (velocity_minus.get_gradient(q) * grid_velocity.get_value(q)) * normal;

          pressure_minus.submit_value(flux, q);
        }

        pressure_minus.integrate_scatter(dealii::EvaluationFlags::values, dst);
      }
    }
    else
    {
      pressure_minus.reinit(face);
      velocity_minus.reinit(face);

      velocity_minus.gather_evaluate(src,
                                     dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);

      if(use_ale_form_convective_term())
      {
        grid_velocity.reinit(face);
        grid_velocity.gather_evaluate(this->convective_kernel->get_grid_velocity(),
                                      dealii::EvaluationFlags::values);
      }

      for(const unsigned int q : pressure_minus.quadrature_point_indices())
      {
        const auto normal = pressure_minus.normal_vector(q);

        // On here we do not know the gradient of the velocity so use the value of the gradient from
        // the inside instead
        tensor const grad_u  = velocity_minus.get_gradient(q);
        vector       u_minus = velocity_minus.get_value(q);
        // see use_ale_form_convective_term()
        if(use_ale_form_convective_term())
          u_minus -= grid_velocity.get_value(q);
        scalar const convective_flux = (grad_u * u_minus) * normal;

        pressure_minus.submit_value(convective_flux, q);
      }

      pressure_minus.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::rhs_ppe_div_term_body_forces_add(
  VectorType &   dst,
  double const & time) const
{
  this->evaluation_time = time;

  VectorType src_dummy;
  this->get_matrix_free().loop(&This::local_rhs_ppe_div_term_body_forces_cell,
                               &This::local_rhs_ppe_div_term_body_forces_inner_face,
                               &This::local_rhs_ppe_div_term_body_forces_boundary_face,
                               this,
                               dst,
                               src_dummy);
}



template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_div_term_body_forces_cell(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & cell_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  CellIntegrator<dim, 1, Number> integrator(matrix_free, dof_index_pressure, quad_index_pressure);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; cell++)
  {
    integrator.reinit(cell);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

      // evaluate right-hand side
      vector const rhs =
        FunctionEvaluator<1, dim, Number>::value(*(this->field_functions->right_hand_side),
                                                 q_points,
                                                 this->evaluation_time);

      integrator.submit_gradient(rhs, q);
    }
    integrator.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
  }
}



template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_div_term_body_forces_inner_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP pressure_minus(matrix_free, true, dof_index_pressure, quad_index_pressure);
  FaceIntegratorP pressure_plus(matrix_free, false, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure_minus.reinit(face);
    pressure_plus.reinit(face);

    for(unsigned int q = 0; q < pressure_minus.n_q_points; ++q)
    {
      dealii::Point<dim, scalar> q_points = pressure_minus.quadrature_point(q);

      // evaluate right-hand side
      vector const rhs =
        FunctionEvaluator<1, dim, Number>::value(*(this->field_functions->right_hand_side),
                                                 q_points,
                                                 this->evaluation_time);
      vector const normal = pressure_minus.normal_vector(q);
      scalar const flux   = rhs * normal;

      pressure_minus.submit_value(-flux, q);
      pressure_plus.submit_value(flux, q);
    }
    pressure_minus.integrate_scatter(dealii::EvaluationFlags::values, dst);
    pressure_plus.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}


template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_div_term_body_forces_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(matrix_free, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    BoundaryTypeU boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    if(boundary_type == BoundaryTypeU::Dirichlet or boundary_type == BoundaryTypeU::DirichletCached)
    {
      // Do nothing on Dirichlet boudary as the boundary face term cancels with the boundary
      // condition.
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      integrator.reinit(face);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate right-hand side
        vector rhs =
          FunctionEvaluator<1, dim, Number>::value(*(this->field_functions->right_hand_side),
                                                   q_points,
                                                   this->evaluation_time);

        scalar flux_times_normal = rhs * integrator.normal_vector(q);
        integrator.submit_value(-flux_times_normal, q);
      }
      integrator.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
    }
  }
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::rhs_ppe_nbc_add(VectorType &       dst,
                                                          VectorType const & src,
                                                          double const &     time,
                                                          Number const       gamma_dt) const
{
  this->evaluation_time = time;
  this->gamma0_dt       = gamma_dt;

  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_add_boundary_face,
                               this,
                               dst,
                               src);
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_nbc_add_boundary_face(
  dealii::MatrixFree<dim, Number> const & data,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int dof_index_velocity  = this->get_dof_index_velocity();
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_velocity = this->get_quad_index_velocity_standard();

  FaceIntegratorU omega(data, true, dof_index_velocity, quad_index_velocity);
  FaceIntegratorP integrator_pressure(data, true, dof_index_pressure, quad_index_velocity);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    dealii::types::boundary_id const boundary_id = data.get_boundary_id(face);
    BoundaryTypeU                    boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(boundary_id);

    if(boundary_type == BoundaryTypeU::Dirichlet or boundary_type == BoundaryTypeU::DirichletCached)
    {
      omega.reinit(face);
      omega.gather_evaluate(src, dealii::EvaluationFlags::gradients);

      integrator_pressure.reinit(face);

      vector g = vector();

      for(unsigned int q = 0; q < integrator_pressure.n_q_points; ++q)
      {
        // First handle the time derivative
        if(boundary_type == BoundaryTypeU::Dirichlet)
        {
          auto bc = this->boundary_descriptor->velocity->dirichlet_bc.find(boundary_id)->second;
          auto const q_point = integrator_pressure.quadrature_point(q);

          g = FunctionEvaluator<1, dim, Number>::value(*bc, q_point, this->evaluation_time);
        }
        else if(boundary_type == BoundaryTypeU::DirichletCached)
        {
          auto bc = this->boundary_descriptor->velocity->get_dirichlet_cached_data();

          g = FunctionEvaluator<1, dim, Number>::value(*bc, face, q, quad_index_velocity);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }
        vector const normal = integrator_pressure.normal_vector(q);
        vector const dudt   = this->gamma0_dt * g;
        scalar const h_dudt = -dudt * normal;

        // Now add the curl-curl term
        scalar const viscosity  = this->get_viscosity_boundary_face(face, q);
        vector const curl_omega = CurlCompute<dim, FaceIntegratorU>::compute(omega, q);

        scalar const h_viscosity = -(viscosity * curl_omega) * normal;

        integrator_pressure.submit_value(h_dudt + h_viscosity, q);
      }
      integrator_pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      // Nothing to do
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
    }
  }
}

/*
 * With a variable viscosity nu and div(u) = 0, the viscous term in the pressure
 * Poisson equation reads
 *
 *   div(2 nu sym_grad(u)) = - curl(nu curl(u)) + 2 grad(u)^T grad(nu),
 *
 * see Pacheco, Schussnig, Fries, Comput. Methods Appl. Mech. Engrg. 382 (2021)
 * 113888. The first term is divergence-free and enters via the Neumann
 * boundary condition on Gamma_D only, where
 *
 *   - curl(nu curl(u)) = - nu curl(curl(u)) + (grad(u) - grad(u)^T) grad(nu)
 *
 * with the first part in rhs_ppe_nbc_add(). This function adds the second
 * part on Gamma_D and the second term G = 2 grad(u)^T grad(nu), like the
 * convective term: (G, grad(q)) in the cells and the central flux on inner
 * faces and velocity Neumann boundaries. On Gamma_D, the boundary flux of G
 * cancels with the Neumann boundary condition. All terms vanish for a constant
 * viscosity.
 */
template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::rhs_ppe_variable_viscosity_add(
  VectorType &       dst,
  VectorType const & velocity,
  VectorType const & viscosity_in)
{
  // The inner faces also read the viscosity of the neighbor.
  viscosity_in.update_ghost_values();
  this->viscosity = &viscosity_in;

  this->get_matrix_free().loop(&This::local_rhs_ppe_variable_viscosity_cell,
                               &This::local_rhs_ppe_variable_viscosity_inner_face,
                               &This::local_rhs_ppe_variable_viscosity_boundary_face,
                               this,
                               dst,
                               velocity);

  this->viscosity = nullptr;
  viscosity_in.zero_out_ghost_values();
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_variable_viscosity_cell(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  unsigned int const dof_index_velocity        = this->get_dof_index_velocity();
  unsigned int const dof_index_velocity_scalar = this->get_dof_index_velocity_scalar();
  unsigned int const dof_index_pressure        = this->get_dof_index_pressure();
  unsigned int const quad_index                = this->get_quad_index_velocity_standard();

  CellIntegrator<dim, 1, Number>   pressure(matrix_free, dof_index_pressure, quad_index);
  CellIntegrator<dim, dim, Number> velocity(matrix_free, dof_index_velocity, quad_index);
  CellIntegrator<dim, 1, Number>   viscosity_integrator(matrix_free,
                                                      dof_index_velocity_scalar,
                                                      quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    pressure.reinit(cell);
    velocity.reinit(cell);
    velocity.gather_evaluate(src, dealii::EvaluationFlags::gradients);
    viscosity_integrator.reinit(cell);
    viscosity_integrator.gather_evaluate(*this->viscosity, dealii::EvaluationFlags::gradients);

    for(unsigned int const q : pressure.quadrature_point_indices())
    {
      vector const flux =
        Number(2.0) * (transpose(velocity.get_gradient(q)) * viscosity_integrator.get_gradient(q));
      pressure.submit_gradient(flux, q);
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
  }
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_variable_viscosity_inner_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int const dof_index_velocity        = this->get_dof_index_velocity();
  unsigned int const dof_index_velocity_scalar = this->get_dof_index_velocity_scalar();
  unsigned int const dof_index_pressure        = this->get_dof_index_pressure();
  unsigned int const quad_index                = this->get_quad_index_velocity_standard();

  FaceIntegratorP pressure_minus(matrix_free, true, dof_index_pressure, quad_index);
  FaceIntegratorP pressure_plus(matrix_free, false, dof_index_pressure, quad_index);
  FaceIntegratorU velocity_minus(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorU velocity_plus(matrix_free, false, dof_index_velocity, quad_index);
  FaceIntegratorP viscosity_minus(matrix_free, true, dof_index_velocity_scalar, quad_index);
  FaceIntegratorP viscosity_plus(matrix_free, false, dof_index_velocity_scalar, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure_minus.reinit(face);
    pressure_plus.reinit(face);
    velocity_minus.reinit(face);
    velocity_minus.gather_evaluate(src, dealii::EvaluationFlags::gradients);
    velocity_plus.reinit(face);
    velocity_plus.gather_evaluate(src, dealii::EvaluationFlags::gradients);
    viscosity_minus.reinit(face);
    viscosity_minus.gather_evaluate(*this->viscosity, dealii::EvaluationFlags::gradients);
    viscosity_plus.reinit(face);
    viscosity_plus.gather_evaluate(*this->viscosity, dealii::EvaluationFlags::gradients);

    for(unsigned int const q : pressure_minus.quadrature_point_indices())
    {
      vector const normal = pressure_minus.normal_vector(q);

      vector const flux_minus =
        Number(2.0) * (transpose(velocity_minus.get_gradient(q)) * viscosity_minus.get_gradient(q));
      vector const flux_plus =
        Number(2.0) * (transpose(velocity_plus.get_gradient(q)) * viscosity_plus.get_gradient(q));
      scalar const flux = Number(0.5) * (flux_minus + flux_plus) * normal;

      pressure_minus.submit_value(-flux, q);
      pressure_plus.submit_value(flux, q);
    }

    pressure_minus.integrate_scatter(dealii::EvaluationFlags::values, dst);
    pressure_plus.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::local_rhs_ppe_variable_viscosity_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int const dof_index_velocity        = this->get_dof_index_velocity();
  unsigned int const dof_index_velocity_scalar = this->get_dof_index_velocity_scalar();
  unsigned int const dof_index_pressure        = this->get_dof_index_pressure();
  unsigned int const quad_index                = this->get_quad_index_velocity_standard();

  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);
  FaceIntegratorU velocity(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorP viscosity_integrator(matrix_free, true, dof_index_velocity_scalar, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    BoundaryTypeU const boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    bool const dirichlet_boundary =
      boundary_type == BoundaryTypeU::Dirichlet or boundary_type == BoundaryTypeU::DirichletCached;

    AssertThrow(dirichlet_boundary or boundary_type == BoundaryTypeU::Neumann,
                dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

    pressure.reinit(face);
    velocity.reinit(face);
    velocity.gather_evaluate(src, dealii::EvaluationFlags::gradients);
    viscosity_integrator.reinit(face);
    viscosity_integrator.gather_evaluate(*this->viscosity, dealii::EvaluationFlags::gradients);

    for(unsigned int const q : pressure.quadrature_point_indices())
    {
      vector const normal  = pressure.normal_vector(q);
      tensor const grad_u  = velocity.get_gradient(q);
      vector const grad_nu = viscosity_integrator.get_gradient(q);

      scalar flux;
      if(dirichlet_boundary)
      {
        // (grad(u) - grad(u)^T) grad(nu) of the Neumann boundary condition
        flux = ((grad_u - transpose(grad_u)) * grad_nu) * normal;
      }
      else
      {
        // boundary flux of G from the interior
        flux = -(Number(2.0) * (transpose(grad_u) * grad_nu)) * normal;
      }

      pressure.submit_value(flux, q);
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
OperatorConsistentSplitting<dim, Number>::apply_helmholtz_operator(VectorType &       dst,
                                                                   VectorType const & src) const
{
  this->momentum_operator.vmult(dst, src);
}

template class OperatorConsistentSplitting<2, float>;
template class OperatorConsistentSplitting<2, double>;

template class OperatorConsistentSplitting<3, float>;
template class OperatorConsistentSplitting<3, double>;

} // namespace IncNS
} // namespace ExaDG
