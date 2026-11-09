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

#include <exadg/incompressible_navier_stokes/spatial_discretization/turbulence_model.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/viscosity_model_functions.h>
#include <exadg/operators/quadrature.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
TurbulenceModel<dim, Number>::TurbulenceModel()
{
}

template<int dim, typename Number>
TurbulenceModel<dim, Number>::~TurbulenceModel()
{
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  dealii::Mapping<dim> const &                           mapping_in,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
  TurbulenceModelData const &                            turbulence_model_data_in,
  unsigned int const                                     dof_index_velocity_in)
{
  Base::initialize(matrix_free_in, viscous_kernel_in, dof_index_velocity_in);

  turbulence_model_data = turbulence_model_data_in;

  turbulence_model_data.check();

  calculate_filter_width(mapping_in);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::set_viscosity(VectorType const & velocity) const
{
  this->viscous_kernel->set_constant_coefficient(this->viscous_kernel->get_data().viscosity);

  this->add_viscosity(velocity);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::add_viscosity(VectorType const & velocity) const
{
  VectorType dummy;

  this->matrix_free->loop(&This::cell_loop_set_coefficients,
                          &This::face_loop_set_coefficients,
                          &This::boundary_face_loop_set_coefficients,
                          this,
                          dummy,
                          velocity);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      cell_range) const
{
  CellIntegratorU integrator(matrix_free,
                             this->dof_index_velocity,
                             this->viscous_kernel->get_quad_index());

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    // get filter width for this cell
    scalar filter_width = integrator.read_cell_data(this->filter_width_vector);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate velocity gradient
      tensor velocity_gradient = integrator.get_gradient(q);

      // get the current viscosity
      scalar viscosity = this->viscous_kernel->get_viscosity_cell(cell, q);

      ViscosityModels::add_turbulent_viscosity(viscosity,
                                               filter_width,
                                               velocity_gradient,
                                               turbulence_model_data);

      // set the coefficients
      this->viscous_kernel->set_coefficient_cell(cell, q, viscosity);
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) const
{
  FaceIntegratorU integrator_m(matrix_free,
                               true,
                               this->dof_index_velocity,
                               this->viscous_kernel->get_quad_index());
  FaceIntegratorU integrator_p(matrix_free,
                               false,
                               this->dof_index_velocity,
                               this->viscous_kernel->get_quad_index());

  // loop over all interior faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    integrator_m.read_dof_values(src);
    integrator_p.read_dof_values(src);

    // we only need the gradient
    integrator_m.evaluate(dealii::EvaluationFlags::gradients);
    integrator_p.evaluate(dealii::EvaluationFlags::gradients);

    // get filter width for this cell and the neighbor
    scalar filter_width          = integrator_m.read_cell_data(this->filter_width_vector);
    scalar filter_width_neighbor = integrator_p.read_cell_data(this->filter_width_vector);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      // calculate velocity gradient for both elements adjacent to the current face
      tensor velocity_gradient          = integrator_m.get_gradient(q);
      tensor velocity_gradient_neighbor = integrator_p.get_gradient(q);

      // get the coefficients
      scalar viscosity          = this->viscous_kernel->get_coefficient_face(face, q);
      scalar viscosity_neighbor = this->viscous_kernel->get_coefficient_face_neighbor(face, q);

      ViscosityModels::add_turbulent_viscosity(viscosity,
                                               filter_width,
                                               velocity_gradient,
                                               turbulence_model_data);

      ViscosityModels::add_turbulent_viscosity(viscosity_neighbor,
                                               filter_width_neighbor,
                                               velocity_gradient_neighbor,
                                               turbulence_model_data);

      // set the coefficients
      this->viscous_kernel->set_coefficient_face(face, q, viscosity);
      this->viscous_kernel->set_coefficient_face_neighbor(face, q, viscosity_neighbor);
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::boundary_face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) const
{
  FaceIntegratorU integrator(matrix_free,
                             true,
                             this->dof_index_velocity,
                             this->viscous_kernel->get_quad_index());

  // loop over all boundary faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    // get filter width for this cell
    scalar filter_width = integrator.read_cell_data(this->filter_width_vector);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate velocity gradient
      tensor velocity_gradient = integrator.get_gradient(q);

      // get the coefficients
      scalar viscosity = this->viscous_kernel->get_coefficient_face(face, q);

      ViscosityModels::add_turbulent_viscosity(viscosity,
                                               filter_width,
                                               velocity_gradient,
                                               turbulence_model_data);

      // set the coefficients
      this->viscous_kernel->set_coefficient_face(face, q, viscosity);
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::calculate_filter_width(dealii::Mapping<dim> const & mapping)
{
  unsigned int n_cells =
    this->matrix_free->n_cell_batches() + this->matrix_free->n_ghost_cell_batches();

  filter_width_vector.resize(n_cells);

  dealii::DoFHandler<dim> const & dof_handler =
    this->matrix_free->get_dof_handler(this->dof_index_velocity);
  ElementType const element_type = get_element_type(dof_handler.get_triangulation());
  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(element_type, this->viscous_kernel->get_degree() + 1);

  dealii::FEValues<dim> fe_values(mapping,
                                  dof_handler.get_fe(),
                                  *quadrature,
                                  dealii::update_JxW_values);

  // loop over all cells
  for(unsigned int i = 0; i < n_cells; ++i)
  {
    for(unsigned int v = 0; v < this->matrix_free->n_active_entries_per_cell_batch(i); ++v)
    {
      typename dealii::DoFHandler<dim>::cell_iterator cell =
        this->matrix_free->get_cell_iterator(i, v, this->dof_index_velocity);
      fe_values.reinit(cell);

      filter_width_vector[i][v] =
        ViscosityModels::compute_filter_width(fe_values, this->viscous_kernel->get_degree());
    }
  }
}

template class TurbulenceModel<2, float>;
template class TurbulenceModel<2, double>;
template class TurbulenceModel<3, float>;
template class TurbulenceModel<3, double>;

} // namespace IncNS
} // namespace ExaDG
