/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/operators/boundary_mass_operator.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
BoundaryMassOperator<dim, n_components, Number>::BoundaryMassOperator() : scaling_factor(1.0)
{
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  BoundaryMassOperatorData<dim> const &             data)
{
  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = kernel.get_integrator_flags();
  this->boundary_ids = data.boundary_ids;
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::set_scaling_factor(Number const & number)
{
  scaling_factor = number;
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::apply_scale(VectorType &       dst,
                                                             Number const &     factor,
                                                             VectorType const & src) const
{
  scaling_factor = factor;

  this->apply(dst, src);

  scaling_factor = 1.0;
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::apply_scale_add(VectorType &       dst,
                                                                 Number const &     factor,
                                                                 VectorType const & src) const
{
  scaling_factor = factor;

  this->apply_add(dst, src);

  scaling_factor = 1.0;
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  (void)integrator;
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::do_boundary_integral(IntegratorFace &                   integrator_m,
                     OperatorType const &               operator_type,
                     dealii::types::boundary_id const & boundary_id) const
{
  (void)operator_type;

  if(this->boundary_ids.find(boundary_id) != this->boundary_ids.end())
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
	  integrator_m.submit_value(kernel.get_boundary_mass_value(scaling_factor, integrator_m.get_value(q)), q);
    }
  }
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::do_boundary_integral_continuous(IntegratorFace &                   integrator_m,
                     OperatorType const &               operator_type,
                     dealii::types::boundary_id const & boundary_id) const
{
  (void)operator_type;

  if(this->boundary_ids.find(boundary_id) != this->boundary_ids.end())
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
	  integrator_m.submit_value(kernel.get_boundary_mass_value(scaling_factor, integrator_m.get_value(q)), q);
    }
  }
}

template<int dim, int n_components, typename Number>
void
BoundaryMassOperator<dim, n_components, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                                  IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;
}

// scalar
template class BoundaryMassOperator<2, 1, float>;
template class BoundaryMassOperator<2, 1, double>;

template class BoundaryMassOperator<3, 1, float>;
template class BoundaryMassOperator<3, 1, double>;

// dim components
template class BoundaryMassOperator<2, 2, float>;
template class BoundaryMassOperator<2, 2, double>;

template class BoundaryMassOperator<3, 3, float>;
template class BoundaryMassOperator<3, 3, double>;

// dim + 1 components
template class BoundaryMassOperator<2, 3, float>;
template class BoundaryMassOperator<2, 3, double>;

template class BoundaryMassOperator<3, 4, float>;
template class BoundaryMassOperator<3, 4, double>;

} // namespace ExaDG
