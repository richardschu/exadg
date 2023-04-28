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
template<int dim, typename Number, int n_components>
BoundaryMassOperator<dim, Number, n_components>::BoundaryMassOperator() : matrix_free(nullptr), scaling_factor(1.0)
{
}

template<int dim, typename Number, int n_components>
bool
BoundaryMassOperator<dim, Number, n_components>::non_empty() const
{
  return this->ids_normal_coefficients.size() > 0;
}

template<int dim, typename Number, int n_components>
IntegratorFlags
BoundaryMassOperator<dim, Number, n_components>::get_integrator_flags() const
{
  return kernel.get_integrator_flags();
}

template<int dim, typename Number, int n_components>
MappingFlags
BoundaryMassOperator<dim, Number, n_components>::get_mapping_flags()
{
  return BoundaryMassKernel<dim,Number>::get_mapping_flags();
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free_in,
  dealii::AffineConstraints<Number> const & affine_constraints,
  BoundaryMassOperatorData<dim, Number> const &     data)
{
  Base::reinit(matrix_free_in, affine_constraints, data);

  this->integrator_flags        = this->get_integrator_flags();
  this->ids_normal_coefficients = data.ids_normal_coefficients;
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::set_scaling_factor(Number const & number) const
{
  scaling_factor = number;
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::set_ids_normal_coefficients(
  std::map<dealii::types::boundary_id, std::pair<bool, Number>> const & ids_normal_coefficients_in) const
{
  this->ids_normal_coefficients = ids_normal_coefficients_in;
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::do_cell_integral(IntegratorCell & integrator) const
{
  (void)integrator;

  std::cout << "new cell integral \n";

  // do nothing
  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    integrator.begin_dof_values()[i] = dealii::make_vectorized_array<Number>(0.);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    integrator.submit_value(0.0 * integrator.get_value(q), q);

}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::do_face_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::do_boundary_integral(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  (void)operator_type;

  if(auto it{this->ids_normal_coefficients.find(boundary_id)}; it != this->ids_normal_coefficients.end())
  {
	Number scaled_coefficient = it->second.second * scaling_factor;
    bool normal_projection = it->second.first;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      if(normal_projection)
        integrator_m.submit_value(kernel.get_boundary_mass_normal_value(scaled_coefficient,
    		                                                            integrator_m.get_normal_vector(q),
                                                                        integrator_m.get_value(q)),
                                  q);
      else
        integrator_m.submit_value(kernel.get_boundary_mass_value(scaled_coefficient,
      		                                                     integrator_m.get_value(q)),
                                  q);
    }
  }
  else
  {
	for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
	{
	  integrator_m.submit_value(0.0 * integrator_m.get_value(q),q);
	}
  }
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  this->do_boundary_integral(integrator_m, operator_type, boundary_id);
}

template class BoundaryMassOperator<2, float, 2>;
template class BoundaryMassOperator<2, double, 2>;

template class BoundaryMassOperator<3, float, 3>;
template class BoundaryMassOperator<3, double, 3>;
} // namespace ExaDG
