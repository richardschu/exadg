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

#include <exadg/structure/spatial_discretization/operators/boundary_conditions.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/spatial_discretization/operators/linear_operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
void
LinearOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    // Cauchy stresses, only valid for linear elasticity
    symmetric_tensor const sigma =
      material->second_piola_kirchhoff_stress(integrator.get_gradient(q),
                                              integrator.get_current_cell_index(),
                                              q);

    // test with gradients
    integrator.submit_gradient(sigma, q);

    if(this->operator_data.unsteady)
    {
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
    }
  }
}

template<int dim, typename Number>
void
LinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  unsigned int const                 face,
  dealii::types::boundary_id const & boundary_id) const
{
  (void)face;

  // Note that for the spatial integration approach, this is a quasi-Newton method
  // as the finite deformation of the body is ignored in the directional derivative.

  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  vector traction;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // reset traction for each q-point evaluation
    traction = 0.0;

    // integrate standard (stored) traction or exterior pressure on Robin boundaries
    if(boundary_type == BoundaryType::Neumann or boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::inhomogeneous or operator_type == OperatorType::full)
      {
        traction -= calculate_neumann_value<dim, Number>(
          q, integrator_m, boundary_type, boundary_id, this->operator_data.bc, this->time);
      }
    }

    // check boundary ID in `robin_bc` to add boundary mass integrals from Robin boundaries on
    // `BoundaryType::NeumannCached` or `BoundaryType::RobinSpringDashpotPressure`
    if(boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::homogeneous or operator_type == OperatorType::full)
      {
        auto const it = this->operator_data.bc->robin_bc.find(boundary_id);

        if(it != this->operator_data.bc->robin_bc.end())
        {
          RobinParameters const & robin_parameters = it->second;
          Number const displacement_coefficient_k  = robin_parameters.displacement_coefficient_k;

          if(robin_parameters.displacement_normal_projection)
          {
            vector const N = integrator_m.normal_vector(q);
            traction += N * (displacement_coefficient_k * (N * integrator_m.get_value(q)));
          }
          else
          {
            traction += displacement_coefficient_k * integrator_m.get_value(q);
          }

          if(this->operator_data.unsteady)
          {
            Number const velocity_coefficient_c = robin_parameters.velocity_coefficient_c;

            if(robin_parameters.velocity_normal_projection)
            {
              vector const N = integrator_m.normal_vector(q);
              traction += N * (velocity_coefficient_c * this->scaling_factor_mass_boundary *
                               (N * integrator_m.get_value(q)));
            }
            else
            {
              traction += velocity_coefficient_c * this->scaling_factor_mass_boundary *
                          integrator_m.get_value(q);
            }
          }
        }
      }
    }

    integrator_m.submit_value(traction, q);
  }
}

template class LinearOperator<2, float>;
template class LinearOperator<2, double>;

template class LinearOperator<3, float>;
template class LinearOperator<3, double>;

} // namespace Structure
} // namespace ExaDG
