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

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/material/library/incompressible_neo_hookean.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
IncompressibleNeoHookean<dim, Number>::IncompressibleNeoHookean(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  unsigned int const                        dof_index,
  unsigned int const                        quad_index,
  IncompressibleNeoHookeanData<dim> const & data,
  bool const                                spatial_integration,
  bool const                                force_material_residual,
  unsigned int const                        cache_level)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual),
    cache_level(cache_level)
{
  // initialize (potentially variable) shear modulus
  Number const shear_modulus = data.shear_modulus;
  shear_modulus_stored       = dealii::make_vectorized_array<Number>(shear_modulus);

  if(shear_modulus_is_variable)
  {
    // allocate vectors for variable coefficients and initialize with constant values
    shear_modulus_coefficients.initialize(matrix_free, quad_index, false, false);
    shear_modulus_coefficients.set_coefficients(shear_modulus);

    VectorType dummy;
    matrix_free.cell_loop(&IncompressibleNeoHookean<dim, Number>::cell_loop_set_coefficients,
                          this,
                          dummy,
                          dummy);
  }

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement vector.
  if(cache_level > 0)
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }
}

template<int dim, typename Number>
void
IncompressibleNeoHookean<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const &,
  Range const & cell_range) const
{
  IntegratorCell integrator(matrix_free, dof_index, quad_index);

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      dealii::VectorizedArray<Number> shear_modulus_vec =
        FunctionEvaluator<0, dim, Number>::value(*(data.shear_modulus_function),
                                                 integrator.quadrature_point(q),
                                                 0.0 /*time*/);

      // set the coefficients
      shear_modulus_coefficients.set_coefficient_cell(cell, q, shear_modulus_vec);
    }
  }
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q,
  bool const         force_evaluation) const
{
  (void)force_evaluation;

  tensor S;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  tensor const F = get_F<dim, Number>(gradient_displacement);
  scalar const J = determinant(F);
  tensor const C = transpose(F) * F;

  scalar const J_pow = pow(J, static_cast<Number>(-2.0 * one_third));

  S = invert(C) * (-shear_modulus_stored * J_pow * one_third * trace(C) +
                   data.bulk_modulus * 0.5 * (J * J - 1.0));
  add_scaled_identity(S, shear_modulus_stored * J_pow);

  return S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  tensor Dd_S;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  scalar const J   = determinant(deformation_gradient);
  tensor const C   = transpose(deformation_gradient) * deformation_gradient;
  scalar const I_1 = trace(C);

  scalar const J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
  tensor const F_inv = invert(deformation_gradient);
  tensor const C_inv = F_inv * transpose(F_inv);

  scalar const one_over_J_times_Dd_J = trace(F_inv * gradient_increment);
  scalar const Dd_I_1 = 2.0 * trace(transpose(gradient_increment) * deformation_gradient);
  tensor const Dd_F_inv_times_transpose_F_inv = -F_inv * gradient_increment * C_inv;
  tensor const Dd_C_inv =
    Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

  Dd_S = C_inv * (Dd_I_1 * (-shear_modulus_stored * one_third * J_pow) +
                  one_over_J_times_Dd_J *
                    (shear_modulus_stored * one_third * J_pow * 2.0 * one_third * I_1 +
                     data.bulk_modulus * J * J));
  Dd_S += Dd_C_inv * (-shear_modulus_stored * one_third * J_pow * I_1 +
                      data.bulk_modulus * 0.5 * (J * J - 1.0));
  add_scaled_identity(Dd_S,
                      -shear_modulus_stored * one_third * J_pow * 2.0 * one_over_J_times_Dd_J);

  return Dd_S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q,
                                                        bool const         force_evaluation) const
{
  (void)force_evaluation;

  tensor tau;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  tensor const F = get_F<dim, Number>(gradient_displacement);
  scalar const J = determinant(F);

  scalar const J_pow                = pow(J, static_cast<Number>(-2.0 * one_third));
  tensor const F_times_F_transposed = F * transpose(F);

  tau = F_times_F_transposed * (shear_modulus_stored * J_pow);
  add_scaled_identity(tau,
                      (-shear_modulus_stored * J_pow * one_third *
                         trace(F_times_F_transposed) /* = I_1 */
                       + data.bulk_modulus * 0.5 * (J * J - 1.0)));

  return tau;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::contract_with_J_times_C(
  tensor const &     symmetric_gradient_increment,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  tensor result;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  scalar const J     = determinant(deformation_gradient);
  scalar const J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
  tensor const C     = transpose(deformation_gradient) * deformation_gradient;
  scalar const I_1   = trace(C);

  result = symmetric_gradient_increment * (2.0 * one_third * shear_modulus_stored * J_pow * I_1 -
                                           data.bulk_modulus * (J * J - 1.0));

  scalar const factor = (-4.0 * one_third * shear_modulus_stored * J_pow *
                         scalar_product(C, symmetric_gradient_increment)) +
                        ((2.0 * one_third * one_third * shear_modulus_stored * I_1 * J_pow +
                          data.bulk_modulus * J * J) *
                         trace(symmetric_gradient_increment));

  add_scaled_identity(result, factor);

  return result;
}

template class IncompressibleNeoHookean<2, float>;
template class IncompressibleNeoHookean<2, double>;

template class IncompressibleNeoHookean<3, float>;
template class IncompressibleNeoHookean<3, double>;

} // namespace Structure
} // namespace ExaDG
