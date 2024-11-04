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

#ifndef STRUCTURE_MATERIAL_LIBRARY_COMPRESSIBLE_NEO_HOOKEAN
#define STRUCTURE_MATERIAL_LIBRARY_COMPRESSIBLE_NEO_HOOKEAN

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/structure/material/material.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct CompressibleNeoHookeanData : public MaterialData
{
  CompressibleNeoHookeanData(
    MaterialType const &                         type,
    double const &                               shear_modulus,
    double const &                               lambda,
    Type2D const &                               type_two_dim,
    std::shared_ptr<dealii::Function<dim>> const shear_modulus_function = nullptr,
    std::shared_ptr<dealii::Function<dim>> const lambda_function        = nullptr)
    : MaterialData(type),
      shear_modulus(shear_modulus),
      shear_modulus_function(shear_modulus_function),
      lambda(lambda),
      lambda_function(lambda_function),
      type_two_dim(type_two_dim)
  {
  }

  double                                 shear_modulus;
  std::shared_ptr<dealii::Function<dim>> shear_modulus_function;

  double                                 lambda; // first Lamee parameter, often called lambda
  std::shared_ptr<dealii::Function<dim>> lambda_function;

  Type2D type_two_dim;
};

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
class CompressibleNeoHookean : public Material<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>              Range;
  typedef CellIntegrator<dim, dim, Number>                   IntegratorCell;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>          tensor;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetric_tensor;

  CompressibleNeoHookean(dealii::MatrixFree<dim, Number> const & matrix_free,
                         unsigned int const                      dof_index,
                         unsigned int const                      quad_index,
                         CompressibleNeoHookeanData<dim> const & data,
                         bool const                              spatial_integration,
                         bool const                              force_material_residual);

  /*
   * The second Piola-Kirchhoff stress is defined as S = 2 d/dC (Psi),
   * where we have strain energy density function Psi defined as
   *
   * Psi = shear_modulus / 2 * ( I_1 - trace(I) - 2 * ln(J) )
   *       + lambda * ln(J)^2
   *
   * with the classic relations
   *
   * F = I + Grad(displacement) ,
   *
   * J = det(F) ,
   *
   * C = F^T * F ,
   *
   * I_1 = tr(C) ,
   *
   * such that we end up with
   *
   * S = shear modulus * I - ( shear_modulus - 2.0 * lambda * ln(J) ) * C^(-1)
   *
   */
  symmetric_tensor
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const final;

  symmetric_tensor
  second_piola_kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                     unsigned int const cell,
                                     unsigned int const q) const final;

  symmetric_tensor
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const final;

  symmetric_tensor
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const final;

  symmetric_tensor
  kirchhoff_stress(tensor const &     gradient_displacement,
                   unsigned int const cell,
                   unsigned int const q) const final;

  symmetric_tensor
  kirchhoff_stress_eval(tensor const &     gradient_displacement,
                        unsigned int const cell,
                        unsigned int const q) const final;

  symmetric_tensor
  kirchhoff_stress(unsigned int const cell, unsigned int const q) const final;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          tensor const &           gradient_displacement,
                          unsigned int const       cell,
                          unsigned int const       q) const final;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          unsigned int const       cell,
                          unsigned int const       q) const final;

  /*
   * Store linearization data depending on cache level.
   */
  void
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const final;

  scalar
  one_over_J(unsigned int const cell, unsigned int const q) const final;

  tensor
  gradient_displacement(unsigned int const cell, unsigned int const q) const final;

private:
  /*
   * Store factors involving (potentially variable) shear modulus.
   */
  void
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const;

  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_log_J(scalar const & Jm1, unsigned int const cell, unsigned int const q) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_S_stable(tensor const &           gradient_displacement,
                   symmetric_tensor const & C_inv,
                   scalar const &           log_J,
                   scalar const &           shear_modulus,
                   scalar const &           lambda) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_S_unstable(symmetric_tensor const & C_inv,
                     scalar const &           log_J,
                     scalar const &           shear_modulus,
                     scalar const &           lambda) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_tau_stable(tensor const & gradient_displacement,
                     scalar const & log_J,
                     scalar const & shear_modulus,
                     scalar const & lambda) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_tau_unstable(tensor const & F,
                       scalar const & log_J,
                       scalar const & shear_modulus,
                       scalar const & lambda) const;

  unsigned int dof_index;
  unsigned int quad_index;

  CompressibleNeoHookeanData<dim> const & data;

  mutable scalar shear_modulus_stored;
  mutable scalar lambda_stored;

  // cache coefficients for spatially varying material parameters
  bool                                 parameters_are_variable;
  mutable VariableCoefficients<scalar> shear_modulus_coefficients;
  mutable VariableCoefficients<scalar> lambda_coefficients;

  // cache linearization data depending on cache_level and spatial_integration
  bool spatial_integration;
  bool force_material_residual;

  // required for nonlinear operator
  mutable VariableCoefficients<scalar> one_over_J_coefficients;
  mutable VariableCoefficients<tensor> gradient_displacement_coefficients;

  // scalar cache level
  mutable VariableCoefficients<scalar> log_J_coefficients;

  // tensor cache level
  mutable VariableCoefficients<symmetric_tensor> kirchhoff_stress_coefficients;
  mutable VariableCoefficients<symmetric_tensor> second_piola_kirchhoff_stress_coefficients;
  mutable VariableCoefficients<tensor>           F_inv_coefficients;
  mutable VariableCoefficients<symmetric_tensor> C_inv_coefficients;
};
} // namespace Structure
} // namespace ExaDG

#endif /* STRUCTURE_MATERIAL_LIBRARY_COMPRESSIBLE_NEO_HOOKEAN */
