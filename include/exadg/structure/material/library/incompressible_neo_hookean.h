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

#ifndef STRUCTURE_MATERIAL_LIBRARY_INCOMPRESSIBLE_NEO_HOOKEAN
#define STRUCTURE_MATERIAL_LIBRARY_INCOMPRESSIBLE_NEO_HOOKEAN

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
struct IncompressibleNeoHookeanData : public MaterialData
{
  IncompressibleNeoHookeanData(
    MaterialType const &                         type,
    double const &                               shear_modulus,
    double const &                               bulk_modulus,
    Type2D const &                               type_two_dim,
    std::shared_ptr<dealii::Function<dim>> const shear_modulus_function = nullptr)
    : MaterialData(type),
      shear_modulus(shear_modulus),
      shear_modulus_function(shear_modulus_function),
      bulk_modulus(bulk_modulus),
      type_two_dim(type_two_dim)
  {
  }

  double                                 shear_modulus;
  std::shared_ptr<dealii::Function<dim>> shear_modulus_function;

  double bulk_modulus;
  Type2D type_two_dim;
};

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
class IncompressibleNeoHookean : public Material<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>              Range;
  typedef CellIntegrator<dim, dim, Number>                   IntegratorCell;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>          tensor;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetric_tensor;

  IncompressibleNeoHookean(dealii::MatrixFree<dim, Number> const &   matrix_free,
                           unsigned int const                        dof_index,
                           unsigned int const                        quad_index,
                           IncompressibleNeoHookeanData<dim> const & data,
                           bool const                                spatial_integration,
                           bool const                                force_material_residual);

  /*
   * The second Piola-Kirchhoff stress is defined as S = S_vol + S_iso (Flory split),
   * where we have strain energy density functions Psi_vol and Psi_iso defined as
   *
   * Psi_vol = bulk_modulus / 4 * ( J^2 - 1 - ln(J) )
   *
   * and
   *
   * Psi_iso = shear_modulus / 2 * ( I_1_bar - trace(I) )
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
   * I_1_bar = J^(-2/3) * I_1
   *
   * such that we end up with
   *
   * S_vol = bulk_modulus / 2 * (J^2 - 1) C^(-1)
   *
   * S_iso = J^(-2/3) * ( I - 1/3 * I_1 * C^(-1) )
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

  /*
   * Helper functions to replace frequently appearing terms.
   */
  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_c1(scalar const &           Jm1,
         scalar const &           J_pow,
         symmetric_tensor const & E,
         scalar const &           shear_modulus_stored,
         unsigned int const       cell,
         unsigned int const       q) const;

  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_c2(scalar const &           Jm1,
         scalar const &           J_pow,
         symmetric_tensor const & E,
         scalar const &           shear_modulus_stored,
         unsigned int const       cell,
         unsigned int const       q) const;

  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_J_pow(scalar const & Jm1, unsigned int const cell, unsigned int const q) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_S_stable(tensor const &           gradient_displacement,
                   symmetric_tensor const & C_inv,
                   scalar const &           J_pow,
                   scalar const &           Jm1,
                   scalar const &           shear_modulus) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_S_unstable(symmetric_tensor const & C_inv,
                     scalar const &           J_pow,
                     scalar const &           c1,
                     scalar const &           shear_modulus) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_tau_stable(tensor const & gradient_displacement,
                     scalar const & Jm1,
                     scalar const & J_pow,
                     scalar const & shear_modulus) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_tau_unstable(tensor const & F,
                       scalar const & J_pow,
                       scalar const & c1,
                       scalar const & shear_modulus) const;

  unsigned int dof_index;
  unsigned int quad_index;

  IncompressibleNeoHookeanData<dim> const & data;
  Number const                              bulk_modulus;

  mutable scalar shear_modulus_stored;

  // cache coefficients for spatially varying material parameters
  bool                                 shear_modulus_is_variable;
  mutable VariableCoefficients<scalar> shear_modulus_coefficients;

  // cache linearization data depending on cache_level and spatial_integration
  bool spatial_integration;
  bool force_material_residual;

  // required for nonlinear operator
  mutable VariableCoefficients<scalar> one_over_J_coefficients;
  mutable VariableCoefficients<tensor> gradient_displacement_coefficients;

  // scalar cache level
  mutable VariableCoefficients<scalar> Jm1_coefficients;
  mutable VariableCoefficients<scalar> J_pow_coefficients;
  mutable VariableCoefficients<scalar> c1_coefficients;
  mutable VariableCoefficients<scalar> c2_coefficients;

  // tensor cache level
  mutable VariableCoefficients<symmetric_tensor> kirchhoff_stress_coefficients;
  mutable VariableCoefficients<symmetric_tensor> C_coefficients;

  mutable VariableCoefficients<symmetric_tensor> second_piola_kirchhoff_stress_coefficients;
  mutable VariableCoefficients<tensor>           F_inv_coefficients;
  mutable VariableCoefficients<symmetric_tensor> C_inv_coefficients;
};
} // namespace Structure
} // namespace ExaDG

#endif /* STRUCTURE_MATERIAL_LIBRARY_INCOMPRESSIBLE_NEO_HOOKEAN */
