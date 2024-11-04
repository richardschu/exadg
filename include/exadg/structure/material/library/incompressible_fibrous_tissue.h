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

#ifndef STRUCTURE_MATERIAL_LIBRARY_INCOMPRESSIBLE_FIBROUS_TISSUE
#define STRUCTURE_MATERIAL_LIBRARY_INCOMPRESSIBLE_FIBROUS_TISSUE

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
struct IncompressibleFibrousTissueData : public MaterialData
{
  typedef dealii::LinearAlgebra::distributed::Vector<float> VectorType;

  IncompressibleFibrousTissueData(
    MaterialType const &                           type,
    double const &                                 shear_modulus,
    double const &                                 bulk_modulus,
    double const &                                 fiber_angle_phi_in_degree,
    double const &                                 fiber_H_11,
    double const &                                 fiber_H_22,
    double const &                                 fiber_H_33,
    double const &                                 fiber_k_1,
    double const &                                 fiber_k_2,
    double const &                                 fiber_switch_limit,
    std::shared_ptr<std::vector<VectorType> const> e1_orientations,
    std::shared_ptr<std::vector<VectorType> const> e2_orientations,
    std::vector<unsigned int> const                degree_per_level,
    double const &                                 point_tolerance,
    Type2D const &                                 type_two_dim,
    std::shared_ptr<dealii::Function<dim>> const   shear_modulus_function = nullptr)
    : MaterialData(type),
      shear_modulus(shear_modulus),
      shear_modulus_function(shear_modulus_function),
      bulk_modulus(bulk_modulus),
      fiber_angle_phi_in_degree(fiber_angle_phi_in_degree),
      fiber_H_11(fiber_H_11),
      fiber_H_22(fiber_H_22),
      fiber_H_33(fiber_H_33),
      fiber_k_1(fiber_k_1),
      fiber_k_2(fiber_k_2),
      fiber_switch_limit(fiber_switch_limit),
      e1_orientations(e1_orientations),
      e2_orientations(e2_orientations),
      degree_per_level(degree_per_level),
      point_tolerance(point_tolerance),
      type_two_dim(type_two_dim)
  {
  }

  double                                 shear_modulus;
  std::shared_ptr<dealii::Function<dim>> shear_modulus_function;

  double bulk_modulus;

  double fiber_angle_phi_in_degree;
  double fiber_H_11;
  double fiber_H_22;
  double fiber_H_33;
  double fiber_k_1;
  double fiber_k_2;
  double fiber_switch_limit;

  std::shared_ptr<std::vector<VectorType> const> e1_orientations;
  std::shared_ptr<std::vector<VectorType> const> e2_orientations;
  std::vector<unsigned int> const                degree_per_level;
  double                                         point_tolerance;

  Type2D type_two_dim;
};

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
class IncompressibleFibrousTissue : public Material<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>              Range;
  typedef CellIntegrator<dim, dim, Number>                   IntegratorCell;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>          vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>          tensor;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetric_tensor;

  IncompressibleFibrousTissue(dealii::MatrixFree<dim, Number> const &      matrix_free,
                              unsigned int const                           dof_index,
                              unsigned int const                           quad_index,
                              IncompressibleFibrousTissueData<dim> const & data,
                              bool const                                   spatial_integration,
                              bool const                                   force_material_residual);

  /*
   * The second Piola-Kirchhoff stress is defined as S = S_vol + S_iso + S_c
   * (Flory split to the ground matrix, collagen contribution considers full
   * deformation gradient), where we have strain energy density functions
   * Psi_vol and Psi_iso defined as
   *
   * Psi_vol = bulk_modulus/4 * ( J^2 - 1 - ln(J) )
   *
   * Psi_iso = shear_modulus/2 * ( I_1_bar - trace(I) )
   *
   * Psi_c   = sum_(i=4,6) k_1/(2*k_2) * ( exp(k_2*E_i^2) - 1 ) if I_i > 1 ,
   *                       0 else.
   *
   * Here, we have the classic relations
   *
   * F = I + Grad(displacement) ,
   *
   * J = det(F) ,
   *
   * C = F^T * F ,
   *
   * I_1 = tr(C) ,
   *
   * I_1_bar = J^(-2/3) * I_1 ,
   *
   * I_i = (M_1 (x) M_1) : C ,
   *
   * with mean fiber direction M_1 for fiber family i. We compute a
   * strain-like measure for each fiber family i,
   *
   * E_i = H_i : (C - I) ,
   *
   * structure tensor
   *
   * H_i = sum_(j=1,2,3) H_jj * M_j (x) M_j ,
   *
   * for fiber family i, mean fiber direction M_1, tangential fiber
   * direction M_2 and orthogonal (no dispersion) fiber direction M_3.
   * The coefficients H_jj, j = 1,2,3, are derived from fiber dispersion
   * parameters a and b,
   *
   * H_11 = (1 - H_33)/2 * (1 + Bessel_1(a) / Bessel_0(a)) ,
   *
   * H_22 = (1 - H_33)/2 * (1 - Bessel_1(a) / Bessel_0(a)) ,
   *
   * H_33 = 1/(4*b) - exp(-2*b) / ( sqrt(2*pi*b) * erf(sqrt(2*b)) ) ,
   *
   * where Bessel_1() and Bessel_0() are the Bessel functions of first
   * kind of order 0 and 1, and erf() is the error function, such that
   * given the fiber parameters, Hjj, j = 1,2,3 can be computed once up
   * front. We thus end up with
   *
   * S_vol = bulk_modulus/2 * (J^2 - 1) C^(-1)
   *
   * S_iso = J^(-2/3) * ( I - 1/3 * I_1 * C^(-1) )
   *
   * S_c   = sum_(i=4,6) 2*k_1 * exp(k_2*E_i^2) * E_i * H_i if I_i > 1 ,
   *                     0 else.
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
         scalar const &           shear_modulus,
         unsigned int const       cell,
         unsigned int const       q) const;

  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_c2(scalar const &           Jm1,
         scalar const &           J_pow,
         symmetric_tensor const & E,
         scalar const &           shear_modulus,
         unsigned int const       cell,
         unsigned int const       q) const;

  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_c3(vector const &           M_1,
         symmetric_tensor const & E,
         scalar const &           E_i,
         unsigned int const       i,
         unsigned int const       cell,
         unsigned int const       q) const;

  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_J_pow(scalar const & Jm1, unsigned int const cell, unsigned int const q) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_structure_tensor(vector const & M_1, vector const & M_2) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  get_structure_tensor(vector const &     M_1,
                       unsigned int const i,
                       unsigned int const cell,
                       unsigned int const q) const;

  DEAL_II_ALWAYS_INLINE scalar
  compute_fiber_switch(vector const & M_1, symmetric_tensor const & E) const;

  template<bool force_evaluation>
  DEAL_II_ALWAYS_INLINE scalar
  get_E_i(symmetric_tensor const & H_i,
          symmetric_tensor const & E,
          unsigned int const       i,
          unsigned int const       cell,
          unsigned int const       q) const;

  DEAL_II_ALWAYS_INLINE Number
  compute_numerical_upper_bound(Number const & fiber_k_1) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_S_ground_matrix_stable(tensor const &           gradient_displacement,
                                 symmetric_tensor const & C_inv,
                                 scalar const &           J_pow,
                                 scalar const &           Jm1,
                                 scalar const &           shear_modulus) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_S_ground_matrix_unstable(symmetric_tensor const & C_inv,
                                   scalar const &           J_pow,
                                   scalar const &           c1,
                                   scalar const &           shear_modulus) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_S_fiber_i(scalar const & c3, scalar const & E_i, symmetric_tensor const & H_i) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_tau_stable(symmetric_tensor const & S_fiber,
                     tensor const &           F,
                     symmetric_tensor const & e,
                     scalar const &           Jm1,
                     scalar const &           J_pow,
                     scalar const &           shear_modulus) const;

  DEAL_II_ALWAYS_INLINE symmetric_tensor
  compute_tau_unstable(symmetric_tensor const & S_fiber,
                       tensor const &           F,
                       scalar const &           J_pow,
                       scalar const &           c1,
                       scalar const &           shear_modulus) const;

  unsigned int dof_index;
  unsigned int quad_index;

  IncompressibleFibrousTissueData<dim> const & data;
  Number const                                 bulk_modulus;
  Number const                                 fiber_k_1;
  Number const                                 fiber_k_2;
  Number const                                 fiber_angle_phi_in_degree;
  Number const                                 fiber_H_11;
  Number const                                 fiber_H_22;
  Number const                                 fiber_H_33;
  Number const                                 fiber_switch_limit;

  mutable scalar shear_modulus_stored;

  bool const orientation_vectors_provided;

  // cache coefficients for spatially varying material parameters
  bool                                 shear_modulus_is_variable;
  mutable VariableCoefficients<scalar> shear_modulus_coefficients;

  // cache linearization data depending on cache_level and spatial_integration
  bool spatial_integration;
  bool force_material_residual;

  // required for nonlinear operator
  mutable VariableCoefficients<scalar> one_over_J_coefficients;
  mutable VariableCoefficients<tensor> gradient_displacement_coefficients;

  mutable std::vector<VariableCoefficients<vector>> fiber_direction_M_1;
  mutable std::vector<VariableCoefficients<vector>> fiber_direction_M_2;

  std::shared_ptr<VectorType> e1_orientation;
  std::shared_ptr<VectorType> e2_orientation;

  // scalar cache level
  mutable VariableCoefficients<scalar> Jm1_coefficients;
  mutable VariableCoefficients<scalar> J_pow_coefficients;
  mutable VariableCoefficients<scalar> c1_coefficients;
  mutable VariableCoefficients<scalar> c2_coefficients;

  mutable std::vector<VariableCoefficients<scalar>> c3_coefficients;
  mutable std::vector<VariableCoefficients<scalar>> E_i_coefficients;

  // tensor cache level
  mutable VariableCoefficients<symmetric_tensor> kirchhoff_stress_coefficients;
  mutable VariableCoefficients<symmetric_tensor> C_coefficients;

  mutable VariableCoefficients<symmetric_tensor> second_piola_kirchhoff_stress_coefficients;
  mutable VariableCoefficients<tensor>           F_inv_coefficients;
  mutable VariableCoefficients<symmetric_tensor> C_inv_coefficients;

  mutable std::vector<VariableCoefficients<symmetric_tensor>> fiber_structure_tensor;
  mutable std::vector<VariableCoefficients<symmetric_tensor>> F_times_H_i_times_FT_coefficients;
};
} // namespace Structure
} // namespace ExaDG

#endif /* STRUCTURE_MATERIAL_LIBRARY_INCOMPRESSIBLE_FIBROUS_TISSUE */
