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

#define N_FIBER_FAMILIES 2

// deal.II
#include <deal.II/base/vectorization.h>
#include <deal.II/fe/mapping_fe.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/material/library/incompressible_fibrous_tissue.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

// ExaDG-Bio
#define LINK_TO_EXADGBIO
#ifdef LINK_TO_EXADGBIO
#  include "../../../../../../exadg-bio/include/match_cell_data.h"
#endif

namespace ExaDG
{
namespace Structure
{
template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  IncompressibleFibrousTissue(dealii::MatrixFree<dim, Number> const &      matrix_free,
                              unsigned int const                           dof_index,
                              unsigned int const                           quad_index,
                              IncompressibleFibrousTissueData<dim> const & data,
                              bool const                                   spatial_integration,
                              bool const                                   force_material_residual)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    bulk_modulus(data.bulk_modulus),
    fiber_k_1(data.fiber_k_1),
    fiber_k_2(data.fiber_k_2),
    fiber_angle_phi_in_degree(data.fiber_angle_phi_in_degree),
    fiber_H_11(data.fiber_H_11),
    fiber_H_22(data.fiber_H_22),
    fiber_H_33(data.fiber_H_33),
    fiber_switch_limit(data.fiber_switch_limit),
    orientation_vectors_provided(data.e1_orientations != nullptr or
                                 data.e2_orientations != nullptr),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual)
{
  // initialize (potentially variable) shear modulus
  Number const shear_modulus = data.shear_modulus;
  shear_modulus_stored       = dealii::make_vectorized_array<Number>(shear_modulus);

  if(shear_modulus_is_variable)
  {
    // Allocate vectors for variable coefficients and initialize with constant values.
    shear_modulus_coefficients.initialize(matrix_free, quad_index, false, false);
    shear_modulus_coefficients.set_coefficients(shear_modulus);
  }

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement
  // vector if possible.
  if constexpr(cache_level > 0)
  {
    Jm1_coefficients.initialize(matrix_free, quad_index, false, false);
    Jm1_coefficients.set_coefficients(0.0);

    J_pow_coefficients.initialize(matrix_free, quad_index, false, false);
    J_pow_coefficients.set_coefficients(1.0);

    c1_coefficients.initialize(matrix_free, quad_index, false, false);
    c1_coefficients.set_coefficients(-shear_modulus * ONE_THIRD * static_cast<Number>(dim));
    c2_coefficients.initialize(matrix_free, quad_index, false, false);
    c2_coefficients.set_coefficients(shear_modulus * TWO_NINTHS * static_cast<Number>(dim) +
                                     bulk_modulus);

    // Set scalar linearization data for fiber contribution initially.
    E_i_coefficients.resize(N_FIBER_FAMILIES);
    c3_coefficients.resize(N_FIBER_FAMILIES);
    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      E_i_coefficients[i].initialize(matrix_free, quad_index, false, false);
      E_i_coefficients[i].set_coefficients(0.0);
      c3_coefficients[i].initialize(matrix_free, quad_index, false, false);
      c3_coefficients[i].set_coefficients(2.0 * fiber_k_1);
    }

    if(spatial_integration)
    {
      one_over_J_coefficients.initialize(matrix_free, quad_index, false, false);
      one_over_J_coefficients.set_coefficients(1.0);
    }

    if constexpr(cache_level > 1)
    {
      if(spatial_integration)
      {
        kirchhoff_stress_coefficients.initialize(matrix_free, quad_index, false, false);
        kirchhoff_stress_coefficients.set_coefficients(get_zero_symmetric_tensor<dim, Number>());

        C_coefficients.initialize(matrix_free, quad_index, false, false);
        C_coefficients.set_coefficients(get_identity_symmetric_tensor<dim, Number>());

        // Set tensorial linearization data for fiber contribution initially.
        // Note that these have to be corrected in cell_loop_set_coefficients().
        F_times_H_i_times_FT_coefficients.resize(N_FIBER_FAMILIES);
        for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
        {
          F_times_H_i_times_FT_coefficients[i].initialize(matrix_free, quad_index, false, false);
          F_times_H_i_times_FT_coefficients[i].set_coefficients(
            get_zero_symmetric_tensor<dim, Number>());
        }

        if(force_material_residual)
        {
          gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
          gradient_displacement_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

          second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                                quad_index,
                                                                false,
                                                                false);
          second_piola_kirchhoff_stress_coefficients.set_coefficients(
            get_zero_symmetric_tensor<dim, Number>());
        }
      }
      else
      {
        gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
        gradient_displacement_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

        F_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        F_inv_coefficients.set_coefficients(get_identity_tensor<dim, Number>());

        C_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        C_inv_coefficients.set_coefficients(get_identity_symmetric_tensor<dim, Number>());

        second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                              quad_index,
                                                              false,
                                                              false);
        second_piola_kirchhoff_stress_coefficients.set_coefficients(
          get_zero_symmetric_tensor<dim, Number>());
      }

      AssertThrow(cache_level < 3, dealii::ExcMessage("Cache level > 2 not implemented."));
    }
  }

  // The vectors created or imported from the binary files in the application
  // increase in h-level and then in p-level (ph-Multigrid expected here).
  if(orientation_vectors_provided)
  {
    AssertThrow(data.e1_orientations != nullptr and data.e2_orientations != nullptr,
                dealii::ExcMessage("Provide orientation vectors for both e1 and e2."));

    AssertThrow(data.e1_orientations->size() > 0,
                dealii::ExcMessage("Provide orientation vectors or `nullptr`."));

    AssertThrow(data.e1_orientations->size() == data.e2_orientations->size(),
                dealii::ExcMessage("Provide orientation vectors for all levels for e1 and e2."));

    for(unsigned int i = 0; i < data.e1_orientations->size(); ++i)
    {
      AssertThrow((*data.e1_orientations)[i].size() == (*data.e2_orientations)[i].size(),
                  dealii::ExcMessage(
                    "Provide e1 and e2 orientation vectors of equal size for each level."));
    }

    AssertThrow(data.e1_orientations->size() == data.degree_per_level.size(),
                dealii::ExcMessage("Provide degree for all levels for e1 and e2."));

    typedef typename IncompressibleFibrousTissueData<dim>::VectorType VectorTypeOrientation;

    e1_orientation = std::make_shared<VectorType>();
    e2_orientation = std::make_shared<VectorType>();
    matrix_free.initialize_dof_vector(*e1_orientation, dof_index);
    matrix_free.initialize_dof_vector(*e2_orientation, dof_index);

#ifdef LINK_TO_EXADGBIO
    // Read the suitable vector from binary format.
    dealii::DoFHandler<dim> const & dof_handler = matrix_free.get_dof_handler(dof_index);
    unsigned int const              degree      = dof_handler.get_fe().base_element(0).degree;
    MPI_Comm const &                mpi_comm    = dof_handler.get_communicator();
    dealii::ConditionalOStream      pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    // Match the initialized vector with the given vectors.
    bool found_match = false;
    for(unsigned int i = 0; i < data.e1_orientations->size(); ++i)
    {
      if(e1_orientation->size() == (*data.e1_orientations)[i].size() and
         degree == data.degree_per_level[i])
      {
        pcout << "Filling vector of size " << e1_orientation->size() << " (degree = " << degree
              << ").\n";
        found_match = true;
        e1_orientation->copy_locally_owned_data_from((*data.e1_orientations)[i]);
        e2_orientation->copy_locally_owned_data_from((*data.e2_orientations)[i]);
      }
    }

    if(not found_match)
    {
      (*e1_orientation) = 0.0;
      e1_orientation->add(1.0);
      (*e1_orientation) = 0.0;
      e1_orientation->add(1.0);

      pcout << "Overwritten orientation vector of size " << e1_orientation->size()
            << " with dummy data.\n\n\n";
    }
    else
    {
      pcout << "|E1|_2 = " << e1_orientation->l2_norm() << "\n"
            << "|E2|_2 = " << e2_orientation->l2_norm() << "\n\n";
    }
#else
    AssertThrow(not orientation_vectors_provided,
                dealii::ExcMessage(
                  "You must link against ExaDG-Bio to enable user-defined material orientations."));
#endif

    e1_orientation->update_ghost_values();
    e2_orientation->update_ghost_values();
  }

  // Set the coefficients on the integration point level.
  VectorType dummy;
  matrix_free.cell_loop(
    &IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
      cell_loop_set_coefficients,
    this,
    dummy,
    dummy);

  // Release vectors after initialization, since cell data is stored.
  if(orientation_vectors_provided)
  {
    e1_orientation.reset();
    e2_orientation.reset();
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
void
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const &,
                             Range const & cell_range) const
{
  IntegratorCell integrator(matrix_free, dof_index, quad_index);
  IntegratorCell integrator_e1(matrix_free, dof_index, quad_index);
  IntegratorCell integrator_e2(matrix_free, dof_index, quad_index);

  // The material coordinate system is initialized constant in the entire domain.
  // Phi is the angle from the circumferential vector E_1 towards the longitudinal
  // vector E_2.
  vector                 M_3;
  std::vector<vector>    M_1(N_FIBER_FAMILIES), M_2(N_FIBER_FAMILIES);
  dealii::Tensor<1, dim> E_1_default, E_2_default;
  for(unsigned int d = 0; d < dim; d++)
  {
    Number const reciprocal_norm = 1.0 / std::sqrt(static_cast<Number>(dim));
    E_1_default[d]               = reciprocal_norm;
    E_2_default[d]               = reciprocal_norm;
  }

  AssertThrow(N_FIBER_FAMILIES <= 2,
              dealii::ExcMessage(
                "Using more than two fiber families, the mean directions overlap."));
  std::vector<Number> fiber_sin_phi(N_FIBER_FAMILIES);
  std::vector<Number> fiber_cos_phi(N_FIBER_FAMILIES);
  {
    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      fiber_sin_phi[i] =
        std::sin(pow(-1.0, i + 1) * fiber_angle_phi_in_degree * dealii::numbers::PI / 180.0);
      fiber_cos_phi[i] =
        std::cos(pow(-1.0, i + 1) * fiber_angle_phi_in_degree * dealii::numbers::PI / 180.0);

      M_1[i] = fiber_cos_phi[i] * E_1_default - fiber_sin_phi[i] * E_2_default;
      M_2[i] = fiber_sin_phi[i] * E_1_default + fiber_cos_phi[i] * E_2_default;
    }

    dealii::Tensor<1, dim> E_3 = cross_product_3d(E_1_default, E_2_default);
    M_3                        = E_3;
  }

  // Store only the minimal amount for the general case of having a field of
  // material coordinate systems. The mean fiber direction is always needed, since
  // I_i_star(M_1) is used as a fiber switch.
  fiber_direction_M_1.resize(N_FIBER_FAMILIES);

  for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
  {
    fiber_direction_M_1[i].initialize(matrix_free, quad_index, false, false);
  }

  if constexpr(cache_level < 2)
  {
    // The structure tensor is reconstructed from M1 and M2 on the fly.
    fiber_direction_M_2.resize(N_FIBER_FAMILIES);
    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      fiber_direction_M_2[i].initialize(matrix_free, quad_index, false, false);
    }
  }
  else
  {
    // The structure tensors are stored for each fiber family.
    fiber_structure_tensor.resize(N_FIBER_FAMILIES);
    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      fiber_structure_tensor[i].initialize(matrix_free, quad_index, false, false);
    }
  }

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    if(orientation_vectors_provided)
    {
      integrator_e1.reinit(cell);
      integrator_e1.read_dof_values(*e1_orientation);
      integrator_e1.evaluate(dealii::EvaluationFlags::values);
      integrator_e2.reinit(cell);
      integrator_e2.read_dof_values(*e2_orientation);
      integrator_e2.evaluate(dealii::EvaluationFlags::values);
    }

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // set shear modulus coefficients.
      if(shear_modulus_is_variable)
      {
        scalar shear_modulus_vec =
          FunctionEvaluator<0, dim, Number>::value(*(data.shear_modulus_function),
                                                   integrator.quadrature_point(q),
                                                   0.0 /*time*/);
        shear_modulus_coefficients.set_coefficient_cell(cell, q, shear_modulus_vec);
      }

      // Update the fiber mean directions.
      if(orientation_vectors_provided)
      {
        vector E_1 = integrator_e1.get_value(q);
        vector E_2 = integrator_e2.get_value(q);

        // Guard a potential zero norm, since we want to allow for
        // non-normed orientation vectors given by the user.
        Number constexpr tol = 1e-10;
        scalar E_1_norm      = E_1.norm();
        scalar E_2_norm      = E_2.norm();

        // factor_default = norm < tol ? 1.0 : 0.0
        scalar E_1_default_factor =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            E_1_norm,
            dealii::make_vectorized_array<Number>(static_cast<Number>(tol)),
            dealii::make_vectorized_array<Number>(static_cast<Number>(1.0)),
            dealii::make_vectorized_array<Number>(static_cast<Number>(0.0)));

        scalar E_2_default_factor =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            E_2_norm,
            dealii::make_vectorized_array<Number>(static_cast<Number>(tol)),
            dealii::make_vectorized_array<Number>(static_cast<Number>(1.0)),
            dealii::make_vectorized_array<Number>(static_cast<Number>(0.0)));

        // Add tol to norm to make sure we do not get undefined behavior.
        E_1 =
          E_1_default_factor * E_1_default +
          ((dealii::make_vectorized_array<Number>(static_cast<Number>(1.0)) - E_1_default_factor) /
           (E_1_norm + tol)) *
            E_1;
        E_2 =
          E_2_default_factor * E_2_default +
          ((dealii::make_vectorized_array<Number>(static_cast<Number>(1.0)) - E_2_default_factor) /
           (E_2_norm + tol)) *
            E_2;

        for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
        {
          M_1[i] = fiber_cos_phi[i] * E_1 - fiber_sin_phi[i] * E_2;
          M_2[i] = fiber_sin_phi[i] * E_1 + fiber_cos_phi[i] * E_2;
        }
      }
      else
      {
        // The mean fiber directions are never updated after initialization
        // with the default E1 and E2 vectors.
      }

      // Fill the fiber orientation vectors or the structure tensor directly
      // based on the Euclidean material coordinate system and correct all
      // corresponding stored data initialized wrongly before.
      if constexpr(cache_level < 2)
      {
        for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
        {
          fiber_direction_M_1[i].set_coefficient_cell(cell, q, M_1[i]);
          fiber_direction_M_2[i].set_coefficient_cell(cell, q, M_2[i]);
        }
      }
      else
      {
        for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
        {
          fiber_direction_M_1[i].set_coefficient_cell(cell, q, M_1[i]);

          symmetric_tensor const H_i = compute_structure_tensor(M_1[i], M_2[i]);
          fiber_structure_tensor[i].set_coefficient_cell(cell, q, H_i);

          // Update the products with the actual structure tensor.
          if(spatial_integration)
          {
            F_times_H_i_times_FT_coefficients[i].set_coefficient_cell(cell,
                                                                      q,
                                                                      /* I * */ H_i /* * I^T */);
          }
        }
      }
    }
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::get_c1(
  scalar const &           Jm1,
  scalar const &           J_pow,
  symmetric_tensor const & E,
  scalar const &           shear_modulus,
  unsigned int const       cell,
  unsigned int const       q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    return ((0.5 * bulk_modulus) * compute_JJm1<Number, stable_formulation>(Jm1) -
            shear_modulus * ONE_THIRD * J_pow * compute_I_1<dim, Number, stable_formulation>(E));
  }
  else
  {
    return c1_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::get_c2(
  scalar const &           Jm1,
  scalar const &           J_pow,
  symmetric_tensor const & E,
  scalar const &           shear_modulus,
  unsigned int const       cell,
  unsigned int const       q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    return (bulk_modulus * (compute_JJm1<Number, stable_formulation>(Jm1) + 1.0) +
            TWO_NINTHS * shear_modulus * J_pow * compute_I_1<dim, Number, stable_formulation>(E));
  }
  else
  {
    return c2_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::get_c3(
  vector const &           M_1,
  symmetric_tensor const & E,
  scalar const &           E_i,
  unsigned int const       i,
  unsigned int const       cell,
  unsigned int const       q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    scalar const fiber_switch = compute_fiber_switch(M_1, E);

    // Enforce an upper bound for the computed value.
    return (fiber_switch * (2.0 * fiber_k_1) *
            exp_limited<Number>(fiber_k_2 * E_i * E_i, compute_numerical_upper_bound(fiber_k_1)));
  }
  else
  {
    return c3_coefficients[i].get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::get_J_pow(
  scalar const &     Jm1,
  unsigned int const cell,
  unsigned int const q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    if constexpr(cache_level == 0)
    {
      // Compute the inverse third root of J^2 via Newton's method:
      //
      // f(x) = x - J^(-2/3) = 0
      // or equivalently
      // f(x) = x^(-3) - J^2 = 0
      // with
      // f'(x) = - 3 x^(-4)
      // and hence the Newton update
      // x_np1 = 4/3 * x_n - J^2/3 * x_n^4.
      //
      // J^(-2/3) = 1 is a good initial guess, since we enforce J = 1
      // implicitly via the constitutive model. We hard-code 3 steps,
      // which were in most tests enough, but might be risky and hence
      // not pay off enough for cache_level > 1 since we are storing
      // related variables anyways.

      scalar const J_sqrd_over_three = ((Jm1 + 2.0) * Jm1 + 1.0) * ONE_THIRD;

      // The first iteration simplifies due to the initial guess x_0 = 1.0,
      // but a one-term Taylor series expansion is more accurate in tests.
      // scalar J_pow = dealii::make_vectorized_array(static_cast<Number>(FOUR_THIRDS));
      // J_pow -= J_sqrd_over_three;
      scalar J_pow = dealii::make_vectorized_array(static_cast<Number>(1.0));
      J_pow -= TWO_THIRDS * Jm1;

      // The remaining Newton iterations are identical.
      scalar J_pow_to_the_fourth = J_pow * J_pow;
      J_pow_to_the_fourth *= J_pow_to_the_fourth;
      J_pow = FOUR_THIRDS * J_pow - J_sqrd_over_three * J_pow_to_the_fourth;

      J_pow_to_the_fourth = J_pow * J_pow;
      J_pow_to_the_fourth *= J_pow_to_the_fourth;
      J_pow = FOUR_THIRDS * J_pow - J_sqrd_over_three * J_pow_to_the_fourth;

      return J_pow;
    }
    else
    {
      return fast_approx_powp1<Number, stable_formulation>(Jm1, static_cast<Number>(-TWO_THIRDS));
    }
  }
  else
  {
    return J_pow_coefficients.get_coefficient_cell(cell, q);
  }
}

// Load the structure tensor (constituents) depending on the cache_level.
// Note that here, we *never* enforce evaluation compared to comparable
// member functions, since the structure tensor or its constituents
// are considered material parameters.
template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_structure_tensor(vector const & M_1, vector const & M_2) const
{
  vector const M_3 = cross_product_3d(M_1, M_2);

  // The structure tensor is given as:
  // H = H_11 * M1 (x) M1 + H_22 * M2 (x) M2 + H_22 * M2 (x) M2
  // Note that j >= i, since we operate on a `symmetric_tensor`.
  symmetric_tensor result;
  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = i; j < dim; ++j)
    {
      result[i][j] =
        fiber_H_11 * M_1[i] * M_1[j] + fiber_H_22 * M_2[i] * M_2[j] + fiber_H_33 * M_3[i] * M_3[j];
    }
  }

  return result;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  get_structure_tensor(vector const & M_1, unsigned int i, unsigned int cell, unsigned int q) const
{
  if constexpr(cache_level < 2)
  {
    vector const M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
    return compute_structure_tensor(M_1, M_2);
  }
  else
  {
    (void)M_1;
    return fiber_structure_tensor[i].get_coefficient_cell(cell, q);
  }
}

// Function to evaluate the fiber_switch, not using/loading any data.
template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_fiber_switch(vector const & M_1, symmetric_tensor const & E) const
{
  if constexpr(stable_formulation)
  {
    // I_i_star = 2 * (M_1 (x) M_1) : E + tr(M_1 (x) M_1)
    // clang-format off
    // Unrolled diagonal part:
    scalar I_i_star = M_1[0] * M_1[0] * (2.0 * E[0][0] + 1.0);
    I_i_star       += M_1[1] * M_1[1] * (2.0 * E[1][1] + 1.0);
    if constexpr(dim == 3)
    {
      I_i_star     += M_1[2] * M_1[2] * (2.0 * E[2][2] + 1.0);
    }

    // Unrolled off-diagonal part, use symmetry:
    I_i_star   += 4.0 * M_1[0] * M_1[1] * E[0][1];
    if constexpr(dim == 3)
    {
      I_i_star += 4.0 * M_1[0] * M_1[2] * E[0][2];
      I_i_star += 4.0 * M_1[1] * M_1[2] * E[1][2];
    }
    // clang-format on

    // fiber_switch = I_i_star < fiber_switch_limit ? 0.0 : 1.0
    scalar const fiber_switch = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      I_i_star,
      dealii::make_vectorized_array<Number>(fiber_switch_limit),
      dealii::make_vectorized_array<Number>(static_cast<Number>(0.0)),
      dealii::make_vectorized_array<Number>(static_cast<Number>(1.0)));

    return fiber_switch;
  }
  else
  {
    // I_i_star = (M_1 (x) M_1) : C
    symmetric_tensor C = 2.0 * E;
    add_scaled_identity(C, static_cast<Number>(1.0));

    // clang-format off
    // Unrolled diagonal part:
    scalar I_i_star = M_1[0] * M_1[0] * C[0][0];
    I_i_star       += M_1[1] * M_1[1] * C[1][1];
    if constexpr(dim == 3)
    {
      I_i_star     += M_1[2] * M_1[2] * C[2][2];
    }

    // Unrolled off-diagonal part, use symmetry:
    I_i_star   += 2.0 * M_1[0] * M_1[1] * C[0][1];
    if constexpr(dim == 3)
    {
      I_i_star += 2.0 * M_1[0] * M_1[2] * C[0][2];
      I_i_star += 2.0 * M_1[1] * M_1[2] * C[1][2];
    }
    // clang-format on

    // fiber_switch = I_i_star < fiber_switch_limit ? 0.0 : 1.0
    scalar const fiber_switch = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      I_i_star,
      dealii::make_vectorized_array<Number>(fiber_switch_limit),
      dealii::make_vectorized_array<Number>(static_cast<Number>(0.0)),
      dealii::make_vectorized_array<Number>(static_cast<Number>(1.0)));

    return fiber_switch;
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::get_E_i(
  symmetric_tensor const & H_i,
  symmetric_tensor const & E,
  unsigned int const       i,
  unsigned int const       cell,
  unsigned int const       q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    if constexpr(stable_formulation)
    {
      return (2.0 * scalar_product(H_i, E));
    }
    else
    {
      // Note that we add and subtract I to simulate computing C - I.
      symmetric_tensor C = 2.0 * E;
      add_scaled_identity(C, static_cast<Number>(1.0));
      return scalar_product(H_i, C - get_identity_symmetric_tensor<dim, Number>());
    }
  }
  else
  {
    return E_i_coefficients[i].get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline Number
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_numerical_upper_bound(Number const & fiber_k_1) const
{
  return (std::max(static_cast<Number>(1e10), fiber_k_1 * fiber_k_1 * fiber_k_1));
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
void
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const
{
  AssertThrow(cache_level > 0 and cache_level < 3,
              dealii::ExcMessage("0 < cache level < 3 expected."));

  for(unsigned int q = 0; q < integrator_lin->n_q_points; ++q)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    tensor Grad_d_lin = integrator_lin->get_gradient(q);
    auto [F, Jm1] = compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(Grad_d_lin);

    // Overwrite computed values with admissible stored ones
    if constexpr(check_type == 2)
    {
      tensor F_old = gradient_displacement_coefficients.get_coefficient_cell(cell, q);
      add_scaled_identity(F_old, static_cast<Number>(1.0));

      bool update_J = false;
      for(unsigned int i = 0; i < Jm1.size(); ++i)
      {
        if(Jm1[i] + 1.0 <= get_J_tol<Number>())
        {
          update_J = true;

          for(unsigned int j = 0; j < dim; ++j)
          {
            for(unsigned int k = 0; k < dim; ++k)
            {
              F[j][k][i]          = F_old[j][k][i];
              Grad_d_lin[j][k][i] = F_old[j][k][i];
            }
            Grad_d_lin[j][j][i] -= 1.0;
          }
        }
      }

      if(update_J)
      {
        AssertThrow(stable_formulation == false,
                    dealii::ExcMessage("Storing F_old does not allow for a stable recovery of J."));

        Jm1 = determinant(F) - 1.0;
      }
    }

    Jm1_coefficients.set_coefficient_cell(cell, q, Jm1);

    scalar const J_pow = get_J_pow<true /* force_evaluation */>(Jm1, cell, q);
    J_pow_coefficients.set_coefficient_cell(cell, q, J_pow);

    if(spatial_integration)
    {
      one_over_J_coefficients.set_coefficient_cell(cell, q, 1.0 / (Jm1 + 1.0));
    }

    symmetric_tensor const E =
      compute_E_scaled<dim, Number, Number, stable_formulation>(Grad_d_lin, 1.0);
    scalar const c1 =
      get_c1<true /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    scalar const c2 =
      get_c2<true /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);

    c1_coefficients.set_coefficient_cell(cell, q, c1);
    c2_coefficients.set_coefficient_cell(cell, q, c2);

    // Fiber and tensorial linearization data.
    if constexpr(cache_level == 1)
    {
      for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
      {
        vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
        vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
        symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);

        scalar const E_i = get_E_i<true /* force_evaluation */>(H_i, E, i, cell, q);
        E_i_coefficients[i].set_coefficient_cell(cell, q, E_i);
        scalar const c3 = get_c3<true /* force_evaluation */>(M_1, E, E_i, i, cell, q);
        c3_coefficients[i].set_coefficient_cell(cell, q, c3);
      }
    }
    else
    {
      tensor const           F_inv = invert(F);
      symmetric_tensor const C_inv = compute_C_inv(F_inv);

      // Set all but the fiber and stress linearization data.
      if(spatial_integration)
      {
        symmetric_tensor const C = compute_HT_times_H(F);
        C_coefficients.set_coefficient_cell(cell, q, C);
        if(force_material_residual)
        {
          gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);
        }
      }
      else
      {
        gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);
        F_inv_coefficients.set_coefficient_cell(cell, q, F_inv);
        C_inv_coefficients.set_coefficient_cell(cell, q, C_inv);
      }

      // Set scalar linearization data for fiber contribution and stresses.
      symmetric_tensor S_fiber;
      for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
      {
        vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
        symmetric_tensor const H_i = fiber_structure_tensor[i].get_coefficient_cell(cell, q);

        scalar const E_i = get_E_i<true /* force_evaluation */>(H_i, E, i, cell, q);
        E_i_coefficients[i].set_coefficient_cell(cell, q, E_i);

        scalar const c3 = get_c3<true /* force_evaluation */>(M_1, E, E_i, i, cell, q);
        c3_coefficients[i].set_coefficient_cell(cell, q, c3);

        if(spatial_integration)
        {
          F_times_H_i_times_FT_coefficients[i].set_coefficient_cell(cell,
                                                                    q,
                                                                    compute_push_forward(H_i, F));
        }

        // Note that both stress tensors require S_fiber.
        S_fiber += compute_S_fiber_i(c3, E_i, H_i);
      }

      if(spatial_integration)
      {
        if constexpr(stable_formulation)
        {
          symmetric_tensor const e =
            compute_e_scaled<dim, Number, Number, stable_formulation>(Grad_d_lin, 1.0);
          symmetric_tensor const tau =
            compute_tau_stable(S_fiber, F, e, Jm1, J_pow, shear_modulus_stored);
          kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);
        }
        else
        {
          symmetric_tensor const tau =
            compute_tau_unstable(S_fiber, F, J_pow, c1, shear_modulus_stored);
          kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);
        }
      }

      if(not spatial_integration or force_material_residual)
      {
        if constexpr(stable_formulation)
        {
          S_fiber +=
            compute_S_ground_matrix_stable(Grad_d_lin, C_inv, J_pow, Jm1, shear_modulus_stored);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S_fiber);
        }
        else
        {
          S_fiber += compute_S_ground_matrix_unstable(C_inv, J_pow, c1, shear_modulus_stored);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S_fiber);
        }
      }
    }
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const
{
  if constexpr(cache_level == 0)
  {
    return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
  }
  else if constexpr(cache_level == 1)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    tensor const F =
      compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
    tensor const           F_inv = invert(F);
    scalar const           Jm1   = Jm1_coefficients.get_coefficient_cell(cell, q);
    symmetric_tensor const C_inv = compute_C_inv(F_inv);
    scalar const           J_pow = get_J_pow<false /* force_evaluation */>(Jm1, cell, q);

    if constexpr(stable_formulation)
    {
      symmetric_tensor S = compute_S_ground_matrix_stable(
        gradient_displacement, C_inv, J_pow, Jm1, shear_modulus_stored);

      for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
      {
        vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
        vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
        symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);
        scalar const           E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
        scalar const           c3  = c3_coefficients[i].get_coefficient_cell(cell, q);

        S += compute_S_fiber_i(c3, E_i, H_i);
      }

      return S;
    }
    else
    {
      scalar const c1 = c1_coefficients.get_coefficient_cell(cell, q);

      symmetric_tensor S = compute_S_ground_matrix_unstable(C_inv, J_pow, c1, shear_modulus_stored);

      for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
      {
        vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
        vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
        symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);
        scalar const           E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
        scalar const           c3  = c3_coefficients[i].get_coefficient_cell(cell, q);

        S += compute_S_fiber_i(c3, E_i, H_i);
      }

      return S;
    }
  }
  else
  {
    AssertThrow(cache_level < 2,
                dealii::ExcMessage("This `cache_level` stores tensorial quantities, "
                                   "use the dedicated function."));

    return second_piola_kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                     unsigned int const cell,
                                     unsigned int const q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  auto const [F, Jm1] =
    compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
  tensor const           F_inv = invert(F);
  symmetric_tensor const C_inv = compute_C_inv(F_inv);
  scalar const           J_pow = get_J_pow<true /* force_evaluation */>(Jm1, cell, q);
  symmetric_tensor const E =
    compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);

  if constexpr(stable_formulation)
  {
    symmetric_tensor S = compute_S_ground_matrix_stable(
      gradient_displacement, C_inv, J_pow, Jm1, shear_modulus_stored);

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      symmetric_tensor const H_i = get_structure_tensor(M_1, i, cell, q);
      scalar const           E_i = get_E_i<true /* force_evaluation */>(H_i, E, i, cell, q);
      scalar const           c3  = get_c3<true /* force_evaluation */>(M_1, E, E_i, i, cell, q);

      S += compute_S_fiber_i(c3, E_i, H_i);
    }

    return S;
  }
  else
  {
    scalar const c1 =
      get_c1<true /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);

    symmetric_tensor S = compute_S_ground_matrix_unstable(C_inv, J_pow, c1, shear_modulus_stored);

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      symmetric_tensor const H_i = get_structure_tensor(M_1, i, cell, q);
      scalar const           E_i = get_E_i<true /* force_evaluation */>(H_i, E, i, cell, q);
      scalar const           c3  = get_c3<true /* force_evaluation */>(M_1, E, E_i, i, cell, q);

      S += compute_S_fiber_i(c3, E_i, H_i);
    }

    return S;
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                   "this `cache_level` does not store tensorial quantities."));

    return (std::numeric_limits<Number>::quiet_NaN() *
            get_identity_symmetric_tensor<dim, Number>());
  }
  else
  {
    return second_piola_kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_S_ground_matrix_stable(tensor const &           gradient_displacement,
                                 symmetric_tensor const & C_inv,
                                 scalar const &           J_pow,
                                 scalar const &           Jm1,
                                 scalar const &           shear_modulus) const
{
  symmetric_tensor S =
    compute_E_scaled<dim, Number, scalar, stable_formulation>(gradient_displacement,
                                                              2.0 * shear_modulus * J_pow);

  add_scaled_identity(S,
                      -ONE_THIRD * trace(S) +
                        (0.5 * bulk_modulus) * compute_JJm1<Number, stable_formulation>(Jm1));

  return compute_symmetric_product(C_inv, S);
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_S_ground_matrix_unstable(symmetric_tensor const & C_inv,
                                   scalar const &           J_pow,
                                   scalar const &           c1,
                                   scalar const &           shear_modulus) const
{
  symmetric_tensor S = c1 * C_inv;

  add_scaled_identity(S, shear_modulus * J_pow);

  return S;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_S_fiber_i(scalar const & c3, scalar const & E_i, symmetric_tensor const & H_i) const
{
  return ((c3 * E_i) * H_i);
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }


  if constexpr(cache_level == 0)
  {
    auto const [F, Jm1] =
      compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
    symmetric_tensor const E =
      compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);
    scalar const J_pow = get_J_pow<false /* force_evaluation */>(Jm1, cell, q);
    scalar const c1 =
      get_c1<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    scalar const c2 =
      get_c2<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    tensor const           F_inv = invert(F);
    symmetric_tensor const C_inv = compute_C_inv(F_inv);

    tensor const F_inv_times_gradient_increment       = F_inv * gradient_increment;
    scalar const one_over_J_times_Dd_J                = trace(F_inv_times_gradient_increment);
    tensor const transpose_gradient_increment_times_F = transpose(gradient_increment) * F;

    symmetric_tensor Dd_S =
      C_inv * (c2 * one_over_J_times_Dd_J - (TWO_THIRDS * shear_modulus_stored * J_pow) *
                                              trace(transpose_gradient_increment_times_F));
    Dd_S += c1 * compute_H_plus_HT(-F_inv_times_gradient_increment * C_inv);
    add_scaled_identity(Dd_S, -shear_modulus_stored * TWO_THIRDS * J_pow * one_over_J_times_Dd_J);

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);
      scalar const           E_i = get_E_i<false /* force_evaluation */>(H_i, E, i, cell, q);
      scalar const           c3  = get_c3<false /* force_evaluation */>(M_1, E, E_i, i, cell, q);

      Dd_S += H_i * (c3 * ((2.0 * fiber_k_2) * E_i * E_i + 1.0) *
                     scalar_product(H_i, compute_H_plus_HT(transpose_gradient_increment_times_F)));
    }

    return Dd_S;
  }
  else if constexpr(cache_level == 1)
  {
    tensor const F =
      compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
    scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const           c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const           c2    = c2_coefficients.get_coefficient_cell(cell, q);
    tensor const           F_inv = invert(F);
    symmetric_tensor const C_inv = compute_C_inv(F_inv);

    tensor const F_inv_times_gradient_increment       = F_inv * gradient_increment;
    scalar const one_over_J_times_Dd_J                = trace(F_inv_times_gradient_increment);
    tensor const transpose_gradient_increment_times_F = transpose(gradient_increment) * F;

    symmetric_tensor Dd_S =
      C_inv * (c2 * one_over_J_times_Dd_J - (TWO_THIRDS * shear_modulus_stored * J_pow) *
                                              trace(transpose_gradient_increment_times_F));
    Dd_S += c1 * compute_H_plus_HT(-F_inv_times_gradient_increment * C_inv);
    add_scaled_identity(Dd_S, -shear_modulus_stored * TWO_THIRDS * J_pow * one_over_J_times_Dd_J);

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);
      scalar const           E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      scalar const           c3  = c3_coefficients[i].get_coefficient_cell(cell, q);

      Dd_S += H_i * (c3 * ((2.0 * fiber_k_2) * E_i * E_i + 1.0) *
                     scalar_product(H_i, compute_H_plus_HT(transpose_gradient_increment_times_F)));
    }

    return Dd_S;
  }
  else
  {
    // Note that we could load F, but F is needed in the nonlinear operator regardless,
    // so we pass it in and use it here instead of loading it twice and not passing it.
    tensor const F =
      compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
    scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const           c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const           c2    = c2_coefficients.get_coefficient_cell(cell, q);
    tensor const           F_inv = F_inv_coefficients.get_coefficient_cell(cell, q);
    symmetric_tensor const C_inv = C_inv_coefficients.get_coefficient_cell(cell, q);

    tensor const F_inv_times_gradient_increment       = F_inv * gradient_increment;
    scalar const one_over_J_times_Dd_J                = trace(F_inv_times_gradient_increment);
    tensor const transpose_gradient_increment_times_F = transpose(gradient_increment) * F;

    symmetric_tensor Dd_S =
      C_inv * (c2 * one_over_J_times_Dd_J - (TWO_THIRDS * shear_modulus_stored * J_pow) *
                                              trace(transpose_gradient_increment_times_F));
    Dd_S += c1 * compute_H_plus_HT(-F_inv_times_gradient_increment * C_inv);
    add_scaled_identity(Dd_S, -shear_modulus_stored * TWO_THIRDS * J_pow * one_over_J_times_Dd_J);

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      symmetric_tensor const H_i = fiber_structure_tensor[i].get_coefficient_cell(cell, q);
      scalar const           E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      scalar const           c3  = c3_coefficients[i].get_coefficient_cell(cell, q);

      Dd_S += H_i * (c3 * ((2.0 * fiber_k_2) * E_i * E_i + 1.0) *
                     scalar_product(H_i, compute_H_plus_HT(transpose_gradient_increment_times_F)));
    }

    return Dd_S;
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  kirchhoff_stress(tensor const &     gradient_displacement,
                   unsigned int const cell,
                   unsigned int const q) const
{
  if constexpr(cache_level == 0)
  {
    return kirchhoff_stress_eval(gradient_displacement, cell, q);
  }
  else if constexpr(cache_level == 1)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    symmetric_tensor tau;
    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);
      scalar const           E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      scalar const           c3  = c3_coefficients[i].get_coefficient_cell(cell, q);

      tau += compute_S_fiber_i(c3, E_i, H_i);
    }

    if constexpr(stable_formulation)
    {
      tensor const F =
        compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
      scalar const           Jm1   = Jm1_coefficients.get_coefficient_cell(cell, q);
      scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
      symmetric_tensor const e =
        compute_e_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);

      tau = compute_tau_stable(tau, F, e, Jm1, J_pow, shear_modulus_stored);
    }
    else
    {
      tensor const F =
        compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
      scalar const J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
      scalar const c1    = c1_coefficients.get_coefficient_cell(cell, q);

      tau = compute_tau_unstable(tau, F, J_pow, c1, shear_modulus_stored);
    }

    return tau;
  }
  else
  {
    AssertThrow(cache_level < 2,
                dealii::ExcMessage("This `cache_level` stores tensorial quantities, "
                                   "use the dedicated function."));

    return kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  kirchhoff_stress_eval(tensor const &     gradient_displacement,
                        unsigned int const cell,
                        unsigned int const q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  symmetric_tensor const E =
    compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);

  symmetric_tensor tau;
  for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
  {
    vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
    symmetric_tensor const H_i = get_structure_tensor(M_1, i, cell, q);
    scalar const           E_i = get_E_i<true /* force_evaluation */>(H_i, E, i, cell, q);
    scalar const           c3  = get_c3<true /* force_evaluation */>(M_1, E, E_i, i, cell, q);

    tau += compute_S_fiber_i(c3, E_i, H_i);
  }

  auto const [F, Jm1] =
    compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);

  scalar const J_pow = get_J_pow<true /* force_evaluation */>(Jm1, cell, q);
  if constexpr(stable_formulation)
  {
    symmetric_tensor const e =
      compute_e_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);
    tau = compute_tau_stable(tau, F, e, Jm1, J_pow, shear_modulus_stored);
  }
  else
  {
    scalar const c1 =
      get_c1<true /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    tau = compute_tau_unstable(tau, F, J_pow, c1, shear_modulus_stored);
  }

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  kirchhoff_stress(unsigned int const cell, unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                   "this `cache_level` does not store tensorial quantities."));

    return (std::numeric_limits<Number>::quiet_NaN() *
            get_identity_symmetric_tensor<dim, Number>());
  }
  else
  {
    return kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_tau_stable(symmetric_tensor const & S_fiber,
                     tensor const &           F,
                     symmetric_tensor const & e,
                     scalar const &           Jm1,
                     scalar const &           J_pow,
                     scalar const &           shear_modulus) const
{
  // Assume that `S_fiber` holds fiber term pull-back, i.e.,
  // sum_(j=4,6) c_3 * E_i * H_i
  // to apply the push-forward only once for the sum of all terms.

  // Push forward fiber contribution.
  symmetric_tensor tau = compute_push_forward(S_fiber, F);

  // Add remaining terms.
  tau += (2.0 * shear_modulus * J_pow) * e;

  add_scaled_identity(tau,
                      -TWO_THIRDS * shear_modulus * J_pow * trace(e) +
                        (0.5 * bulk_modulus) * compute_JJm1<Number, stable_formulation>(Jm1));

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  compute_tau_unstable(symmetric_tensor const & S_fiber,
                       tensor const &           F,
                       scalar const &           J_pow,
                       scalar const &           c1,
                       scalar const &           shear_modulus) const
{
  // Assume that `S_fiber` holds fiber term pull-back, i.e.,
  // sum_(j=4,6) c_3 * E_i * H_i
  // to apply the push-forward only once for the sum of all terms.
  symmetric_tensor tau = S_fiber;

  // Add isochoric term, i.e.,
  // mu * J^(-2/3) * I
  // to apply push forward in next step.
  add_scaled_identity(tau, shear_modulus * J_pow);

  // Push forward the summed terms.
  tau = compute_push_forward(tau, F);

  // Add the remaining term.
  add_scaled_identity(tau, c1);

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          tensor const &           gradient_displacement,
                          unsigned int const       cell,
                          unsigned int const       q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  if constexpr(cache_level == 0)
  {
    auto const [F, Jm1] =
      compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
    symmetric_tensor const E =
      compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);
    symmetric_tensor C = 2.0 * E;
    add_scaled_identity<dim, Number, Number>(C, 1.0);
    scalar const J_pow = get_J_pow<false /* force_evaluation */>(Jm1, cell, q);
    scalar const c1 =
      get_c1<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    scalar const c2 =
      get_c2<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);

    symmetric_tensor result = symmetric_gradient_increment * (-2.0 * c1);
    result +=
      ((-4.0 * ONE_THIRD) * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) * C;
    add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);
      scalar const           E_i = get_E_i<false /* force_evaluation */>(H_i, E, i, cell, q);
      scalar const           c3  = get_c3<false /* force_evaluation */>(M_1, E, E_i, i, cell, q);
      symmetric_tensor const F_times_H_i_times_FT = compute_push_forward(H_i, F);

      result += (c3 * ((4.0 * fiber_k_2) * E_i * E_i + 2.0) *
                 scalar_product(F_times_H_i_times_FT, symmetric_gradient_increment)) *
                F_times_H_i_times_FT;
    }

    bound_tensor(result, compute_numerical_upper_bound(fiber_k_1));
    return result;
  }
  else if constexpr(cache_level == 1)
  {
    tensor const F =
      compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
    symmetric_tensor const C     = compute_HT_times_H(F);
    scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const           c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const           c2    = c2_coefficients.get_coefficient_cell(cell, q);

    symmetric_tensor result = symmetric_gradient_increment * (-2.0 * c1);
    result +=
      ((-4.0 * ONE_THIRD) * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) * C;
    add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      vector const           M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      vector const           M_2 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      symmetric_tensor const H_i = compute_structure_tensor(M_1, M_2);
      scalar const           E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      scalar const           c3  = c3_coefficients[i].get_coefficient_cell(cell, q);
      symmetric_tensor const F_times_H_i_times_FT = compute_push_forward(H_i, F);

      result += (c3 * ((4.0 * fiber_k_2) * E_i * E_i + 2.0) *
                 scalar_product(F_times_H_i_times_FT, symmetric_gradient_increment)) *
                F_times_H_i_times_FT;
    }

    bound_tensor(result, compute_numerical_upper_bound(fiber_k_1));
    return result;
  }
  else
  {
    AssertThrow(cache_level < 2,
                dealii::ExcMessage("This function is not the optimal choice for `cache_level` 2."));

    return contract_with_J_times_C(symmetric_gradient_increment, cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          unsigned int const       cell,
                          unsigned int const       q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function cannot be called with `cache_level` < 2."));

    return (std::numeric_limits<Number>::quiet_NaN() *
            get_identity_symmetric_tensor<dim, Number>());
  }
  else
  {
    symmetric_tensor const C     = C_coefficients.get_coefficient_cell(cell, q);
    scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const           c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const           c2    = c2_coefficients.get_coefficient_cell(cell, q);

    symmetric_tensor result = symmetric_gradient_increment * (-2.0 * c1);
    result +=
      ((-4.0 * ONE_THIRD) * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) * C;
    add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

    for(unsigned int i = 0; i < N_FIBER_FAMILIES; ++i)
    {
      scalar const           E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      scalar const           c3  = c3_coefficients[i].get_coefficient_cell(cell, q);
      symmetric_tensor const F_times_H_i_times_FT =
        F_times_H_i_times_FT_coefficients[i].get_coefficient_cell(cell, q);

      result += (c3 * ((4.0 * fiber_k_2) * E_i * E_i + 2.0) *
                 scalar_product(F_times_H_i_times_FT, symmetric_gradient_increment)) *
                F_times_H_i_times_FT;
    }

    bound_tensor(result, compute_numerical_upper_bound(fiber_k_1));
    return result;
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::one_over_J(
  unsigned int const cell,
  unsigned int const q) const
{
  AssertThrow(spatial_integration and cache_level > 0,
              dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return (one_over_J_coefficients.get_coefficient_cell(cell, q));
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number, check_type, stable_formulation, cache_level>::
  gradient_displacement(unsigned int const cell, unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("Cannot access precomputed deformation gradient."));
  }

  return (gradient_displacement_coefficients.get_coefficient_cell(cell, q));
}

// clang-format off
// Note that the higher check types (third template argument) are missing.
template class IncompressibleFibrousTissue<2, float,  0, true,  0>;
template class IncompressibleFibrousTissue<2, float,  0, true,  1>;
template class IncompressibleFibrousTissue<2, float,  0, true,  2>;

template class IncompressibleFibrousTissue<2, float,  0, false, 0>;
template class IncompressibleFibrousTissue<2, float,  0, false, 1>;
template class IncompressibleFibrousTissue<2, float,  0, false, 2>;

template class IncompressibleFibrousTissue<2, double, 0, true,  0>;
template class IncompressibleFibrousTissue<2, double, 0, true,  1>;
template class IncompressibleFibrousTissue<2, double, 0, true,  2>;

template class IncompressibleFibrousTissue<2, double, 0, false, 0>;
template class IncompressibleFibrousTissue<2, double, 0, false, 1>;
template class IncompressibleFibrousTissue<2, double, 0, false, 2>;

template class IncompressibleFibrousTissue<3, float,  0, true,  0>;
template class IncompressibleFibrousTissue<3, float,  0, true,  1>;
template class IncompressibleFibrousTissue<3, float,  0, true,  2>;

template class IncompressibleFibrousTissue<3, float,  0, false, 0>;
template class IncompressibleFibrousTissue<3, float,  0, false, 1>;
template class IncompressibleFibrousTissue<3, float,  0, false, 2>;

template class IncompressibleFibrousTissue<3, double, 0, true,  0>;
template class IncompressibleFibrousTissue<3, double, 0, true,  1>;
template class IncompressibleFibrousTissue<3, double, 0, true,  2>;

template class IncompressibleFibrousTissue<3, double, 0, false, 0>;
template class IncompressibleFibrousTissue<3, double, 0, false, 1>;
template class IncompressibleFibrousTissue<3, double, 0, false, 2>;
// clang-format on

} // namespace Structure
} // namespace ExaDG
