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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// ExaDG
#include <exadg/operators/structure_calculators.h>
#include <exadg/structure/material/material_handler.h>
#include <exadg/utilities/tensor_utilities.h>

namespace ExaDG
{
template<int dim, typename Number>
DisplacementJacobianCalculator<dim, Number>::DisplacementJacobianCalculator()
  : matrix_free(nullptr), dof_index_vector(0), dof_index_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_vector_in,
  unsigned int const                      dof_index_scalar_in,
  unsigned int const                      quad_index_in)
{
  matrix_free      = &matrix_free_in;
  dof_index_vector = dof_index_vector_in;
  dof_index_scalar = dof_index_scalar_in;
  quad_index       = quad_index_in;
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::compute_projection_rhs(
  VectorType &       dst_scalar_valued,
  VectorType const & src_vector_valued) const
{
  dst_scalar_valued = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst_scalar_valued, src_vector_valued);
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst_scalar_valued,
  VectorType const &                            src_vector_valued,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_scalar, quad_index, 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    // Do not enforce constraints on the `src` vector, as constraints are already applied and
    // `dealii::MatrixFree` object stores constraints relevant in linear systemes, not necessarily
    // constraints suitable for a DoF vector corresponding to the solution (e.g., in Newton's
    // method).
    integrator_vector.read_dof_values_plain(src_vector_valued);
    integrator_vector.evaluate(dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_vector.n_q_points; q++)
    {
      tensor const gradient_displacement = integrator_vector.get_gradient(q);
      tensor const F                     = Structure::compute_F(gradient_displacement);
      scalar const Jacobian              = determinant(F);

      integrator_scalar.submit_value(Jacobian, q);
    }

    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst_scalar_valued);
  }
}

template<int dim, typename Number>
MaxPrincipalStressCalculator<dim, Number>::MaxPrincipalStressCalculator()
  : matrix_free(nullptr),
    dof_index_vector(0),
    dof_index_scalar(0),
    quad_index(0),
    elasticity_operator_base(nullptr)
{
}

template<int dim, typename Number>
void
MaxPrincipalStressCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  unsigned int const                                     dof_index_vector_in,
  unsigned int const                                     dof_index_scalar_in,
  unsigned int const                                     quad_index_in,
  Structure::ElasticityOperatorBase<dim, Number> const & elasticity_operator_base_in)
{
  matrix_free              = &matrix_free_in;
  dof_index_vector         = dof_index_vector_in;
  dof_index_scalar         = dof_index_scalar_in;
  quad_index               = quad_index_in;
  elasticity_operator_base = &elasticity_operator_base_in;
}

template<int dim, typename Number>
void
MaxPrincipalStressCalculator<dim, Number>::compute_projection_rhs(
  VectorType &       dst_scalar_valued,
  VectorType const & src_vector_valued) const
{
  dst_scalar_valued = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst_scalar_valued, src_vector_valued);
}

template<int dim, typename Number>
void
MaxPrincipalStressCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst_scalar_valued,
  VectorType const &                            src_vector_valued,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_scalar, quad_index, 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    Structure::Material<dim, Number> const & material =
      elasticity_operator_base->get_material_in_cell(matrix_free, cell);

    integrator_vector.reinit(cell);
    // Do not enforce constraints on the `src` vector, as constraints are already applied and the
    // `dealii::MatrixFree` object might store different constraints.
    integrator_vector.read_dof_values_plain(src_vector_valued);
    integrator_vector.evaluate(dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_vector.n_q_points; q++)
    {
      tensor const           gradient_displacement = integrator_vector.get_gradient(q);
      tensor const           F                     = Structure::compute_F(gradient_displacement);
      scalar const           Jacobian              = determinant(F);
      symmetric_tensor const S =
        material.second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
      symmetric_tensor const sigma = Structure::compute_push_forward(S, F) / Jacobian;

      // Loop over vectorization length to use `dealii::eigenvalues()`.
      scalar max_eigenvalue;
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); v++)
      {
        dealii::SymmetricTensor<2, dim, Number> sigma_v;
        for(unsigned int i = 0; i < dim; ++i)
        {
          for(unsigned int j = 0; j < dim; ++j)
          {
            sigma_v[i][j] = sigma[i][j][v];
          }
        }
        std::array<Number, dim> eigenvalues_v = dealii::eigenvalues(sigma_v);

        max_eigenvalue[v] = eigenvalues_v[0];
      }

      integrator_scalar.submit_value(max_eigenvalue, q);
    }

    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst_scalar_valued);
  }
}

template<int dim, typename Number>
LocalStressCalculator<dim, Number>::LocalStressCalculator()
  : matrix_free(nullptr),
    dof_index_vector(0),
    quad_index(0),
    local_stress_direction(LocalStressDirection::Undefined),
    elasticity_operator_base(nullptr)
{
}

template<int dim, typename Number>
void
LocalStressCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  unsigned int const                                     dof_index_vector_in,
  unsigned int const                                     dof_index_vector_postprocessing_in,
  unsigned int const                                     quad_index_in,
  LocalStressDirection const &                           local_stress_direction_in,
  Structure::ElasticityOperatorBase<dim, Number> const & elasticity_operator_base_in)
{
  matrix_free                     = &matrix_free_in;
  dof_index_vector                = dof_index_vector_in;
  dof_index_vector_postprocessing = dof_index_vector_postprocessing_in;
  quad_index                      = quad_index_in;
  local_stress_direction          = local_stress_direction_in;
  elasticity_operator_base        = &elasticity_operator_base_in;
}

template<int dim, typename Number>
void
LocalStressCalculator<dim, Number>::compute_projection_rhs(
  VectorType &       dst_vector_postprocessing_valued,
  VectorType const & src_vector_valued) const
{
  dst_vector_postprocessing_valued = 0;

  matrix_free->cell_loop(&This::cell_loop,
                         this,
                         dst_vector_postprocessing_valued,
                         src_vector_valued);
}

template<int dim, typename Number>
void
LocalStressCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst_vector_postprocessing_valued,
  VectorType const &                            src_vector_valued,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
  CellIntegratorVector integrator_vector_postprocessing(matrix_free,
                                                        dof_index_vector_postprocessing,
                                                        quad_index,
                                                        0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    Structure::Material<dim, Number> const & material =
      elasticity_operator_base->get_material_in_cell(matrix_free, cell);

    integrator_vector_postprocessing.reinit(cell);

    if(local_stress_direction == LocalStressDirection::Full or
       local_stress_direction == LocalStressDirection::Normal or
       local_stress_direction == LocalStressDirection::InPlane)
    {
      integrator_vector.reinit(cell);
      // Do not enforce constraints on the `src` vector, as constraints are already applied and the
      // `dealii::MatrixFree` object might store different constraints.
      integrator_vector.read_dof_values_plain(src_vector_valued);
      integrator_vector.evaluate(dealii::EvaluationFlags::gradients);
    }

    for(unsigned int q = 0; q < integrator_vector_postprocessing.n_q_points; q++)
    {
      // get local material coordinate system
      std::vector<vector> const E1_E2 = material.get_material_orientation_E1_E2(cell, q);

      if(local_stress_direction == LocalStressDirection::E1_orientation_only)
      {
        // Only get the E1 material orientation vector.
        integrator_vector_postprocessing.submit_value(E1_E2[0], q);
      }
      else if(local_stress_direction == LocalStressDirection::E2_orientation_only)
      {
        // Only get the E2 material orientation vector.
        integrator_vector_postprocessing.submit_value(E1_E2[1], q);
      }
      else
      {
        // compute stress
        tensor const           gradient_displacement = integrator_vector.get_gradient(q);
        tensor const           F                     = Structure::compute_F(gradient_displacement);
        scalar const           Jacobian              = determinant(F);
        symmetric_tensor const S =
          material.second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
        symmetric_tensor const sigma = Structure::compute_push_forward(S, F) / Jacobian;

        // compute spatial normal on cutting plain
        vector e3 = cross_product_3d(F * E1_E2[0], F * E1_E2[1]);
        e3 /= e3.norm();

        // compute traction vector, cutting plain is defined via e3
        vector traction = sigma * e3;
        if(local_stress_direction == LocalStressDirection::Full)
        {
          // nothing to do, the full spatial traction vector is requested
        }
        else if(local_stress_direction == LocalStressDirection::Normal)
        {
          // component of the Cauchy traction vector in e3 direction
          traction = (traction * e3) * e3;
        }
        else if(local_stress_direction == LocalStressDirection::InPlane)
        {
          // sum of the in-plain (e1--e2) components of the Cauchy traction vector
          traction -= (traction * e3) * e3; // subtract e3 component
        }
        else
        {
          AssertThrow(false,
                      dealii::ExcMessage("The `LocalStressDirection` has to be "
                                         "defined to extract traction components."));
        }

        integrator_vector_postprocessing.submit_value(traction, q);
      }
    }

    integrator_vector_postprocessing.integrate_scatter(dealii::EvaluationFlags::values,
                                                       dst_vector_postprocessing_valued);
  }
}

template class DisplacementJacobianCalculator<2, float>;
template class DisplacementJacobianCalculator<2, double>;

template class DisplacementJacobianCalculator<3, float>;
template class DisplacementJacobianCalculator<3, double>;

template class LocalStressCalculator<2, float>;
template class LocalStressCalculator<2, double>;

template class LocalStressCalculator<3, float>;
template class LocalStressCalculator<3, double>;

template class MaxPrincipalStressCalculator<2, float>;
template class MaxPrincipalStressCalculator<2, double>;

template class MaxPrincipalStressCalculator<3, float>;
template class MaxPrincipalStressCalculator<3, double>;

} // namespace ExaDG
