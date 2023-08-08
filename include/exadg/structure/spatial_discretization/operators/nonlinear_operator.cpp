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
#include <exadg/structure/spatial_discretization/operators/nonlinear_operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  OperatorData<dim> const &                 data)
{
  Base::initialize(matrix_free, affine_constraints, data);

  integrator_lin = std::make_shared<IntegratorCell>(*this->matrix_free,
                                                    this->operator_data.dof_index_inhomogeneous,
                                                    this->operator_data.quad_index);
  // it should not make a difference here whether we use dof_index or dof_index_inhomogeneous
  this->matrix_free->initialize_dof_vector(displacement_lin, this->operator_data.dof_index);
  displacement_lin.update_ghost_values();
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::evaluate_nonlinear(VectorType & dst, VectorType const & src) const
{
  this->matrix_free->loop(&This::cell_loop_nonlinear,
                          &This::face_loop_nonlinear,
                          &This::boundary_face_loop_nonlinear,
                          this,
                          dst,
                          src,
                          true);
}

template<int dim, typename Number>
bool
NonLinearOperator<dim, Number>::valid_deformation(VectorType const & displacement) const
{
  Number dst = 0.0;

  // dst has to remain zero for a valid deformation state
  this->matrix_free->cell_loop(&This::cell_loop_valid_deformation,
                               this,
                               dst,
                               displacement,
                               false /* no zeroing of dst vector */);

  // sum over all MPI processes
  Number valid = 0.0;
  valid        = dealii::Utilities::MPI::sum(
    dst, this->matrix_free->get_dof_handler(this->operator_data.dof_index).get_communicator());

  return (valid == 0.0);
}


template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_solution_linearization(VectorType const & vector) const
{
  // Only update linearized operator if deformation state is valid. It is better to continue
  // with an old deformation state in the linearized operator than with an invalid one.
  if(valid_deformation(vector))
  {
    displacement_lin = vector;
    displacement_lin.update_ghost_values();
  }
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::VectorType const &
NonLinearOperator<dim, Number>::get_solution_linearization() const
{
  return displacement_lin;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::update_ghost_values_linearization_vector() const
{
  // See the deal.II documentation:
  // Return whether the vector currently is in a state where ghost values can be read or not. This
  // is the same functionality as other parallel vectors have. If this method returns false, this
  // only means that read-access to ghost elements is prohibited whereas write access is still
  // possible (to those entries specified as ghosts during initialization), not that there are no
  // ghost elements at all.
  if(not displacement_lin.has_ghost_elements())
  {
    displacement_lin.update_ghost_values();
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell_nonlinear(IntegratorCell &   integrator,
                                                      unsigned int const cell) const
{
  integrator.reinit(cell);

  this->material_handler.reinit(*this->matrix_free, cell);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator_inhom(matrix_free,
                                  this->operator_data.dof_index_inhomogeneous,
                                  this->operator_data.quad_index);

  IntegratorCell integrator(matrix_free,
                            this->operator_data.dof_index,
                            this->operator_data.quad_index);

  auto const unsteady_flag = this->operator_data.unsteady ? dealii::EvaluationFlags::values :
                                                            dealii::EvaluationFlags::nothing;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    reinit_cell_nonlinear(integrator_inhom, cell);
    integrator.reinit(cell);

    integrator_inhom.gather_evaluate(src, unsteady_flag | dealii::EvaluationFlags::gradients);

    do_cell_integral_nonlinear(integrator_inhom);

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_inhom.integrate(unsteady_flag | dealii::EvaluationFlags::gradients,
                               integrator.begin_dof_values());
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_valid_deformation(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  Number &                                dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator(matrix_free,
                            this->operator_data.dof_index_inhomogeneous,
                            this->operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    reinit_cell_nonlinear(integrator, cell);

    integrator.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // material displacement gradient
      tensor const Grad_d = integrator.get_gradient(q);

      // material deformation gradient
      tensor const F     = get_F<dim, Number>(Grad_d);
      scalar const det_F = determinant(F);
      for(unsigned int v = 0; v < det_F.size(); ++v)
      {
        // if deformation is invalid, add a positive value to dst
        if(det_F[v] <= 0.0)
          dst += 1.0;
      }
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::face_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::boundary_face_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorFace integrator_m_inhom(matrix_free,
                                    true,
                                    this->operator_data.dof_index_inhomogeneous,
                                    this->operator_data.quad_index);

  IntegratorFace integrator_m = IntegratorFace(matrix_free,
                                               true,
                                               this->operator_data.dof_index,
                                               this->operator_data.quad_index);

  // apply Neumann or Robin BCs
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(integrator_m_inhom, face);
    integrator_m.reinit(face);

    // In case of a pull-back of the traction vector, we need to evaluate the displacement gradient
    // to obtain the surface area ratio da/dA. We write the integrator flags explicitly in this case
    // since they depend on the parameter pull_back_traction. On Robin boundaries, we need the
    // solution values.
    bool const is_on_robin_boundary =
      this->operator_data.bc->get_boundary_type(matrix_free.get_boundary_id(face)) ==
      BoundaryType::RobinSpringDashpotPressure;
    if(this->operator_data.pull_back_traction or is_on_robin_boundary)
    {
      integrator_m_inhom.gather_evaluate(src,
                                         dealii::EvaluationFlags::gradients |
                                           dealii::EvaluationFlags::values);
    }

    do_boundary_integral_continuous(integrator_m_inhom,
                                    OperatorType::full,
                                    matrix_free.get_boundary_id(face));

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_m_inhom.integrate(this->integrator_flags.face_integrate,
                                 integrator_m.begin_dof_values());
    integrator_m.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral_nonlinear(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  // loop over all quadrature points
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    // material displacement gradient
    tensor const Grad_d = integrator.get_gradient(q);

    // material deformation gradient
    tensor const F = get_F<dim, Number>(Grad_d);

    // Green-Lagrange strains
    tensor const E = get_E<dim, Number>(F);

    // 2. Piola-Kirchhoff stresses
    tensor const S = material->evaluate_stress(E, integrator.get_current_cell_index(), q);

    // 1st Piola-Kirchhoff stresses P = F * S
    tensor const P = F * S;

    // Grad_v : P
    integrator.submit_gradient(P, q);

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
NonLinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector traction;

    if(boundary_type == BoundaryType::Neumann or boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::inhomogeneous or operator_type == OperatorType::full)
      {
        traction -= calculate_neumann_value<dim, Number>(
          q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->time);

        if(this->operator_data.pull_back_traction)
        {
          tensor F = get_F<dim, Number>(integrator.get_gradient(q));
          vector N = integrator.get_normal_vector(q);
          // da/dA * n = det F F^{-T} * N := n_star
          // -> da/dA = n_star.norm()
          vector n_star = determinant(F) * transpose(invert(F)) * N;
          // t_0 = da/dA * t
          traction *= n_star.norm();
        }
      }
    }

    if(boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::homogeneous or operator_type == OperatorType::full)
      {
        auto const it = this->operator_data.bc->robin_k_c_p_param.find(boundary_id);

        if(it != this->operator_data.bc->robin_k_c_p_param.end())
        {
          bool const   normal_projection_displacement = it->second.first[0];
          double const coefficient_displacement       = it->second.second[0];

          if(normal_projection_displacement)
          {
            vector const N = integrator.get_normal_vector(q);
            traction += N * (coefficient_displacement * (N * integrator.get_value(q)));
          }
          else
          {
            traction += coefficient_displacement * integrator.get_value(q);
          }

          if(this->operator_data.unsteady)
          {
            bool const   normal_projection_velocity = it->second.first[1];
            double const coefficient_velocity       = it->second.second[1];

            if(normal_projection_velocity)
            {
              vector const N = integrator.get_normal_vector(q);
              traction += N * (coefficient_velocity * this->scaling_factor_mass_boundary *
                               (N * integrator.get_value(q)));
            }
            else
            {
              traction +=
                coefficient_velocity * this->scaling_factor_mass_boundary * integrator.get_value(q);
            }
          }
        }
      }
    }

    integrator.submit_value(traction, q);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell_derived(IntegratorCell &   integrator,
                                                    unsigned int const cell) const
{
  Base::reinit_cell_derived(integrator, cell);

  integrator_lin->reinit(cell);

  integrator_lin->read_dof_values(displacement_lin);
  integrator_lin->evaluate(dealii::EvaluationFlags::gradients);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  // loop over all quadrature points
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    // kinematics
    tensor const Grad_delta = integrator.get_gradient(q);

    tensor const F_lin = get_F<dim, Number>(integrator_lin->get_gradient(q));

    // Green-Lagrange strains
    tensor const E_lin = get_E<dim, Number>(F_lin);

    // 2nd Piola-Kirchhoff stresses
    tensor const S_lin = material->evaluate_stress(E_lin, integrator.get_current_cell_index(), q);

    // directional derivative of 1st Piola-Kirchhoff stresses P

    // 1. elastic and initial displacement stiffness contributions
    tensor delta_P =
      F_lin *
      material->apply_C(transpose(F_lin) * Grad_delta, integrator.get_current_cell_index(), q);

    // 2. geometric (or initial stress) stiffness contribution
    delta_P += Grad_delta * S_lin;

    // Grad_v : delta_P
    integrator.submit_gradient(delta_P, q);

    if(this->operator_data.unsteady)
    {
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
    }
  }
}

template class NonLinearOperator<2, float>;
template class NonLinearOperator<2, double>;

template class NonLinearOperator<3, float>;
template class NonLinearOperator<3, double>;

} // namespace Structure
} // namespace ExaDG
