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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_OPERATORS_INVERSE_MASS_OPERATOR_H_
#define EXADG_OPERATORS_INVERSE_MASS_OPERATOR_H_

// deal.II
#include <deal.II/fe/fe_data.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/inverse_mass_parameters.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>

namespace ExaDG
{
template<typename Number>
struct InverseMassOperatorData
{
  InverseMassOperatorData()
    : dof_index(0),
      quad_index(0),
      coefficient_is_variable(false),
      consider_inverse_coefficient(false),
      variable_coefficients(nullptr)
  {
  }

  // Get optimal in the sense of (most likely) fastest implementation type of the inverse mass
  // operator depending on the approximation space.
  template<int dim>
  static InverseMassType
  get_optimal_inverse_mass_type(dealii::FiniteElement<dim> const & fe)
  {
    if(fe.conforms(dealii::FiniteElementData<dim>::L2))
    {
      if(fe.reference_cell().is_hyper_cube())
      {
        return InverseMassType::MatrixfreeOperator;
      }
      else
      {
        return InverseMassType::ElementwiseKrylovSolver;
      }
    }
    else
    {
      return InverseMassType::GlobalKrylovSolver;
    }
  }

  // Parameters referring to dealii::MatrixFree
  unsigned int dof_index;
  unsigned int quad_index;

  InverseMassParameters parameters;

  // Enable variable coefficients.
  bool coefficient_is_variable;

  // Consider the regular form of the coefficient (1) or its inverse (2):
  // (1) : (u_h , v_h * c)_Omega
  // (2) : (u_h , v_h / c)_Omega
  bool consider_inverse_coefficient;

  VariableCoefficients<dealii::VectorizedArray<Number>> const * variable_coefficients;
};

template<int dim, int n_components, typename Number>
class InverseMassOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef InverseMassOperator<dim, n_components, Number> This;

  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    InverseMassAsMatrixFreeOperator;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  InverseMassOperator();

  unsigned int
  get_n_iter_global_last() const;

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free_in,
             InverseMassOperatorData<Number> const     inverse_mass_operator_data,
             dealii::AffineConstraints<Number> const * constraints = nullptr);

  /**
   * Updates the inverse mass operator. This function recomputes the preconditioners in case
   * the geometry has changed (e.g. the mesh has been deformed).
   */
  void
  update();

  // dst = M^-1 * src
  void
  apply(VectorType & dst, VectorType const & src) const;

  // dst = scaling_factor * (M^-1 * src)
  void
  apply_scale(VectorType & dst, double const scaling_factor, VectorType const & src) const;


private:
  void
  cell_loop_matrix_free_operator(dealii::MatrixFree<dim, Number> const &,
                                 VectorType &       dst,
                                 VectorType const & src,
                                 Range const &      cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index, quad_index;

  InverseMassParameters data;

  // Variable coefficients not managed by this class.
  bool coefficient_is_variable;
  bool consider_inverse_coefficient;

  VariableCoefficients<dealii::VectorizedArray<Number>> const * variable_coefficients;

  // Solver and preconditioner for solving a global linear system of equations for all degrees of
  // freedom.
  std::shared_ptr<PreconditionerBase<Number>>     global_preconditioner;
  std::shared_ptr<Krylov::SolverBase<VectorType>> global_solver;

  // Iterations needed in global Krylov solver at last inverse application.
  mutable unsigned int n_iter_global_last = dealii::numbers::invalid_unsigned_int;

  // Block-Jacobi preconditioner used as cell-wise inverse for L2-conforming spaces. In this case,
  // the mass matrix is block-diagonal and a block-Jacobi preconditioner inverts the mass operator
  // exactly (up to solver tolerances). The implementation of the block-Jacobi preconditioner can be
  // matrix-based or matrix-free, depending on the parameters specified.
  std::shared_ptr<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>
    block_jacobi_preconditioner;

  // MassOperator as underlying operator for the cell-wise or global iterative solves.
  MassOperator<dim, n_components, Number> mass_operator;
};

/*
 * Inverse mass operator for H(div)-conforming space:
 *
 * This class applies the inverse mass operator by solving the mass system as a global linear system
 * of equations for all degrees of freedom. It is used in case the mass operator is not
 * block-diagonal and can not be inverted element-wise (e.g. H(div)-conforming space).
 */
// template<int dim, int n_components, typename Number>
// class InverseMassOperatorHdiv
// {
// private:
//   typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

// public:
//   void
//   initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
//              dealii::AffineConstraints<Number> const & constraints,
//              InverseMassOperatorDataHdiv const         inverse_mass_operator_data)
//   {
//     // mass operator
//     MassOperatorData<dim> mass_operator_data;
//     mass_operator_data.dof_index  = inverse_mass_operator_data.dof_index;
//     mass_operator_data.quad_index = inverse_mass_operator_data.quad_index;
//     mass_operator.initialize(matrix_free, constraints, mass_operator_data);

//     solver_control =
//       dealii::ReductionControl(inverse_mass_operator_data.parameters.solver_data.max_iter,
//                                inverse_mass_operator_data.parameters.solver_data.abs_tol,
//                                inverse_mass_operator_data.parameters.solver_data.rel_tol);
//     preconditioner_type = inverse_mass_operator_data.parameters.preconditioner;

//     if(preconditioner_type == PreconditionerMass::PointJacobi)
//     {
//       preconditioner =
//         std::make_shared<JacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
//           mass_operator, true /* initialize_preconditioner */);
//     }
//     else if(preconditioner_type == PreconditionerMass::LumpedDiagonal)
//     {
//       VectorType tmp;
//       mass_operator.initialize_dof_vector(tmp);
//       mass_operator.initialize_dof_vector(lumped_diagonal.get_vector());
//       tmp = 1.;
//       mass_operator.vmult(lumped_diagonal.get_vector(), tmp);
//       for(Number & entry : lumped_diagonal.get_vector())
//         if(entry > 1e-30)
//           entry = 1.0 / entry;
//         else
//           entry = 1.;
//     }
//   }

//   /**
//    * This function applies the inverse mass operator. Note that this function allows identical
//    dst,
//    * src vector, i.e. the function can be called like apply(dst, dst).
//    */
//   unsigned int
//   apply(VectorType & dst, VectorType const & src) const
//   {
//     VectorType temp;

//     // Note that the inverse mass operator might be called like inverse_mass.apply(dst, dst),
//     // i.e. with identical destination and source vectors. In this case, we need to make sure
//     // that the result is still correct.
//     const VectorType * src_ptr;
//     if(&dst == &src)
//     {
//       temp    = src;
//       src_ptr = &temp;
//     }
//     else
//     {
//       src_ptr = &src;
//     }

//     dealii::SolverCG<VectorType> solver(solver_control);
//     if(preconditioner_type == PreconditionerMass::None)
//     {
//       solver.solve(mass_operator, dst, *src_ptr, dealii::PreconditionIdentity());
//     }
//     else if(preconditioner_type == PreconditionerMass::PointJacobi)
//     {
//       AssertThrow(preconditioner.get() != nullptr,
//                   dealii::ExcMessage("Preconditioner not initialized!"));
//       solver.solve(mass_operator, dst, *src_ptr, *preconditioner);
//     }
//     else if(preconditioner_type == PreconditionerMass::LumpedDiagonal)
//     {
//       solver.solve(mass_operator, dst, *src_ptr, lumped_diagonal);
//     }
//     else
//       AssertThrow(false,
//                   dealii::ExcMessage(
//                     "Preconditioner type for Hdiv inverse mass matrix not recognized"));

//     return solver_control.last_step();
//   }

// private:
//   // Solver/preconditioner for mass system solving a global linear system of equations for all
//   // degrees of freedom.
//   std::shared_ptr<PreconditionerBase<Number>> preconditioner;
//   dealii::DiagonalMatrix<VectorType>          lumped_diagonal;
//   dealii::ReductionControl mutable solver_control;

//   // We need a MassOperator as underlying operator.
//   MassOperator<dim, n_components, Number> mass_operator;

//   PreconditionerMass preconditioner_type;
// };

} // namespace ExaDG

#endif /* EXADG_OPERATORS_INVERSE_MASS_OPERATOR_H_ */
