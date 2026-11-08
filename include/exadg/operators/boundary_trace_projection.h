/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
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

#ifndef EXADG_OPERATORS_BOUNDARY_TRACE_PROJECTION_H_
#define EXADG_OPERATORS_BOUNDARY_TRACE_PROJECTION_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

// C/C++
#include <functional>
#include <set>

namespace ExaDG
{
/*
 * L2 projection onto the trace of a finite element space on the boundaries
 * `boundary_ids`: solves M dst = rhs with the conjugate gradient method, where
 * M is the boundary mass matrix of these boundaries, integrated with the
 * quadrature rule `quad_index` in the configuration described by the mapping
 * of `matrix_free`. The vector `rhs` contains the integrals of the function to
 * be projected tested with the shape functions. Degrees of freedom without
 * support on these boundaries are not part of the trace space and set to zero.
 * The constraints of `dof_index` are applied as in any matrix-free operator.
 * The vector `dst` is initialized with the layout of `dof_index`. Returns the
 * number of iterations.
 */
template<int dim, int n_components, typename Number>
unsigned int
solve_boundary_mass_system(dealii::LinearAlgebra::distributed::Vector<Number> &       dst,
                           dealii::LinearAlgebra::distributed::Vector<Number> const & rhs,
                           dealii::MatrixFree<dim, Number> const &                    matrix_free,
                           unsigned int const                                         dof_index,
                           unsigned int const                                         quad_index,
                           std::set<dealii::types::boundary_id> const &               boundary_ids,
                           double const relative_tolerance)
{
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  // degrees of freedom with support on the boundaries
  VectorType trace_dofs;
  matrix_free.initialize_dof_vector(trace_dofs, dof_index);
  {
    dealii::DoFHandler<dim> const & dof_handler = matrix_free.get_dof_handler(dof_index);

    std::vector<dealii::types::global_dof_index> dof_indices;
    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(not cell->is_locally_owned())
        continue;

      for(unsigned int const f : cell->face_indices())
      {
        if(cell->at_boundary(f) and
           boundary_ids.find(cell->face(f)->boundary_id()) != boundary_ids.end())
        {
          dealii::FiniteElement<dim> const & fe = cell->get_fe();
          dof_indices.resize(fe.n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);
          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          {
            if(fe.has_support_on_face(i, f))
              trace_dofs(dof_indices[i]) = 1.0;
          }
        }
      }
    }
    trace_dofs.compress(dealii::VectorOperation::add);

    // indicator independent of the number of contributing cells
    for(unsigned int i = 0; i < trace_dofs.locally_owned_size(); ++i)
      trace_dofs.local_element(i) = trace_dofs.local_element(i) > 0.0 ? 1.0 : 0.0;
  }

  // boundary mass operator, extended by the identity for all degrees of
  // freedom not in the trace space
  dealii::FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator(matrix_free,
                                                                        true,
                                                                        dof_index,
                                                                        quad_index);

  auto const apply_boundary_mass = [&](VectorType & dst_, VectorType const & src_) {
    dst_ = 0.0;
    src_.update_ghost_values();

    for(unsigned int face = matrix_free.n_inner_face_batches();
        face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
        ++face)
    {
      if(boundary_ids.find(matrix_free.get_boundary_id(face)) == boundary_ids.end())
        continue;

      integrator.reinit(face);
      integrator.gather_evaluate(src_, dealii::EvaluationFlags::values);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        integrator.submit_value(integrator.get_value(q), q);
      integrator.integrate_scatter(dealii::EvaluationFlags::values, dst_);
    }

    dst_.compress(dealii::VectorOperation::add);
    src_.zero_out_ghost_values();
  };

  // The row sums of the boundary mass matrix serve as diagonal preconditioner.
  // They are positive for the trace degrees of freedom of nodal elements, a
  // zero row sum indicates a constrained degree of freedom.
  VectorType row_sums;
  matrix_free.initialize_dof_vector(row_sums, dof_index);
  apply_boundary_mass(row_sums, trace_dofs);

  VectorType is_trace_dof;
  matrix_free.initialize_dof_vector(is_trace_dof, dof_index);
  dealii::DiagonalMatrix<VectorType> preconditioner;
  matrix_free.initialize_dof_vector(preconditioner.get_vector(), dof_index);
  for(unsigned int i = 0; i < row_sums.locally_owned_size(); ++i)
  {
    bool const trace_dof = trace_dofs.local_element(i) > 0.0 and row_sums.local_element(i) > 0.0;

    is_trace_dof.local_element(i) = trace_dof ? 1.0 : 0.0;
    preconditioner.get_vector().local_element(i) =
      trace_dof ? 1.0 / row_sums.local_element(i) : 1.0;
  }

  // M restricted to the trace space plus the identity on its complement
  struct Operator
  {
    void
    vmult(VectorType & dst_, VectorType const & src_) const
    {
      src_trace = src_;
      for(unsigned int i = 0; i < src_.locally_owned_size(); ++i)
        src_trace.local_element(i) *= is_trace_dof_.local_element(i);

      apply(dst_, src_trace);

      for(unsigned int i = 0; i < dst_.locally_owned_size(); ++i)
      {
        if(is_trace_dof_.local_element(i) == 0.0)
          dst_.local_element(i) = src_.local_element(i);
      }
    }

    std::function<void(VectorType &, VectorType const &)> apply;
    VectorType const &                                    is_trace_dof_;
    VectorType &                                          src_trace;
  };
  VectorType src_trace;
  matrix_free.initialize_dof_vector(src_trace, dof_index);
  Operator const boundary_mass{apply_boundary_mass, is_trace_dof, src_trace};

  // right-hand side restricted to the trace space, `rhs` has the same locally
  // owned degrees of freedom
  VectorType rhs_trace;
  matrix_free.initialize_dof_vector(rhs_trace, dof_index);
  AssertThrow(rhs.locally_owned_size() == rhs_trace.locally_owned_size(),
              dealii::ExcMessage("Vector layout does not match dof_index."));
  for(unsigned int i = 0; i < rhs.locally_owned_size(); ++i)
    rhs_trace.local_element(i) = is_trace_dof.local_element(i) * rhs.local_element(i);

  matrix_free.initialize_dof_vector(dst, dof_index);

  double const rhs_norm = rhs_trace.l2_norm();
  if(rhs_norm == 0.0)
    return 0;

  dealii::SolverControl        solver_control(1000, relative_tolerance * rhs_norm);
  dealii::SolverCG<VectorType> solver(solver_control);
  solver.solve(boundary_mass, dst, rhs_trace, preconditioner);

  return solver_control.last_step();
}

} // namespace ExaDG

#endif /* EXADG_OPERATORS_BOUNDARY_TRACE_PROJECTION_H_ */
