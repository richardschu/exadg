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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_

// C/C++
#include <algorithm>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/solver_control.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/solvers/enum_types.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
struct SolverData
{
  SolverData() : max_iter(1e3), abs_tol(1e-20), rel_tol(1e-6), max_krylov_size(30)
  {
  }

  SolverData(unsigned int const max_iter_,
             double const       abs_tol_,
             double const       rel_tol_,
             unsigned int const max_krylov_size_ = 30)
    : max_iter(max_iter_), abs_tol(abs_tol_), rel_tol(rel_tol_), max_krylov_size(max_krylov_size_)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Maximum number of iterations", max_iter);
    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);
    print_parameter(pcout, "Relative tolerance reference", rel_tol_reference);
    print_parameter(pcout, "Maximum size of Krylov space", max_krylov_size);
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;
  // see enum declaration
  RelTolReference rel_tol_reference = RelTolReference::InitialResidual;
  // only relevant for GMRES type solvers
  unsigned int max_krylov_size;
};

/*
 * Creates the control of an iterative solver for the system with right-hand side `rhs`.
 */
template<typename VectorType>
dealii::ReductionControl
create_solver_control(unsigned int const    max_iter,
                      double const          abs_tol,
                      double const          rel_tol,
                      RelTolReference const rel_tol_reference,
                      VectorType const &    rhs)
{
  if(rel_tol_reference == RelTolReference::RightHandSide)
  {
    // `dealii::ReductionControl` stops once
    // |r_k| <= reduce * |r_0| or |r_k| <= tol.
    // The reduction 0.0 disables the relative criterion.
    double constexpr rel_tol_impossible = 0.0;
    return dealii::ReductionControl(max_iter,
                                    std::max(abs_tol, rel_tol * rhs.l2_norm()),
                                    rel_tol_impossible);
  }
  else
  {
    return dealii::ReductionControl(max_iter, abs_tol, rel_tol);
  }
}
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_ */
