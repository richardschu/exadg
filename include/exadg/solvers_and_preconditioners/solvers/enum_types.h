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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_ENUM_TYPES_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
/*
 * Reference norm of the relative solver tolerance:
 * an iterative solver stops once the residual norm
 * |b - A x_k| drops below max(abs_tol, rel_tol * reference), where
 *
 *  - `InitialResidual`: reference = |b - A x_0|.
 *    A better initial guess x_0 tightens the target.
 *
 *  - `RightHandSide`: reference = |b|.
 *    A better initial guess saves iterations.
 *
 * Both are identical for a zero initial guess.
 */
enum class RelTolReference
{
  InitialResidual,
  RightHandSide
};

namespace Elementwise
{
/*
 * Elementwise solver for block Jacobi preconditioner
 */
enum class Solver
{
  Undefined,
  CG,
  GMRES
};

} // namespace Elementwise
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_ENUM_TYPES_H_ */
