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

// ExaDG
#include <exadg/operators/operator_base.templates.h>

namespace ExaDG
{
template class OperatorBase<3, double, 1>;
template class OperatorBase<3, double, 3>;
template class OperatorBase<3, double, 4>;
template class OperatorBase<3, double, 5>;

#ifdef DEAL_II_WITH_PETSC
typedef typename dealii::PETScWrappers::MPI::SparseMatrix PETScSparseMatrix;

template void
OperatorBase<3, double, 1>::internal_calculate_system_matrix(PETScSparseMatrix &) const;
template void
OperatorBase<3, double, 2>::internal_calculate_system_matrix(PETScSparseMatrix &) const;
template void
OperatorBase<3, double, 3>::internal_calculate_system_matrix(PETScSparseMatrix &) const;
template void
OperatorBase<3, double, 4>::internal_calculate_system_matrix(PETScSparseMatrix &) const;
template void
OperatorBase<3, double, 5>::internal_calculate_system_matrix(PETScSparseMatrix &) const;
#endif

#ifdef DEAL_II_WITH_TRILINOS
typedef typename dealii::TrilinosWrappers::SparseMatrix TrilinosSparseMatrix;

template void
OperatorBase<3, double, 1>::internal_calculate_system_matrix(TrilinosSparseMatrix &) const;
template void
OperatorBase<3, double, 2>::internal_calculate_system_matrix(TrilinosSparseMatrix &) const;
template void
OperatorBase<3, double, 3>::internal_calculate_system_matrix(TrilinosSparseMatrix &) const;
template void
OperatorBase<3, double, 4>::internal_calculate_system_matrix(TrilinosSparseMatrix &) const;
template void
OperatorBase<3, double, 5>::internal_calculate_system_matrix(TrilinosSparseMatrix &) const;
#endif

} // namespace ExaDG
