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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_
#define EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/operators/inverse_mass_parameters.h>

namespace ExaDG
{
namespace GridToGridProjection
{
// Parameters controlling the grid-to-grid projection.
template<int dim>
struct GridToGridProjectionData
{
  GridToGridProjectionData()
    : rpe_data(),
      solver_data(SolverData(1e3, 1e-20, 1e-6, LinearSolver::CG)),
      preconditioner(PreconditionerMass::PointJacobi),
      amg_data(AMGData()),
      additional_quadrature_points(1)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "RPE tolerance", rpe_data.tolerance);
    print_parameter(pcout, "RPE enforce unique mapping", rpe_data.enforce_unique_mapping);
    print_parameter(pcout, "RPE rtree level", rpe_data.rtree_level);

    // These parameters play only a role if an iterative scheme is used for projection.
    // That is for `InverseMassType != InverseMassType::MatrixfreeOperator` determined at runtime.
    solver_data.print(pcout);
    print_parameter(pcout, "Preconditioner", preconditioner);
    amg_data.print(pcout);
  }

  typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData rpe_data;
  SolverData                                                                  solver_data;
  PreconditionerMass                                                          preconditioner;
  AMGData                                                                     amg_data;

  // Number of additional integration points used for sampling the source grid.
  // The default `additional_quadrature_points = 1` considers `fe_degree + 1` quadrature points in
  // 1D using the `fe_degree` of the target grid's finite element.
  unsigned int additional_quadrature_points;
};

/**
 * Utility function to perform matrix-free grid-to-grid projection. We assume we only have a single
 * `dealii::FiniteElement` per `dealii::DoFHandler`. This function requires a suitable `MatrixFree`
 * object.
 */
template<int dim, typename Number, typename VectorType>
void
do_grid_to_grid_projection(
  std::shared_ptr<dealii::Mapping<dim> const> const &  source_mapping,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & target_matrix_free,
  std::vector<std::vector<VectorType *>> & target_vectors_per_dof_handler,
  GridToGridProjectionData<dim> const &    data);

/**
 * Same as the function above, but creates a suitable `MatrixFree` object, which is expensive.
 */
template<int dim, typename Number, typename VectorType>
void
grid_to_grid_projection(
  std::shared_ptr<dealii::Mapping<dim> const> const &  source_mapping,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::shared_ptr<dealii::Mapping<dim> const> const &  target_mapping,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  std::vector<std::vector<VectorType *>> &             target_vectors_per_dof_handler,
  GridToGridProjectionData<dim> const &                data);

} // namespace GridToGridProjection
} // namespace ExaDG

#endif /* EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_ */
