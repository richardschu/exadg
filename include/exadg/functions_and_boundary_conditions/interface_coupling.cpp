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

// deal.II
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/postprocessor/write_output.h>

namespace ExaDG
{
template<int rank, int dim, typename Number>
InterfaceCoupling<rank, dim, Number>::InterfaceCoupling() : dof_handler_src(nullptr)
{
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::setup(
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>> interface_data_dst_,
  dealii::DoFHandler<dim> const &                            dof_handler_src_,
  dealii::Mapping<dim> const &                               mapping_src_,
  std::vector<bool> const &                                  marked_vertices_src_,
  double const                                               tolerance_)
{
  AssertThrow(interface_data_dst_.get(),
              dealii::ExcMessage("Received uninitialized variable. Aborting."));

  if(marked_vertices_src_.size() > 0)
  {
    AssertThrow(marked_vertices_src_.size() ==
                  (unsigned int)dof_handler_src_.get_triangulation().n_vertices(),
                dealii::ExcMessage("Vector marked_vertices_src_ has invalid size."));
  }

  interface_data_dst = interface_data_dst_;
  dof_handler_src    = &dof_handler_src_;

  for(auto quad_index : interface_data_dst->get_quad_indices())
  {
    // exchange quadrature points with their owners
    typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData rpe_data;
    rpe_data.tolerance              = tolerance_;
    rpe_data.enforce_unique_mapping = false;
    rpe_data.rtree_level            = 0;
    rpe_data.marked_vertices        = [marked_vertices_src_]() { return marked_vertices_src_; };

    map_evaluator.emplace(
      quad_index, std::make_unique<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(rpe_data));

    auto const * points = &interface_data_dst->get_array_q_points(quad_index);

    map_evaluator[quad_index]->reinit(*points, dof_handler_src_.get_triangulation(), mapping_src_);

    if(not map_evaluator[quad_index]->all_points_found())
    {
      // get vector of points not found
      std::vector<dealii::Point<dim>> points_not_found;
      points_not_found.reserve(points->size());
      unsigned int n_points_not_found = 0;
      for(unsigned int i = 0; i < points->size(); ++i)
      {
        if(not map_evaluator[quad_index]->point_found(i))
        {
          n_points_not_found += 1;
          points_not_found.push_back((*points)[i]);
        }
      }
      MPI_Comm const mpi_comm = dof_handler_src->get_mpi_communicator();
      n_points_not_found      = dealii::Utilities::MPI::sum(n_points_not_found, mpi_comm);

      std::string const file_name =
        "interface_coupling_quad_index_" + dealii::Utilities::to_string(quad_index);

      write_grid(dof_handler_src->get_triangulation(),
                 mapping_src_,
                 4,
                 "./",
                 file_name,
                 0 /* counter */,
                 mpi_comm);

      write_points_in_dummy_triangulation(
        points_not_found, "./", file_name, 0 /* counter */, mpi_comm);

      AssertThrow(map_evaluator[quad_index]->all_points_found(),
                  dealii::ExcMessage(std::string("Setup of InterfaceCoupling was not successful: " +
                                                 std::to_string(n_points_not_found) +
                                                 " points have not been found.")));
    }
  }
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::update_data(VectorType const & dof_vector_src)
{
  dof_vector_src.update_ghost_values();

  for(auto quadrature_index : interface_data_dst->get_quad_indices())
  {
    std::vector<
      typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const
      solution_src =
        dealii::VectorTools::point_values<n_components>(*map_evaluator[quadrature_index],
                                                        *dof_handler_src,
                                                        dof_vector_src,
                                                        dealii::VectorTools::EvaluationFlags::avg);

    auto & array_solution_dst = interface_data_dst->get_array_solution(quadrature_index);

    AssertThrow(solution_src.size() == array_solution_dst.size(),
                dealii::ExcMessage("Vectors must have the same length."));

    for(unsigned int i = 0; i < solution_src.size(); ++i)
      array_solution_dst[i] = solution_src[i];
  }
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::integrate_transposed(
  VectorType &                                           dof_vector_src,
  quad_index const                                       quad_index_dst,
  std::vector<dealii::Tensor<rank, dim, double>> const & values_dst,
  dealii::AffineConstraints<Number> const &              constraints_src) const
{
  using PointEvaluation = dealii::FEPointEvaluation<n_components, dim, dim, Number>;
  using value_type      = typename PointEvaluation::value_type;

  auto const it = map_evaluator.find(quad_index_dst);
  AssertThrow(it != map_evaluator.end(),
              dealii::ExcMessage("Specified quad_index_dst = " + std::to_string(quad_index_dst) +
                                 " does not exist in map_evaluator."));
  dealii::Utilities::MPI::RemotePointEvaluation<dim> const & evaluator = *it->second;

  // Points found in several cells are evaluated as the average over these
  // cells in update_data(). The transpose distributes the value of such a point
  // equally to these cells.
  std::vector<unsigned int> const & point_ptrs = evaluator.get_point_ptrs();
  AssertThrow(values_dst.size() + 1 == point_ptrs.size(),
              dealii::ExcMessage("Vectors must have the same length."));

  std::vector<value_type> values(values_dst.size());
  for(unsigned int i = 0; i < values_dst.size(); ++i)
  {
    unsigned int const n_cells = point_ptrs[i + 1] - point_ptrs[i];
    if(n_cells > 0)
      values[i] = value_type(values_dst[i]) / static_cast<Number>(n_cells);
  }

  dof_vector_src = 0.0;
  dof_vector_src.zero_out_ghost_values();

  dealii::FiniteElement<dim> const & fe = dof_handler_src->get_fe();
  PointEvaluation point_evaluation(evaluator.get_mapping(), fe, dealii::update_values);

  dealii::Vector<Number>                       cell_vector(fe.n_dofs_per_cell());
  std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

  auto const integrate_function =
    [&](dealii::ArrayView<value_type const> const &                                   values_cells,
        typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::CellData const & cell_data) {
      for(auto const cell_index : cell_data.cell_indices())
      {
        auto const cell =
          cell_data.get_active_cell_iterator(cell_index)->as_dof_handler_iterator(*dof_handler_src);

        auto const unit_points = cell_data.get_unit_points(cell_index);
        auto const values_cell = cell_data.get_data_view(cell_index, values_cells);

        point_evaluation.reinit(cell, unit_points);
        for(unsigned int q = 0; q < unit_points.size(); ++q)
          point_evaluation.submit_value(values_cell[q], q);

        point_evaluation.test_and_sum(dealii::make_array_view(cell_vector),
                                      dealii::EvaluationFlags::values);

        cell->get_dof_indices(dof_indices);
        constraints_src.distribute_local_to_global(cell_vector, dof_indices, dof_vector_src);
      }
    };

  std::vector<value_type> buffer;
  evaluator.template process_and_evaluate<value_type>(values, buffer, integrate_function);

  dof_vector_src.compress(dealii::VectorOperation::add);
}

template class InterfaceCoupling<0, 2, float>;
template class InterfaceCoupling<1, 2, float>;
template class InterfaceCoupling<0, 3, float>;
template class InterfaceCoupling<1, 3, float>;

template class InterfaceCoupling<0, 2, double>;
template class InterfaceCoupling<1, 2, double>;
template class InterfaceCoupling<0, 3, double>;
template class InterfaceCoupling<1, 3, double>;

} // namespace ExaDG
