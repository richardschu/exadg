/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
 *  Schussnig
 *
 *  This file is dual-licensed under the Apache-2.0 with LLVM Exception (see
 *  https://spdx.org/licenses/Apache-2.0.html and
 *  https://spdx.org/licenses/LLVM-exception.html) and the GNU General Public
 *  License as published by the Free Software Foundation, either version 3 of
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License in the top-level LICENSE file for
 *  more details.
 *  ______________________________________________________________________
 */

#pragma once

#include <deal.II/base/floating_point_comparator.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/matrix_free.h>


namespace Helper
{
inline void
print_time(const double        time,
           const std::string & name,
           const MPI_Comm      communicator,
           const double        total_time = 0.)
{
  dealii::Utilities::MPI::MinMaxAvg data = dealii::Utilities::MPI::min_max_avg(time, communicator);

  if(dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
  {
    const unsigned int n_digits = static_cast<unsigned int>(
      std::ceil(std::log10(dealii::Utilities::MPI::n_mpi_processes(communicator))));
    std::cout << std::left << std::setw(29) << name << " " << std::setw(11) << data.min << " [p"
              << std::setw(n_digits) << data.min_index << "] " << std::setw(11) << data.avg << " "
              << std::setw(11) << data.max << " [p" << std::setw(n_digits) << data.max_index << "]";
    if(total_time > 0)
      std::cout << " " << data.avg * 100. / total_time << "%";
    std::cout << std::endl;
  }
}

// function to convert from a map, with keys associated to the buckets by
// which we sliced the index space, length chunk_size_zero_vector, and
// values equal to the slice index which are touched by the respective
// partition, to a "vectors-of-vectors" like data structure. Rather than
// using the vectors, we set up a sparsity-pattern like structure where
// one index specifies the start index (range_list_index), and the other
// the actual ranges (range_list).
inline void
convert_map_to_range_list(const unsigned int                                        n_partitions,
                          const unsigned int                                        chunk_size,
                          const std::map<unsigned int, std::vector<unsigned int>> & ranges_in,
                          std::vector<unsigned int> &                          range_list_index,
                          std::vector<std::pair<unsigned int, unsigned int>> & range_list,
                          const unsigned int                                   max_size)
{
  range_list_index.resize(n_partitions + 1);
  range_list_index[0] = 0;
  range_list.clear();
  for(unsigned int partition = 0; partition < n_partitions; ++partition)
  {
    auto it = ranges_in.find(partition);
    if(it != ranges_in.end())
    {
      for(unsigned int i = 0; i < it->second.size(); ++i)
      {
        const unsigned int first_i = i;
        while(i + 1 < it->second.size() && it->second[i + 1] == it->second[i] + 1)
          ++i;
        range_list.emplace_back(std::min(it->second[first_i] * chunk_size, max_size),
                                std::min((it->second[i] + 1) * chunk_size, max_size));
      }
      range_list_index[partition + 1] = range_list.size();
    }
    else
      range_list_index[partition + 1] = range_list_index[partition];
  }
}

template<int dim>
std::vector<unsigned int>
compute_vectorization_category(const dealii::Triangulation<dim> & tria)
{
  std::vector<unsigned int> cell_vectorization_category;
  if(tria.n_levels() > 2)
  {
    cell_vectorization_category.resize(tria.n_active_cells(),
                                       dealii::numbers::invalid_unsigned_int);
    unsigned int                             next_category = 0;
    std::array<std::vector<unsigned int>, 8> next_cells;
    std::vector<unsigned int>                sorted_next_cells;
    for(const auto & grandparent : tria.cell_iterators_on_level(tria.n_levels() - 3))
      if(grandparent->has_children())
      {
        for(auto & entry : next_cells)
          entry.clear();
        for(unsigned int c0 = 0; c0 < grandparent->n_children(); ++c0)
        {
          const auto parent = grandparent->child(c0);
          if(parent->has_children())
            for(unsigned int c = 0; c < parent->n_children(); ++c)
            {
              const auto cell = parent->child(c);
              Assert(cell->is_active(), dealii::ExcInternalError());
              if(cell->is_locally_owned())
              {
                unsigned int boundary_type = 0;
                for(unsigned int d = 0; d < dim; ++d)
                  if(cell->at_boundary(2 * d) ||
                     cell->neighbor(2 * d)->subdomain_id() != cell->subdomain_id())
                  {
                    boundary_type += dealii::Utilities::pow(2, d);
                  }
                next_cells[boundary_type].push_back(cell->active_cell_index());
              }
            }
        }
        unsigned int n_cells = 0;
        for(const std::vector<unsigned int> & cells : next_cells)
          n_cells += cells.size();
        sorted_next_cells.clear();
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[7].begin(),
                                 next_cells[7].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[6].begin(),
                                 next_cells[6].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[5].begin(),
                                 next_cells[5].end());
        unsigned int count_4 = 0, count_2 = 0, count_1 = 0;
        while(sorted_next_cells.size() % 8 != 0 && count_4 < next_cells[4].size())
        {
          sorted_next_cells.push_back(next_cells[4][count_4]);
          ++count_4;
        }
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[3].begin(),
                                 next_cells[3].end());
        // avoid having 9 cells of one kind that spills two 8-sized lanes,
        // so peel off one of each here
        if(sorted_next_cells.size() % 8 != 0)
          if(count_2 < next_cells[2].size())
          {
            sorted_next_cells.push_back(next_cells[2][count_2]);
            ++count_2;
          }
        if(sorted_next_cells.size() % 8 != 0)
          if(count_1 < next_cells[1].size())
          {
            sorted_next_cells.push_back(next_cells[1][count_1]);
            ++count_1;
          }
        while(sorted_next_cells.size() % 8 != 0 && count_1 < next_cells[1].size())
        {
          sorted_next_cells.push_back(next_cells[1][count_1]);
          ++count_1;
        }
        while(sorted_next_cells.size() % 8 != 0 && count_2 < next_cells[2].size())
        {
          sorted_next_cells.push_back(next_cells[2][count_2]);
          ++count_2;
        }
        // possibly move some insertions of category 0 in between to fill
        // up and avoid further exchange steps
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[2].begin() + count_2,
                                 next_cells[2].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[4].begin() + count_4,
                                 next_cells[4].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[1].begin() + count_1,
                                 next_cells[1].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[0].begin(),
                                 next_cells[0].end());
        AssertThrow(sorted_next_cells.size() == n_cells,
                    dealii::ExcDimensionMismatch(sorted_next_cells.size(), n_cells));

        for(unsigned int i = 0; i < n_cells; ++next_category)
          for(unsigned int j = 0; j < 16 && i < n_cells; ++j, ++i)
            cell_vectorization_category[sorted_next_cells[i]] = next_category;
      }
  }
  return cell_vectorization_category;
}

} // namespace Helper


namespace Extruded
{
using namespace dealii;
template<int dim, typename Number>
struct MappingInfo
{
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

  void
  reinit(const Mapping<dim> &                                         mapping,
         const Quadrature<1> &                                        quadrature_1d,
         const Triangulation<dim> &                                   triangulation,
         const std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> cell_level_index)
  {
    underlying_quadrature = quadrature_1d;

    FloatingPointComparator<double> comparator(triangulation.begin()->diameter() * 1e-10);
    std::map<Point<2>, std::array<int, 3>, FloatingPointComparator<double>> unique_cells(
      comparator);

    std::vector<unsigned int> geometry_index(triangulation.n_active_cells(),
                                             numbers::invalid_unsigned_int);
    // collect possible compression of Jacobian data due to extrusion in z
    // direction by checking the position of the first cell vertex in the xy
    // plane. First check locally owned cells in exactly the order matrix-free
    // loops visit them, then for ghosts to get the complete face data also in
    // parallel computations
    for(unsigned int c = 0; c < cell_level_index.size(); ++c)
      for(unsigned int v = 0;
          v < n_lanes && cell_level_index[c][v][0] != numbers::invalid_unsigned_int;
          ++v)
      {
        const typename Triangulation<dim>::active_cell_iterator cell(&triangulation,
                                                                     cell_level_index[c][v][0],
                                                                     cell_level_index[c][v][1]);
        const auto vertices = mapping.get_vertices(cell);
        Point<2>   p(vertices[0][0], vertices[0][1]);
        const auto position = unique_cells.find(p);
        if(position == unique_cells.end())
        {
          geometry_index[cell->active_cell_index()] = unique_cells.size();
          unique_cells.insert(std::make_pair(
            p, std::array<int, 3>{{(int)unique_cells.size(), cell->level(), cell->index()}}));
        }
        else
          geometry_index[cell->active_cell_index()] = position->second[0];
      }
    for(const auto & cell : triangulation.active_cell_iterators())
      if(cell->is_ghost())
      {
        const auto vertices = mapping.get_vertices(cell);
        Point<2>   p(vertices[0][0], vertices[0][1]);
        const auto position = unique_cells.find(p);
        if(position == unique_cells.end())
        {
          geometry_index[cell->active_cell_index()] = unique_cells.size();
          unique_cells.insert(std::make_pair(
            p, std::array<int, 3>{{(int)unique_cells.size(), cell->level(), cell->index()}}));
        }
        else
          geometry_index[cell->active_cell_index()] = position->second[0];
      }

    const unsigned int n_q_points_1d = quadrature_1d.size();
    const unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);
    mapping_data_index.resize(cell_level_index.size());
    face_mapping_data_index.resize(cell_level_index.size());
    for(unsigned int c = 0; c < cell_level_index.size(); ++c)
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        if(cell_level_index[c][v][0] == numbers::invalid_unsigned_int)
        {
          for(unsigned int f = 0; f < 4; ++f)
            for(unsigned int s = 0; s < 2; ++s)
              face_mapping_data_index[c][f][s][v] = face_mapping_data_index[c][f][s][0];
          continue;
        }
        const typename Triangulation<dim>::active_cell_iterator dcell(&triangulation,
                                                                      cell_level_index[c][v][0],
                                                                      cell_level_index[c][v][1]);
        mapping_data_index[c][v] = geometry_index[dcell->active_cell_index()] * n_q_points_2d;
        for(unsigned int f = 0; f < 4; ++f)
        {
          face_mapping_data_index[c][f][0][v] =
            (geometry_index[dcell->active_cell_index()] * 4 + f) * n_q_points_1d;
          const bool at_boundary           = dcell->at_boundary(f);
          const bool has_periodic_neighbor = at_boundary && dcell->has_periodic_neighbor(f);
          if(at_boundary == false || has_periodic_neighbor)
          {
            const auto neighbor =
              has_periodic_neighbor ? dcell->periodic_neighbor(f) : dcell->neighbor(f);
            const unsigned int neighbor_face_idx = has_periodic_neighbor ?
                                                     dcell->periodic_neighbor_face_no(f) :
                                                     dcell->neighbor_face_no(f);
            face_mapping_data_index[c][f][1][v] =
              (geometry_index[neighbor->active_cell_index()] * 4 + neighbor_face_idx) *
              n_q_points_1d;
          }
          else
            face_mapping_data_index[c][f][1][v] = face_mapping_data_index[c][f][0][v];
        }
      }

    quad_weights_z.resize(n_q_points_1d);
    for(unsigned int q = 0; q < n_q_points_1d; ++q)
      quad_weights_z[q] = quadrature_1d.weight(q);

    Quadrature<dim - 1> face_quadrature(quadrature_1d);
    Quadrature<2>       quadrature_2d(quadrature_1d);
    quad_weights_xy.resize(quadrature_2d.size());
    for(unsigned int q = 0; q < quadrature_2d.size(); ++q)
      quad_weights_xy[q] = quadrature_2d.weight(q);

    std::vector<Point<dim>> points(quadrature_2d.size());
    for(unsigned int i = 0; i < quadrature_2d.size(); ++i)
      for(unsigned int d = 0; d < 2; ++d)
        points[i][d] = quadrature_2d.point(i)[d];

    FE_Nothing<dim>   dummy_fe;
    FEValues<dim>     fe_values(mapping,
                            dummy_fe,
                            Quadrature<dim>(points),
                            update_jacobians | update_jacobian_grads | update_quadrature_points);
    FEFaceValues<dim> fe_face_values(mapping,
                                     dummy_fe,
                                     face_quadrature,
                                     update_quadrature_points | update_jacobians |
                                       update_JxW_values | update_normal_vectors |
                                       update_jacobian_grads);

    quadrature_points.resize(n_q_points_2d * unique_cells.size());
    jacobians_xy.resize(n_q_points_2d * unique_cells.size());
    inv_jacobians_xy.resize(n_q_points_2d * unique_cells.size());
    cell_JxW_xy.resize(n_q_points_2d * unique_cells.size());
    jacobian_grads.resize(n_q_points_2d * unique_cells.size());

    face_quadrature_points.resize(4 * n_q_points_1d * unique_cells.size());
    face_jacobians_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_jacobian_grads.resize(4 * n_q_points_1d * unique_cells.size());
    face_normal_vector_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_jxn_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_JxW_xy.resize(4 * n_q_points_1d * unique_cells.size());

    ip_penalty_factors.resize(unique_cells.size());
    for(const auto & [_, index] : unique_cells)
    {
      const typename Triangulation<dim>::cell_iterator cell(&triangulation, index[1], index[2]);
      fe_values.reinit(cell);
      AssertDimension(geometry_index[cell->active_cell_index()], index[0]);
      double cell_volume = 0;
      for(unsigned int q = 0; q < n_q_points_2d; ++q)
      {
        const DerivativeForm<1, dim, dim> jacobian = fe_values.jacobian(q);
        cell_volume += jacobian.determinant() * face_quadrature.weight(q);
        const DerivativeForm<1, dim, dim> inv_jacobian = jacobian.covariant_form();
        const unsigned int                data_idx     = index[0] * n_q_points_2d + q;
        cell_JxW_xy[data_idx] =
          (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]) *
          face_quadrature.weight(q);
        if(dim == 3)
        {
          h_z         = jacobian[2][2];
          h_z_inverse = 1. / h_z;
        }
        for(unsigned int d = 0; d < 2; ++d)
          for(unsigned int e = 0; e < 2; ++e)
          {
            jacobians_xy[data_idx][d][e]     = jacobian[d][e];
            inv_jacobians_xy[data_idx][d][e] = inv_jacobian[d][e];
          }
        for(unsigned int d = 0; d < 2; ++d)
          quadrature_points[data_idx][d] = fe_values.quadrature_point(q)[d];

        const auto jac_grad = fe_values.jacobian_grad(q);
        for(unsigned int d = 0; d < 2; ++d)
        {
          jacobian_grads[data_idx][0][d] = jac_grad[d][0][0];
          jacobian_grads[data_idx][1][d] = jac_grad[d][1][1];
          jacobian_grads[data_idx][2][d] = jac_grad[d][0][1];
        }
      }

      double surface_area = 0;
      for(unsigned int face = 0; face < 4; ++face)
      {
        fe_face_values.reinit(cell, face);
        double face_factor =
          (cell->at_boundary(face) && !cell->has_periodic_neighbor(face)) ? 1. : 0.5;
        for(unsigned int q = 0; q < face_quadrature.size(); ++q)
          surface_area += face_factor * fe_face_values.JxW(q);

        for(unsigned int qx = 0; qx < n_q_points_1d; ++qx)
        {
          // take switched coordinates xz on y faces into account
          const unsigned int q        = (face < 2 || dim == 2) ? qx : qx * n_q_points_1d;
          const unsigned int data_idx = (index[0] * 4 + face) * n_q_points_1d + qx;
          const auto         jac      = fe_face_values.jacobian(q);
          const auto         inv_jac  = jac.covariant_form();
          for(unsigned int d = 0; d < 2; ++d)
            face_quadrature_points[data_idx][d] = fe_face_values.quadrature_point(q)[d];
          for(unsigned int d = 0; d < 2; ++d)
            face_jxn_xy[data_idx][d] = inv_jac[0][d] * fe_face_values.normal_vector(q)[0] +
                                       inv_jac[1][d] * fe_face_values.normal_vector(q)[1];
          face_JxW_xy[data_idx] = std::sqrt(jac[0][1 - face / 2] * jac[0][1 - face / 2] +
                                            jac[1][1 - face / 2] * jac[1][1 - face / 2]) *
                                  quadrature_1d.weight(qx);
          for(unsigned int d = 0; d < 2; ++d)
            for(unsigned int e = 0; e < 2; ++e)
              face_jacobians_xy[data_idx][d][e] = jac[d][e];
          const auto jac_grad = fe_face_values.jacobian_grad(q);
          for(unsigned int d = 0; d < 2; ++d)
          {
            face_jacobian_grads[data_idx][0][d] = jac_grad[d][0][0];
            face_jacobian_grads[data_idx][1][d] = jac_grad[d][1][1];
            face_jacobian_grads[data_idx][2][d] = jac_grad[d][0][1];
            face_normal_vector_xy[data_idx][d]  = fe_face_values.normal_vector(q)[d];
          }
        }
      }
      // take the two faces in z direction into account; they are always in
      // periodic direction so do not check for boundary
      if(dim == 3)
        surface_area += 2 * 0.5 * cell_volume / h_z;

      ip_penalty_factors[index[0]] = surface_area / cell_volume;
    }
  }

  std::size_t
  memory_consumption() const
  {
    return MemoryConsumption::memory_consumption(mapping_data_index) +
           MemoryConsumption::memory_consumption(quadrature_points) +
           MemoryConsumption::memory_consumption(jacobians_xy) +
           MemoryConsumption::memory_consumption(inv_jacobians_xy) +
           MemoryConsumption::memory_consumption(jacobian_grads) +
           MemoryConsumption::memory_consumption(cell_JxW_xy) +
           MemoryConsumption::memory_consumption(face_mapping_data_index) +
           MemoryConsumption::memory_consumption(face_quadrature_points) +
           MemoryConsumption::memory_consumption(face_normal_vector_xy) +
           MemoryConsumption::memory_consumption(face_jacobians_xy) +
           MemoryConsumption::memory_consumption(face_jacobian_grads) +
           MemoryConsumption::memory_consumption(face_jxn_xy) +
           MemoryConsumption::memory_consumption(face_JxW_xy) +
           MemoryConsumption::memory_consumption(quad_weights_xy) +
           MemoryConsumption::memory_consumption(quad_weights_z) +
           MemoryConsumption::memory_consumption(ip_penalty_factors) + sizeof(this);
  }

  Number                                            h_z;
  Number                                            h_z_inverse;
  std::vector<std::array<unsigned int, n_lanes>>    mapping_data_index;
  AlignedVector<Point<2, Number>>                   quadrature_points;
  AlignedVector<Tensor<2, 2, Number>>               jacobians_xy;
  AlignedVector<Tensor<2, 2, Number>>               inv_jacobians_xy;
  AlignedVector<Number>                             cell_JxW_xy;
  AlignedVector<Tensor<1, 3, Tensor<1, 2, Number>>> jacobian_grads;
  std::vector<ndarray<unsigned int, 4, 2, n_lanes>> face_mapping_data_index;
  AlignedVector<Point<2, Number>>                   face_quadrature_points;
  AlignedVector<Tensor<1, 2, Number>>               face_normal_vector_xy;
  AlignedVector<Tensor<2, 2, Number>>               face_jacobians_xy;
  AlignedVector<Tensor<1, 3, Tensor<1, 2, Number>>> face_jacobian_grads;
  AlignedVector<Tensor<1, 2, Number>>               face_jxn_xy;
  AlignedVector<Number>                             face_JxW_xy;
  std::vector<Number>                               quad_weights_xy;
  std::vector<Number>                               quad_weights_z;
  std::vector<Number>                               ip_penalty_factors;
  Quadrature<1>                                     underlying_quadrature;
};

} // namespace Extruded
