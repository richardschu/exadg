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
#include <deal.II/base/memory_space_data.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>

#include "momentum_operator_rt.h"


namespace LaplaceOperator
{
using namespace dealii;



template<int dim, typename Number = double>
class LaplaceOperatorDG : public EnableObserverPointer
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  LaplaceOperatorDG() = default;

  template<typename OtherNumber>
  void
  reinit(const Mapping<dim> &                   mapping,
         const DoFHandler<dim> &                dof_handler,
         const AffineConstraints<OtherNumber> & constraints,
         const std::vector<unsigned int> &      cell_vectorization_category,
         const Quadrature<1> &                  quadrature,
         const MPI_Comm                         communicator_shared = MPI_COMM_SELF)
  {
    this->dof_handler                                   = &dof_handler;
    const FiniteElement<dim> &                       fe = dof_handler.get_fe();
    MatrixFree<dim, Number>                          matrix_free;
    typename MatrixFree<dim, Number>::AdditionalData mf_data;
    mf_data.cell_vectorization_category          = cell_vectorization_category;
    mf_data.cell_vectorization_categories_strict = true;
    mf_data.overlap_communication_computation    = false;
    mf_data.initialize_mapping                   = false;
    matrix_free.reinit(MappingQ1<dim>(), dof_handler, constraints, quadrature, mf_data);
    {
      std::array<unsigned int, n_lanes> default_dof_indices;
      for(unsigned int & a : default_dof_indices)
        a = numbers::invalid_unsigned_int;
      dof_indices.resize(matrix_free.n_cell_batches(), default_dof_indices);
    }
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        dof_indices[cell][v] =
          matrix_free.get_dof_info()
            .dof_indices_contiguous[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                   [cell * n_lanes + v];
    partitioner = std::make_shared<Utilities::MPI::Partitioner>(dof_handler.locally_owned_dofs(),
                                                                dof_handler.get_mpi_communicator());

    {
      ndarray<unsigned int, 2 * dim, n_lanes> default_argument;
      for(unsigned int i = 0; i < 2 * dim; ++i)
        for(unsigned int j = 0; j < n_lanes; ++j)
          default_argument[i][j] = numbers::invalid_unsigned_int;
      neighbor_cells.resize(matrix_free.n_cell_batches(), default_argument);
      mpi_exchange_data_on_faces.resize(matrix_free.n_cell_batches(), default_argument);
    }
    {
      IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

      std::vector<unsigned int> cell_indices(dof_handler.get_triangulation().n_active_cells(),
                                             numbers::invalid_unsigned_int);
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < n_lanes; ++v)
          cell_indices[matrix_free.get_cell_iterator(cell, v)->active_cell_index()] =
            cell * VectorizedArray<Number>::size() + v;

      std::map<unsigned int, std::vector<std::array<types::global_dof_index, 5>>> proc_neighbors;
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        {
          const auto dcell = matrix_free.get_cell_iterator(cell, v);
          for(unsigned int f = 0; f < 2 * dim; ++f)
          {
            const bool at_boundary           = dcell->at_boundary(f);
            const bool has_periodic_neighbor = at_boundary && dcell->has_periodic_neighbor(f);
            if(at_boundary == false || has_periodic_neighbor)
            {
              const auto neighbor =
                has_periodic_neighbor ? dcell->periodic_neighbor(f) : dcell->neighbor(f);
              if(neighbor->is_locally_owned())
              {
                AssertIndexRange(neighbor->active_cell_index(), cell_indices.size());
                neighbor_cells[cell][f][v] = cell_indices[neighbor->active_cell_index()];
              }
              else
              {
                std::array<types::global_dof_index, 5> neighbor_data;
                neighbor_data[0] = cell * n_lanes + v;
                neighbor_data[1] = has_periodic_neighbor ? dcell->periodic_neighbor_face_no(f) :
                                                           dcell->neighbor_face_no(f);
                neighbor_data[2] = dcell->global_active_cell_index();
                neighbor_data[3] = neighbor->global_active_cell_index();
                neighbor_data[4] = f;
                proc_neighbors[neighbor->subdomain_id()].push_back(neighbor_data);
                // set dummy
                neighbor_cells[cell][f][v] = cell_indices[dcell->active_cell_index()];
              }
            }
          }
        }

      const unsigned int nn = fe.degree + 1;
      // data: projected to face, 2 because of values and derivatives
      const unsigned int data_per_face = 2 * Utilities::pow(nn, dim - 1);

      // find out how the neighbor wants us to send the data -> we sort by the
      // global cell index. We use a reverse order of processes as the data
      // access might favor nearby access to cell data.
      send_data_process.clear();
      send_data_cell_index.clear();
      send_data_face_index.clear();
      {
        std::size_t sizes = 0;
        for(const auto & it : proc_neighbors)
          sizes += it.second.size();
        send_data_cell_index.reserve(sizes);
        send_data_face_index.reserve(sizes);
      }
      unsigned int offset = 0;
      for(auto it = proc_neighbors.rbegin(); it != proc_neighbors.rend(); ++it)
      {
        std::sort(it->second.begin(),
                  it->second.end(),
                  [](const std::array<types::global_dof_index, 5> & a,
                     const std::array<types::global_dof_index, 5> & b)
                  {
                    if(a[4] < b[4])
                      return true;
                    else if(a[4] == b[4] && a[3] < b[3])
                      return true;
                    else
                      return false;
                  });
        send_data_process.emplace_back(it->first, it->second.size());
        for(unsigned int i = 0; i < it->second.size(); ++i)
        {
          send_data_cell_index.emplace_back(it->second[i][0]);
          send_data_face_index.emplace_back(it->second[i][4]);
        }

        // finally figure out where to read the imported face data from,
        // which is done by sorting in the way the neighboring process would
        // have run the sorting we just did a few lines up
        std::sort(it->second.begin(),
                  it->second.end(),
                  [](const std::array<types::global_dof_index, 5> & a,
                     const std::array<types::global_dof_index, 5> & b)
                  {
                    if(a[1] < b[1])
                      return true;
                    else if(a[1] == b[1] && a[2] < b[2])
                      return true;
                    else
                      return false;
                  });

        for(unsigned int i = 0; i < it->second.size(); ++i)
        {
          const unsigned int cell                               = it->second[i][0] / n_lanes;
          const unsigned int v                                  = it->second[i][0] % n_lanes;
          mpi_exchange_data_on_faces[cell][it->second[i][4]][v] = offset;

          offset += data_per_face;
        }
      }
      import_values.clear();
      import_values.resize_fast(send_data_cell_index.size() * data_per_face);
      export_values.clear();
      export_values.resize_fast(send_data_cell_index.size() * data_per_face);
    }

    compute_vector_access_pattern();

    FloatingPointComparator<double> comparator(dof_handler.get_triangulation().begin()->diameter() *
                                               1e-10);
    std::map<Point<2>, std::array<int, 3>, FloatingPointComparator<double>> unique_cells(
      comparator);

    std::vector<unsigned int> geometry_index(dof_handler.get_triangulation().n_active_cells(),
                                             numbers::invalid_unsigned_int);
    // collect possible compression of Jacobian data due to extrusion in z
    // direction by checking the position of the first cell vertex in the xy
    // plane. First check locally owned cells in exactly the order matrix-free
    // loops visit them, then for ghosts to get the complete face data also in
    // parallel computations
    for(unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(c); ++v)
      {
        const auto cell     = matrix_free.get_cell_iterator(c, v);
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
    for(const auto & cell : dof_handler.active_cell_iterators())
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

    const unsigned int n_q_points_1d = matrix_free.get_shape_info().data[0].n_q_points_1d;
    const unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);
    mapping_data_index.resize(matrix_free.n_cell_batches());
    face_mapping_data_index.resize(matrix_free.n_cell_batches());
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
      {
        const auto dcell            = matrix_free.get_cell_iterator(cell, v);
        mapping_data_index[cell][v] = geometry_index[dcell->active_cell_index()] * n_q_points_2d;
        for(unsigned int f = 0; f < 4; ++f)
        {
          face_mapping_data_index[cell][f][0][v] =
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
            face_mapping_data_index[cell][f][1][v] =
              (geometry_index[neighbor->active_cell_index()] * 4 + neighbor_face_idx) *
              n_q_points_1d;
          }
          else
            face_mapping_data_index[cell][f][1][v] = face_mapping_data_index[cell][f][0][v];
        }
      }
      for(unsigned int v = matrix_free.n_active_entries_per_cell_batch(cell); v < n_lanes; ++v)
        for(unsigned int f = 0; f < 4; ++f)
          for(unsigned int s = 0; s < 2; ++s)
            face_mapping_data_index[cell][f][s][v] = face_mapping_data_index[cell][f][s][0];
    }

    QGauss<1>               quadrature_1d(n_q_points_1d);
    Quadrature<dim - 1>     face_quadrature(quadrature_1d);
    std::vector<Point<dim>> points(face_quadrature.size());
    for(unsigned int i = 0; i < face_quadrature.size(); ++i)
      for(unsigned int d = 0; d < 2; ++d)
        points[i][d] = face_quadrature.point(i)[d];

    FE_Nothing<dim>   dummy_fe;
    FEValues<dim>     fe_values(mapping, dummy_fe, Quadrature<dim>(points), update_jacobians);
    FEFaceValues<dim> fe_face_values(mapping,
                                     dummy_fe,
                                     face_quadrature,
                                     update_jacobians | update_JxW_values | update_normal_vectors);

    jacobians_xy.resize(n_q_points_2d * unique_cells.size());
    cell_JxW_xy.resize(n_q_points_2d * unique_cells.size());
    face_jxn_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_JxW_xy.resize(4 * n_q_points_1d * unique_cells.size());

    ip_penalty_factors.resize(unique_cells.size());
    for(const auto & [_, index] : unique_cells)
    {
      const typename Triangulation<dim>::cell_iterator cell(&dof_handler.get_triangulation(),
                                                            index[1],
                                                            index[2]);
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
            jacobians_xy[data_idx][d][e] = inv_jacobian[d][e];
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
            face_jxn_xy[data_idx][d] = inv_jac[0][d] * fe_face_values.normal_vector(q)[0] +
                                       inv_jac[1][d] * fe_face_values.normal_vector(q)[1];
          face_JxW_xy[data_idx] = std::sqrt(jac[0][1 - face / 2] * jac[0][1 - face / 2] +
                                            jac[1][1 - face / 2] * jac[1][1 - face / 2]) *
                                  quadrature_1d.weight(qx);
        }
      }
      // take the two faces in z direction into account; they are always in
      // periodic direction so do not check for boundary
      if(dim == 3)
        surface_area += 2 * 0.5 * cell_volume / h_z;

      ip_penalty_factors[index[0]] = surface_area / cell_volume;
    }

    {
      quad_weights_h_z.resize(n_q_points_1d);
      QGauss<1> quad(n_q_points_1d);
      for(unsigned int q = 0; q < quad.size(); ++q)
        quad_weights_h_z[q] = quad.weight(q) * h_z;

      std::vector<Polynomials::Polynomial<double>> basis =
        Polynomials::generate_complete_Lagrange_basis(quad.get_points());
      interpolate_quad_to_boundary[0].resize(n_q_points_1d);
      interpolate_quad_to_boundary[1].resize(n_q_points_1d);
      std::vector<double> val_and_der(2);
      for(unsigned int i = 0; i < basis.size(); ++i)
      {
        basis[i].value(0., val_and_der);
        interpolate_quad_to_boundary[0][i][0] = val_and_der[0];
        interpolate_quad_to_boundary[0][i][1] = val_and_der[1];
        basis[i].value(1., val_and_der);
        interpolate_quad_to_boundary[1][i][0] = val_and_der[0];
        interpolate_quad_to_boundary[1][i][1] = val_and_der[1];
      }
    }

    shape_info = matrix_free.get_shape_info();

    cell_level_index.resize(matrix_free.n_cell_batches());
    AssertDimension(cell_level_index.size(), dof_indices.size());
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int lane = 0; lane < n_lanes; ++lane)
        if(lane >= matrix_free.n_active_entries_per_cell_batch(cell))
          cell_level_index[cell][lane] = {
            {numbers::invalid_unsigned_int, numbers::invalid_unsigned_int}};
        else
        {
          const auto iter                 = matrix_free.get_cell_iterator(cell, lane);
          cell_level_index[cell][lane][0] = iter->level();
          cell_level_index[cell][lane][1] = iter->index();
        }

    detect_dependencies_of_face_integrals();

    set_parameters(0., 1.0);

    for(double & t : timings)
      t = 0.0;
  }

  ~LaplaceOperatorDG()
  {
    if(timings[0] > 0)
    {
      const MPI_Comm comm = dof_handler->get_mpi_communicator();
      std::cout << std::defaultfloat << std::setprecision(3);
      const double total_time =
        Utilities::MPI::sum(timings[9], comm) / timings[0] / Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for DG Laplace operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[0])
                  << " evaluations [t_total=" << total_time * timings[0] << "s]" << std::endl;
      if(Utilities::MPI::n_mpi_processes(comm) > 1)
      {
        RTOperator::print_time(timings[3] / timings[0],
                               "Pack/send data dg ghosts",
                               comm,
                               total_time);
        RTOperator::print_time(timings[5] / timings[0], "MPI_Waitall dg ghosts", comm, total_time);
        RTOperator::print_time(timings[4] / timings[0], "Pre-loop before ghosts", comm, total_time);
      }

      RTOperator::print_time(timings[7] / timings[0], "Matrix-free loop", comm, total_time);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
    if(timings[10] > 0)
    {
      const MPI_Comm comm       = MPI_COMM_WORLD;
      const double   total_time = Utilities::MPI::sum(timings[14], comm) / timings[10] /
                                Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for DG mass operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[10])
                  << " evaluations [t_total=" << total_time * timings[10] << "s]" << std::endl;

      RTOperator::print_time(timings[12] / timings[10], "Matrix-free loop", comm, total_time);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
  }

  void
  initialize_dof_vector(VectorType & vec) const
  {
    vec.reinit(partitioner);
  }

  void
  set_parameters(const double factor_mass, const double factor_laplace)
  {
    this->factor_mass    = factor_mass;
    this->factor_laplace = factor_laplace;
  }

  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    vmult(
      dst,
      src,
      [&dst](const unsigned int start, const unsigned int end) {
        std::memset(dst.begin() + start,
                    0,
                    (end - start) * sizeof(typename VectorType::value_type));
      },
      [](const unsigned int, const unsigned int) {});
  }

  void
  compute_diagonal(VectorType & dst) const
  {
    initialize_dof_vector(dst);
    const unsigned int n_cell_batches = dof_indices.size();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      diagonal_operation(cell, dst);
    dst.compress(VectorOperation::add);
  }

  void
  set_penalty_parameters(const Number additional_factor)
  {
    // Same as in ExaDG::IP::get_penalty_factor<dim> but split off to test
    // this function independently of ExaDG
    const Number factor =
      additional_factor * Utilities::fixed_power<2>(dof_handler->get_fe().degree + 1.0);
    this->penalty_parameters.resize(dof_indices.size());
    const unsigned int n_q_points_1d = shape_info.data[0].n_q_points_1d;
    for(unsigned int cell = 0; cell < dof_indices.size(); ++cell)
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        for(unsigned int face = 0; face < 4; ++face)
        {
          const unsigned int idx = face_mapping_data_index[cell][face][0][v] / n_q_points_1d / 4;
          AssertThrow(idx < ip_penalty_factors.size(),
                      ExcIndexRange(0, ip_penalty_factors.size(), idx));
          const unsigned int neigh_idx =
            face_mapping_data_index[cell][face][1][v] / n_q_points_1d / 4;
          AssertThrow(neigh_idx < ip_penalty_factors.size(),
                      ExcIndexRange(0, ip_penalty_factors.size(), neigh_idx));
          this->penalty_parameters[cell][face][v] =
            std::max(ip_penalty_factors[idx], ip_penalty_factors[neigh_idx]) * factor;
        }
        for(unsigned int face = 4; face < 2 * dim; ++face)
          this->penalty_parameters[cell][face][v] =
            ip_penalty_factors[mapping_data_index[cell][v] / Utilities::pow(n_q_points_1d, 2)] *
            factor;
      }
  }

  types::global_dof_index
  m() const
  {
    return partitioner->size();
  }

  Number
  el(const types::global_dof_index, const types::global_dof_index) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  void
  vmult(VectorType &                                                        dst,
        const VectorType &                                                  src,
        const std::function<void(const unsigned int, const unsigned int)> & before_loop,
        const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
  {
    Timer total_timer;
    if(factor_laplace != 0.)
      timings[0] += 1;
    else
      timings[10] += 1;
    Timer time;

    const unsigned int n_cell_batches = dof_indices.size();

    if(factor_laplace != 0.)
    {
      for(unsigned int range = cell_loop_pre_list_index[n_cell_batches];
          range < cell_loop_pre_list_index[n_cell_batches + 1];
          ++range)
        before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);

      timings[4] += time.wall_time();
    }

    time.restart();

    // only do the data exchange for face integral if we have Laplacian contribution
    if(factor_laplace != 0. && Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      const int nn = shape_info.data[0].fe_degree + 1;

      // data: everything projected to face, 2 because of values and derivatives
      const int data_per_face = 2 * Utilities::pow(nn, dim - 1);

      std::vector<MPI_Request> requests(2 * send_data_process.size());
      unsigned int             offset = 0;
      for(unsigned int p = 0; p < send_data_process.size(); ++p)
      {
        MPI_Irecv(&import_values[offset * data_per_face],
                  send_data_process[p].second * data_per_face * sizeof(Number),
                  MPI_BYTE,
                  send_data_process[p].first,
                  send_data_process[p].first + 47,
                  src.get_mpi_communicator(),
                  &requests[p]);
        offset += send_data_process[p].second;
      }
      AssertDimension(offset * data_per_face, import_values.size());

      if(nn == 2)
        vmult_pack_and_send_data<2>(src, requests);
      else if(nn == 3)
        vmult_pack_and_send_data<3>(src, requests);
      else if(nn == 4)
        vmult_pack_and_send_data<4>(src, requests);
#ifndef DEBUG
      else if(nn == 5)
        vmult_pack_and_send_data<5>(src, requests);
      else if(nn == 6)
        vmult_pack_and_send_data<6>(src, requests);
      else if(nn == 7)
        vmult_pack_and_send_data<7>(src, requests);
      else if(nn == 8)
        vmult_pack_and_send_data<8>(src, requests);
      else if(nn == 9)
        vmult_pack_and_send_data<9>(src, requests);
#endif
      else
        AssertThrow(false, ExcNotImplemented());

      timings[3] += time.wall_time();

      time.restart();
      if(!requests.empty())
        MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
      timings[5] += time.wall_time();
    }

    time.restart();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(factor_laplace != 0.)
        for(unsigned int range = cell_loop_pre_list_index[cell];
            range < cell_loop_pre_list_index[cell + 1];
            ++range)
          before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);
      else
        for(unsigned int range = cell_loop_mass_pre_list_index[cell];
            range < cell_loop_mass_pre_list_index[cell + 1];
            ++range)
          before_loop(cell_loop_mass_pre_list[range].first, cell_loop_mass_pre_list[range].second);

      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 1)
        do_cell_operation<1, true>(cell, src, dst);
      else if(degree == 2)
        do_cell_operation<2, true>(cell, src, dst);
      else if(degree == 3)
        do_cell_operation<3, true>(cell, src, dst);
      else if(degree == 4)
        do_cell_operation<4, true>(cell, src, dst);
#ifndef DEBUG
      else if(degree == 5)
        do_cell_operation<5, true>(cell, src, dst);
      else if(degree == 6)
        do_cell_operation<6, true>(cell, src, dst);
      else if(degree == 7)
        do_cell_operation<7, true>(cell, src, dst);
      else if(degree == 8)
        do_cell_operation<8, true>(cell, src, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
      for(unsigned int range = cell_loop_post_list_index[cell];
          range < cell_loop_post_list_index[cell + 1];
          ++range)
        after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);
    }

    if(factor_laplace != 0)
      timings[7] += time.wall_time();
    else
      timings[12] += time.wall_time();

    if(factor_laplace != 0)
    {
      timings[8] += time.wall_time();
      timings[9] += total_timer.wall_time();
    }
    else
    {
      timings[13] += time.wall_time();
      timings[14] += total_timer.wall_time();
    }
  }

  std::size_t
  memory_consumption() const
  {
    return MemoryConsumption::memory_consumption(dof_indices) +
           MemoryConsumption::memory_consumption(neighbor_cells) +
           MemoryConsumption::memory_consumption(mpi_exchange_data_on_faces) +
           MemoryConsumption::memory_consumption(import_values) +
           MemoryConsumption::memory_consumption(export_values) +
           MemoryConsumption::memory_consumption(all_owned_faces) +
           MemoryConsumption::memory_consumption(mapping_data_index) +
           MemoryConsumption::memory_consumption(jacobians_xy) +
           MemoryConsumption::memory_consumption(cell_JxW_xy) +
           MemoryConsumption::memory_consumption(face_mapping_data_index) +
           MemoryConsumption::memory_consumption(face_jxn_xy) +
           MemoryConsumption::memory_consumption(face_JxW_xy) +
           MemoryConsumption::memory_consumption(quad_weights_h_z) +
           MemoryConsumption::memory_consumption(interpolate_quad_to_boundary) +
           MemoryConsumption::memory_consumption(send_data_process) +
           MemoryConsumption::memory_consumption(send_data_cell_index) +
           MemoryConsumption::memory_consumption(send_data_face_index) +
           MemoryConsumption::memory_consumption(cell_loop_pre_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_pre_list) +
           MemoryConsumption::memory_consumption(cell_loop_mass_pre_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_mass_pre_list) +
           MemoryConsumption::memory_consumption(cell_loop_post_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_post_list) +
           MemoryConsumption::memory_consumption(face_flux_buffer) +
           MemoryConsumption::memory_consumption(face_flux_buffer_index) +
           MemoryConsumption::memory_consumption(all_left_face_fluxes_from_buffer);
  }

private:
  static constexpr unsigned int                    n_lanes = VectorizedArray<Number>::size();
  ObserverPointer<const DoFHandler<dim>>           dof_handler;
  internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
  std::vector<std::array<unsigned int, n_lanes>>   dof_indices;
  std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>> neighbor_cells;
  std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>> mpi_exchange_data_on_faces;
  mutable AlignedVector<Number>                                import_values;
  mutable AlignedVector<Number>                                export_values;
  Table<2, unsigned char>                                      all_owned_faces;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  mutable std::array<double, 15> timings;

  Number factor_mass;
  Number factor_laplace;

  Number                                            h_z;
  Number                                            h_z_inverse;
  std::vector<std::array<unsigned int, n_lanes>>    mapping_data_index;
  AlignedVector<Tensor<2, 2, Number>>               jacobians_xy;
  AlignedVector<Number>                             cell_JxW_xy;
  std::vector<ndarray<unsigned int, 4, 2, n_lanes>> face_mapping_data_index;
  AlignedVector<Tensor<1, 2, Number>>               face_jxn_xy;
  AlignedVector<Number>                             face_JxW_xy;
  std::vector<Number>                               quad_weights_h_z;
  std::array<std::vector<std::array<Number, 2>>, 2> interpolate_quad_to_boundary;

  std::vector<Number>                                         ip_penalty_factors;
  AlignedVector<std::array<VectorizedArray<Number>, 2 * dim>> penalty_parameters;

  std::vector<std::pair<unsigned int, unsigned int>> send_data_process;
  std::vector<unsigned int>                          send_data_cell_index;
  std::vector<unsigned char>                         send_data_face_index;

  std::vector<unsigned int>                          cell_loop_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_pre_list;
  std::vector<unsigned int>                          cell_loop_mass_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_mass_pre_list;
  std::vector<unsigned int>                          cell_loop_post_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_post_list;

  std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> cell_level_index;

  mutable AlignedVector<Number>                        face_flux_buffer;
  std::vector<ndarray<unsigned int, 2 * dim, n_lanes>> face_flux_buffer_index;
  std::vector<std::array<bool, dim>>                   all_left_face_fluxes_from_buffer;

  unsigned int
  n_active_entries_per_cell_batch(const unsigned int cell) const
  {
    AssertIndexRange(cell, dof_indices.size());
    unsigned int index = n_lanes - 1;
    while(cell_level_index[cell][index][0] == numbers::invalid_unsigned_int)
    {
      Assert(index > 0, ExcInternalError());
      --index;
    }
    return index + 1;
  }

  typename DoFHandler<dim>::cell_iterator
  get_cell_iterator(const unsigned int cell, const unsigned int lane) const
  {
    return typename DoFHandler<dim>::cell_iterator(&dof_handler->get_triangulation(),
                                                   cell_level_index[cell][lane][0],
                                                   cell_level_index[cell][lane][1],
                                                   dof_handler.get());
  }

  void
  compute_vector_access_pattern()
  {
    const unsigned int        n_dofs     = partitioner->locally_owned_size();
    constexpr unsigned int    chunk_size = 128;
    std::vector<unsigned int> touched_first_by((n_dofs + chunk_size - 1) / chunk_size,
                                               numbers::invalid_unsigned_int);
    std::vector<unsigned int> touched_mass_first_by((n_dofs + chunk_size - 1) / chunk_size,
                                                    numbers::invalid_unsigned_int);
    std::vector<unsigned int> touched_last_by((n_dofs + chunk_size - 1) / chunk_size,
                                              numbers::invalid_unsigned_int);
    const unsigned int        n_cell_batches = dof_indices.size();
    const unsigned int        dofs_on_cell   = dof_handler->get_fe().dofs_per_cell;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int face = 0; face < 2 * dim; ++face)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int c = neighbor_cells[cell][face][v];
          if(c != numbers::invalid_unsigned_int)
          {
            const unsigned int idx = dof_indices[c / n_lanes][c % n_lanes];
            if(idx < n_dofs)
            {
              const unsigned int first = idx / chunk_size;
              const unsigned int last  = (idx + dofs_on_cell - 1) / chunk_size;
              for(unsigned int j = first; j <= last; ++j)
                if(touched_first_by[j] == numbers::invalid_unsigned_int)
                  touched_first_by[j] = cell;
            }
          }
        }
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx = dof_indices[cell][v];
        if(idx < n_dofs)
        {
          const unsigned int first = idx / chunk_size;
          const unsigned int last  = (idx + dofs_on_cell - 1) / chunk_size;
          for(unsigned int j = first; j <= last; ++j)
          {
            if(touched_first_by[j] == numbers::invalid_unsigned_int)
              touched_first_by[j] = cell;
            if(touched_mass_first_by[j] == numbers::invalid_unsigned_int)
              touched_mass_first_by[j] = cell;
            touched_last_by[j] = cell;
          }
        }
      }
    }

    // ensure that all indices are touched at least during the last round
    for(const auto & index : touched_first_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());
    for(const auto & index : touched_mass_first_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());
    for(const auto & index : touched_last_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());
    {
      unsigned int n_batches_half = 0, n_batches_10 = 0;
      for(unsigned int i = 0; i < touched_first_by.size(); ++i)
        if(touched_last_by[i] - touched_first_by[i] > n_cell_batches / 2)
          ++n_batches_half;
        else if(touched_last_by[i] - touched_first_by[i] > 10)
          ++n_batches_10;

      RTOperator::print_time(static_cast<double>(n_batches_half) / touched_first_by.size(),
                             "Pre-/post distance > 1/2 size",
                             MPI_COMM_WORLD);
      RTOperator::print_time(static_cast<double>(n_batches_half + n_batches_10) /
                               touched_first_by.size(),
                             "Pre-/post distance > 10",
                             MPI_COMM_WORLD);
    }

    for(const unsigned int cell_lane : send_data_cell_index)
    {
      const unsigned int cell = cell_lane / n_lanes;
      const unsigned int lane = cell_lane % n_lanes;
      const unsigned int idx  = dof_indices[cell][lane];
      if(idx < n_dofs)
      {
        const unsigned int first = idx / chunk_size;
        const unsigned int last  = (idx + dofs_on_cell - 1) / chunk_size;
        for(unsigned int j = first; j <= last; ++j)
          if(touched_first_by[j] != n_cell_batches)
            touched_first_by[j] = n_cell_batches;
      }
    }

#if 0
    std::cout << "data locality quality: ";
    for (unsigned int i = 0; i < touched_first_by.size(); ++i)
      std::cout << touched_last_by[i] - touched_first_by[i] << "  ";
    std::cout << std::endl;
#endif

    {
      unsigned int n_batches_half = 0, n_batches_10 = 0;
      for(unsigned int i = 0; i < touched_first_by.size(); ++i)
        if(touched_last_by[i] - touched_first_by[i] > n_cell_batches / 2)
          ++n_batches_half;
        else if(touched_last_by[i] - touched_first_by[i] > 10)
          ++n_batches_10;

      RTOperator::print_time(static_cast<double>(n_batches_half) / touched_first_by.size(),
                             "Pre-/post distance > 1/2 size",
                             MPI_COMM_WORLD);
      RTOperator::print_time(static_cast<double>(n_batches_half + n_batches_10) /
                               touched_first_by.size(),
                             "Pre-/post distance > 10",
                             MPI_COMM_WORLD);
    }

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_pre;
    for(unsigned int i = 0; i < touched_first_by.size(); ++i)
      chunk_must_do_pre[touched_first_by[i]].push_back(i);
    RTOperator::convert_map_to_range_list(n_cell_batches + 1,
                                          chunk_size,
                                          chunk_must_do_pre,
                                          cell_loop_pre_list_index,
                                          cell_loop_pre_list,
                                          n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_mass_pre;
    for(unsigned int i = 0; i < touched_mass_first_by.size(); ++i)
      chunk_must_do_mass_pre[touched_mass_first_by[i]].push_back(i);
    RTOperator::convert_map_to_range_list(n_cell_batches,
                                          chunk_size,
                                          chunk_must_do_mass_pre,
                                          cell_loop_mass_pre_list_index,
                                          cell_loop_mass_pre_list,
                                          n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_post;
    for(unsigned int i = 0; i < touched_last_by.size(); ++i)
      chunk_must_do_post[touched_last_by[i]].push_back(i);
    RTOperator::convert_map_to_range_list(n_cell_batches,
                                          chunk_size,
                                          chunk_must_do_post,
                                          cell_loop_post_list_index,
                                          cell_loop_post_list,
                                          n_dofs);

#if 0
    for (unsigned int i = 0; i < n_cell_batches + 2; ++i)
      {
        std::cout << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " pre ";
        for (unsigned int j = cell_loop_pre_list_index[i];
             j < cell_loop_pre_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_pre_list[j].first << ","
                    << cell_loop_pre_list[j].second << ") ";
        if (i < n_cell_batches + 1)
          {
        std::cout << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " pre mass ";
        for (unsigned int j = cell_loop_mass_pre_list_index[i];
             j < cell_loop_mass_pre_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_mass_pre_list[j].first << ","
                    << cell_loop_mass_pre_list[j].second << ") ";
        std::cout << "post ";
        for (unsigned int j = cell_loop_post_list_index[i];
             j < cell_loop_post_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_post_list[j].first << ","
                    << cell_loop_post_list[j].second << ")";
        std::cout << std::endl;
      }
      }
#endif
  }

  template<int nn, int n_points_1d>
  void
  read_cell_values(const Number *                            src_vector,
                   const std::array<unsigned int, n_lanes> & dof_indices,
                   VectorizedArray<Number> *                 out) const
  {
    static_assert(nn > 1, "Degree 0 not supported");
    constexpr unsigned int          dofs_per_plane = nn * nn;
    constexpr unsigned int          n_points_2d    = n_points_1d * n_points_1d;
    VectorizedArray<Number>         cell_data[dofs_per_plane];
    const Number * DEAL_II_RESTRICT shape_data = shape_info.data[0].shape_values_eo.data();

    for(unsigned int i_z = 0; i_z < (dim > 2 ? nn : 1); ++i_z)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + i_z * dofs_per_plane,
                                    dof_indices.data(),
                                    cell_data);
      VectorizedArray<Number> * out_z = out + i_z * n_points_2d;

      // perform interpolation in x direction
      for(unsigned int i_y = 0; i_y < nn; ++i_y)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              1,
                                              1,
                                              true,
                                              false>(shape_data,
                                                     cell_data + i_y * nn,
                                                     out_z + i_y * n_points_1d);
      }

      // perform interpolation in y direction
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_1d,
                                              n_points_1d,
                                              true,
                                              false>(shape_data, out_z + i_x, out_z + i_x);
    }
    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_2d; ++i)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_2d,
                                              n_points_2d,
                                              true,
                                              false>(shape_data, out + i, out + i);
      }
  }

  template<int nn, int n_points_1d>
  void
  integrate_cell_scatter(const dealii::ndarray<unsigned int, n_lanes> & dof_indices,
                         const unsigned int                             n_filled_lanes,
                         VectorizedArray<Number> *                      quad_values,
                         Number *                                       dst_vector) const
  {
    constexpr unsigned int  dofs_per_plane = nn * nn;
    constexpr unsigned int  n_points_2d    = n_points_1d * n_points_1d;
    VectorizedArray<Number> cell_data[n_points_2d];

    const Number * DEAL_II_RESTRICT shape_data = shape_info.data[0].shape_values_eo.data();
    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_1d * n_points_1d; ++i)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_2d,
                                              n_points_2d,
                                              false,
                                              false>(shape_data, quad_values + i, quad_values + i);
      }
    for(unsigned int i_z = 0; i_z < (dim > 2 ? nn : 1); ++i_z)
    {
      VectorizedArray<Number> * out_z = quad_values + i_z * n_points_2d;
      // perform integration in y direction
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_1d,
                                              n_points_1d,
                                              false,
                                              false>(shape_data, out_z + i_x, out_z + i_x);

      // perform integration in x direction
      for(unsigned int i_y = 0; i_y < nn; ++i_y)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              1,
                                              1,
                                              false,
                                              false>(shape_data,
                                                     out_z + i_y * n_points_1d,
                                                     cell_data + i_y * nn);
      }
      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(
          true, dofs_per_plane, cell_data, dof_indices.data(), dst_vector + i_z * dofs_per_plane);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          Number * dst_ptr = dst_vector + dof_indices[v] + i_z * dofs_per_plane;
          for(unsigned int i = 0; i < dofs_per_plane; ++i)
            dst_ptr[i] += cell_data[i][v];
        }
    }
  }

  template<int nn>
  void
  read_face_values(const Number *                            src_vector,
                   const unsigned int                        face,
                   const std::array<unsigned int, n_lanes> & dof_indices,
                   VectorizedArray<Number> *                 out) const
  {
    const unsigned int face_direction = face / 2;
    const unsigned int side           = face % 2;

    constexpr unsigned int  dofs_per_face  = Utilities::pow(nn, dim - 1);
    constexpr unsigned int  dofs_per_plane = nn * nn;
    VectorizedArray<Number> data_cell[dofs_per_face];

    const Number * DEAL_II_RESTRICT shape_data = shape_info.data[0].shape_data_on_face[side].data();

    for(unsigned int i2 = 0; i2 < (dim > 2 ? nn : 1); ++i2)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + i2 * dofs_per_plane,
                                    dof_indices.data(),
                                    data_cell);

      if(face_direction < 2)
      {
        VectorizedArray<Number> * out_val  = out + i2 * nn;
        VectorizedArray<Number> * out_grad = out + dofs_per_face + i2 * nn;
        if(face_direction == 0)
        {
          for(unsigned int i1 = 0; i1 < nn; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1 * nn] * shape_data[0];
            VectorizedArray<Number> sum_g = data_cell[i1 * nn] * shape_data[nn];
            for(unsigned int i = 1; i < nn; ++i)
            {
              sum_v += data_cell[i1 * nn + i] * shape_data[i];
              sum_g += data_cell[i1 * nn + i] * shape_data[nn + i];
            }

            out_val[i1]  = sum_v;
            out_grad[i1] = sum_g;
          }
        }
        else
        {
          // face_direction == 1
          for(unsigned int i1 = 0; i1 < nn; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1] * shape_data[0];
            VectorizedArray<Number> sum_g = data_cell[i1] * shape_data[nn];
            for(unsigned int i = 1; i < nn; ++i)
            {
              sum_v += data_cell[i * nn + i1] * shape_data[i];
              sum_g += data_cell[i * nn + i1] * shape_data[nn + i];
            }

            out_val[i1]  = sum_v;
            out_grad[i1] = sum_g;
          }
        }
      }
      else
      {
        Assert(dim == 3, ExcInternalError());
        Assert(face_direction == 2, ExcInternalError());
        VectorizedArray<Number> *     out_val  = out;
        VectorizedArray<Number> *     out_grad = out + dofs_per_face;
        const VectorizedArray<Number> val_i2   = shape_data[i2];
        const VectorizedArray<Number> grad_i2  = shape_data[i2 + nn];
        if(i2 == 0)
          for(unsigned int i = 0; i < dofs_per_face; ++i)
          {
            out_val[i]  = val_i2 * data_cell[i];
            out_grad[i] = grad_i2 * data_cell[i];
          }
        else
          for(unsigned int i = 0; i < dofs_per_face; ++i)
          {
            out_val[i] += val_i2 * data_cell[i];
            out_grad[i] += grad_i2 * data_cell[i];
          }
      }
    }
  }



  template<int nn>
  void
  vmult_pack_and_send_data(const VectorType & src, std::vector<MPI_Request> & requests) const
  {
    constexpr int data_per_face = 2 * Utilities::pow(nn, dim - 1);

    std::array<VectorizedArray<Number>, 2 * nn * nn> tmp_vec;

    unsigned int offset = 0;
    for(unsigned int p = 0; p < send_data_process.size(); ++p)
    {
      const unsigned int my_faces = send_data_process[p].second;
      for(unsigned int count = offset; count < offset + my_faces;)
      {
        const unsigned int face = send_data_face_index[count];

        // to use vectorized code path, must check that faces are
        // available
        unsigned int n_faces = 1;
        if(count + n_lanes <= offset + my_faces &&
           send_data_face_index[count + n_lanes - 1] == face)
          n_faces = n_lanes;
        else
          while(count + n_faces < offset + my_faces &&
                send_data_face_index[count + n_faces] == face)
            ++n_faces;
        AssertIndexRange(n_faces, n_lanes + 1);

        std::array<unsigned int, n_lanes> dof_indices_vec;
        std::array<unsigned int, n_lanes> indices;
        for(unsigned int v = 0; v < n_faces; ++v, ++count)
        {
          indices[v]              = count * data_per_face;
          const unsigned int cell = send_data_cell_index[count] / n_lanes;
          const unsigned int lane = send_data_cell_index[count] % n_lanes;
          AssertDimension(face, send_data_face_index[count]);
          dof_indices_vec[v] = dof_indices[cell][lane];
        }
        for(unsigned int v = n_faces; v < n_lanes; ++v)
        {
          dof_indices_vec[v] = dof_indices_vec[0];
        }
        read_face_values<nn>(src.begin(), face, dof_indices_vec, tmp_vec.data());

        // copy to dedicated data field
        if(n_faces == n_lanes)
          vectorized_transpose_and_store(
            false, data_per_face, tmp_vec.data(), indices.data(), export_values.data());
        else
          for(unsigned int v = 0; v < n_faces; ++v)
            for(unsigned int i = 0; i < data_per_face; ++i)
              export_values[indices[v] + i] = tmp_vec[i][v];
      }

      MPI_Isend(&export_values[offset * data_per_face],
                my_faces * data_per_face * sizeof(Number),
                MPI_BYTE,
                send_data_process[p].first,
                src.get_partitioner()->this_mpi_process() + 47,
                src.get_mpi_communicator(),
                &requests[send_data_process.size() + p]);
      offset += my_faces;
    }
    AssertDimension(offset * data_per_face, export_values.size());
  }

  void
  diagonal_operation(const unsigned int cell, VectorType & dst) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 1)
      do_cell_operation<1, false>(cell, dst, dst);
    else if(degree == 2)
      do_cell_operation<2, false>(cell, dst, dst);
    else if(degree == 3)
      do_cell_operation<3, false>(cell, dst, dst);
    else if(degree == 4)
      do_cell_operation<4, false>(cell, dst, dst);
#ifndef DEBUG
    else if(degree == 5)
      do_cell_operation<5, false>(cell, dst, dst);
    else if(degree == 6)
      do_cell_operation<6, false>(cell, dst, dst);
    else if(degree == 7)
      do_cell_operation<7, false>(cell, dst, dst);
    else if(degree == 8)
      do_cell_operation<8, false>(cell, dst, dst);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  template<int degree, bool compute_exterior>
  void
  do_cell_operation(const unsigned int cell, const VectorType & src, VectorType & dst) const
  {
    constexpr unsigned int dofs_per_cell = Utilities::pow(degree + 1, dim);
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data    = shape_info.data[0];

    VectorizedArray<Number> quad_values[n_points], out_values[n_points];
    if(compute_exterior)
    {
      read_cell_values<degree + 1, n_q_points_1d>(src.begin(), dof_indices[cell], quad_values);

      if(factor_laplace != 0)
        compute_cell_lapl<n_q_points_1d>(shape_data, cell, quad_values, out_values);
      else
        AssertThrow(false, ExcNotImplemented("Pure mass case not implemented yet"));

      // Face integrals if Laplace factor is positive
      if(factor_laplace != 0.)
        for(unsigned int f = 0; f < 2 * dim; ++f)
          compute_face<degree, true>(shape_data, src, cell, f, quad_values, out_values);

      integrate_cell_scatter<degree + 1, n_q_points_1d>(dof_indices[cell],
                                                        n_active_entries_per_cell_batch(cell),
                                                        out_values,
                                                        dst.begin());
    }
    else
    {
      const unsigned int n_entries_per_batch = n_active_entries_per_cell_batch(cell);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        std::array<unsigned int, dim> tensor_i;
        constexpr unsigned int        nn = degree + 1;
        if(dim == 2)
        {
          tensor_i[0] = i % nn;
          tensor_i[1] = i / nn;
        }
        else if(dim == 3)
        {
          tensor_i[0] = i % nn;
          tensor_i[1] = (i / nn) % nn;
          tensor_i[2] = i / (nn * nn);
        }
        for(unsigned int q2 = 0, q = 0; q2 < (dim > 2 ? n_q_points_1d : 1); ++q2)
          for(unsigned int q1 = 0; q1 < (dim > 1 ? n_q_points_1d : 1); ++q1)
            for(unsigned int q0 = 0; q0 < n_q_points_1d; ++q0, ++q)
              quad_values[q] =
                ((dim == 3 ? shape_data.shape_values[tensor_i[2] * nn + q2] : Number(1.0)) *
                 shape_data.shape_values[tensor_i[1] * nn + q1]) *
                shape_data.shape_values[tensor_i[0] * nn + q0];

        if(factor_laplace != 0)
          compute_cell_lapl<n_q_points_1d>(shape_data, cell, quad_values, out_values);

        if(factor_laplace != 0.)
          for(unsigned int f = 0; f < 2 * dim; ++f)
            compute_face<degree, false>(shape_data, src, cell, f, quad_values, out_values);

        VectorizedArray<Number> sum = 0;
        for(unsigned int q2 = 0, q = 0; q2 < (dim > 2 ? n_q_points_1d : 1); ++q2)
          for(unsigned int q1 = 0; q1 < (dim > 1 ? n_q_points_1d : 1); ++q1)
          {
            VectorizedArray<Number> inner_sum = {};
            for(unsigned int q0 = 0; q0 < n_q_points_1d; ++q0)
              inner_sum += shape_data.shape_values[tensor_i[0] * nn + q0] * quad_values[q];
            sum += ((dim == 3 ? shape_data.shape_values[tensor_i[2]] : Number(1.0)) *
                    shape_data.shape_values[tensor_i[1]]) *
                   inner_sum;
          }

        // write diagonal entry to global vector
        for(unsigned int v = 0; v < n_entries_per_batch; ++v)
          dst.local_element(dof_indices[cell][v] + i) = sum[v];
      }
    }
  }

  template<int n_q_points_1d>
  void
  compute_cell_lapl(const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data,
                    const unsigned int                                                 cell,
                    const VectorizedArray<Number> *                                    quad_values,
                    VectorizedArray<Number> * out_values) const
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);

    VectorizedArray<Number> grad_y[n_q_points];
    VectorizedArray<Number> grad_x[Utilities::pow(n_q_points_1d, dim - 1)];
    VectorizedArray<Number> grad_z[n_q_points_1d];

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_data_index[cell][v] * 4;

    const Number factor_mass = this->factor_mass;
    const Number factor_lapl = this->factor_laplace;

    constexpr unsigned int nn         = n_q_points_1d;
    const Number *         shape_grad = shape_data.shape_gradients_collocation_eo.data();
    for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              nn,
                                              Utilities::pow(nn, dim - 1),
                                              true,
                                              false>(shape_grad,
                                                     quad_values + (i1 * n_q_points_2d + i0),
                                                     grad_y + (i1 * nn + i0));

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              1,
                                              (dim == 3 ? nn : 1),
                                              true,
                                              false>(shape_grad,
                                                     quad_values + (qy * nn + i0 * n_q_points_2d),
                                                     grad_x + i0);

      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy;
        vectorized_load_and_transpose(4,
                                      &jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &inv_jac_xy[0][0]);
        VectorizedArray<Number> JxW_xy;
        JxW_xy.gather(&cell_JxW_xy[q1], mapping_data_index[cell].data());

        if constexpr(dim == 2)
        {
          const VectorizedArray<Number> val     = quad_values[q1];
          const VectorizedArray<Number> grad[2] = {grad_x[qx], grad_y[q1]};

          VectorizedArray<Number> grad_real[dim];
          for(unsigned int d = 0; d < dim; ++d)
            grad_real[d] = inv_jac_xy[d][0] * grad[0] + inv_jac_xy[d][1] * grad[1];

          for(unsigned int e = 0; e < dim; ++e)
          {
            grad_real[e] *= JxW_xy * factor_lapl;
          }

          grad_x[qx]     = inv_jac_xy[0][0] * grad_real[0] + inv_jac_xy[1][0] * grad_real[1];
          grad_y[q1]     = inv_jac_xy[0][1] * grad_real[0] + inv_jac_xy[1][1] * grad_real[1];
          out_values[q1] = val * (JxW_xy * factor_mass);
        }
        else // now to dim == 3
        {
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                nn * nn,
                                                1,
                                                true,
                                                false>(shape_grad, quad_values + q1, grad_z);

          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
          {
            const VectorizedArray<Number> val     = quad_values[q];
            const VectorizedArray<Number> grad[3] = {grad_x[qz + qx * nn],
                                                     grad_y[qz * nn + qy * nn * nn + qx],
                                                     grad_z[qz]};

            VectorizedArray<Number> grad_real[dim];
            for(unsigned int d = 0; d < 2; ++d)
              grad_real[d] = inv_jac_xy[d][0] * grad[0] + inv_jac_xy[d][1] * grad[1];
            grad_real[2] = h_z_inverse * grad[2];

            const Number                  weight_h_z   = quad_weights_h_z[qz];
            const VectorizedArray<Number> factor_deriv = (JxW_xy * factor_lapl) * weight_h_z;
            const VectorizedArray<Number> factor_ma    = (JxW_xy * factor_mass) * weight_h_z;

            out_values[q] = val * factor_ma;
            for(unsigned int e = 0; e < dim; ++e)
            {
              grad_real[e] *= factor_deriv;
            }

            grad_x[qz + qx * nn] =
              inv_jac_xy[0][0] * grad_real[0] + inv_jac_xy[1][0] * grad_real[1];
            grad_y[qz * nn + qy * nn * nn + qx] =
              inv_jac_xy[0][1] * grad_real[0] + inv_jac_xy[1][1] * grad_real[1];
            grad_z[qz] = h_z_inverse * grad_real[2];
          }

          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                1,
                                                nn * nn,
                                                false,
                                                true>(shape_grad, grad_z, out_values + q1);
        }
      }

      for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              (dim == 3 ? nn : 1),
                                              1,
                                              false,
                                              true>(shape_grad,
                                                    grad_x + i0,
                                                    out_values + qy * nn + i0 * n_q_points_2d);
    }

    for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              Utilities::pow(nn, dim - 1),
                                              nn,
                                              false,
                                              true>(shape_grad,
                                                    grad_y + i1 * nn + i0,
                                                    out_values + i1 * n_q_points_2d + i0);
  }

  template<int degree, bool compute_exterior>
  void
  compute_face(const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data,
               const VectorType &                                                 src,
               const unsigned int                                                 cell,
               const unsigned int                                                 f,
               const VectorizedArray<Number> *                                    quad_values,
               VectorizedArray<Number> *                                          out_values) const
  {
    constexpr bool use_face_buffer = true && compute_exterior;

    if(use_face_buffer && f % 2 == 0 && all_left_face_fluxes_from_buffer[cell][f / 2])
      return;

    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    constexpr unsigned int n_lanes       = VectorizedArray<Number>::size();

    std::array<unsigned int, n_lanes> dof_indices_neighbor;
    unsigned int                      lane_with_neighbor = 0;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int)
      {
        lane_with_neighbor = v;
        break;
      }

    // TODO: 0: interior face, 1: Neumann, -1: Dirichlet, apply from actual b.c. in operator
    VectorizedArray<Number> boundary_mask;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] == numbers::invalid_unsigned_int)
        boundary_mask[v] = 1.;
      else
        boundary_mask[v] = 0.;

    if(compute_exterior)
    {
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        // use a dummy neighbor with nearby data for cells at boundary to make
        // sure vectorized functions continue working
        const unsigned int neighbor_idx =
          neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int ?
            neighbor_cells[cell][f][v] :
            neighbor_cells[cell][f][lane_with_neighbor];
        if(neighbor_idx != numbers::invalid_unsigned_int)
          dof_indices_neighbor[v] = dof_indices[neighbor_idx / n_lanes][neighbor_idx % n_lanes];
        else
          dof_indices_neighbor[v] = dof_indices[0][0];
      }
    }

    VectorizedArray<Number> face_array[2 * Utilities::pow(n_q_points_1d, dim - 1)];
    constexpr int           nn            = degree + 1;
    constexpr int           dofs_per_face = Utilities::pow(nn, dim - 1);
    const Number            factor_lapl   = this->factor_laplace;

    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_g({}, shape_data.shape_gradients_collocation_eo.data(), {});
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     nn,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval(shape_data.shape_values_eo.data(), {}, {});

    const unsigned int      face_direction  = f / 2;
    constexpr unsigned int  n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);
    VectorizedArray<Number> scratch_data[n_q_points_face + 1];
    VectorizedArray<Number> values_face[2][n_q_points_face + 1];
    VectorizedArray<Number> grads_face[2][n_q_points_face * dim + 1];

    if constexpr(compute_exterior)
    {
      read_face_values<degree + 1>(src.begin(),
                                   f + (f % 2 ? -1 : 1),
                                   dof_indices_neighbor,
                                   face_array);
      // For faces located on MPI-remote processes, we replace the dummy
      // data read out above by the data sent to us
      for(unsigned int v = 0; v < n_lanes; ++v)
        if(mpi_exchange_data_on_faces[cell][f][v] != numbers::invalid_unsigned_int)
          for(unsigned int i = 0; i < 2 * dofs_per_face; ++i)
          {
            face_array[i][v] = import_values[mpi_exchange_data_on_faces[cell][f][v] + i];
          }

      eval.template values<0, true, false>(face_array, scratch_data);
      eval.template values<1, true, false>(scratch_data, values_face[1]);
      eval.template values<0, true, false>(face_array + nn * nn, scratch_data);
      eval.template values<1, true, false, dim>(scratch_data, grads_face[1] + face_direction);
    }

    // interpolate from interior to face
    const std::array<Number, 2> * shape = interpolate_quad_to_boundary[f % 2].data();

    for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
      for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
      {
        unsigned int idx = 0;
        if(face_direction == 0)
          idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
        else if(face_direction == 1)
          idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
        else
          idx = i0 + i1 * n_q_points_1d;

        const VectorizedArray<Number> * my_vals = quad_values + idx;
        VectorizedArray<Number>         v0      = shape[0][0] * my_vals[0];
        VectorizedArray<Number>         d0      = shape[0][1] * my_vals[0];
        const unsigned int              stride  = Utilities::pow(n_q_points_1d, face_direction);
        for(unsigned int i = 1; i < n_q_points_1d; ++i)
        {
          v0 += shape[i][0] * my_vals[i * stride + 0];
          d0 += shape[i][1] * my_vals[i * stride + 0];
        }
        const unsigned int val_idx   = i1 * n_q_points_1d + i0;
        values_face[0][val_idx]      = v0;
        const unsigned int deriv_idx = val_idx * dim + face_direction;
        grads_face[0][deriv_idx]     = d0;
      }

    // Only need some tangential derivative in xy plane, as the domain is
    // extruded in z direction
    if(face_direction < 2)
      for(unsigned int side = 0; side < (compute_exterior ? 2 : 1); ++side)
      {
        const unsigned int first  = face_direction == 0 ? 1 : 0;
        const unsigned int second = face_direction == 1 ? 2 : first + 1;
        eval_g.template gradients<0, true, false, dim>(values_face[side], grads_face[side] + first);
        // This is a logical contradiction to the above if statement, so
        // this is actually never executed, but we keep it for
        // completeness
        if(face_direction == 2)
          eval_g.template gradients<1, true, false, dim>(values_face[side],
                                                         grads_face[side] + second);
      }

    const Number                  h_z_inverse = this->h_z_inverse;
    const VectorizedArray<Number> sigmaF      = penalty_parameters[cell][f] * factor_lapl;

    if(face_direction < 2)
    {
      std::array<unsigned int, n_lanes> shifted_data_indices;
      std::array<unsigned int, n_lanes> shifted_data_indices_neigh;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx          = face_mapping_data_index[cell][f][0][v];
        shifted_data_indices[v]         = idx * 2;
        const unsigned int neighbor_idx = face_mapping_data_index[cell][f][1][v];
        shifted_data_indices_neigh[v]   = neighbor_idx * 2;
      }

      for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
      {
        Tensor<1, 2, VectorizedArray<Number>> jac_x_normal[2];
        vectorized_load_and_transpose(2,
                                      &face_jxn_xy[q1][0],
                                      shifted_data_indices.data(),
                                      &jac_x_normal[0][0]);
        vectorized_load_and_transpose(2,
                                      &face_jxn_xy[q1][0],
                                      shifted_data_indices_neigh.data(),
                                      &jac_x_normal[1][0]);

        VectorizedArray<Number> JxW_xy;
        JxW_xy.gather(face_JxW_xy.data() + q1, face_mapping_data_index[cell][f][0].data());

        for(unsigned int qz = 0, q = q1; qz < n_q_points_1d; ++qz, q += n_q_points_1d)
        {
          VectorizedArray<Number>       val[2]     = {values_face[0][q], values_face[1][q]};
          const VectorizedArray<Number> grad[2][2] = {
            {grads_face[0][q * dim], grads_face[0][q * dim + 1]},
            {grads_face[1][q * dim], grads_face[1][q * dim + 1]}};

          VectorizedArray<Number> normal_derivatives[2];
          for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
          {
            normal_derivatives[s] =
              jac_x_normal[s][0] * grad[s][0] + jac_x_normal[s][1] * grad[s][1];
          }

          if(compute_exterior)
          {
            val[1] = boundary_mask * val[0] + (Number(1.0) - std::abs(boundary_mask)) * val[1];
            normal_derivatives[1] = boundary_mask * normal_derivatives[0] +
                                    (Number(1.0) - std::abs(boundary_mask)) * normal_derivatives[1];
          }

          const VectorizedArray<Number> integrate_factor = JxW_xy * quad_weights_h_z[qz];

          // physical terms
          const VectorizedArray<Number> effective_factor =
            integrate_factor * make_vectorized_array<Number>(0.5 * factor_lapl);

          const auto viscous_value_flux =
            (integrate_factor * sigmaF) * (val[0] - val[1]) -
            effective_factor * (normal_derivatives[0] - normal_derivatives[1]);

          const auto viscous_gradient_flux = effective_factor * (val[1] - val[0]);

          // apply test functions via transpose of the above
          for(unsigned int d = 0; d < 2; ++d)
            grads_face[0][q * dim + d] = jac_x_normal[0][d] * viscous_gradient_flux;
          values_face[0][q] = viscous_value_flux;
          if(use_face_buffer)
          {
            for(unsigned int d = 0; d < 2; ++d)
              grads_face[1][q * dim + d] = jac_x_normal[1][d] * (-viscous_gradient_flux);
            values_face[1][q] = -viscous_value_flux;
          }
        }
      }
    }
    else
    {
      const Number normal_sign = (f % 2) ? 1 : -1;

      std::array<unsigned int, n_lanes> shifted_data_indices;
      for(unsigned int v = 0; v < n_lanes; ++v)
        shifted_data_indices[v] = mapping_data_index[cell][v] * 4;

      for(unsigned int q = 0; q < n_q_points_2d; ++q)
      {
        VectorizedArray<Number> JxW_xy;
        JxW_xy.gather(&cell_JxW_xy[q], mapping_data_index[cell].data());

        const VectorizedArray<Number> val[2]  = {values_face[0][q], values_face[1][q]};
        const VectorizedArray<Number> grad[2] = {grads_face[0][q * dim + 2],
                                                 grads_face[1][q * dim + 2]};

        VectorizedArray<Number> normal_derivatives[2];
        for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
          normal_derivatives[s] = (h_z_inverse * normal_sign) * grad[s];

        // physical terms
        const VectorizedArray<Number> effective_factor =
          JxW_xy * make_vectorized_array<Number>(0.5 * factor_lapl);
        const auto viscous_value_flux =
          (JxW_xy * sigmaF) * (val[0] - val[1]) -
          effective_factor * (normal_derivatives[0] + normal_derivatives[1]);

        const auto viscous_gradient_flux = effective_factor * (val[1] - val[0]);

        // transpose of evaluate part
        values_face[0][q]          = viscous_value_flux;
        grads_face[0][q * dim + 2] = normal_sign * h_z_inverse * viscous_gradient_flux;
        if(use_face_buffer)
        {
          values_face[1][q]          = -values_face[0][q];
          grads_face[1][q * dim + 2] = grads_face[0][q * dim + 2];
        }
      }
    }

    if(face_direction < 2)
      for(unsigned int s = 0; s < ((use_face_buffer && f % 2) ? 2 : 1); ++s)
      {
        const unsigned int first  = face_direction == 0 ? 1 : 0;
        const unsigned int second = face_direction == 1 ? 2 : first + 1;
        eval_g.template gradients<0, false, true, dim>(grads_face[s] + first, values_face[s]);
        if(face_direction == 2)
          eval_g.template gradients<1, false, true, dim>(grads_face[s] + second, values_face[s]);
      }

    // fill flux buffer
    if(use_face_buffer && f % 2 == 1)
    {
      vectorized_transpose_and_store(false,
                                     n_q_points_face,
                                     values_face[1],
                                     face_flux_buffer_index[cell][f].data(),
                                     face_flux_buffer.data());
      for(unsigned int q = 0; q < n_q_points_face; ++q)
        values_face[1][q] = grads_face[1][q * dim + face_direction];
      vectorized_transpose_and_store(false,
                                     n_q_points_face,
                                     values_face[1],
                                     face_flux_buffer_index[cell][f].data(),
                                     face_flux_buffer.data() + n_q_points_face);
    }
    if(use_face_buffer && all_left_face_fluxes_from_buffer[cell][face_direction])
    {
      vectorized_load_and_transpose(n_q_points_face,
                                    face_flux_buffer.data() + n_q_points_face,
                                    face_flux_buffer_index[cell][f - 1].data(),
                                    values_face[1]);
      for(unsigned int q = 0; q < n_q_points_face; ++q)
        grads_face[1][q * dim + face_direction] = values_face[1][q];
      vectorized_load_and_transpose(n_q_points_face,
                                    face_flux_buffer.data(),
                                    face_flux_buffer_index[cell][f - 1].data(),
                                    values_face[1]);
    }

    // interpolate from face to interior
    const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction);
    dealii::ndarray<VectorizedArray<Number>, n_q_points_1d, 2> shape_vec;
    for(unsigned int i = 0; i < n_q_points_1d; ++i)
    {
      shape_vec[i][0] = shape[i][0];
      shape_vec[i][1] = shape[i][1];
    }
    if(use_face_buffer && f % 2 == 1 && all_left_face_fluxes_from_buffer[cell][face_direction])
      for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
        {
          unsigned int idx;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
          else
            idx = i0 + i1 * n_q_points_1d;

          VectorizedArray<Number> *     vals      = out_values + idx;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> v0        = values_face[0][val_idx];
          const VectorizedArray<Number> d0        = grads_face[0][deriv_idx];
          const VectorizedArray<Number> vo0       = values_face[1][val_idx];
          const VectorizedArray<Number> do0       = grads_face[1][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
            vals[i * stride] += shape_vec[i][0] * v0 + shape_vec[i][1] * d0 +
                                shape_vec[n_q_points_1d - 1 - i][0] * vo0 -
                                shape_vec[n_q_points_1d - 1 - i][1] * do0;
        }
    else
      for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
        {
          unsigned int idx;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
          else
            idx = i0 + i1 * n_q_points_1d;

          VectorizedArray<Number> *     vals      = out_values + idx;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const VectorizedArray<Number> v0        = values_face[0][val_idx];
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> d0        = grads_face[0][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
            vals[i * stride] += shape_vec[i][0] * v0 + shape_vec[i][1] * d0;
        }
  }

  unsigned int
  find_first_zero_bit(const std::uint64_t x)
  {
    std::uint64_t y = ~x;
    if(y == 0)
      return 64;

      // count number of trailing zero bits
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(y);
#else

    unsigned int pos = 0;
    while((y & 1) == 0)
    {
      y >>= 1;
      ++pos;
    }
    return pos;

#endif
  }

  void
  detect_dependencies_of_face_integrals()
  {
    Timer                      time;
    const Triangulation<dim> & tria = dof_handler->get_triangulation();

    std::vector<int>   touch_length(tria.n_raw_faces(), 0);
    const unsigned int n_cell_batches = dof_indices.size();
    for(unsigned int c = 0, count = 1; c < n_cell_batches; ++c)
      for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
      {
        const auto cell = get_cell_iterator(c, v);
        for(unsigned int f = 0; f < 2 * dim; ++f)
        {
          const unsigned int face_idx = cell->face(f)->index();
          const unsigned int periodic_idx =
            (cell->at_boundary(f) && cell->has_periodic_neighbor(f)) ?
              cell->periodic_neighbor(f)->face(cell->periodic_neighbor_face_no(f))->index() :
              face_idx;
          if(touch_length[face_idx] == 0)
          {
            touch_length[face_idx]     = -count;
            touch_length[periodic_idx] = -count;
          }
          else
          {
            AssertThrow(touch_length[face_idx] < 0, ExcInternalError());
            touch_length[face_idx]     = count + touch_length[face_idx];
            touch_length[periodic_idx] = touch_length[face_idx];
          }
        }
        ++count;
      }

#if 0
    // Some debugging to check the distances of faces from first to last touch
    std::map<unsigned int, unsigned int> distances;
    for (const int a : touch_length)
      if (a > 0)
        {
          if (distances.find(a) == distances.end())
            distances[a] = 1;
          else
            ++distances[a];
        }
    std::cout << "faces: ";
    for (const auto it : distances)
      std::cout << it.first << "|" << it.second << "  ";
    std::cout << std::endl;
#endif

    face_flux_buffer_index.resize(dof_indices.size());
    all_left_face_fluxes_from_buffer.resize(dof_indices.size());
    const unsigned int n_data_per_face = Utilities::pow(quad_weights_h_z.size(), dim - 1) * 2;

    constexpr unsigned int     long_range_start = 16;
    std::vector<std::uint64_t> face_storage(1);
    std::vector<unsigned int>  face_index_in_storage(tria.n_raw_faces(), 0);
    unsigned int               size_of_storage = 1;
    for(unsigned int c = 0; c < n_cell_batches; ++c)
      for(unsigned int d = 0; d < dim; ++d)
      {
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
        {
          const auto cell = get_cell_iterator(c, v);

          // For cells at the right boundary (including those adjacent to
          // remote MPI processes), set the index to 0 to indicate a sink
          // that will receive vectorized writes but not get picked up
          // again later. This is one single array and thus acceptable.
          if(cell->at_boundary(2 * d + 1) ||
             cell->neighbor(2 * d + 1)->subdomain_id() != cell->subdomain_id())
          {
            face_flux_buffer_index[c][2 * d + 1][v] = 0;
            continue;
          }
          const unsigned int face_idx = cell->face(2 * d + 1)->index();

          const auto add_entry = [&](const unsigned int position)
          {
            const unsigned int entry_within_vector = position / 64;
            const unsigned int bit_within_entry    = position % 64;

            // flag current bit as 'occupied'
            face_storage[entry_within_vector] |= std::uint64_t(1) << bit_within_entry;
            face_index_in_storage[face_idx]         = position + 1;
            face_flux_buffer_index[c][2 * d + 1][v] = (position + 1) * n_data_per_face;
            size_of_storage                         = std::max(size_of_storage, position + 2);
          };

          // Only short-distance faces are allowed in the first part of
          // the vector to ensure we have a high cache locality in some
          // part of the final data vector for the flux
          // buffer. Furthermore, the complexity of the linear search we
          // do in this array does not grow too large.
          if(touch_length[face_idx] < 40)
          {
            const unsigned int pos = find_first_zero_bit(face_storage[0]);
            if(pos < 64)
            {
              add_entry(pos);
              continue;
            }
          }
          else
          {
            // should only look in later part
            unsigned int pos = find_first_zero_bit(face_storage[0] >> long_range_start);
            pos += long_range_start;
            if(pos < 64)
            {
              add_entry(pos);
              continue;
            }
          }
          // if we did not find it in the first part, we need to perform a
          // linear search in the storage array, which we accelerate by
          // the bit-processing function
          bool added_this = false;
          for(unsigned int s = 1; s < face_storage.size(); ++s)
          {
            const unsigned int pos = find_first_zero_bit(face_storage[s]);
            if(pos < 64)
            {
              add_entry(pos + 64 * s);
              added_this = true;
              break;
            }
          }
          if(!added_this)
          {
            const unsigned int pos = face_storage.size() * 64;
            face_storage.emplace_back(1);
            face_index_in_storage[face_idx]         = pos + 1;
            face_flux_buffer_index[c][2 * d + 1][v] = (pos + 1) * n_data_per_face;
            size_of_storage += 1;
          }
        }
        bool face_missing = false;
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
        {
          const auto         cell     = get_cell_iterator(c, v);
          const unsigned int face_idx = cell->face(2 * d)->index();
          if(face_index_in_storage[face_idx] > 0)
          {
            AssertIndexRange(face_index_in_storage[face_idx], size_of_storage);
            face_flux_buffer_index[c][2 * d][v] = face_index_in_storage[face_idx] * n_data_per_face;
            const unsigned int index_in_storage = (face_index_in_storage[face_idx] - 1) / 64;
            const unsigned int bit_index        = (face_index_in_storage[face_idx] - 1) % 64;
            face_storage[index_in_storage] &= ~(std::uint64_t(1) << bit_index);
          }
          else
            face_missing = true;
        }
        all_left_face_fluxes_from_buffer[c][d] = (face_missing == false);
      }
    face_flux_buffer.resize(n_data_per_face * size_of_storage);

    unsigned int count_0 = 0, count_1 = 0;
    for(const auto & a : all_left_face_fluxes_from_buffer)
      for(const char b : a)
        if(b == 0)
          ++count_0;
        else
          ++count_1;

    RTOperator::print_time(static_cast<double>(count_1) / (count_0 + count_1),
                           "Face flux buffer efficiency",
                           MPI_COMM_WORLD);

#if 0
    std::cout << "Identified storage of length " << size_of_storage << " with "
              << face_flux_buffer.size() * sizeof(Number) << " bytes, quality: "
              << count_1 << " of " << (count_0 + count_1) << std::endl;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Time analyze dependencies: " << time.wall_time() << std::endl;
#endif
  }
};


} // namespace LaplaceOperator
