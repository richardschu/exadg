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


namespace RTOperator
{
void
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
void
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


template<int dim, int degree, typename Number>
void
distribute_local_to_global_rt_compressed(
  const dealii::VectorizedArray<Number> * local_array,
  const unsigned int                      n_filled_lanes,
  const dealii::ndarray<unsigned int, 2 * dim + 1, dealii::VectorizedArray<Number>::size()> &
           dof_indices,
  Number * dst_vector)
{
  constexpr unsigned int n_lanes       = dealii::VectorizedArray<Number>::size();
  constexpr unsigned int dofs_per_face = dealii::Utilities::pow(degree, dim - 1);
  constexpr unsigned int dofs_per_comp = dofs_per_face * (degree + 1);

  dealii::VectorizedArray<Number> data[dofs_per_face];
  for(unsigned int f = 0; f < 2; ++f)
  {
    for(unsigned int i = 0; i < dofs_per_face; ++i)
      data[i] = local_array[i * (degree + 1) + f * degree];

    // check if indices unconstrained
    bool all_indices_unconstrained = n_filled_lanes == n_lanes;
    for(const unsigned int i : dof_indices[f])
      if(i == dealii::numbers::invalid_unsigned_int)
      {
        all_indices_unconstrained = false;
        break;
      }
    if(all_indices_unconstrained)
      vectorized_transpose_and_store(true, dofs_per_face, data, dof_indices[f].data(), dst_vector);
    else
    {
      for(unsigned int v = 0; v < n_filled_lanes; ++v)
        if(dof_indices[f][v] != dealii::numbers::invalid_unsigned_int)
        {
          Number * dst_ptr = dst_vector + dof_indices[f][v];
          for(unsigned int i = 0; i < dofs_per_face; ++i)
            dst_ptr[i] += data[i][v];
        }
    }
  }
  if(dim == 3)
  {
    for(unsigned int f = 0; f < 2; ++f)
    {
      for(unsigned int i = 0; i < degree; ++i)
        for(unsigned int j = 0; j < degree; ++j)
          data[i * degree + j] =
            local_array[dofs_per_comp + i * (degree + 1) * degree + j + f * degree * degree];

      // check if indices unconstrained
      bool all_indices_unconstrained = n_filled_lanes == n_lanes;
      for(const unsigned int i : dof_indices[2 + f])
        if(i == dealii::numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_transpose_and_store(
          true, dofs_per_face, data, dof_indices[2 + f].data(), dst_vector);
      else
      {
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          if(dof_indices[2 + f][v] != dealii::numbers::invalid_unsigned_int)
          {
            Number * dst_ptr = dst_vector + dof_indices[2 + f][v];
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              dst_ptr[i] += data[i][v];
          }
      }
    }
  }

  for(unsigned int f = 0; f < 2; ++f)
  {
    const dealii::VectorizedArray<Number> * data =
      local_array + (dim - 1) * dofs_per_comp + f * degree * dofs_per_face;

    // check if indices unconstrained
    bool all_indices_unconstrained = n_filled_lanes == n_lanes;
    for(const unsigned int i : dof_indices[2 * dim - 2 + f])
      if(i == dealii::numbers::invalid_unsigned_int)
      {
        all_indices_unconstrained = false;
        break;
      }
    if(all_indices_unconstrained)
      vectorized_transpose_and_store(
        true, dofs_per_face, data, dof_indices[2 * dim - 2 + f].data(), dst_vector);
    else
    {
      for(unsigned int v = 0; v < n_filled_lanes; ++v)
        if(dof_indices[2 * dim - 2 + f][v] != dealii::numbers::invalid_unsigned_int)
        {
          Number * dst_ptr = dst_vector + dof_indices[2 * dim - 2 + f][v];
          for(unsigned int i = 0; i < dofs_per_face; ++i)
            dst_ptr[i] += data[i][v];
        }
    }
  }

  dealii::VectorizedArray<Number> data_2[dofs_per_comp - 2 * dofs_per_face];
  for(unsigned int i = 0; i < dofs_per_face; ++i)
    for(unsigned int j = 1; j < degree; ++j)
      data_2[i * (degree - 1) + j - 1] = local_array[i * (degree + 1) + j];

  if(n_filled_lanes == n_lanes)
    vectorized_transpose_and_store(
      true, dofs_per_comp - 2 * dofs_per_face, data_2, dof_indices[2 * dim].data(), dst_vector);
  else
    for(unsigned int v = 0; v < n_filled_lanes; ++v)
    {
      Number * dst_ptr = dst_vector + dof_indices[2 * dim][v];
      for(unsigned int i = 0; i < dofs_per_comp - 2 * dofs_per_face; ++i)
        dst_ptr[i] += data_2[i][v];
    }
  if(dim == 3)
    for(unsigned int i = 0; i < degree; ++i)
    {
      const dealii::VectorizedArray<Number> * data =
        local_array + dofs_per_comp + i * degree * (degree + 1) + degree;
      Number * dst_ptr = dst_vector + dofs_per_comp - 2 * dofs_per_face + i * (degree - 1) * degree;
      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(
          true, degree * (degree - 1), data, dof_indices[2 * dim].data(), dst_ptr);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          for(unsigned int i = 0; i < degree * (degree - 1); ++i)
            dst_ptr[i] += data[i][v];
        }
    }

  const dealii::VectorizedArray<Number> * my_data =
    local_array + (dim - 1) * dofs_per_comp + dofs_per_face;
  Number * dst_ptr = dst_vector + (dim - 1) * dofs_per_comp - (2 * dim - 2) * dofs_per_face;

  if(n_filled_lanes == n_lanes)
    vectorized_transpose_and_store(
      true, dofs_per_comp - 2 * dofs_per_face, my_data, dof_indices[2 * dim].data(), dst_ptr);
  else
    for(unsigned int v = 0; v < n_filled_lanes; ++v)
    {
      for(unsigned int i = 0; i < dofs_per_comp - 2 * dofs_per_face; ++i)
        dst_ptr[i] += my_data[i][v];
    }
}


using namespace dealii;



namespace mpi_shared_memory
{
template<typename Number>
void
allocate_shared_recv_data(MemorySpace::MemorySpaceData<Number, MemorySpace::Host> & data,
                          const unsigned int                                        expected_size,
                          const MPI_Comm communicator_shared)
{
  const unsigned int n_shared_ranks = Utilities::MPI::n_mpi_processes(communicator_shared);
  const unsigned int rank_shared    = Utilities::MPI::this_mpi_process(communicator_shared);

  MPI_Win  mpi_window;
  Number * data_this;

  std::vector<Number *> others(n_shared_ranks);

  MPI_Info info;
  int      ierr = MPI_Info_create(&info);
  AssertThrowMPI(ierr);

  ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");
  AssertThrowMPI(ierr);

  const std::size_t align_by = 64;

  std::size_t allocated_bytes =
    ((expected_size * sizeof(Number) + align_by) / sizeof(Number)) * sizeof(Number);

  ierr = MPI_Win_allocate_shared(
    allocated_bytes, sizeof(Number), info, communicator_shared, &data_this, &mpi_window);
  AssertThrowMPI(ierr);

  for(unsigned int i = 0; i < n_shared_ranks; ++i)
  {
    int        disp_unit;
    MPI_Aint   segment_size;
    const auto ierr = MPI_Win_shared_query(mpi_window, i, &segment_size, &disp_unit, &others[i]);
    AssertDimension(disp_unit, sizeof(Number));
    AssertThrowMPI(ierr);
  }

  Number * ptr_unaligned = others[rank_shared];
  Number * ptr_aligned   = ptr_unaligned;

  AssertThrow(std::align(align_by,
                         expected_size * sizeof(Number),
                         reinterpret_cast<void *&>(ptr_aligned),
                         allocated_bytes) != nullptr,
              dealii::ExcNotImplemented());

  unsigned int              n_align_local = ptr_aligned - ptr_unaligned;
  std::vector<unsigned int> n_align_sm(n_shared_ranks);

  ierr = MPI_Allgather(
    &n_align_local, 1, MPI_UNSIGNED, n_align_sm.data(), 1, MPI_UNSIGNED, communicator_shared);
  AssertThrowMPI(ierr);

  for(unsigned int i = 0; i < n_shared_ranks; ++i)
    others[i] += n_align_sm[i];

  std::vector<unsigned int> allocated_sizes(n_shared_ranks);

  ierr = MPI_Allgather(
    &expected_size, 1, MPI_UNSIGNED, allocated_sizes.data(), 1, MPI_UNSIGNED, communicator_shared);
  AssertThrowMPI(ierr);

  data.values_sm.resize(n_shared_ranks);
  for(unsigned int i = 0; i < n_shared_ranks; ++i)
    data.values_sm[i] = ArrayView<const Number>(others[i], allocated_sizes[i]);

  data.values = Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
    ptr_aligned, expected_size);

  // Kokkos will not free the memory because the memory is
  // unmanaged. Instead we use a shared pointer to take care of
  // that.
  data.values_sm_ptr = {ptr_aligned, [mpi_window](Number *) mutable {
                          // note: we are creating here a copy of
                          // the window other approaches led to
                          // segmentation faults
                          const auto ierr = MPI_Win_free(&mpi_window);
                          AssertThrowMPI(ierr);
                        }};
}
} // namespace mpi_shared_memory



template<int dim, typename Number = double>
class RaviartThomasOperatorBase : public EnableObserverPointer
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  RaviartThomasOperatorBase() = default;

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
      const MPI_Comm                comm               = dof_handler.get_mpi_communicator();
      IndexSet                      locally_owned_dofs = dof_handler.locally_owned_dofs();
      const types::global_dof_index n_unconstrained_owned_dofs =
        locally_owned_dofs.n_elements() - matrix_free.get_constrained_dofs().size();
      const types::global_dof_index first_constrained_dof =
        matrix_free.get_constrained_dofs().empty() ? n_unconstrained_owned_dofs :
                                                     matrix_free.get_constrained_dofs()[0];

      AssertThrow(first_constrained_dof == n_unconstrained_owned_dofs,
                  ExcMessage("Expected all constrained DoFs to be sorted to "
                             "the end of locally owned range, but found " +
                             std::to_string(first_constrained_dof) + " vs " +
                             std::to_string(n_unconstrained_owned_dofs)));

      // Assign DoF numbers for the unconstrained degrees of freedom only
      std::pair<types::global_dof_index, types::global_dof_index> positions =
        Utilities::MPI::partial_and_total_sum(n_unconstrained_owned_dofs, comm);
      std::vector<types::global_dof_index> dof_numbers(locally_owned_dofs.n_elements(),
                                                       numbers::invalid_dof_index);
      types::global_dof_index              counter = positions.first;
      for(unsigned int i = 0; i < n_unconstrained_owned_dofs; ++i)
        dof_numbers[i] = counter++;

      // Extract ghost entries for which we need to query the numbers from
      // remote processes - we do not know the start indices of the respective
      // processes even though we could query the DoF index owner through the
      // triangulation, so we need to perform a lookup anyway and do that by a
      // ghost exchange of all data. While there, also extract a compressed
      // representation, taking the first number on each entity (face, cell)
      // in global index space, which we later translate to local numbers

      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
      std::vector<dealii::ndarray<types::global_dof_index, 2 * dim + 1, n_lanes>>
                                           dof_indices_per_entity(matrix_free.n_cell_batches());
      std::vector<types::global_dof_index> ghost_indices;
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        {
          matrix_free.get_cell_iterator(cell, v)->get_dof_indices(local_dof_indices);

          // Adjust dof indices due to periodicity
          for(types::global_dof_index & a : local_dof_indices)
          {
            const auto line = constraints.get_constraint_entries(a);
            if(line != nullptr && line->size() == 1 && (*line)[0].second == Number(1.0))
              a = (*line)[0].first;
          }
          for(types::global_dof_index a : local_dof_indices)
            if(!locally_owned_dofs.is_element(a) && !constraints.is_constrained(a))
              ghost_indices.push_back(a);

          const unsigned int dofs_per_face = fe.dofs_per_face;
          for(unsigned int f = 0; f < 2 * dim; ++f)
          {
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              AssertThrow(local_dof_indices[f * dofs_per_face + i] ==
                            local_dof_indices[f * dofs_per_face] + i,
                          ExcInternalError());

            dof_indices_per_entity[cell][f][v] = local_dof_indices[f * dofs_per_face];
          }
          const unsigned int start_cell_dofs = 2 * dim * dofs_per_face;
          for(unsigned int i = 0; i < fe.dofs_per_cell - start_cell_dofs; ++i)
            AssertThrow(local_dof_indices[start_cell_dofs + i] ==
                          local_dof_indices[start_cell_dofs] + i,
                        ExcInternalError());
          dof_indices_per_entity[cell][2 * dim][v] = local_dof_indices[start_cell_dofs];
        }

      IndexSet ghost_index_set(locally_owned_dofs.size());
      ghost_index_set.add_indices(ghost_indices.begin(), ghost_indices.end());
      ghost_index_set.compress();
      Utilities::MPI::Partitioner partitioner_dofs(locally_owned_dofs, ghost_index_set, comm);

      std::vector<types::global_dof_index> tmp_array(partitioner_dofs.n_import_indices());
      std::vector<types::global_dof_index> numbers_ghosts(partitioner_dofs.n_ghost_indices());
      std::vector<MPI_Request>             requests;
      partitioner_dofs.export_to_ghosted_array_start(3,
                                                     make_const_array_view(dof_numbers),
                                                     make_array_view(tmp_array),
                                                     make_array_view(numbers_ghosts),
                                                     requests);
      partitioner_dofs.export_to_ghosted_array_finish(make_array_view(numbers_ghosts), requests);

      IndexSet owned_dofs(positions.second);
      owned_dofs.add_range(positions.first, positions.first + n_unconstrained_owned_dofs);
      std::vector<types::global_dof_index> compressed_ghost;
      compressed_ghost.reserve(numbers_ghosts.size());
      for(const types::global_dof_index a : numbers_ghosts)
        if(a != numbers::invalid_dof_index)
          compressed_ghost.push_back(a);
      IndexSet ghost_dofs(positions.second);
      ghost_dofs.add_indices(compressed_ghost.begin(), compressed_ghost.end());
      ghost_dofs.compress();

      partitioner = std::make_shared<Utilities::MPI::Partitioner>(owned_dofs, ghost_dofs, comm);

      {
        ndarray<unsigned int, 2 * dim, n_lanes> default_argument;
        for(unsigned int i = 0; i < 2 * dim; ++i)
          for(unsigned int j = 0; j < n_lanes; ++j)
            default_argument[i][j] = numbers::invalid_unsigned_int;
        neighbor_cells.resize(matrix_free.n_cell_batches(), default_argument);
        mpi_exchange_data_on_faces.resize(matrix_free.n_cell_batches(), default_argument);
      }
      {
        ndarray<unsigned int, 2 * dim + 1, n_lanes> default_argument;
        for(unsigned int i = 0; i < 2 * dim + 1; ++i)
          for(unsigned int j = 0; j < n_lanes; ++j)
            default_argument[i][j] = numbers::invalid_unsigned_int;
        dof_indices.resize(matrix_free.n_cell_batches(), default_argument);
      }

      std::vector<unsigned int> cell_indices(dof_handler.get_triangulation().n_active_cells(),
                                             numbers::invalid_unsigned_int);
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          if(v >= matrix_free.n_active_entries_per_cell_batch(cell))
          {
            // fill in valid data to simplify work with vector access
            for(unsigned int f = 0; f < 2 * dim + 1; ++f)
              dof_indices[cell][f][v] = dof_indices[cell][f][0];
            continue;
          }

          for(unsigned int f = 0; f < 2 * dim + 1; ++f)
          {
            const types::global_dof_index index    = dof_indices_per_entity[cell][f][v];
            const unsigned int            my_index = partitioner_dofs.global_to_local(index);
            types::global_dof_index       number_compressed;
            if(my_index < locally_owned_dofs.n_elements())
              number_compressed = dof_numbers[my_index];
            else
              number_compressed = numbers_ghosts[my_index - dof_numbers.size()];
            Assert(number_compressed != numbers::invalid_dof_index ||
                     constraints.is_constrained(index),
                   ExcInternalError());
            if(number_compressed != numbers::invalid_dof_index)
              dof_indices[cell][f][v] = partitioner->global_to_local(number_compressed);
            else
              dof_indices[cell][f][v] = numbers::invalid_unsigned_int;
          }

          cell_indices[matrix_free.get_cell_iterator(cell, v)->active_cell_index()] =
            cell * VectorizedArray<Number>::size() + v;
        }

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

      const unsigned int n_t = fe.degree;
      const unsigned int n_n = n_t + 1;
      // data: projected to face, 2 because of values and derivatives
      const unsigned int data_per_face =
        2 * (Utilities::pow(n_t, dim - 1) + n_n * Utilities::pow(n_t, dim - 2) * (dim - 1));

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
                     const std::array<types::global_dof_index, 5> & b) {
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
                     const std::array<types::global_dof_index, 5> & b) {
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
    FEValues<dim>     fe_values(mapping,
                            dummy_fe,
                            Quadrature<dim>(points),
                            update_jacobians | update_jacobian_grads);
    FEFaceValues<dim> fe_face_values(mapping,
                                     dummy_fe,
                                     face_quadrature,
                                     update_jacobians | update_JxW_values | update_jacobian_grads |
                                       update_normal_vectors);

    jacobians_xy.resize(n_q_points_2d * unique_cells.size());
    jacobian_grads.resize(n_q_points_2d * unique_cells.size());
    face_jacobians_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_jacobian_grads.resize(4 * n_q_points_1d * unique_cells.size());
    face_normal_vector_xy.resize(4 * n_q_points_1d * unique_cells.size());

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
        const unsigned int data_idx = index[0] * n_q_points_2d + q;
        if(dim == 3)
        {
          h_z         = jacobian[2][2];
          h_z_inverse = 1. / h_z;
        }
        for(unsigned int d = 0; d < 2; ++d)
          for(unsigned int e = 0; e < 2; ++e)
            jacobians_xy[data_idx][d][e] = jacobian[d][e];
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

    {
      quad_weights_xy.resize(n_q_points_2d);
      QGauss<2> quad(n_q_points_1d);
      for(unsigned int q = 0; q < quad.size(); ++q)
        quad_weights_xy[q] = quad.weight(q);
    }
    {
      quad_weights_z.resize(n_q_points_1d);
      QGauss<1> quad(n_q_points_1d);
      for(unsigned int q = 0; q < quad.size(); ++q)
        quad_weights_z[q] = quad.weight(q);

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

  ~RaviartThomasOperatorBase()
  {
    if(timings[0] > 0)
    {
      const MPI_Comm comm = dof_handler->get_mpi_communicator();
      std::cout << std::defaultfloat << std::setprecision(3);
      const double total_time =
        Utilities::MPI::sum(timings[9], comm) / timings[0] / Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for RT Laplace operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[0])
                  << " evaluations [t_total=" << total_time * timings[0] << "s]" << std::endl;
      print_time(timings[1] / timings[0], "Update cell ghost values", comm, total_time);
      if(Utilities::MPI::n_mpi_processes(comm) > 1)
      {
        print_time(timings[3] / timings[0], "Pack/send data dg ghosts", comm, total_time);
        print_time(timings[5] / timings[0], "MPI_Waitall dg ghosts", comm, total_time);
        print_time(timings[4] / timings[0], "Pre-loop before ghosts", comm, total_time);
      }

      print_time(timings[7] / timings[0], "Matrix-free loop", comm, total_time);
      print_time(timings[8] / timings[0], "Compress cell ghost values", comm, total_time);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
    if(timings[10] > 0)
    {
      const MPI_Comm comm       = MPI_COMM_WORLD;
      const double   total_time = Utilities::MPI::sum(timings[14], comm) / timings[10] /
                                Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for mass operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[10])
                  << " evaluations [t_total=" << total_time * timings[10] << "s]" << std::endl;
      print_time(timings[11] / timings[10], "Update cell ghost values", comm, total_time);

      print_time(timings[12] / timings[10], "Matrix-free loop", comm, total_time);
      print_time(timings[13] / timings[10], "Compress cell ghost values", comm, total_time);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
  }

  void
  initialize_dof_vector(VectorType & vec) const
  {
    vec.reinit(partitioner);
  }

  template<typename VectorType1, typename VectorType2>
  void
  copy_mf_to_this_vector(const VectorType1 & src, VectorType2 & dst) const
  {
    Assert(partitioner->is_compatible(*dst.get_partitioner()), ExcInternalError());
    AssertIndexRange(dst.locally_owned_size(), src.locally_owned_size() + 1);
    std::copy(src.begin(), src.begin() + dst.locally_owned_size(), dst.begin());
  }

  template<typename VectorType1, typename VectorType2>
  void
  copy_this_to_mf_vector(const VectorType1 & src, VectorType2 & dst) const
  {
    Assert(partitioner->is_compatible(*src.get_partitioner()), ExcInternalError());
    AssertIndexRange(src.locally_owned_size(), dst.locally_owned_size() + 1);
    std::copy(src.begin(), src.begin() + src.locally_owned_size(), dst.begin());
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
  vmult_add(VectorType & dst, const VectorType & src) const
  {
    vmult(
      dst,
      src,
      [](const unsigned int, const unsigned int) {},
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

    for(unsigned int range = cell_loop_pre_list_index[n_cell_batches];
        range < cell_loop_pre_list_index[n_cell_batches + 1];
        ++range)
      before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);

    if(factor_laplace != 0.)
    {
      src.update_ghost_values_start();
      timings[1] += time.wall_time();
      time.restart();

      for(unsigned int range = cell_loop_pre_list_index[n_cell_batches + 1];
          range < cell_loop_pre_list_index[n_cell_batches + 2];
          ++range)
        before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);

      timings[4] += time.wall_time();
      time.restart();
      src.update_ghost_values_finish();
      timings[1] += time.wall_time();
    }
    else
    {
      src.update_ghost_values();
      timings[11] += time.wall_time();
    }

    time.restart();

    // only do the data exchange for face integral if we have Laplacian contribution
    if(factor_laplace != 0. && Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      const int n_t = shape_info.data[0].fe_degree;
      const int n_n = n_t + 1;

      // data: everything projected to face, 2 because of values and derivatives
      const int data_per_face =
        2 * (Utilities::pow(n_t, dim - 1) + n_n * Utilities::pow(n_t, dim - 2) * (dim - 1));

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

      if(n_t == 2)
        vmult_pack_and_send_data<2>(src, requests);
      else if(n_t == 3)
        vmult_pack_and_send_data<3>(src, requests);
      else if(n_t == 4)
        vmult_pack_and_send_data<4>(src, requests);
#ifndef DEBUG
      else if(n_t == 5)
        vmult_pack_and_send_data<5>(src, requests);
      else if(n_t == 6)
        vmult_pack_and_send_data<6>(src, requests);
      else if(n_t == 7)
        vmult_pack_and_send_data<7>(src, requests);
      else if(n_t == 8)
        vmult_pack_and_send_data<8>(src, requests);
      else if(n_t == 9)
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
      if(degree == 2)
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
      else if(degree == 9)
        do_cell_operation<9, true>(cell, src, dst);
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

    time.restart();
    dst.compress_start(0, VectorOperation::add);
    src.zero_out_ghost_values();
    dst.compress_finish(VectorOperation::add);

    for(unsigned int range = cell_loop_post_list_index[n_cell_batches];
        range < cell_loop_post_list_index[n_cell_batches + 1];
        ++range)
      after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);

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
           MemoryConsumption::memory_consumption(jacobian_grads) +
           MemoryConsumption::memory_consumption(face_mapping_data_index) +
           MemoryConsumption::memory_consumption(face_normal_vector_xy) +
           MemoryConsumption::memory_consumption(face_jacobians_xy) +
           MemoryConsumption::memory_consumption(face_jacobian_grads) +
           MemoryConsumption::memory_consumption(quad_weights_xy) +
           MemoryConsumption::memory_consumption(quad_weights_z) +
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
  std::vector<dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes>> dof_indices;
  std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>>     neighbor_cells;
  std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>>     mpi_exchange_data_on_faces;
  mutable AlignedVector<Number>                                    import_values;
  mutable AlignedVector<Number>                                    export_values;
  Table<2, unsigned char>                                          all_owned_faces;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  mutable std::array<double, 15> timings;

  Number factor_mass;
  Number factor_laplace;

  Number                                            h_z;
  Number                                            h_z_inverse;
  std::vector<std::array<unsigned int, n_lanes>>    mapping_data_index;
  AlignedVector<Tensor<2, 2, Number>>               jacobians_xy;
  AlignedVector<Tensor<1, 3, Tensor<1, 2, Number>>> jacobian_grads;
  std::vector<ndarray<unsigned int, 4, 2, n_lanes>> face_mapping_data_index;
  AlignedVector<Tensor<1, 2, Number>>               face_normal_vector_xy;
  AlignedVector<Tensor<2, 2, Number>>               face_jacobians_xy;
  AlignedVector<Tensor<1, 3, Tensor<1, 2, Number>>> face_jacobian_grads;
  std::vector<Number>                               quad_weights_xy;
  std::vector<Number>                               quad_weights_z;
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
    const unsigned int        dofs_on_quad   = dof_handler->get_fe().dofs_per_face;
    const unsigned int dofs_on_hex = dof_handler->get_fe().dofs_per_cell - 2 * dim * dofs_on_quad;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int face = 0; face < 2 * dim; ++face)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int c = neighbor_cells[cell][face][v];
          if(c != numbers::invalid_unsigned_int)
          {
            for(unsigned int i = 0; i < 2 * dim; ++i)
            {
              const unsigned int idx = dof_indices[c / n_lanes][i][c % n_lanes];
              if(idx < n_dofs)
              {
                const unsigned int first = idx / chunk_size;
                const unsigned int last  = (idx + dofs_on_quad - 1) / chunk_size;
                for(unsigned int j = first; j <= last; ++j)
                  if(touched_first_by[j] == numbers::invalid_unsigned_int)
                    touched_first_by[j] = cell;
              }
            }
            const unsigned int idx = dof_indices[c / n_lanes][2 * dim][c % n_lanes];
            if(idx < n_dofs)
            {
              const unsigned int first = idx / chunk_size;
              const unsigned int last  = (idx + dofs_on_hex - 1) / chunk_size;
              for(unsigned int j = first; j <= last; ++j)
                if(touched_first_by[j] == numbers::invalid_unsigned_int)
                  touched_first_by[j] = cell;
            }
          }
        }
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        for(unsigned int i = 0; i < 2 * dim; ++i)
        {
          const unsigned int idx = dof_indices[cell][i][v];
          if(idx < n_dofs)
          {
            const unsigned int first = idx / chunk_size;
            const unsigned int last  = (idx + dofs_on_quad - 1) / chunk_size;
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
        const unsigned int idx = dof_indices[cell][2 * dim][v];
        if(idx < n_dofs)
        {
          const unsigned int first = idx / chunk_size;
          const unsigned int last  = (idx + dofs_on_hex - 1) / chunk_size;
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

      print_time(static_cast<double>(n_batches_half) / touched_first_by.size(),
                 "Pre-/post distance > 1/2 size",
                 MPI_COMM_WORLD);
      print_time(static_cast<double>(n_batches_half + n_batches_10) / touched_first_by.size(),
                 "Pre-/post distance > 10",
                 MPI_COMM_WORLD);
    }

    // set the import indices in the data to be exchanged via the partitioner
    // to one index higher to indicate that we want to process them first and
    // to two indices higher for the face integrals.
    for(auto it : partitioner->import_indices())
      for(unsigned int i = it.first; i < it.second; ++i)
        touched_first_by[i / chunk_size] = n_cell_batches;
    for(auto it : partitioner->import_indices())
      for(unsigned int i = it.first; i < it.second; ++i)
        touched_mass_first_by[i / chunk_size] = n_cell_batches;

    for(const unsigned int cell_lane : send_data_cell_index)
    {
      const unsigned int cell = cell_lane / n_lanes;
      const unsigned int lane = cell_lane % n_lanes;
      for(unsigned int i = 0; i < 2 * dim; ++i)
      {
        const unsigned int idx = dof_indices[cell][i][lane];
        if(idx < n_dofs)
        {
          const unsigned int first = idx / chunk_size;
          const unsigned int last  = (idx + dofs_on_quad - 1) / chunk_size;
          for(unsigned int j = first; j <= last; ++j)
            if(touched_first_by[j] != n_cell_batches)
              touched_first_by[j] = n_cell_batches + 1;
        }
      }
      const unsigned int idx = dof_indices[cell][2 * dim][lane];
      if(idx < n_dofs)
      {
        const unsigned int first = idx / chunk_size;
        const unsigned int last  = (idx + dofs_on_hex - 1) / chunk_size;
        for(unsigned int j = first; j <= last; ++j)
          if(touched_first_by[j] != n_cell_batches)
            touched_first_by[j] = n_cell_batches + 1;
      }
    }

    // since the write is only on the cell indices, we only need to pick up
    // those indices with access to the vector partitioner in the post list
    for(auto it : partitioner->import_indices())
      for(unsigned int i = it.first; i < it.second; ++i)
        touched_last_by[i / chunk_size] = n_cell_batches;

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

      print_time(static_cast<double>(n_batches_half) / touched_first_by.size(),
                 "Pre-/post distance > 1/2 size",
                 MPI_COMM_WORLD);
      print_time(static_cast<double>(n_batches_half + n_batches_10) / touched_first_by.size(),
                 "Pre-/post distance > 10",
                 MPI_COMM_WORLD);
    }

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_pre;
    for(unsigned int i = 0; i < touched_first_by.size(); ++i)
      chunk_must_do_pre[touched_first_by[i]].push_back(i);
    convert_map_to_range_list(n_cell_batches + 2,
                              chunk_size,
                              chunk_must_do_pre,
                              cell_loop_pre_list_index,
                              cell_loop_pre_list,
                              n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_mass_pre;
    for(unsigned int i = 0; i < touched_mass_first_by.size(); ++i)
      chunk_must_do_mass_pre[touched_mass_first_by[i]].push_back(i);
    convert_map_to_range_list(n_cell_batches + 1,
                              chunk_size,
                              chunk_must_do_mass_pre,
                              cell_loop_mass_pre_list_index,
                              cell_loop_mass_pre_list,
                              n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_post;
    for(unsigned int i = 0; i < touched_last_by.size(); ++i)
      chunk_must_do_post[touched_last_by[i]].push_back(i);
    convert_map_to_range_list(n_cell_batches + 1,
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

  template<int n_t, int n_points_1d>
  void
  read_cell_values(const Number *                                              src_vector,
                   const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & dof_indices,
                   VectorizedArray<Number> *                                   out) const
  {
    static_assert(n_t > 1, "Degree 0 not supported");
    constexpr unsigned int n_n                = n_t + 1;
    constexpr unsigned int dofs_per_face      = dealii::Utilities::pow(n_t, dim - 1);
    constexpr unsigned int dofs_per_plane     = n_t * (n_t - 1);
    constexpr unsigned int cell_dofs_per_comp = dofs_per_plane * (dim > 2 ? n_t : 1);

    VectorizedArray<Number> data_f[2][dofs_per_face];
    VectorizedArray<Number> data_cell[dofs_per_plane];
    VectorizedArray<Number> data_points[n_points_1d * n_points_1d];
    VectorizedArray<Number> data_interpolate[4 * n_n];

    const Number * DEAL_II_RESTRICT shape_data_n = shape_info.data[0].shape_values_eo.data();
    const Number * DEAL_II_RESTRICT shape_data_t = shape_info.data[1].shape_values_eo.data();
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const unsigned int i : dof_indices[f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_load_and_transpose(dofs_per_face, src_vector, dof_indices[f].data(), data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int idx = dof_indices[f][v];
          if(idx != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = src_vector[idx + i];
        }
      }
    }
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + i_z * dofs_per_plane,
                                    dof_indices[2 * dim].data(),
                                    data_cell);

      // perform interpolation in x direction
      for(unsigned int i_y = 0; i_y < n_t; ++i_y)
      {
        data_interpolate[0]   = data_f[0][i_z * n_t + i_y];
        data_interpolate[n_t] = data_f[1][i_z * n_t + i_y];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_interpolate[1 + i] = data_cell[i_y * (n_t - 1) + i];
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              1,
                                              1,
                                              true,
                                              false>(shape_data_n,
                                                     data_interpolate,
                                                     data_points + i_y * n_points_1d);
      }

      // perform interpolation in y direction
      VectorizedArray<Number> * out_z = out + i_z * dim * n_points_1d * n_points_1d;
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              n_points_1d,
                                              dim * n_points_1d,
                                              true,
                                              false>(shape_data_t,
                                                     data_points + i_x,
                                                     out_z + i_x * dim);
    }

    // y component
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const unsigned int i : dof_indices[2 + f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_load_and_transpose(dofs_per_face,
                                      src_vector,
                                      dof_indices[2 + f].data(),
                                      data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int idx = dof_indices[2 + f][v];
          if(idx != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = src_vector[idx + i];
        }
      }
    }
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + cell_dofs_per_comp + i_z * dofs_per_plane,
                                    dof_indices[2 * dim].data(),
                                    data_cell);

      // perform interpolation in y direction
      for(unsigned int i_x = 0; i_x < n_t; ++i_x)
      {
        data_interpolate[0]   = data_f[0][i_z * n_t + i_x];
        data_interpolate[n_t] = data_f[1][i_z * n_t + i_x];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_interpolate[1 + i] = data_cell[i * n_t + i_x];
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              1,
                                              n_t,
                                              true,
                                              false>(shape_data_n,
                                                     data_interpolate,
                                                     data_points + i_x);
      }

      // perform interpolation in x direction
      VectorizedArray<Number> * out_z = out + i_z * dim * n_points_1d * n_points_1d + 1;
      for(unsigned int i_y = 0; i_y < n_points_1d; ++i_y)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              1,
                                              dim,
                                              true,
                                              false>(shape_data_t,
                                                     data_points + i_y * n_t,
                                                     out_z + i_y * n_points_1d * dim);
    }
    if constexpr(dim == 3)
    {
      for(unsigned int i = 0; i < n_points_1d * n_points_1d; ++i)
        for(unsigned int j = 0; j < 2; ++j)
        {
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                n_points_1d * n_points_1d * dim,
                                                n_points_1d * n_points_1d * dim,
                                                true,
                                                false>(shape_data_t,
                                                       out + i * 3 + j,
                                                       out + i * 3 + j);
        }
      for(unsigned int f = 0; f < 2; ++f)
      {
        bool all_indices_unconstrained = true;
        for(const unsigned int i : dof_indices[4 + f])
          if(i == numbers::invalid_unsigned_int)
          {
            all_indices_unconstrained = false;
            break;
          }
        if(all_indices_unconstrained)
          vectorized_load_and_transpose(dofs_per_face,
                                        src_vector,
                                        dof_indices[4 + f].data(),
                                        data_f[f]);
        else
        {
          for(unsigned int i = 0; i < dofs_per_face; ++i)
            data_f[f][i] = 0;
          for(unsigned int v = 0; v < n_lanes; ++v)
          {
            const unsigned int idx = dof_indices[4 + f][v];
            if(idx != numbers::invalid_unsigned_int)
              for(unsigned int i = 0; i < dofs_per_face; ++i)
                data_f[f][i][v] = src_vector[idx + i];
          }
        }
      }

      AssertThrow(dofs_per_face >= 4, ExcNotImplemented());
      for(unsigned int i2 = 0; i2 < dofs_per_face; i2 += 4)
      {
        const unsigned int i = std::min(i2, dofs_per_face - 4);

        for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
        {
          vectorized_load_and_transpose(4,
                                        src_vector + 2 * cell_dofs_per_comp + i +
                                          i1 * dofs_per_face,
                                        dof_indices[2 * dim].data(),
                                        data_interpolate + (i1 + 1) * 4);
        }
        for(unsigned int j = 0; j < 4; ++j)
        {
          data_interpolate[j]           = data_f[0][i + j];
          data_interpolate[j + 4 * n_t] = data_f[1][i + j];
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_n,
                                                n_points_1d,
                                                4,
                                                n_points_1d * n_points_1d * dim,
                                                true,
                                                false>(shape_data_n,
                                                       data_interpolate + j,
                                                       out + (i + j) * dim + 2);
        }
      }
      for(unsigned int i2 = 0; i2 < n_points_1d; ++i2)
      {
        VectorizedArray<Number> * out_i2 = out + i2 * n_points_1d * n_points_1d * dim + 2;
        for(int i1 = n_t - 1; i1 >= 0; --i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                dim,
                                                dim,
                                                true,
                                                false>(shape_data_t,
                                                       out_i2 + i1 * n_t * dim,
                                                       out_i2 + i1 * n_points_1d * dim);
        for(unsigned int i1 = 0; i1 < n_points_1d; ++i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                dim * n_points_1d,
                                                dim * n_points_1d,
                                                true,
                                                false>(shape_data_t,
                                                       out_i2 + i1 * dim,
                                                       out_i2 + i1 * dim);
      }
    }
  }

  template<int n_t, int n_points_1d>
  void
  integrate_cell_scatter(const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & dof_indices,
                         const unsigned int                                          n_filled_lanes,
                         VectorizedArray<Number> *                                   quad_values,
                         Number * dst_vector) const
  {
    static_assert(n_t > 1, "Degree 0 not supported");
    constexpr unsigned int n_n                = n_t + 1;
    constexpr unsigned int dofs_per_face      = dealii::Utilities::pow(n_t, dim - 1);
    constexpr unsigned int dofs_per_plane     = n_t * (n_t - 1);
    constexpr unsigned int cell_dofs_per_comp = dofs_per_plane * (dim > 2 ? n_t : 1);

    VectorizedArray<Number> data_f[2][dofs_per_face];
    VectorizedArray<Number> data_cell[dofs_per_plane];
    VectorizedArray<Number> data_points[n_points_1d * n_points_1d];
    VectorizedArray<Number> data_interpolate[n_n];

    const Number * DEAL_II_RESTRICT shape_data_n = shape_info.data[0].shape_values_eo.data();
    const Number * DEAL_II_RESTRICT shape_data_t = shape_info.data[1].shape_values_eo.data();

    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_1d * n_points_1d; ++i)
      {
        for(unsigned int j = 0; j < 2; ++j)
        {
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                n_points_1d * n_points_1d * dim,
                                                n_points_1d * n_points_1d * dim,
                                                false,
                                                false>(shape_data_t,
                                                       quad_values + i * 3 + j,
                                                       quad_values + i * 3 + j);
        }
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              n_points_1d * n_points_1d * dim,
                                              n_points_1d * n_points_1d * dim,
                                              false,
                                              false>(shape_data_n,
                                                     quad_values + i * 3 + 2,
                                                     quad_values + i * 3 + 2);
      }

    // x component
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      // perform interpolation in y direction
      VectorizedArray<Number> * in_z = quad_values + i_z * dim * n_points_1d * n_points_1d;
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              dim * n_points_1d,
                                              n_points_1d,
                                              false,
                                              false>(shape_data_t,
                                                     in_z + i_x * dim,
                                                     data_points + i_x);

      // perform interpolation in x direction
      for(unsigned int i_y = 0; i_y < n_t; ++i_y)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              1,
                                              1,
                                              false,
                                              false>(shape_data_n,
                                                     data_points + i_y * n_points_1d,
                                                     data_interpolate);
        data_f[0][i_z * n_t + i_y] = data_interpolate[0];
        data_f[1][i_z * n_t + i_y] = data_interpolate[n_t];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_cell[i_y * (n_t - 1) + i] = data_interpolate[1 + i];
      }

      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(true,
                                       dofs_per_plane,
                                       data_cell,
                                       dof_indices[2 * dim].data(),
                                       dst_vector + i_z * dofs_per_plane);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          Number * dst_ptr = dst_vector + dof_indices[2 * dim][v] + i_z * dofs_per_plane;
          for(unsigned int i = 0; i < dofs_per_plane; ++i)
            dst_ptr[i] += data_cell[i][v];
        }
    }
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = n_filled_lanes == n_lanes;
      for(const unsigned int i : dof_indices[f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_transpose_and_store(
          true, dofs_per_face, data_f[f], dof_indices[f].data(), dst_vector);
      else
      {
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          if(dof_indices[f][v] != numbers::invalid_unsigned_int)
          {
            Number * dst_ptr = dst_vector + dof_indices[f][v];
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              dst_ptr[i] += data_f[f][i][v];
          }
      }
    }

    // y component
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      // perform interpolation in x direction
      VectorizedArray<Number> * in_z = quad_values + i_z * dim * n_points_1d * n_points_1d + 1;
      for(unsigned int i_y = 0; i_y < n_points_1d; ++i_y)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              dim,
                                              1,
                                              false,
                                              false>(shape_data_t,
                                                     in_z + i_y * n_points_1d * dim,
                                                     data_points + i_y * n_t);

      // perform interpolation in y direction
      for(unsigned int i_x = 0; i_x < n_t; ++i_x)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              n_t,
                                              1,
                                              false,
                                              false>(shape_data_n,
                                                     data_points + i_x,
                                                     data_interpolate);
        data_f[0][i_z * n_t + i_x] = data_interpolate[0];
        data_f[1][i_z * n_t + i_x] = data_interpolate[n_t];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_cell[i * n_t + i_x] = data_interpolate[1 + i];
      }

      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(true,
                                       dofs_per_plane,
                                       data_cell,
                                       dof_indices[2 * dim].data(),
                                       dst_vector + cell_dofs_per_comp + i_z * dofs_per_plane);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          Number * dst_ptr =
            dst_vector + dof_indices[2 * dim][v] + i_z * dofs_per_plane + cell_dofs_per_comp;
          for(unsigned int i = 0; i < dofs_per_plane; ++i)
            dst_ptr[i] += data_cell[i][v];
        }
    }
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = n_filled_lanes == n_lanes;
      for(const unsigned int i : dof_indices[2 + f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_transpose_and_store(
          true, dofs_per_face, data_f[f], dof_indices[2 + f].data(), dst_vector);
      else
      {
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          if(dof_indices[2 + f][v] != numbers::invalid_unsigned_int)
          {
            Number * dst_ptr = dst_vector + dof_indices[2 + f][v];
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              dst_ptr[i] += data_f[f][i][v];
          }
      }
    }

    if constexpr(dim == 3)
      for(unsigned int i2 = 0; i2 < n_n; ++i2)
      {
        VectorizedArray<Number> * in_i2 = quad_values + i2 * n_points_1d * n_points_1d * dim + 2;
        for(unsigned int i1 = 0; i1 < n_points_1d; ++i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                dim * n_points_1d,
                                                n_points_1d,
                                                false,
                                                false>(shape_data_t,
                                                       in_i2 + i1 * dim,
                                                       data_points + i1);
        for(unsigned int i1 = 0; i1 < n_t; ++i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                1,
                                                1,
                                                false,
                                                false>(shape_data_t,
                                                       data_points + i1 * n_points_1d,
                                                       data_points + i1 * n_t);

        if(i2 == 0 || i2 == n_t)
        {
          const unsigned int f = i2 == 0 ? 0 : 1;

          // check if indices unconstrained
          bool all_indices_unconstrained = n_filled_lanes == n_lanes;
          for(const unsigned int i : dof_indices[4 + f])
            if(i == numbers::invalid_unsigned_int)
            {
              all_indices_unconstrained = false;
              break;
            }
          if(all_indices_unconstrained)
            vectorized_transpose_and_store(
              true, dofs_per_face, data_points, dof_indices[4 + f].data(), dst_vector);
          else
          {
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              if(dof_indices[4 + f][v] != numbers::invalid_unsigned_int)
              {
                Number * dst_ptr = dst_vector + dof_indices[4 + f][v];
                for(unsigned int i = 0; i < dofs_per_face; ++i)
                  dst_ptr[i] += data_points[i][v];
              }
          }
        }
        else
        {
          if(n_filled_lanes == n_lanes)
            vectorized_transpose_and_store(true,
                                           dofs_per_face,
                                           data_points,
                                           dof_indices[2 * dim].data(),
                                           dst_vector + 2 * cell_dofs_per_comp +
                                             (i2 - 1) * dofs_per_face);
          else
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
            {
              Number * dst_ptr = dst_vector + dof_indices[2 * dim][v] + (i2 - 1) * dofs_per_face +
                                 2 * cell_dofs_per_comp;
              for(unsigned int i = 0; i < dofs_per_face; ++i)
                dst_ptr[i] += data_points[i][v];
            }
        }
      }
  }

  template<int n_t>
  void
  read_face_values(const Number *                                              src_vector,
                   const unsigned int                                          face,
                   const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & dof_indices,
                   VectorizedArray<Number> *                                   out) const
  {
    const unsigned int face_direction = face / 2;
    const unsigned int side           = face % 2;

    constexpr unsigned int n_n            = n_t + 1;
    constexpr unsigned int dofs_per_face  = dealii::Utilities::pow(n_t, dim - 1);
    constexpr unsigned int dofs_per_plane = n_t * (n_t - 1);

    VectorizedArray<Number> data_f[2][dofs_per_face];
    VectorizedArray<Number> data_cell[n_t * n_t];

    constexpr unsigned int cell_dofs_per_comp = Utilities::pow(n_t, dim - 1) * (n_t - 1);

    const Number * DEAL_II_RESTRICT shape_data_n =
      shape_info.data[0].shape_data_on_face[side].data();
    const Number * DEAL_II_RESTRICT shape_data_t =
      shape_info.data[1].shape_data_on_face[side].data();

    // x component
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const unsigned int i : dof_indices[f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_load_and_transpose(dofs_per_face, src_vector, dof_indices[f].data(), data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int idx = dof_indices[f][v];
          if(idx != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = src_vector[idx + i];
        }
      }
    }
    if(face_direction < 2)
    {
      for(unsigned int i2 = 0; i2 < (dim > 2 ? n_t : 1); ++i2)
      {
        vectorized_load_and_transpose(dofs_per_plane,
                                      src_vector + i2 * dofs_per_plane,
                                      dof_indices[2 * dim].data(),
                                      data_cell);
        if(face_direction == 0)
        {
          VectorizedArray<Number> * out_val  = out + i2 * n_t;
          VectorizedArray<Number> * out_grad = out + dofs_per_face + i2 * n_t;
          for(unsigned int i1 = 0; i1 < n_t; ++i1)
          {
            VectorizedArray<Number> sum_v = data_f[0][i2 * n_t + i1] * shape_data_n[0] +
                                            data_f[1][i2 * n_t + i1] * shape_data_n[n_t];
            VectorizedArray<Number> sum_g = data_f[0][i2 * n_t + i1] * shape_data_n[n_n] +
                                            data_f[1][i2 * n_t + i1] * shape_data_n[n_n + n_t];
            for(unsigned int i = 0; i < n_t - 1; ++i)
            {
              sum_v += data_cell[i1 * (n_t - 1) + i] * shape_data_n[i + 1];
              sum_g += data_cell[i1 * (n_t - 1) + i] * shape_data_n[n_n + i + 1];
            }

            out_val[i1]  = sum_v;
            out_grad[i1] = sum_g;
          }
        }
        else
        {
          // face_direction == 1
          VectorizedArray<Number> * out_val  = out + i2 * n_n;
          VectorizedArray<Number> * out_grad = out + n_t * n_n + i2 * n_n;
          for(unsigned int f = 0; f < 2; ++f)
          {
            VectorizedArray<Number> sum_v = data_f[f][i2 * n_t] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_f[f][i2 * n_t] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_f[f][i2 * n_t + i] * shape_data_t[i];
              sum_g += data_f[f][i2 * n_t + i] * shape_data_t[n_t + i];
            }
            out_val[f * n_t]  = sum_v;
            out_grad[f * n_t] = sum_g;
          }
          for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_cell[i1] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_cell[i * (n_t - 1) + i1] * shape_data_t[i];
              sum_g += data_cell[i * (n_t - 1) + i1] * shape_data_t[n_t + i];
            }

            out_val[i1 + 1]  = sum_v;
            out_grad[i1 + 1] = sum_g;
          }
        }
      }
    }
    else
    {
      Assert(dim == 3, ExcInternalError());
      Assert(face_direction == 2, ExcInternalError());
      VectorizedArray<Number> * out_val  = out;
      VectorizedArray<Number> * out_grad = out + n_t * n_n;
      for(unsigned int f = 0; f < 2; ++f)
        for(unsigned int i2 = 0; i2 < n_t; ++i2)
        {
          VectorizedArray<Number> sum_v = data_f[f][i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = data_f[f][i2] * shape_data_t[n_t];
          for(unsigned int i = 1; i < n_t; ++i)
          {
            sum_v += data_f[f][i2 + i * n_t] * shape_data_t[i];
            sum_g += data_f[f][i2 + i * n_t] * shape_data_t[n_t + i];
          }
          out_val[i2 * n_n + f * n_t]  = sum_v;
          out_grad[i2 * n_n + f * n_t] = sum_g;
        }
      std::array<unsigned int, dofs_per_plane> indices_of_cell_data;
      for(unsigned int i2 = 0; i2 < n_t; ++i2)
        for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          indices_of_cell_data[i2 * (n_t - 1) + i1] = i2 * (n_t + 1) + i1 + 1;
      if(dofs_per_plane >= 4)
        for(unsigned int i2 = 0; i2 < dofs_per_plane; i2 += 4)
        {
          const unsigned int      i = std::min(i2, dofs_per_plane - 4);
          VectorizedArray<Number> entries[4];
          vectorized_load_and_transpose(4, src_vector + i, dof_indices[2 * dim].data(), entries);
          std::array<VectorizedArray<Number>, 4> sum_v, sum_g;
          for(unsigned int e = 0; e < 4; ++e)
          {
            sum_v[e] = entries[e] * shape_data_t[0];
            sum_g[e] = entries[e] * shape_data_t[n_t];
          }
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            vectorized_load_and_transpose(4,
                                          src_vector + i + i1 * dofs_per_plane,
                                          dof_indices[2 * dim].data(),
                                          entries);
            for(unsigned int e = 0; e < 4; ++e)
            {
              sum_v[e] += entries[e] * shape_data_t[i1];
              sum_g[e] += entries[e] * shape_data_t[n_t + i1];
            }
          }
          for(unsigned int e = 0; e < 4; ++e)
          {
            out_val[indices_of_cell_data[i + e]]  = sum_v[e];
            out_grad[indices_of_cell_data[i + e]] = sum_g[e];
          }
        }
      else
      {
        VectorizedArray<Number> tmp[cell_dofs_per_comp];
        vectorized_load_and_transpose(cell_dofs_per_comp,
                                      src_vector,
                                      dof_indices[2 * dim].data(),
                                      tmp);
        for(unsigned int i2 = 0; i2 < dofs_per_plane; ++i2)
        {
          VectorizedArray<Number> sum_v = tmp[i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = tmp[i2] * shape_data_t[n_t];
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            sum_v += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[i1];
            sum_g += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[n_t + i1];
          }
          out_val[indices_of_cell_data[i2]]  = sum_v;
          out_grad[indices_of_cell_data[i2]] = sum_g;
        }
      }
    }

    out += (face_direction == 0) ? 2 * n_t * n_t : 2 * n_t * n_n;

    // y component
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const unsigned int i : dof_indices[2 + f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_load_and_transpose(dofs_per_face,
                                      src_vector,
                                      dof_indices[f + 2].data(),
                                      data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int idx = dof_indices[f + 2][v];
          if(idx != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = src_vector[idx + i];
        }
      }
    }
    if(face_direction < 2)
    {
      for(unsigned int i2 = 0; i2 < (dim > 2 ? n_t : 1); ++i2)
      {
        vectorized_load_and_transpose(dofs_per_plane,
                                      src_vector + cell_dofs_per_comp + i2 * dofs_per_plane,
                                      dof_indices[2 * dim].data(),
                                      data_cell);
        if(face_direction == 0)
        {
          VectorizedArray<Number> * out_val  = out + i2 * n_n;
          VectorizedArray<Number> * out_grad = out + n_t * n_n + i2 * n_n;
          for(unsigned int f = 0; f < 2; ++f)
          {
            VectorizedArray<Number> sum_v = data_f[f][i2 * n_t] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_f[f][i2 * n_t] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_f[f][i2 * n_t + i] * shape_data_t[i];
              sum_g += data_f[f][i2 * n_t + i] * shape_data_t[n_t + i];
            }
            out_val[f * n_t]  = sum_v;
            out_grad[f * n_t] = sum_g;
          }
          for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1 * n_t] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_cell[i1 * n_t] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_cell[i1 * n_t + i] * shape_data_t[i];
              sum_g += data_cell[i1 * n_t + i] * shape_data_t[n_t + i];
            }

            out_val[i1 + 1]  = sum_v;
            out_grad[i1 + 1] = sum_g;
          }
        }
        else
        {
          // face_direction == 1
          VectorizedArray<Number> * out_val  = out + i2 * n_t;
          VectorizedArray<Number> * out_grad = out + dofs_per_face + i2 * n_t;
          for(unsigned int i1 = 0; i1 < n_t; ++i1)
          {
            VectorizedArray<Number> sum_v = data_f[0][i2 * n_t + i1] * shape_data_n[0] +
                                            data_f[1][i2 * n_t + i1] * shape_data_n[n_t];
            VectorizedArray<Number> sum_g = data_f[0][i2 * n_t + i1] * shape_data_n[n_n] +
                                            data_f[1][i2 * n_t + i1] * shape_data_n[n_n + n_t];
            for(unsigned int i = 0; i < n_t - 1; ++i)
            {
              sum_v += data_cell[i * n_t + i1] * shape_data_n[i + 1];
              sum_g += data_cell[i * n_t + i1] * shape_data_n[n_n + i + 1];
            }

            out_val[i1]  = sum_v;
            out_grad[i1] = sum_g;
          }
        }
      }
    }
    else
    {
      Assert(dim == 3, ExcInternalError());
      Assert(face_direction == 2, ExcInternalError());
      VectorizedArray<Number> * out_val  = out;
      VectorizedArray<Number> * out_grad = out + n_t * n_n;
      for(unsigned int f = 0; f < 2; ++f)
        for(unsigned int i2 = 0; i2 < n_t; ++i2)
        {
          VectorizedArray<Number> sum_v = data_f[f][i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = data_f[f][i2] * shape_data_t[n_t];
          for(unsigned int i = 1; i < n_t; ++i)
          {
            sum_v += data_f[f][i2 + i * n_t] * shape_data_t[i];
            sum_g += data_f[f][i2 + i * n_t] * shape_data_t[n_t + i];
          }
          out_val[i2 + f * n_t * n_t]  = sum_v;
          out_grad[i2 + f * n_t * n_t] = sum_g;
        }
      std::array<unsigned int, dofs_per_plane> indices_of_cell_data;
      for(unsigned int i2 = 0; i2 < n_t - 1; ++i2)
        for(unsigned int i1 = 0; i1 < n_t; ++i1)
          indices_of_cell_data[i2 * n_t + i1] = (i2 + 1) * n_t + i1;
      if(dofs_per_plane >= 4)
        for(unsigned int i2 = 0; i2 < dofs_per_plane; i2 += 4)
        {
          const unsigned int      i = std::min(i2, dofs_per_plane - 4);
          VectorizedArray<Number> entries[4];
          vectorized_load_and_transpose(4,
                                        src_vector + cell_dofs_per_comp + i,
                                        dof_indices[2 * dim].data(),
                                        entries);
          std::array<VectorizedArray<Number>, 4> sum_v, sum_g;
          for(unsigned int e = 0; e < 4; ++e)
          {
            sum_v[e] = entries[e] * shape_data_t[0];
            sum_g[e] = entries[e] * shape_data_t[n_t];
          }
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            vectorized_load_and_transpose(4,
                                          src_vector + cell_dofs_per_comp + i + i1 * dofs_per_plane,
                                          dof_indices[2 * dim].data(),
                                          entries);
            for(unsigned int e = 0; e < 4; ++e)
            {
              sum_v[e] += entries[e] * shape_data_t[i1];
              sum_g[e] += entries[e] * shape_data_t[n_t + i1];
            }
          }
          for(unsigned int e = 0; e < 4; ++e)
          {
            out_val[indices_of_cell_data[i + e]]  = sum_v[e];
            out_grad[indices_of_cell_data[i + e]] = sum_g[e];
          }
        }
      else
      {
        VectorizedArray<Number> tmp[cell_dofs_per_comp];
        vectorized_load_and_transpose(cell_dofs_per_comp,
                                      src_vector + cell_dofs_per_comp,
                                      dof_indices[2 * dim].data(),
                                      tmp);
        for(unsigned int i2 = 0; i2 < dofs_per_plane; ++i2)
        {
          VectorizedArray<Number> sum_v = tmp[i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = tmp[i2] * shape_data_t[n_t];
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            sum_v += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[i1];
            sum_g += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[n_t + i1];
          }
          out_val[indices_of_cell_data[i2]]  = sum_v;
          out_grad[indices_of_cell_data[i2]] = sum_g;
        }
      }
    }

    // z component
    out += (face_direction == 1) ? 2 * n_t * n_t : 2 * n_t * n_n;
    if(dim > 2)
    {
      if(face_direction < 2)
      {
        for(unsigned int f = 0; f < 2; ++f)
        {
          // check if indices unconstrained
          bool all_indices_unconstrained = true;
          for(const unsigned int i : dof_indices[4 + f])
            if(i == numbers::invalid_unsigned_int)
            {
              all_indices_unconstrained = false;
              break;
            }
          if(all_indices_unconstrained)
            vectorized_load_and_transpose(dofs_per_face,
                                          src_vector,
                                          dof_indices[f + 4].data(),
                                          data_cell);
          else
          {
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_cell[i] = 0;
            for(unsigned int v = 0; v < n_lanes; ++v)
            {
              const unsigned int idx = dof_indices[f + 4][v];
              if(idx != numbers::invalid_unsigned_int)
                for(unsigned int i = 0; i < dofs_per_face; ++i)
                  data_cell[i][v] = src_vector[idx + i];
            }
          }
          if(face_direction == 0)
            for(unsigned int i2 = 0; i2 < n_t; ++i2)
            {
              VectorizedArray<Number> sum_v = data_cell[i2 * n_t] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i2 * n_t] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i2 * n_t + i] * shape_data_t[i];
                sum_g += data_cell[i2 * n_t + i] * shape_data_t[n_t + i];
              }
              out[f * n_t * n_t + i2]             = sum_v;
              out[n_n * n_t + f * n_t * n_t + i2] = sum_g;
            }
          else
            for(unsigned int i2 = 0; i2 < n_t; ++i2)
            {
              VectorizedArray<Number> sum_v = data_cell[i2] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i2] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i2 + i * n_t] * shape_data_t[i];
                sum_g += data_cell[i2 + i * n_t] * shape_data_t[n_t + i];
              }
              out[f * n_t * n_t + i2]             = sum_v;
              out[n_n * n_t + f * n_t * n_t + i2] = sum_g;
            }
        }
        for(unsigned int i2 = 0; i2 < n_t - 1; ++i2)
        {
          vectorized_load_and_transpose(dofs_per_face,
                                        src_vector + 2 * cell_dofs_per_comp + i2 * dofs_per_face,
                                        dof_indices[2 * dim].data(),
                                        data_cell);
          if(face_direction == 0)
            for(unsigned int i1 = 0; i1 < n_t; ++i1)
            {
              VectorizedArray<Number> sum_v = data_cell[i1 * n_t] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i1 * n_t] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i1 * n_t + i] * shape_data_t[i];
                sum_g += data_cell[i1 * n_t + i] * shape_data_t[n_t + i];
              }
              out[(i2 + 1) * n_t + i1]             = sum_v;
              out[n_t * n_n + (i2 + 1) * n_t + i1] = sum_g;
            }
          else
            for(unsigned int i1 = 0; i1 < n_t; ++i1)
            {
              VectorizedArray<Number> sum_v = data_cell[i1] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i1] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i1 + i * n_t] * shape_data_t[i];
                sum_g += data_cell[i1 + i * n_t] * shape_data_t[n_t + i];
              }
              out[(i2 + 1) * n_t + i1]             = sum_v;
              out[n_t * n_n + (i2 + 1) * n_t + i1] = sum_g;
            }
        }
      }
      else
      {
        // face_direction == 2
        for(unsigned int f = 0; f < 2; ++f)
        {
          // check if indices unconstrained
          bool all_indices_unconstrained = true;
          for(const unsigned int i : dof_indices[4 + f])
            if(i == numbers::invalid_unsigned_int)
            {
              all_indices_unconstrained = false;
              break;
            }
          if(all_indices_unconstrained)
            vectorized_load_and_transpose(dofs_per_face,
                                          src_vector,
                                          dof_indices[f + 4].data(),
                                          data_f[f]);
          else
          {
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i] = 0;
            for(unsigned int v = 0; v < n_lanes; ++v)
            {
              const unsigned int idx = dof_indices[f + 4][v];
              if(idx != numbers::invalid_unsigned_int)
                for(unsigned int i = 0; i < dofs_per_face; ++i)
                  data_f[f][i][v] = src_vector[idx + i];
            }
          }
        }
        AssertThrow(dofs_per_face >= 4, ExcNotImplemented());
        for(unsigned int i2 = 0; i2 < dofs_per_face; i2 += 4)
        {
          const unsigned int                     i = std::min(i2, dofs_per_face - 4);
          std::array<VectorizedArray<Number>, 4> sum_v, sum_g;
          for(unsigned int e = 0; e < 4; ++e)
          {
            sum_v[e] = data_f[0][i + e] * shape_data_n[0] + data_f[1][i + e] * shape_data_n[n_t];
            sum_g[e] =
              data_f[0][i + e] * shape_data_n[n_n] + data_f[1][i + e] * shape_data_n[n_n + n_t];
          }
          for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          {
            VectorizedArray<Number> entries[4];
            vectorized_load_and_transpose(4,
                                          src_vector + 2 * cell_dofs_per_comp + i +
                                            i1 * dofs_per_face,
                                          dof_indices[2 * dim].data(),
                                          entries);
            for(unsigned int e = 0; e < 4; ++e)
            {
              sum_v[e] += entries[e] * shape_data_n[i1 + 1];
              sum_g[e] += entries[e] * shape_data_n[n_n + i1 + 1];
            }
          }
          for(unsigned int e = 0; e < 4; ++e)
          {
            out[i + e]                 = sum_v[e];
            out[i + e + dofs_per_face] = sum_g[e];
          }
        }
      }
    }
  }



  template<int n_t>
  void
  vmult_pack_and_send_data(const VectorType & src, std::vector<MPI_Request> & requests) const
  {
    constexpr int n_n = n_t + 1;
    constexpr int data_per_face =
      2 * (Utilities::pow(n_t, dim - 1) + n_n * Utilities::pow(n_t, dim - 2) * (dim - 1));

    std::array<VectorizedArray<Number>, dim * 3 * n_n * n_t> tmp_vec;

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

        dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> dof_indices_vec;
        std::array<unsigned int, n_lanes>                   indices;
        for(unsigned int v = 0; v < n_faces; ++v, ++count)
        {
          indices[v]              = count * data_per_face;
          const unsigned int cell = send_data_cell_index[count] / n_lanes;
          const unsigned int lane = send_data_cell_index[count] % n_lanes;
          AssertDimension(face, send_data_face_index[count]);
          for(unsigned int f = 0; f < 2 * dim + 1; ++f)
            dof_indices_vec[f][v] = dof_indices[cell][f][lane];
        }
        for(unsigned int v = n_faces; v < n_lanes; ++v)
        {
          for(unsigned int f = 0; f < 2 * dim + 1; ++f)
            dof_indices_vec[f][v] = dof_indices_vec[f][0];
        }
        read_face_values<n_t>(src.begin(), face, dof_indices_vec, tmp_vec.data());

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
    AssertDimension(degree, shape_info.data[1].fe_degree + 1);
    if(degree == 2)
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
    else if(degree == 9)
      do_cell_operation<9, false>(cell, dst, dst);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  template<int degree, bool compute_exterior>
  void
  do_cell_operation(const unsigned int cell, const VectorType & src, VectorType & dst) const
  {
    constexpr unsigned int dofs_per_component = Utilities::pow(degree, dim - 1) * (degree + 1);
    constexpr unsigned int dofs_per_cell      = dim * dofs_per_component;
    constexpr unsigned int n_q_points_1d      = degree + 1;
    constexpr unsigned int n_points           = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data         = shape_info.data;

    VectorizedArray<Number> quad_values[dim * n_points], out_values[dim * n_points];
    if(compute_exterior)
    {
      read_cell_values<degree, n_q_points_1d>(src.begin(), dof_indices[cell], quad_values);

      if(factor_laplace != 0)
        compute_cell_lapl<n_q_points_1d>(shape_data, cell, quad_values, out_values);
      else
        compute_cell_mass<n_q_points_1d>(cell, quad_values, out_values);

      // Face integrals if Laplace factor is positive
      if(factor_laplace != 0.)
        for(unsigned int f = 0; f < 2 * dim; ++f)
          compute_face<degree, true>(shape_data, src, cell, f, quad_values, out_values);

      integrate_cell_scatter<degree, n_q_points_1d>(dof_indices[cell],
                                                    n_active_entries_per_cell_batch(cell),
                                                    out_values,
                                                    dst.begin());
    }
    else
    {
      VectorizedArray<Number> diagonal[dofs_per_cell];
      for(unsigned int d = 0; d < dim; ++d)
        for(unsigned int i = 0; i < dofs_per_component; ++i)
        {
          internal::EvaluatorTensorProductAnisotropic<dim, degree, n_q_points_1d, true> eval;
          for(unsigned int j = 0; j < dofs_per_component; ++j)
            out_values[j] = 0;
          out_values[i] = 1;
          if(d == 0)
          {
            // always put the data last for the component in question,
            // the rest will be zero
            VectorizedArray<Number> * values = quad_values + (dim - 1) * n_points;
            eval.template normal<0>(shape_data[0], out_values, values);
            eval.template tangential<1, 0>(shape_data[1], values, values);
            if constexpr(dim > 2)
              eval.template tangential<2, 0>(shape_data[1], values, values);
          }
          else if(d == 1)
          {
            VectorizedArray<Number> * values = quad_values + (dim - 1) * n_points;
            eval.template normal<1>(shape_data[0], out_values, values);
            eval.template tangential<0, 1>(shape_data[1], values, values);
            if constexpr(dim > 2)
              eval.template tangential<2, 1>(shape_data[1], values, values);
          }
          else if(d == 2)
          {
            VectorizedArray<Number> * values = quad_values + (dim - 1) * n_points;

            eval.template normal<2>(shape_data[0], out_values, values);
            eval.template tangential<0, 2>(shape_data[1], values, values);
            eval.template tangential<1, 2>(shape_data[1], values, values);
          }

          for(unsigned int i = 0; i < n_points; ++i)
          {
            quad_values[i * dim + d] = quad_values[(dim - 1) * n_points + i];
            for(unsigned int e = 0; e < dim; ++e)
              if(e != d)
                quad_values[i * dim + e] = 0;
          }

          if(factor_laplace != 0)
            compute_cell_lapl<n_q_points_1d>(shape_data, cell, quad_values, out_values);
          else
            compute_cell_mass<n_q_points_1d>(cell, quad_values, out_values);

          if(factor_laplace != 0.)
            for(unsigned int f = 0; f < 2 * dim; ++f)
              compute_face<degree, false>(shape_data, src, cell, f, quad_values, out_values);

          for(unsigned int i = 0; i < n_points; ++i)
            quad_values[i] = out_values[i * dim + d];

          {
            internal::EvaluatorTensorProductAnisotropic<dim, degree, n_q_points_1d, false> eval;
            VectorizedArray<Number> * values = quad_values;

            if(d == 0)
            {
              if constexpr(dim > 2)
                eval.template tangential<2, 0>(shape_data[1], values, values);
              eval.template tangential<1, 0>(shape_data[1], values, values);
              eval.template normal<0>(shape_data[0], values, values);
              diagonal[i] = values[i];
            }

            if(d == 1)
            {
              if constexpr(dim > 2)
                eval.template tangential<2, 1>(shape_data[1], values, values);
              eval.template tangential<0, 1>(shape_data[1], values, values);
              eval.template normal<1>(shape_data[0], values, values);
              diagonal[dofs_per_component + i] = values[i];
            }

            if(d == 2)
            {
              eval.template tangential<1, 2>(shape_data[1], values, values);
              eval.template tangential<0, 2>(shape_data[1], values, values);
              eval.template normal<2>(shape_data[0], values, values);
              diagonal[2 * dofs_per_component + i] = values[i];
            }
          }
        }
      distribute_local_to_global_rt_compressed<dim, degree>(diagonal,
                                                            n_active_entries_per_cell_batch(cell),
                                                            dof_indices[cell],
                                                            dst.begin());
    }
  }

  template<int n_q_points_1d>
  void
  compute_cell_lapl(
    const std::vector<internal::MatrixFreeFunctions::UnivariateShapeData<Number>> & shape_data,
    const unsigned int                                                              cell,
    const VectorizedArray<Number> *                                                 quad_values,
    VectorizedArray<Number> * out_values) const
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);

    VectorizedArray<Number> grad_y[n_q_points * dim];
    VectorizedArray<Number> grad_x[Utilities::pow(n_q_points_1d, dim - 1) * dim];
    VectorizedArray<Number> grad_z[n_q_points_1d * dim];

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_data_index[cell][v] * 4;
    std::array<unsigned int, n_lanes> shifted_data_indices_gr;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices_gr[v] = mapping_data_index[cell][v] * 6;

    const Number factor_mass = this->factor_mass;
    const Number factor_lapl = this->factor_laplace;

    constexpr unsigned int nn         = n_q_points_1d;
    const Number *         shape_grad = shape_data[0].shape_gradients_collocation_eo.data();
    for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        for(unsigned int d = 0; d < dim; ++d)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                nn * dim,
                                                Utilities::pow(nn, dim - 1) * dim,
                                                true,
                                                false>(shape_grad,
                                                       quad_values +
                                                         dim * (i1 * n_q_points_2d + i0) + d,
                                                       grad_y + dim * (i1 * nn + i0) + d);

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
        for(unsigned int d = 0; d < dim; ++d)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                dim,
                                                (dim == 3 ? nn : 1) * dim,
                                                true,
                                                false>(shape_grad,
                                                       quad_values +
                                                         dim * (qy * nn + i0 * n_q_points_2d) + d,
                                                       grad_x + dim * i0 + d);

      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number>               inv_jac_det = Number(1.0) / determinant(jac_xy);
        const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy  = transpose(invert(jac_xy));
        Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy;
        vectorized_load_and_transpose(6,
                                      &jacobian_grads[q1][0][0],
                                      shifted_data_indices_gr.data(),
                                      &jac_grad_xy[0][0]);

        // Prepare for -(J^{-T} * jac_grad * J^{-1} * J * values *
        // det(J^{-1})), which can be expressed as a rank-1 update tmp[d] *
        // tmp4[e], where tmp = J * values and tmp4 = (J^{-T} * jac_grad *
        // J^{-1})
        VectorizedArray<Number> jac_grad_rank1[2];
        {
          VectorizedArray<Number> tmp[2];
          for(unsigned int d = 0; d < 2; ++d)
            tmp[d] =
              (inv_jac_xy[0][d] * jac_grad_xy[d][0] + inv_jac_xy[1][d] * jac_grad_xy[d][1] +
               inv_jac_xy[0][1 - d] * jac_grad_xy[2][0] + inv_jac_xy[1][1 - d] * jac_grad_xy[2][1]);
          for(unsigned int d = 0; d < 2; ++d)
            jac_grad_rank1[d] = (tmp[0] * inv_jac_xy[d][0] + tmp[1] * inv_jac_xy[d][1]);
        }

        if constexpr(dim == 2)
        {
          const VectorizedArray<Number> val[2] = {quad_values[q1 * dim], quad_values[q1 * dim + 1]};
          const VectorizedArray<Number> grad[2][2] = {{grad_x[qx * dim], grad_x[qx * dim + 1]},
                                                      {grad_y[q1 * dim], grad_y[q1 * dim + 1]}};
          const VectorizedArray<Number> t0         = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
          const VectorizedArray<Number> t1         = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
          const VectorizedArray<Number> s0         = jac_xy[0][0] * t0 + jac_xy[1][0] * t1;
          const VectorizedArray<Number> s1         = jac_xy[0][1] * t0 + jac_xy[1][1] * t1;

          VectorizedArray<Number> grad_real[dim][dim];
          for(unsigned int d = 0; d < dim; ++d)
          {
            // (J * grad_quad) * J^-1 * det(J^-1), part in braces
            VectorizedArray<Number> tmp[dim];
            for(unsigned int e = 0; e < dim; ++e)
              tmp[e] = (jac_xy[d][0] * grad[0][e] + jac_xy[d][1] * grad[1][e]);

            // Add (jac_grad * values) * J^{-1} * det(J^{-1}), combine
            // terms outside braces with gradient part from above
            tmp[0] += jac_grad_xy[0][d] * val[0];
            tmp[0] += jac_grad_xy[2][d] * val[1];
            tmp[1] += jac_grad_xy[1][d] * val[1];
            tmp[1] += jac_grad_xy[2][d] * val[0];

            for(unsigned int e = 0; e < dim; ++e)
              grad_real[d][e] = (tmp[0] * inv_jac_xy[e][0] + tmp[1] * inv_jac_xy[e][1]);

            VectorizedArray<Number> tmp2 = jac_xy[d][0] * val[0] + jac_xy[d][1] * val[1];

            for(unsigned int e = 0; e < dim; ++e)
            {
              grad_real[d][e] -= jac_grad_rank1[e] * tmp2;

              // multiply by det(J^{-1}) necessary in all
              // contributions above and the factors in the equation
              grad_real[d][e] *= quad_weights_xy[q1] * factor_lapl * inv_jac_det;
            }
          }

          // J * (J^{-1} * (grad_in * factor))
          VectorizedArray<Number> tmp[dim][dim];
          for(unsigned int d = 0; d < dim; ++d)
            for(unsigned int e = 0; e < dim; ++e)
              tmp[d][e] = (inv_jac_xy[0][d] * grad_real[e][0] + inv_jac_xy[1][d] * grad_real[e][1]);

          for(unsigned int d = 0; d < dim; ++d)
          {
            grad_x[qx * dim + d] = (jac_xy[0][0] * tmp[d][0] + jac_xy[1][0] * tmp[d][1]);
            grad_y[q1 * dim + d] = (jac_xy[0][1] * tmp[d][0] + jac_xy[1][1] * tmp[d][1]);
          }

          // jac_grad * (J^{-1} * (grad_in * factor)), re-use part in
          // braces as 'tmp' from above
          VectorizedArray<Number> value_tmp[dim];
          for(unsigned int d = 0; d < dim; ++d)
            value_tmp[d] = (tmp[d][0] * jac_grad_xy[d][0] + tmp[d][1] * jac_grad_xy[d][1] +
                            tmp[1 - d][0] * jac_grad_xy[2][0] + tmp[1 - d][1] * jac_grad_xy[2][1]);

          //   -(grad_in * factor) * J * (J^{-T} * jac_grad * J^{-1})
          // = -(grad_in * factor) * J * ( \------- tmp4 ---------/ )
          for(unsigned int d = 0; d < dim; ++d)
          {
            VectorizedArray<Number> tmp2 =
              (grad_real[d][0] * jac_grad_rank1[0] + grad_real[d][1] * jac_grad_rank1[1]);
            for(unsigned int e = 0; e < dim; ++e)
              value_tmp[e] -= jac_xy[d][e] * tmp2;
          }

          out_values[q1 * dim] =
            value_tmp[0] + s0 * (inv_jac_det * factor_mass * quad_weights_xy[q1]);
          out_values[q1 * dim + 1] =
            value_tmp[1] + s1 * (inv_jac_det * factor_mass * quad_weights_xy[q1]);
        }
        else // now to dim == 3
        {
          for(unsigned int d = 0; d < dim; ++d)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::gradient,
                                                  nn,
                                                  nn,
                                                  nn * nn * dim,
                                                  dim,
                                                  true,
                                                  false>(shape_grad,
                                                         quad_values + q1 * dim + d,
                                                         grad_z + d);

          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
          {
            const unsigned int            ix         = dim * (qz + qx * nn);
            const unsigned int            iy         = dim * (qz * nn + qy * nn * nn + qx);
            const unsigned int            iz         = dim * qz;
            const VectorizedArray<Number> val[3]     = {quad_values[q * dim],
                                                    quad_values[q * dim + 1],
                                                    quad_values[q * dim + 2]};
            const VectorizedArray<Number> grad[3][3] = {
              {grad_x[ix + 0], grad_y[iy + 0], grad_z[iz + 0]},
              {grad_x[ix + 1], grad_y[iy + 1], grad_z[iz + 1]},
              {grad_x[ix + 2], grad_y[iy + 2], grad_z[iz + 2]}};

            Tensor<1, dim, VectorizedArray<Number>> val_real;
            val_real[0] = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
            val_real[1] = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
            val_real[2] = h_z * val[2];

            const Number weight_z = quad_weights_z[qz];
            // need factors without h_z_inverse in code below because
            // it cancels with h_z
            const VectorizedArray<Number> factor_deriv_z =
              (quad_weights_xy[q1] * inv_jac_det * factor_lapl) * weight_z;
            const VectorizedArray<Number> factor_deriv = h_z_inverse * factor_deriv_z;
            const VectorizedArray<Number> factor_ma =
              (quad_weights_xy[q1] * h_z_inverse * inv_jac_det * factor_mass) * weight_z;

            VectorizedArray<Number> grad_real[dim][dim];
            for(unsigned int d = 0; d < 2; ++d)
            {
              // (J * grad_quad) * J^-1 * det(J^-1), part in braces
              VectorizedArray<Number> tmp[dim];
              for(unsigned int e = 0; e < dim; ++e)
                tmp[e] = (jac_xy[d][0] * grad[0][e] + jac_xy[d][1] * grad[1][e]);

              // Add (jac_grad * values) * J^{-1} * det(J^{-1}),
              // combine terms outside braces with gradient part from
              // above
              tmp[0] += jac_grad_xy[0][d] * val[0];
              tmp[0] += jac_grad_xy[2][d] * val[1];
              tmp[1] += jac_grad_xy[1][d] * val[1];
              tmp[1] += jac_grad_xy[2][d] * val[0];

              for(unsigned int e = 0; e < 2; ++e)
                grad_real[d][e] = (tmp[0] * inv_jac_xy[e][0] + tmp[1] * inv_jac_xy[e][1]);

              for(unsigned int e = 0; e < 2; ++e)
              {
                grad_real[d][e] -= jac_grad_rank1[e] * val_real[d];

                // multiply by det(J^{-1}) necessary in all
                // contributions above and the factors in the
                // equation
                grad_real[d][e] *= factor_deriv;
              }

              const VectorizedArray<Number> tmp_z_grad2 =
                (inv_jac_xy[d][0] * grad[2][0] + inv_jac_xy[d][1] * grad[2][1]);

              grad_real[2][d] = (tmp_z_grad2 - jac_grad_rank1[d] * val[2]) * factor_deriv_z;
              grad_real[d][2] = (h_z_inverse * tmp[2]) * factor_deriv;
            }
            grad_real[2][2] = factor_deriv * grad[2][2];

            // For the test function part, the Jacobian determinant
            // that is part of the integration factor cancels with the
            // inverse Jacobian determinant present in the factor of
            // the derivative

            // J * (J^{-1} * (grad_in * factor))
            VectorizedArray<Number> tmp[2][dim];
            for(unsigned int d = 0; d < 2; ++d)
              for(unsigned int e = 0; e < dim; ++e)
                tmp[d][e] =
                  (inv_jac_xy[0][d] * grad_real[e][0] + inv_jac_xy[1][d] * grad_real[e][1]);

            grad_x[ix]     = (jac_xy[0][0] * tmp[0][0] + jac_xy[1][0] * tmp[0][1]);
            grad_x[ix + 1] = (jac_xy[0][1] * tmp[0][0] + jac_xy[1][1] * tmp[0][1]);
            grad_x[ix + 2] = h_z * tmp[0][2];
            grad_y[iy]     = (jac_xy[0][0] * tmp[1][0] + jac_xy[1][0] * tmp[1][1]);
            grad_y[iy + 1] = (jac_xy[0][1] * tmp[1][0] + jac_xy[1][1] * tmp[1][1]);
            grad_y[iy + 2] = h_z * tmp[1][2];
            grad_z[iz]     = (jac_xy[0][0] * (grad_real[0][2] * h_z_inverse) +
                          jac_xy[1][0] * (grad_real[1][2] * h_z_inverse));
            grad_z[iz + 1] = (jac_xy[0][1] * (grad_real[0][2] * h_z_inverse) +
                              jac_xy[1][1] * (grad_real[1][2] * h_z_inverse));
            grad_z[iz + 2] = grad_real[2][2];

            // jac_grad * (J^{-1} * (grad_in * factor)), re-use part in
            // braces as 'tmp' from above
            VectorizedArray<Number> value_tmp[dim];
            for(unsigned int d = 0; d < 2; ++d)
              value_tmp[d] =
                (tmp[d][0] * jac_grad_xy[d][0] + tmp[d][1] * jac_grad_xy[d][1] +
                 tmp[1 - d][0] * jac_grad_xy[2][0] + tmp[1 - d][1] * jac_grad_xy[2][1]);

            //   -(grad_in * factor) * J * (J^{-T} * jac_grad * J^{-1})
            // = -(grad_in * factor) * J * ( \------- tmp4 ---------/ )
            for(unsigned int d = 0; d < dim; ++d)
              val_real[d] *= factor_ma;
            for(unsigned int d = 0; d < 2; ++d)
            {
              VectorizedArray<Number> tmp2 =
                (grad_real[d][0] * jac_grad_rank1[0] + grad_real[d][1] * jac_grad_rank1[1]);
              for(unsigned int e = 0; e < 2; ++e)
                value_tmp[e] += jac_xy[d][e] * (val_real[d] - tmp2);
            }
            value_tmp[2] = h_z * (val_real[2] - grad_real[2][0] * jac_grad_rank1[0] -
                                  grad_real[2][1] * jac_grad_rank1[1]);

            out_values[q * dim]     = value_tmp[0];
            out_values[q * dim + 1] = value_tmp[1];
            out_values[q * dim + 2] = value_tmp[2];
          }

          for(unsigned int d = 0; d < dim; ++d)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::gradient,
                                                  nn,
                                                  nn,
                                                  dim,
                                                  nn * nn * dim,
                                                  false,
                                                  true>(shape_grad,
                                                        grad_z + d,
                                                        out_values + q1 * dim + d);
        }
      }

      for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
        for(unsigned int d = 0; d < dim; ++d)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                (dim == 3 ? nn : 1) * dim,
                                                dim,
                                                false,
                                                true>(shape_grad,
                                                      grad_x + i0 * dim + d,
                                                      out_values +
                                                        (qy * nn + i0 * n_q_points_2d) * dim + d);
    }

    for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        for(unsigned int d = 0; d < dim; ++d)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                Utilities::pow(nn, dim - 1) * dim,
                                                nn * dim,
                                                false,
                                                true>(shape_grad,
                                                      grad_y + (i1 * nn + i0) * dim + d,
                                                      out_values + (i1 * n_q_points_2d + i0) * dim +
                                                        d);
  }

  template<int n_q_points_1d>
  void
  compute_cell_mass(const unsigned int        cell,
                    VectorizedArray<Number> * quad_values,
                    VectorizedArray<Number> * out_values) const
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_data_index[cell][v] * 4;

    const Number factor_mass = this->factor_mass;
    const Number h_z_inverse = this->h_z_inverse;

    constexpr unsigned int nn = n_q_points_1d;

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number> inv_jac_det = Number(1.0) / determinant(jac_xy);

        if constexpr(dim == 2)
        {
          const VectorizedArray<Number> val[2] = {quad_values[q1 * dim], quad_values[q1 * dim + 1]};
          const VectorizedArray<Number> t0     = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
          const VectorizedArray<Number> t1     = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
          const VectorizedArray<Number> s0     = jac_xy[0][0] * t0 + jac_xy[1][0] * t1;
          const VectorizedArray<Number> s1     = jac_xy[0][1] * t0 + jac_xy[1][1] * t1;

          out_values[q1 * dim]     = s0 * (inv_jac_det * factor_mass * quad_weights_xy[q1]);
          out_values[q1 * dim + 1] = s1 * (inv_jac_det * factor_mass * quad_weights_xy[q1]);
        }
        else // now to dim == 3
        {
          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
          {
            const VectorizedArray<Number> val[3] = {quad_values[q * dim],
                                                    quad_values[q * dim + 1],
                                                    quad_values[q * dim + 2]};

            Tensor<1, dim, VectorizedArray<Number>> val_real;
            val_real[0] = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
            val_real[1] = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
            val_real[2] = h_z * val[2];

            const Number weight_z = quad_weights_z[qz];
            // need factors without h_z_inverse in code below because
            // it cancels with h_z
            const VectorizedArray<Number> factor_ma =
              (quad_weights_xy[q1] * h_z_inverse * inv_jac_det * factor_mass) * weight_z;

            // For the test function part, the Jacobian determinant
            // that is part of the integration factor cancels with the
            // inverse Jacobian determinant present in the factor of
            // the derivative

            // jac_grad * (J^{-1} * (grad_in * factor)), re-use part in
            // braces as 'tmp' from above
            VectorizedArray<Number> value_tmp[dim];
            for(unsigned int d = 0; d < dim; ++d)
              val_real[d] *= factor_ma;
            for(unsigned int d = 0; d < 2; ++d)
              value_tmp[d] = jac_xy[0][d] * val_real[0] + jac_xy[1][d] * val_real[1];
            value_tmp[2] = h_z * val_real[2];

            out_values[q * dim]     = value_tmp[0];
            out_values[q * dim + 1] = value_tmp[1];
            out_values[q * dim + 2] = value_tmp[2];
          }
        }
      }
    }
  }

  template<int degree, bool compute_exterior>
  void
  compute_face(
    const std::vector<internal::MatrixFreeFunctions::UnivariateShapeData<Number>> & shape_data,
    const VectorType &                                                              src,
    const unsigned int                                                              cell,
    const unsigned int                                                              f,
    const VectorizedArray<Number> *                                                 quad_values,
    VectorizedArray<Number> * out_values) const
  {
    constexpr bool use_face_buffer = true && compute_exterior;

    if(use_face_buffer && f % 2 == 0 && all_left_face_fluxes_from_buffer[cell][f / 2])
      return;

    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    AssertThrow(n_q_points_2d == quad_weights_xy.size(),
                ExcDimensionMismatch(n_q_points_2d, quad_weights_xy.size()));
    constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

    ndarray<unsigned int, 2 * dim + 1, n_lanes> dof_indices_neighbor;
    unsigned int                                lane_with_neighbor = 0;
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
        boundary_mask[v] = -1.;
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
          for(unsigned int i = 0; i < 2 * dim + 1; ++i)
            dof_indices_neighbor[i][v] =
              dof_indices[neighbor_idx / n_lanes][i][neighbor_idx % n_lanes];
        else
          for(unsigned int i = 0; i < 2 * dim + 1; ++i)
            dof_indices_neighbor[i][v] = dof_indices[0][i][0];
      }
    }

    VectorizedArray<Number> face_array[dim * 2 * Utilities::pow(n_q_points_1d, dim - 1)];
    constexpr int           n_n         = degree + 1;
    constexpr int           n_t         = degree;
    const Number            factor_lapl = this->factor_laplace;

    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_g({}, shape_data[0].shape_gradients_collocation_eo.data(), {});
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     degree,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_iso(shape_data[1].shape_values_eo.data(), {}, {});
    internal::EvaluatorTensorProductAnisotropic<dim - 1, degree, n_q_points_1d, true> eval_aniso;

    const unsigned int      face_direction  = f / 2;
    constexpr unsigned int  n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);
    VectorizedArray<Number> scratch_data[n_q_points_face + 1];
    VectorizedArray<Number> values_face[2][dim][n_q_points_face + 1];
    VectorizedArray<Number> grads_face[2][dim][n_q_points_face * dim + 1];

    if constexpr(compute_exterior)
    {
      read_face_values<degree>(src.begin(), f + (f % 2 ? -1 : 1), dof_indices_neighbor, face_array);

      constexpr dealii::ndarray<int, 3, 3> dofs_per_direction{
        {{{n_n, n_t, n_t}}, {{n_t, n_n, n_t}}, {{n_t, n_t, n_n}}}};

      // For faces located on MPI-remote processes, we replace the dummy
      // data read out above by the data sent to us
      for(unsigned int v = 0; v < n_lanes; ++v)
        if(mpi_exchange_data_on_faces[cell][f][v] != numbers::invalid_unsigned_int)
          for(unsigned int d = 0, c = 0; d < dim; ++d)
          {
            unsigned int dofs_on_this_face = 1;
            for(unsigned int e = 0; e < dim; ++e)
              if(e != d)
                dofs_on_this_face *= dofs_per_direction[f / 2][e];
            for(unsigned int i = 0; i < 2 * dofs_on_this_face; ++i)
            {
              face_array[i + 2 * c][v] =
                import_values[mpi_exchange_data_on_faces[cell][f][v] + i + c * 2];
            }
            c += dofs_on_this_face;
          }

      const VectorizedArray<Number> * values_dofs = face_array + face_direction * n_t * n_n * 2;
      VectorizedArray<Number> *       values      = values_face[1][face_direction];
      VectorizedArray<Number> *       gradients   = grads_face[1][face_direction];
      eval_iso.template values<0, true, false>(values_dofs, scratch_data);
      eval_iso.template values<1, true, false>(scratch_data, values);
      eval_iso.template values<0, true, false>(values_dofs + n_t * n_t, scratch_data);
      eval_iso.template values<1, true, false, dim>(scratch_data, gradients + face_direction);

      if(face_direction == 0)
      {
        values_dofs = face_array + 2 * n_t * n_t;
        values      = values_face[1][1];
        gradients   = grads_face[1][1];
      }
      else
      {
        values_dofs = face_array;
        values      = values_face[1][0];
        gradients   = grads_face[1][0];
      }
      eval_aniso.template normal<0>(shape_data[0], values_dofs, values);
      eval_aniso.template tangential<1, 0>(shape_data[1], values, values);
      eval_aniso.template normal<0>(shape_data[0], values_dofs + n_t * n_n, scratch_data);
      eval_aniso.template tangential<1, 0, dim>(shape_data[1],
                                                scratch_data,
                                                gradients + face_direction);

      if(face_direction < 2)
      {
        values_dofs = face_array + 2 * n_t * n_t + 2 * n_t * n_n;
        values      = values_face[1][2];
        gradients   = grads_face[1][2];
      }
      else
      {
        values_dofs = face_array + 2 * n_t * n_n;
        values      = values_face[1][1];
        gradients   = grads_face[1][1];
      }
      eval_aniso.template normal<1>(shape_data[0], values_dofs, values);
      eval_aniso.template tangential<0, 1>(shape_data[1], values, values);

      eval_aniso.template normal<1>(shape_data[0], values_dofs + n_t * n_n, scratch_data);
      eval_aniso.template tangential<0, 1, dim>(shape_data[1],
                                                scratch_data,
                                                gradients + face_direction);
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

        const VectorizedArray<Number> * my_vals = quad_values + idx * dim;
        VectorizedArray<Number>         v0      = shape[0][0] * my_vals[0];
        VectorizedArray<Number>         d0      = shape[0][1] * my_vals[0];
        VectorizedArray<Number>         v1      = shape[0][0] * my_vals[1];
        VectorizedArray<Number>         d1      = shape[0][1] * my_vals[1];
        VectorizedArray<Number>         v2      = shape[0][0] * my_vals[2];
        VectorizedArray<Number>         d2      = shape[0][1] * my_vals[2];
        const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction) * dim;
        for(unsigned int i = 1; i < n_q_points_1d; ++i)
        {
          v0 += shape[i][0] * my_vals[i * stride + 0];
          d0 += shape[i][1] * my_vals[i * stride + 0];
          v1 += shape[i][0] * my_vals[i * stride + 1];
          d1 += shape[i][1] * my_vals[i * stride + 1];
          v2 += shape[i][0] * my_vals[i * stride + 2];
          d2 += shape[i][1] * my_vals[i * stride + 2];
        }
        const unsigned int val_idx   = i1 * n_q_points_1d + i0;
        values_face[0][0][val_idx]   = v0;
        values_face[0][1][val_idx]   = v1;
        values_face[0][2][val_idx]   = v2;
        const unsigned int deriv_idx = val_idx * dim + face_direction;
        grads_face[0][0][deriv_idx]  = d0;
        grads_face[0][1][deriv_idx]  = d1;
        grads_face[0][2][deriv_idx]  = d2;
      }

    // Only need some tangential derivative in xy plane, as the domain is
    // extruded in z direction
    if(face_direction < 2)
      for(unsigned int side = 0; side < (compute_exterior ? 2 : 1); ++side)
      {
        for(unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int first  = face_direction == 0 ? 1 : 0;
          const unsigned int second = face_direction == 1 ? 2 : first + 1;
          eval_g.template gradients<0, true, false, dim>(values_face[side][d],
                                                         grads_face[side][d] + first);
          // This is a logical contradiction to the above if statement, so
          // this is actually never executed, but we keep it for
          // completeness
          if(face_direction == 2)
            eval_g.template gradients<1, true, false, dim>(values_face[side][d],
                                                           grads_face[side][d] + second);
        }
      }

    const Number                  h_z         = this->h_z;
    const Number                  h_z_inverse = this->h_z_inverse;
    const VectorizedArray<Number> sigmaF      = penalty_parameters[cell][f] * factor_lapl;

    if(face_direction < 2)
    {
      std::array<unsigned int, n_lanes> shifted_data_indices;
      std::array<unsigned int, n_lanes> shifted_data_indices_gr;
      std::array<unsigned int, n_lanes> shifted_data_indices_norm;
      std::array<unsigned int, n_lanes> shifted_data_indices_neigh;
      std::array<unsigned int, n_lanes> shifted_data_indices_gr_neigh;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx           = face_mapping_data_index[cell][f][0][v];
        shifted_data_indices[v]          = idx * 4;
        shifted_data_indices_gr[v]       = idx * 6;
        shifted_data_indices_norm[v]     = idx * 2;
        const unsigned int neighbor_idx  = face_mapping_data_index[cell][f][1][v];
        shifted_data_indices_neigh[v]    = neighbor_idx * 4;
        shifted_data_indices_gr_neigh[v] = neighbor_idx * 6;
      }

      for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy[2];
        vectorized_load_and_transpose(4,
                                      &face_jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0][0]);
        vectorized_load_and_transpose(4,
                                      &face_jacobians_xy[q1][0][0],
                                      shifted_data_indices_neigh.data(),
                                      &jac_xy[1][0][0]);

        const VectorizedArray<Number> area_element_xy =
          std::sqrt(jac_xy[0][0][1 - face_direction] * jac_xy[0][0][1 - face_direction] +
                    jac_xy[0][1][1 - face_direction] * jac_xy[0][1][1 - face_direction]);

        const VectorizedArray<Number> inv_jac_det[2] = {Number(1.0) / determinant(jac_xy[0]),
                                                        Number(1.0) / determinant(jac_xy[1])};
        const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy[2] = {transpose(invert(jac_xy[0])),
                                                                     transpose(invert(jac_xy[1]))};

        Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy[2];
        vectorized_load_and_transpose(6,
                                      &face_jacobian_grads[q1][0][0],
                                      shifted_data_indices_gr.data(),
                                      &jac_grad_xy[0][0][0]);
        if(compute_exterior)
          vectorized_load_and_transpose(6,
                                        &face_jacobian_grads[q1][0][0],
                                        shifted_data_indices_gr_neigh.data(),
                                        &jac_grad_xy[1][0][0]);

        Tensor<1, 2, VectorizedArray<Number>> normal;
        vectorized_load_and_transpose(2,
                                      &face_normal_vector_xy[q1][0],
                                      shifted_data_indices_norm.data(),
                                      &normal[0]);

        Tensor<1, 2, VectorizedArray<Number>> jac_x_normal[2] = {normal * inv_jac_xy[0],
                                                                 normal * inv_jac_xy[1]};

        // Prepare for -(J^{-T} * jac_grad * J^{-1} * J * values *
        // det(J^{-1})), which can be expressed as a rank-1 update
        // tmp[d] * tmp4[e], where tmp = J * values and tmp4 =
        // (J^{-T} * jac_grad * J^{-1})
        VectorizedArray<Number> jac_grad_rank1_normal[2];
        for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
        {
          VectorizedArray<Number> tmp[2];
          for(unsigned int d = 0; d < 2; ++d)
            tmp[d] = (inv_jac_xy[s][0][d] * jac_grad_xy[s][d][0] +
                      inv_jac_xy[s][1][d] * jac_grad_xy[s][d][1] +
                      inv_jac_xy[s][0][1 - d] * jac_grad_xy[s][2][0] +
                      inv_jac_xy[s][1][1 - d] * jac_grad_xy[s][2][1]);
          jac_grad_rank1_normal[s] = (tmp[0] * jac_x_normal[s][0] + tmp[1] * jac_x_normal[s][1]);
        }

        for(unsigned int qz = 0, q = q1; qz < n_q_points_1d; ++qz, q += n_q_points_1d)
        {
          const VectorizedArray<Number> val[2][3] = {
            {values_face[0][0][q], values_face[0][1][q], values_face[0][2][q]},
            {values_face[1][0][q], values_face[1][1][q], values_face[1][2][q]}};
          const VectorizedArray<Number> grad[2][3][2] = {
            {{grads_face[0][0][q * dim], grads_face[0][0][q * dim + 1]},
             {grads_face[0][1][q * dim], grads_face[0][1][q * dim + 1]},
             {grads_face[0][2][q * dim], grads_face[0][2][q * dim + 1]}},
            {{grads_face[1][0][q * dim], grads_face[1][0][q * dim + 1]},
             {grads_face[1][1][q * dim], grads_face[1][1][q * dim + 1]},
             {grads_face[1][2][q * dim], grads_face[1][2][q * dim + 1]}}};

          Tensor<1, dim, VectorizedArray<Number>> val_real[2];
          Tensor<1, dim, VectorizedArray<Number>> normal_derivatives[2];
          for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
          {
            val_real[s][0] = jac_xy[s][0][0] * val[s][0] + jac_xy[s][0][1] * val[s][1];
            val_real[s][1] = jac_xy[s][1][0] * val[s][0] + jac_xy[s][1][1] * val[s][1];

            for(unsigned int d = 0; d < 2; ++d)
            {
              // (J * grad_quad) * J^-1 * det(J^-1), part in
              // braces
              VectorizedArray<Number> tmp[2];
              for(unsigned int e = 0; e < 2; ++e)
                tmp[e] = (jac_xy[s][d][0] * grad[s][0][e] + jac_xy[s][d][1] * grad[s][1][e]);

              // Add (jac_grad * values) * J^{-1} * det(J^{-1}),
              // combine terms outside braces with gradient part
              // from above
              tmp[0] += jac_grad_xy[s][0][d] * val[s][0];
              tmp[0] += jac_grad_xy[s][2][d] * val[s][1];
              tmp[1] += jac_grad_xy[s][1][d] * val[s][1];
              tmp[1] += jac_grad_xy[s][2][d] * val[s][0];

              VectorizedArray<Number> normal_deriv_d =
                tmp[0] * jac_x_normal[s][0] + tmp[1] * jac_x_normal[s][1];

              normal_deriv_d -= jac_grad_rank1_normal[s] * val_real[s][d];

              normal_derivatives[s][d] = (h_z_inverse * inv_jac_det[s]) * normal_deriv_d;
            }

            // jac[2][2] (= h_z) cancels with h_z_inverse contained in
            // determinant
            normal_derivatives[s][2] = inv_jac_det[s] * (jac_x_normal[s][0] * grad[s][2][0] +
                                                         jac_x_normal[s][1] * grad[s][2][1] -
                                                         jac_grad_rank1_normal[s] * val[s][2]);

            for(unsigned int d = 0; d < 2; ++d)
              val_real[s][d] *= inv_jac_det[s] * h_z_inverse;
            val_real[s][2] = inv_jac_det[s] * val[s][2];
          }

          if(compute_exterior)
          {
            val_real[1] =
              boundary_mask * val_real[0] + (Number(1.0) - std::abs(boundary_mask)) * val_real[1];
            normal_derivatives[1] = -boundary_mask * normal_derivatives[0] +
                                    (Number(1.0) - std::abs(boundary_mask)) * normal_derivatives[1];
          }

          // the length element h_z cancels with h_z_inverse of
          // determinant in Piola transformation for test function
          const VectorizedArray<Number> integrate_factor =
            (quad_weights_z[q1] * area_element_xy) * quad_weights_z[qz];

          // physical terms
          const VectorizedArray<Number> effective_factor =
            integrate_factor * make_vectorized_array<Number>(0.5 * factor_lapl);

          const auto viscous_value_flux =
            (integrate_factor * sigmaF) * (val_real[0] - val_real[1]) -
            effective_factor * (normal_derivatives[0] + normal_derivatives[1]);

          const auto viscous_gradient_flux = effective_factor * (val_real[1] - val_real[0]);

          // apply test functions via transpose of the above
          for(unsigned int s = 0; s < (use_face_buffer ? 2 : 1); ++s)
          {
            const auto my_viscous_gradient = viscous_gradient_flux * inv_jac_det[s];
            const auto my_viscous_value =
              viscous_value_flux * (s == 1 ? -inv_jac_det[s] : inv_jac_det[s]);
            VectorizedArray<Number> tmp[2][2];
            for(unsigned int d = 0; d < 2; ++d)
              for(unsigned int e = 0; e < 2; ++e)
                tmp[d][e] = jac_x_normal[s][d] * my_viscous_gradient[e];

            for(unsigned int d = 0; d < 2; ++d)
            {
              grads_face[s][0][q * dim + d] =
                (jac_xy[s][0][0] * tmp[d][0] + jac_xy[s][1][0] * tmp[d][1]);
              grads_face[s][1][q * dim + d] =
                (jac_xy[s][0][1] * tmp[d][0] + jac_xy[s][1][1] * tmp[d][1]);
              grads_face[s][2][q * dim + d] = (h_z * my_viscous_gradient[2]) * jac_x_normal[s][d];
            }

            // jac_grad * (J^{-1} * (grad_in * factor)), re-use
            // part in braces as 'tmp' from above
            VectorizedArray<Number> value_tmp[dim];
            for(unsigned int d = 0; d < 2; ++d)
              value_tmp[d] =
                (tmp[d][0] * jac_grad_xy[s][d][0] + tmp[d][1] * jac_grad_xy[s][d][1] +
                 tmp[1 - d][0] * jac_grad_xy[s][2][0] + tmp[1 - d][1] * jac_grad_xy[s][2][1]);

            //   -(grad_in * factor) * J * (J^{-T} * jac_grad * J^{-1})
            // = -(grad_in * factor) * J * ( \------- tmp4 ---------/ )
            for(unsigned int d = 0; d < 2; ++d)
            {
              VectorizedArray<Number> tmp2 =
                my_viscous_value[d] - my_viscous_gradient[d] * jac_grad_rank1_normal[s];
              for(unsigned int e = 0; e < 2; ++e)
                value_tmp[e] += jac_xy[s][d][e] * tmp2;
            }
            values_face[s][0][q] = value_tmp[0];
            values_face[s][1][q] = value_tmp[1];
            values_face[s][2][q] =
              h_z * (my_viscous_value[2] - my_viscous_gradient[2] * jac_grad_rank1_normal[s]);
          }
        }
      }
    }
    else
    {
      std::array<unsigned int, n_lanes> shifted_data_indices;
      for(unsigned int v = 0; v < n_lanes; ++v)
        shifted_data_indices[v] = mapping_data_index[cell][v] * 4;

      const Number normal_sign = (f % 2) ? 1 : -1;

      for(unsigned int q = 0; q < n_q_points_2d; ++q)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &jacobians_xy[q][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number> inv_jac_det = Number(1.0) / determinant(jac_xy);

        const VectorizedArray<Number> val[2][3] = {
          {values_face[0][0][q], values_face[0][1][q], values_face[0][2][q]},
          {values_face[1][0][q], values_face[1][1][q], values_face[1][2][q]}};
        const VectorizedArray<Number> grad[2][3] = {{grads_face[0][0][q * dim + 2],
                                                     grads_face[0][1][q * dim + 2],
                                                     grads_face[0][2][q * dim + 2]},
                                                    {grads_face[1][0][q * dim + 2],
                                                     grads_face[1][1][q * dim + 2],
                                                     grads_face[1][2][q * dim + 2]}};

        Tensor<1, dim, VectorizedArray<Number>> val_real[2];
        Tensor<1, dim, VectorizedArray<Number>> normal_derivatives[2];
        for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
        {
          for(unsigned int d = 0; d < 2; ++d)
            normal_derivatives[s][d] = (h_z_inverse * h_z_inverse * normal_sign * inv_jac_det) *
                                       (jac_xy[d][0] * grad[s][0] + jac_xy[d][1] * grad[s][1]);
          normal_derivatives[s][2] = (h_z_inverse * normal_sign * inv_jac_det) * grad[s][2];

          for(unsigned int d = 0; d < 2; ++d)
            val_real[s][d] =
              (h_z_inverse * inv_jac_det) * (jac_xy[d][0] * val[s][0] + jac_xy[d][1] * val[s][1]);

          // h_z cancels with factor h_z_inverse from Jacobian
          // determinant
          val_real[s][2] = inv_jac_det * val[s][2];
        }

        const VectorizedArray<Number> integrate_factor = quad_weights_xy[q] * determinant(jac_xy);

        // physical terms
        const VectorizedArray<Number> effective_factor =
          integrate_factor * make_vectorized_array<Number>(0.5 * factor_lapl);
        const auto viscous_value_flux =
          (integrate_factor * sigmaF) * (val_real[0] - val_real[1]) -
          effective_factor * (normal_derivatives[0] + normal_derivatives[1]);

        const auto viscous_gradient_flux = effective_factor * (val_real[1] - val_real[0]);

        // transpose of evaluate part
        for(unsigned int d = 0; d < 2; ++d)
          values_face[0][d][q] =
            (h_z_inverse * inv_jac_det) *
            (jac_xy[0][d] * viscous_value_flux[0] + jac_xy[1][d] * viscous_value_flux[1]);
        values_face[0][2][q] = inv_jac_det * viscous_value_flux[2];
        for(unsigned int d = 0; d < 2; ++d)
          grads_face[0][d][q * dim + 2] =
            (h_z_inverse * h_z_inverse * normal_sign * inv_jac_det) *
            (jac_xy[0][d] * viscous_gradient_flux[0] + jac_xy[1][d] * viscous_gradient_flux[1]);
        grads_face[0][2][q * dim + 2] =
          (h_z_inverse * normal_sign * inv_jac_det) * viscous_gradient_flux[2];
        if(use_face_buffer)
        {
          for(unsigned int d = 0; d < 3; ++d)
            values_face[1][d][q] = -values_face[0][d][q];
          for(unsigned int d = 0; d < 3; ++d)
            grads_face[1][d][q * dim + 2] = grads_face[0][d][q * dim + 2];
        }
      }
    }

    if(face_direction < 2)
      for(unsigned int s = 0; s < ((use_face_buffer && f % 2) ? 2 : 1); ++s)
        for(unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int first  = face_direction == 0 ? 1 : 0;
          const unsigned int second = face_direction == 1 ? 2 : first + 1;
          eval_g.template gradients<0, false, true, dim>(grads_face[s][d] + first,
                                                         values_face[s][d]);
          if(face_direction == 2)
            eval_g.template gradients<1, false, true, dim>(grads_face[s][d] + second,
                                                           values_face[s][d]);
        }

    // fill flux buffer
    if(use_face_buffer && f % 2)
    {
      for(unsigned int d = 0; d < dim; ++d)
        vectorized_transpose_and_store(false,
                                       n_q_points_face,
                                       values_face[1][d],
                                       face_flux_buffer_index[cell][f].data(),
                                       face_flux_buffer.data() + d * n_q_points_face);
      for(unsigned int d = 0; d < dim; ++d)
      {
        for(unsigned int q = 0; q < n_q_points_face; ++q)
          values_face[1][0][q] = grads_face[1][d][q * dim + face_direction];
        vectorized_transpose_and_store(false,
                                       n_q_points_face,
                                       values_face[1][0],
                                       face_flux_buffer_index[cell][f].data(),
                                       face_flux_buffer.data() + (dim + d) * n_q_points_face);
      }
    }
    if(use_face_buffer && all_left_face_fluxes_from_buffer[cell][face_direction])
    {
      for(unsigned int d = 0; d < dim; ++d)
      {
        vectorized_load_and_transpose(n_q_points_face,
                                      face_flux_buffer.data() + (dim + d) * n_q_points_face,
                                      face_flux_buffer_index[cell][f - 1].data(),
                                      values_face[1][0]);
        for(unsigned int q = 0; q < n_q_points_face; ++q)
          grads_face[1][d][q * dim + face_direction] = values_face[1][0][q];
      }
      for(unsigned int d = 0; d < dim; ++d)
        vectorized_load_and_transpose(n_q_points_face,
                                      face_flux_buffer.data() + d * n_q_points_face,
                                      face_flux_buffer_index[cell][f - 1].data(),
                                      values_face[1][d]);
    }

    const std::array<Number, 2> * shape_o = interpolate_quad_to_boundary[0].data();

    // interpolate from face to interior
    const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction) * dim;
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

          VectorizedArray<Number> *     vals      = out_values + idx * dim;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> v0        = values_face[0][0][val_idx];
          const VectorizedArray<Number> v1        = values_face[0][1][val_idx];
          const VectorizedArray<Number> v2        = values_face[0][2][val_idx];
          const VectorizedArray<Number> d0        = grads_face[0][0][deriv_idx];
          const VectorizedArray<Number> d1        = grads_face[0][1][deriv_idx];
          const VectorizedArray<Number> d2        = grads_face[0][2][deriv_idx];
          const VectorizedArray<Number> vo0       = values_face[1][0][val_idx];
          const VectorizedArray<Number> vo1       = values_face[1][1][val_idx];
          const VectorizedArray<Number> vo2       = values_face[1][2][val_idx];
          const VectorizedArray<Number> do0       = grads_face[1][0][deriv_idx];
          const VectorizedArray<Number> do1       = grads_face[1][1][deriv_idx];
          const VectorizedArray<Number> do2       = grads_face[1][2][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
          {
            vals[i * stride + 0] +=
              shape[i][0] * v0 + shape[i][1] * d0 + shape_o[i][0] * vo0 + shape_o[i][1] * do0;
            vals[i * stride + 1] +=
              shape[i][0] * v1 + shape[i][1] * d1 + shape_o[i][0] * vo1 + shape_o[i][1] * do1;
            vals[i * stride + 2] +=
              shape[i][0] * v2 + shape[i][1] * d2 + shape_o[i][0] * vo2 + shape_o[i][1] * do2;
          }
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

          VectorizedArray<Number> *     vals      = out_values + idx * dim;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const VectorizedArray<Number> v0        = values_face[0][0][val_idx];
          const VectorizedArray<Number> v1        = values_face[0][1][val_idx];
          const VectorizedArray<Number> v2        = values_face[0][2][val_idx];
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> d0        = grads_face[0][0][deriv_idx];
          const VectorizedArray<Number> d1        = grads_face[0][1][deriv_idx];
          const VectorizedArray<Number> d2        = grads_face[0][2][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
          {
            vals[i * stride + 0] += shape[i][0] * v0 + shape[i][1] * d0;
            vals[i * stride + 1] += shape[i][0] * v1 + shape[i][1] * d1;
            vals[i * stride + 2] += shape[i][0] * v2 + shape[i][1] * d2;
          }
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
    const unsigned int n_data_per_face = Utilities::pow(quad_weights_z.size(), dim - 1) * 2 * dim;

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

          const auto add_entry = [&](const unsigned int position) {
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

    print_time(static_cast<double>(count_1) / (count_0 + count_1),
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


} // namespace RTOperator
