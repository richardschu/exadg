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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_

// C/C++
#include <fstream>
#include <sstream>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/grid/mapping_dof_vector.h>

namespace ExaDG
{
inline std::string
restart_filename(std::string const & name, MPI_Comm const & mpi_comm)
{
  std::string const rank =
    dealii::Utilities::int_to_string(dealii::Utilities::MPI::this_mpi_process(mpi_comm));

  std::string const filename = name + "." + rank + ".restart";

  return filename;
}

inline void
rename_restart_files(std::string const & filename)
{
  // backup: rename current restart file into restart.old in case something fails while writing
  std::string const from = filename;
  std::string const to   = filename + ".old";

  std::ifstream ifile(from.c_str());
  if((bool)ifile) // rename only if file already exists
  {
    int const error = rename(from.c_str(), to.c_str());

    AssertThrow(error == 0, dealii::ExcMessage("Can not rename file: " + from + " -> " + to));
  }
}

inline void
write_restart_file(std::ostringstream & oss, std::string const & filename)
{
  std::ofstream stream(filename.c_str());

  stream << oss.str() << std::endl;
}

template<typename VectorType>
inline void
print_vector_l2_norm(VectorType const & vector)
{
  MPI_Comm const & mpi_comm = vector.get_mpi_communicator();
  double const     l2_norm  = vector.l2_norm();
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::cout << "    global vector l2 norm: " << std::scientific << std::setprecision(8)
              << std::setw(20) << l2_norm << "\n";
  }
}

/**
 * Utility functions to read and write the local entries of a
 * dealii::LinearAlgebra::distributed::Vector
 * from/to a boost archive per block and entry.
 */
template<typename VectorType, typename BoostInputArchiveType>
inline void
read_distributed_vector(VectorType & vector, BoostInputArchiveType & input_archive)
{
  // Depending on VectorType, we have to loop over the blocks to
  // access the local entries via vector.local_element(i).
  using Number = typename VectorType::value_type;
  if(std::is_same<VectorType, dealii::LinearAlgebra::distributed::Vector<Number>>::value)
  {
    dealii::LinearAlgebra::distributed::Vector<Number> * tmp =
      dynamic_cast<dealii::LinearAlgebra::distributed::Vector<Number> *>(&vector);
    for(unsigned int i = 0; i < tmp->locally_owned_size(); ++i)
    {
      input_archive >> tmp->local_element(i);
    }
  }
  else if(std::is_same<VectorType, dealii::LinearAlgebra::distributed::BlockVector<Number>>::value)
  {
    dealii::LinearAlgebra::distributed::BlockVector<Number> * tmp =
      dynamic_cast<dealii::LinearAlgebra::distributed::BlockVector<Number> *>(&vector);
    for(unsigned int i = 0; i < tmp->n_blocks(); ++i)
    {
      for(unsigned int i = 0; i < tmp->block(i).locally_owned_size(); ++i)
      {
        input_archive >> tmp->block(i).local_element(i);
      }
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Reading into this VectorType not supported."));
  }

  // Print L2 norm to screen for comparison.
  print_vector_l2_norm(vector);
}

template<typename VectorType, typename BoostOutputArchiveType>
inline void
write_distributed_vector(VectorType const & vector, BoostOutputArchiveType & output_archive)
{
  // Print L2 norm to screen for comparison.
  print_vector_l2_norm(vector);

  // Depending on VectorType, we have to loop over the blocks to
  // access the local entries via vector.local_element(i).
  using Number = typename VectorType::value_type;
  if(std::is_same<VectorType, dealii::LinearAlgebra::distributed::Vector<Number>>::value)
  {
    dealii::LinearAlgebra::distributed::Vector<Number> const * tmp =
      dynamic_cast<dealii::LinearAlgebra::distributed::Vector<Number> const *>(&vector);
    for(unsigned int i = 0; i < tmp->locally_owned_size(); ++i)
    {
      output_archive << tmp->local_element(i);
    }
  }
  else if(std::is_same<VectorType, dealii::LinearAlgebra::distributed::BlockVector<Number>>::value)
  {
    dealii::LinearAlgebra::distributed::BlockVector<Number> const * tmp =
      dynamic_cast<dealii::LinearAlgebra::distributed::BlockVector<Number> const *>(&vector);
    for(unsigned int i = 0; i < tmp->n_blocks(); ++i)
    {
      for(unsigned int i = 0; i < tmp->block(i).locally_owned_size(); ++i)
      {
        output_archive << tmp->block(i).local_element(i);
      }
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Writing into this VectorType not supported."));
  }
}

/** Utility function to convert a vector of block vector pointers into a
 * vector of vectors of VectorType pointers, where all vectors from each
 * individual block are summarized in a std::vector.
 * This is useful for solution transfer and serialization.
 */
template<typename VectorType, typename BlockVectorType>
std::vector<std::vector<VectorType *>>
get_vectors_per_block(std::vector<BlockVectorType *> const & block_vectors)
{
  unsigned int const n_blocks = block_vectors.at(0)->n_blocks();
  for(unsigned int i = 0; i < block_vectors.size(); ++i)
  {
    AssertThrow(block_vectors[i]->n_blocks() == n_blocks,
                dealii::ExcMessage("Provided number of blocks per "
                                   "BlockVector must be equal."));
  }

  std::vector<std::vector<VectorType *>> vectors_per_block;
  for(unsigned int i = 0; i < n_blocks; ++i)
  {
    std::vector<VectorType *> vectors;
    for(unsigned int j = 0; j < block_vectors.size(); ++j)
    {
      vectors.push_back(&block_vectors[j]->block(i));
    }
    vectors_per_block.push_back(vectors);
  }

  return vectors_per_block;
}

/**
 * Same as above but input argument is a vector of BlockVectors.
 * Return type is a vector of vector of pointers, i.e, unchanged.
 */
template<typename VectorType, typename BlockVectorType>
std::vector<std::vector<VectorType *>>
get_vectors_per_block(std::vector<BlockVectorType> & block_vectors)
{
  unsigned int const n_blocks = block_vectors.at(0).n_blocks();
  for(unsigned int i = 0; i < block_vectors.size(); ++i)
  {
    AssertThrow(block_vectors[i].n_blocks() == n_blocks,
                dealii::ExcMessage("Provided number of blocks per "
                                   "BlockVector must be equal."));
  }

  std::vector<std::vector<VectorType *>> vectors_per_block;
  for(unsigned int i = 0; i < n_blocks; ++i)
  {
    std::vector<VectorType *> vectors;
    for(unsigned int j = 0; j < block_vectors.size(); ++j)
    {
      vectors.push_back(&block_vectors[j].block(i));
    }
    vectors_per_block.push_back(vectors);
  }

  return vectors_per_block;
}

/** Utility function to setup a BlockVector given a vector
 * of DoFHandlers only containing owned DoFs. This can be used
 * in combination with `get_vectors_per_block()` to obtain vectors
 * of VectorType pointers as required for `dealii::SolutionTransfer`.
 */
template<int dim, typename BlockVectorType>
std::vector<BlockVectorType>
get_block_vectors_from_dof_handlers(
  unsigned int const                                        n_vectors,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers)
{
  unsigned int const n_blocks = dof_handlers.size();

  // Setup first BlockVector
  BlockVectorType block_vector(n_blocks);
  for(unsigned int i = 0; i < n_blocks; ++i)
  {
    block_vector.block(i).reinit(dof_handlers[i]->locally_owned_dofs(),
                                 dof_handlers[i]->get_communicator());
  }
  block_vector.collect_sizes();

  std::vector<BlockVectorType> block_vectors(n_vectors, block_vector);

  return block_vectors;
}

/**
 * Utility function to store a std::vector<VectorType> in a triangulation and serialize.
 * We assume that the Triangulation(s) linked to the DoFHandlers are all identical.
 * Note also that the sequence of vectors and DoFHandlers here and in
 * deserialize_triangulation_and_load_vectors() *must* be identical.
 * This function does not consider a mapping to be stored, if it is
 * not provided within the `dof_handlers` (and hence treated like all other vectors). 
 */
template<int dim, typename VectorType>
inline void
store_vectors_in_triangulation_and_serialize(
  std::string const &                                       filename_base,
  std::vector<std::vector<VectorType const *>> const &      vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers)
{
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to store in triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));
  auto const & triangulation = dof_handlers.at(0)->get_triangulation();

  // Check if all the Triangulation(s) associated with the DoFHandlers point to the same object.
  for(unsigned int i = 1; i < dof_handlers.size(); ++i)
  {
    AssertThrow(&dof_handlers[i]->get_triangulation() == &triangulation,
                dealii::ExcMessage("Triangulations of DoFHandlers are not identical."));
  }

  // Loop over the DoFHandlers and store the vectors in the triangulation.
  std::vector<std::shared_ptr<dealii::parallel::distributed::SolutionTransfer<dim, VectorType>>>
    solution_transfers;
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    for(unsigned int j = 0; j < vectors_per_dof_handler[i].size(); ++j)
    {
      print_vector_l2_norm(*vectors_per_dof_handler[i][j]);
    }
    solution_transfers.push_back(
      std::make_shared<dealii::parallel::distributed::SolutionTransfer<dim, VectorType>>(
        *dof_handlers[i]));
    solution_transfers[i]->prepare_for_serialization(vectors_per_dof_handler[i]);
  }

  // Serialize the triangulation keeping a maximum of two snapshots.
  std::string const filename = filename_base + ".triangulation";
  rename_restart_files(filename);
  triangulation.save(filename);
}

/**
 * Same as the function above, but the mapping is stored for tensor-product elements
 * as one of the vectors, while for any other element type, we ignore the mapping in
 * the projection when deserializing.
 */
template<int dim, typename VectorType>
inline void
store_vectors_in_triangulation_and_serialize(
  std::string const &                                       filename_base,
  std::vector<std::vector<VectorType const *>> const &      vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers,
  dealii::Mapping<dim> const &                              mapping,
  dealii::DoFHandler<dim> const *                           dof_handler_mapping)
{
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to store in triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));

  auto const & triangulation = dof_handlers.at(0)->get_triangulation();

  // Check if all the Triangulation(s) associated with the DoFHandlers point to the same object.
  for(unsigned int i = 1; i < dof_handlers.size(); ++i)
  {
    AssertThrow(&dof_handlers[i]->get_triangulation() == &triangulation,
                dealii::ExcMessage("Triangulations of DoFHandlers are not identical."));
  }

  AssertThrow(triangulation.all_reference_cells_are_hyper_cube(),
              dealii::ExcMessage("Serialization including mapping not "
                                 "supported for non-hypercube cell types."));

  // Initialize vector to hold grid coordinates.
  bool       vector_initialized = false;
  VectorType vector_grid_coordinates;
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    if(dof_handlers[i] == dof_handler_mapping and not vector_initialized)
    {
      // Cheaper setup if we already have a vector given in the input arguments.
      vector_grid_coordinates.reinit(*vectors_per_dof_handler[i][0], true /* omit_zeroing_entries */);
      vector_initialized = true;
      break;
    }
  }

  if(not vector_initialized)
  {
    // More expensive setup extracting the `dealii::IndexSet`.
    dealii::IndexSet const & locally_owned_dofs = dof_handler_mapping->locally_owned_dofs();
    dealii::IndexSet const   locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler_mapping);
    vector_grid_coordinates.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   dof_handler_mapping->get_communicator());
  }

  // Fill vector with mapping.
  unsigned int const mapping_degree = dof_handler_mapping->get_fe().degree;
  MappingDoFVector<dim, typename VectorType::value_type> mapping_dof_vector(mapping_degree);
  mapping_dof_vector.fill_grid_coordinates_vector(mapping,
                                                  vector_grid_coordinates,
                                                  *dof_handler_mapping);

  // Attach vector holding mapping and corresponding `dof_handler_mapping`.
  std::vector<std::vector<VectorType const *>> vectors_per_dof_handler_extended =
    vectors_per_dof_handler;
  std::vector<VectorType const *> tmp = {&vector_grid_coordinates};
  vectors_per_dof_handler_extended.push_back(tmp);

  std::vector<dealii::DoFHandler<dim, dim> const *> dof_handlers_extended = dof_handlers;
  dof_handlers_extended.push_back(dof_handler_mapping);

  // Use utility function that ignores the mapping.
  store_vectors_in_triangulation_and_serialize(filename_base,
                                               vectors_per_dof_handler_extended,
                                               dof_handlers_extended);
}

template<int dim>
inline std::shared_ptr<dealii::Triangulation<dim>>
deserialize_triangulation(std::string const &     filename_base,
                          TriangulationType const triangulation_type,
                          MPI_Comm const &        mpi_communicator)
{
  std::shared_ptr<dealii::Triangulation<dim>> triangulation;

  // Deserialize the checkpointed triangulation,
  if(triangulation_type == TriangulationType::Serial)
  {
    triangulation = std::make_shared<dealii::Triangulation<dim>>();
    triangulation->load(filename_base + ".triangulation");
  }
  else if(triangulation_type == TriangulationType::Distributed)
  {
    // Deserialize the coarse triangulation to be stored by the user
    // during `create_grid` in the respective application.
    dealii::Triangulation<dim, dim> coarse_triangulation;
    try
    {
      coarse_triangulation.load(filename_base + ".coarse_triangulation");
    }
    catch(...)
    {
      AssertThrow(false,
                  dealii::ExcMessage("Deserializing coarse triangulation expected in\n" +
                                     filename_base +
                                     ".coarse_triangulation\n"
                                     "make sure to store the coarse grid during `create_grid`\n"
                                     "in the respective application.h"));
    }

    std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> tmp =
      std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(mpi_communicator);
    tmp->copy_triangulation(coarse_triangulation);
    coarse_triangulation.clear();
    tmp->load(filename_base + ".triangulation");

    triangulation = std::dynamic_pointer_cast<dealii::Triangulation<dim>>(tmp);
  }
  else if(triangulation_type == TriangulationType::FullyDistributed)
  {
    // Note that the number of MPI processes the triangulation was
    // saved with cannot change and hence autopartitioning is disabled.
    std::shared_ptr<dealii::parallel::fullydistributed::Triangulation<dim>> tmp =
      std::make_shared<dealii::parallel::fullydistributed::Triangulation<dim>>(mpi_communicator);
    tmp->load(filename_base + ".triangulation");

    triangulation = std::dynamic_pointer_cast<dealii::Triangulation<dim>>(tmp);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("TriangulationType not supported."));
  }

  return triangulation;
}

/**
 * Utility function to load vectors via `dealii::SolutionTransfer`
 * assuming the `Triangulation` the `DoFHandler` was initialized with
 * actually stores the related data.
 */
template<int dim, typename VectorType>
inline void
load_vectors(std::vector<std::vector<VectorType *>> &                  vectors_per_dof_handler,
             std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers)
{
  // The DoFHandlers and vectors are already initialized and
  // ``vectors_per_dof_handler`` contain only owned DoFs.
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to load into from triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));

  // Loop over the DoFHandlers and load the vectors stored in
  // the triangulation the DoFHandlers were initialized with.
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    dealii::parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(
      *dof_handlers[i]);
    solution_transfer.deserialize(vectors_per_dof_handler[i]);

    for(unsigned int j = 0; j < vectors_per_dof_handler[i].size(); ++j)
    {
      print_vector_l2_norm(*vectors_per_dof_handler[i][j]);
    }
  }
}

/**
 * Same as the above function, but consider for a mapping added as an additional vector
 * added during `store_vectors_in_triangulation_and_serialize()`.
 */
template<int dim, typename VectorType>
inline std::shared_ptr<dealii::Mapping<dim>>
load_vectors(std::vector<std::vector<VectorType *>> &                  vectors_per_dof_handler,
             std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers,
             dealii::DoFHandler<dim> const *                           dof_handler_mapping)
{
  // We need a collective call to `SolutionTransfer::deserialize()` with all vectors in a
  // single container. Hence, create a mapping vector and add a pointer to the input argument.
  dealii::IndexSet const & locally_owned_dofs = dof_handler_mapping->locally_owned_dofs();
  dealii::IndexSet const & locally_relevant_dofs =
    dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler_mapping);
  VectorType vector_grid_coordinates(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     dof_handler_mapping->get_communicator());

  // Standard utility function, sequence as in `store_vectors_in_triangulation_and_serialize()`.
  std::vector<std::vector<VectorType *>> vectors_per_dof_handler_extended = vectors_per_dof_handler;
  std::vector<VectorType *> tmp = {&vector_grid_coordinates};
  vectors_per_dof_handler_extended.push_back(tmp);
  std::vector<dealii::DoFHandler<dim, dim> const *> dof_handlers_extended = dof_handlers;
  dof_handlers_extended.push_back(dof_handler_mapping);

  load_vectors(vectors_per_dof_handler_extended, dof_handlers_extended);

  // Reconstruct the mapping given the deserialized grid coordinate vector.
  std::shared_ptr<dealii::Mapping<dim>> mapping;
  unsigned int const                    mapping_degree = dof_handler_mapping->get_fe().degree;
  MappingDoFVector<dim, typename VectorType::value_type> mapping_dof_vector(mapping_degree);
  mapping_dof_vector.fill_grid_coordinates_vector(*mapping,
                                                  vector_grid_coordinates,
                                                  *dof_handler_mapping);

  return mapping;
}

/**
 * Utility function to perform grid-to-grid projection using `dealii::RemotePointEvaluation`.
 * We assume we only have a single `dealii::FiniteElement` per `dealii::DoFHandler`.
 * The VectorType template argument is assumed no to be of `BlockVector` type.
 * Note that this function initializes a complete `dealii::MatrixFree` object and hence
 */
template<int dim, typename VectorType>
inline void
grid_to_grid_projection(std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
                        std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
                        std::shared_ptr<dealii::Mapping<dim>> const &        source_mapping,
                        std::vector<std::vector<VectorType *>> &             target_vectors_per_dof_handler,
                        std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
                        std::shared_ptr<dealii::Mapping<dim> const> const &  target_mapping)
{
  // Check input dimensions.
  AssertThrow(source_vectors_per_dof_handler.size() == source_dof_handlers.size(),
              dealii::ExcMessage("First dimension of source vector of vectors "
                                 "has to match source DoFHandler count."));
  AssertThrow(target_vectors_per_dof_handler.size() == target_dof_handlers.size(),
              dealii::ExcMessage("First dimension of target vector of vectors "
                                 "has to match target DoFHandler count."));
  AssertThrow(source_dof_handlers.size() == target_dof_handlers.size(),
              dealii::ExcMessage("Target and source DoFHandler counts have to match"));
  AssertThrow(source_vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("Vector of source vectors empty."));
  for(unsigned int i = 0; i < source_vectors_per_dof_handler.size(); ++i)
  {
    AssertThrow(source_vectors_per_dof_handler[i].size() ==
                  target_vectors_per_dof_handler.at(i).size(),
                dealii::ExcMessage("Vectors of source and target vectors need to have same size."));
  }

  // Collect integration points ##+

  // Setup suitable `dealii::MatrixFree` and `dealii::FEEvaluation` objects per
  // `dealii::DoFHandler`.

  // inverse_mass_operator.h
  // mass_operator.h
  // jacobi_preconditioner.h
}


} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_ */
