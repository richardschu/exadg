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
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

// ExaDG
#include <exadg/grid/balanced_granularity_partition_policy.h>
#include <exadg/grid/grid_utilities.h>

namespace ExaDG
{
namespace GridUtilities
{
template<int dim>
void
create_mapping(std::shared_ptr<dealii::Mapping<dim>> & mapping,
               ElementType const &                     element_type,
               unsigned int const &                    mapping_degree)
{
  if(element_type == ElementType::Hypercube)
  {
    mapping = std::make_shared<dealii::MappingQ<dim>>(mapping_degree);
  }
  else if(element_type == ElementType::Simplex)
  {
    mapping = std::make_shared<dealii::MappingFE<dim>>(dealii::FE_SimplexP<dim>(mapping_degree));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter element_type."));
  }
}


template<int dim, typename Number>
void
create_mapping_with_multigrid(std::shared_ptr<dealii::Mapping<dim>> &           mapping,
                              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings,
                              ElementType const &                               element_type,
                              unsigned int const &                              mapping_degree_fine,
                              unsigned int const & mapping_degree_coarse,
                              bool const           involves_h_multigrid)
{
  // create fine mapping
  create_mapping(mapping, element_type, mapping_degree_fine);

  // create coarse mappings if needed
  std::shared_ptr<dealii::Mapping<dim>> coarse_mapping;
  if(involves_h_multigrid and (mapping_degree_coarse != mapping_degree_fine))
  {
    create_mapping(coarse_mapping, element_type, mapping_degree_coarse);
  }
  multigrid_mappings = std::make_shared<MultigridMappings<dim, Number>>(mapping, coarse_mapping);
}


template<int dim>
std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
transform_periodic_face_pairs_to_dof_cell_iterator(
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_faces,
  dealii::DoFHandler<dim> const &                                dof_handler)
{
  typedef typename std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
    PeriodicFacesDoF;

  PeriodicFacesDoF periodic_faces_dof;

  for(auto it : periodic_faces)
  {
    dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>
      face_pair_dof_hander;

    face_pair_dof_hander.cell[0] = it.cell[0]->as_dof_handler_iterator(dof_handler);
    face_pair_dof_hander.cell[1] = it.cell[1]->as_dof_handler_iterator(dof_handler);

    face_pair_dof_hander.face_idx[0] = it.face_idx[0];
    face_pair_dof_hander.face_idx[1] = it.face_idx[1];

    face_pair_dof_hander.orientation = it.orientation;
    face_pair_dof_hander.matrix      = it.matrix;

    periodic_faces_dof.push_back(face_pair_dof_hander);
  }

  return periodic_faces_dof;
}


template<int dim>
void
create_triangulation(
  std::shared_ptr<dealii::Triangulation<dim>> &                  triangulation,
  PeriodicFacePairs<dim> &                                       periodic_face_pairs,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  bool const                                                     construct_multigrid_hierarchy,
  std::function<void(dealii::Triangulation<dim> &,
                     PeriodicFacePairs<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  unsigned int const                                             global_refinements,
  std::vector<unsigned int> const &                              vector_local_refinements)
{
  if(vector_local_refinements.size() != 0)
  {
    AssertThrow(
      data.element_type == ElementType::Hypercube,
      dealii::ExcMessage(
        "Local refinements are currently only supported for meshes composed of hypercube elements."));
  }

  // mesh smoothing
  auto mesh_smoothing = dealii::Triangulation<dim>::none;

  // the mesh used for a simulation should not depend on the preconditioner used
  // to solve the problem (like whether multigrid is used or not). Hence, we
  // always set this parameter for ElementType::Hypercube. It is not posible to
  // set this parameter for ElementType::Simplex due to the implementation in
  // deal.II.
  if(data.element_type == ElementType::Hypercube)
  {
    mesh_smoothing = dealii::Triangulation<dim>::limit_level_difference_at_vertices;
  }

  if(data.triangulation_type == TriangulationType::Serial)
  {
    AssertDimension(dealii::Utilities::MPI::n_mpi_processes(mpi_comm), 1);
    triangulation = std::make_shared<dealii::Triangulation<dim>>(mesh_smoothing);

    lambda_create_triangulation(*triangulation,
                                periodic_face_pairs,
                                global_refinements,
                                vector_local_refinements);
  }
  else if(data.triangulation_type == TriangulationType::Distributed)
  {
    typename dealii::parallel::distributed::Triangulation<dim>::Settings distributed_settings =
      dealii::parallel::distributed::Triangulation<dim>::default_setting;

    if(construct_multigrid_hierarchy)
    {
      distributed_settings =
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy;
    }

    triangulation =
      std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(mpi_comm,
                                                                          mesh_smoothing,
                                                                          distributed_settings);

    lambda_create_triangulation(*triangulation,
                                periodic_face_pairs,
                                global_refinements,
                                vector_local_refinements);
  }
  else if(data.triangulation_type == TriangulationType::FullyDistributed)
  {
    auto const serial_grid_generator = [&](dealii::Triangulation<dim, dim> & tria_serial) {
      lambda_create_triangulation(tria_serial,
                                  periodic_face_pairs,
                                  global_refinements,
                                  vector_local_refinements);
    };

    auto const serial_grid_partitioner = [&](dealii::Triangulation<dim, dim> & tria_serial,
                                             MPI_Comm const                    comm,
                                             unsigned int const                group_size) {
      (void)group_size;
      if(data.partitioning_type == PartitioningType::Metis)
      {
        dealii::GridTools::partition_triangulation(dealii::Utilities::MPI::n_mpi_processes(comm),
                                                   tria_serial);
      }
      else if(data.partitioning_type == PartitioningType::z_order)
      {
        dealii::GridTools::partition_triangulation_zorder(
          dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }
    };

    unsigned int const group_size = 1;

    typename dealii::TriangulationDescription::Settings triangulation_description_setting =
      dealii::TriangulationDescription::default_setting;

    if(construct_multigrid_hierarchy)
    {
      triangulation_description_setting =
        dealii::TriangulationDescription::construct_multigrid_hierarchy;
    }

    triangulation =
      std::make_shared<dealii::parallel::fullydistributed::Triangulation<dim>>(mpi_comm);

    auto const description = dealii::TriangulationDescription::Utilities::
      create_description_from_triangulation_in_groups<dim, dim>(
        serial_grid_generator,
        serial_grid_partitioner,
        triangulation->get_mpi_communicator(),
        group_size,
        mesh_smoothing,
        triangulation_description_setting);

    triangulation->create_triangulation(description);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
  }
}


template<int dim>
void
remove_manifolds(dealii::Triangulation<dim> & tria)
{
  tria.reset_all_manifolds();
  for(auto const & cell : tria.cell_iterators())
    cell->set_all_manifold_ids(dealii::numbers::flat_manifold_id);
}


template<int dim>
void
create_coarse_triangulations_automatically_from_fine_triangulation(
  dealii::Triangulation<dim> const &                               fine_triangulation,
  PeriodicFacePairs<dim> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<dim>> &                            coarse_periodic_face_pairs,
  GridData const &                                                 data)
{
  // In case of a serial or distributed triangulation, deal.II can automatically
  // generate the coarse triangulations.
  AssertThrow(data.triangulation_type == TriangulationType::Serial or
                data.triangulation_type == TriangulationType::Distributed,
              dealii::ExcMessage("Invalid parameter triangulation_type."));

  if(data.triangulation_type == TriangulationType::Serial)
  {
    AssertThrow(
      fine_triangulation.all_reference_cells_are_hyper_cube(),
      dealii::ExcMessage(
        "The create_geometric_coarsening_sequence function of dealii does currently not support "
        "simplicial elements."));

    coarse_triangulations_const =
      dealii::MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        fine_triangulation);
  }
  else if(data.triangulation_type == TriangulationType::Distributed)
  {
    AssertThrow(
      fine_triangulation.all_reference_cells_are_hyper_cube(),
      dealii::ExcMessage(
        "dealii::parallel::distributed::Triangulation does not support simplicial elements."));

    coarse_triangulations_const =
      dealii::MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        fine_triangulation,
        BalancedGranularityPartitionPolicy<dim>(
          dealii::Utilities::MPI::n_mpi_processes(fine_triangulation.get_mpi_communicator())));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
  }

  // deal.II adds the fine triangulation as the last entry of the vector.
  // According to our convention in ExaDG, we only include the triangulations
  // coarser than the fine level. Hence, we remove the last entry of the vector.
  coarse_triangulations_const.pop_back();

  coarse_periodic_face_pairs.resize(coarse_triangulations_const.size());
  for(unsigned int level = 0; level < coarse_periodic_face_pairs.size(); ++level)
  {
    coarse_periodic_face_pairs[level] = fine_periodic_face_pairs;
  }
}


template<int dim>
void
create_coarse_triangulations_for_fully_distributed_triangulation(
  dealii::Triangulation<dim> const &                               fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<dim>> &                            coarse_periodic_face_pairs,
  GridData const &                                                 data,
  std::function<void(dealii::Triangulation<dim> &,
                     PeriodicFacePairs<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const &   lambda_create_triangulation,
  std::vector<unsigned int> const                                  vector_local_refinements)
{
  // In case of a fully distributed triangulation, deal.II cannot automatically
  // generate the coarse triangulations. Create the coarse traingulations using
  // the lambda function and the vector of local refinements.
  AssertThrow(data.triangulation_type == TriangulationType::FullyDistributed,
              dealii::ExcMessage("Invalid parameter triangulation_type."));

  if(fine_triangulation.n_global_levels() >= 2)
  {
    // Resize the empty coarse_triangulations and coarse_periodic_face_pairs
    // vectors.
    std::vector<std::shared_ptr<dealii::Triangulation<dim>>> coarse_triangulations =
      std::vector<std::shared_ptr<dealii::Triangulation<dim>>>(
        fine_triangulation.n_global_levels() - 1);

    coarse_periodic_face_pairs = std::vector<PeriodicFacePairs<dim>>(coarse_triangulations.size());

    // Start one level below the fine triangulation.
    unsigned int              level        = fine_triangulation.n_global_levels() - 2;
    std::vector<unsigned int> refine_local = vector_local_refinements;

    // Undo global refinements.
    if(data.n_refine_global >= 1)
    {
      unsigned int const n_refine_global_start = (unsigned int)(data.n_refine_global - 1);
      for(int refine_global = n_refine_global_start; refine_global >= 0; --refine_global)
      {
        GridUtilities::create_triangulation<dim>(coarse_triangulations[level],
                                                 coarse_periodic_face_pairs[level],
                                                 fine_triangulation.get_mpi_communicator(),
                                                 data,
                                                 false /*construct_multigrid_hierarchy */,
                                                 lambda_create_triangulation,
                                                 refine_global,
                                                 refine_local);

        if(level > 0)
        {
          level--;
        }
      }
    }

    // Undo local refinements.
    if(refine_local.size() > 0)
    {
      while(*std::max_element(refine_local.begin(), refine_local.end()) != 0)
      {
        for(size_t material_id = 0; material_id < refine_local.size(); material_id++)
        {
          if(refine_local[material_id] > 0)
          {
            refine_local[material_id]--;
          }
        }

        GridUtilities::create_triangulation<dim>(coarse_triangulations[level],
                                                 coarse_periodic_face_pairs[level],
                                                 fine_triangulation.get_mpi_communicator(),
                                                 data,
                                                 false /*construct_multigrid_hierarchy */,
                                                 lambda_create_triangulation,
                                                 0 /*refine_global*/,
                                                 refine_local);

        if(level > 0)
        {
          level--;
        }
      }
    }

    AssertThrow(
      level == 0,
      dealii::ExcMessage(
        "There occurred a logical error when creating the geometric coarsening sequence."));

    // Make all entries in the vector of shared pointers const.
    for(auto const & it : coarse_triangulations)
    {
      coarse_triangulations_const.push_back(it);
    }
  }
}


template<int dim>
void
create_coarse_triangulations(
  dealii::Triangulation<dim> const &                               fine_triangulation,
  PeriodicFacePairs<dim> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<dim>> &                            coarse_periodic_face_pairs,
  GridData const &                                                 data,
  std::function<void(dealii::Triangulation<dim> &,
                     PeriodicFacePairs<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const &   lambda_create_triangulation,
  std::vector<unsigned int> const                                  vector_local_refinements)
{
  // In case of a serial or distributed triangulation, deal.II can automatically
  // generate the coarse triangulations, otherwise, the coarse triangulations
  // have to be explicitily created using the provided lambda function and the
  // vector of local refinements.
  if(data.triangulation_type == TriangulationType::Serial or
     data.triangulation_type == TriangulationType::Distributed)
  {
    create_coarse_triangulations_automatically_from_fine_triangulation(fine_triangulation,
                                                                       fine_periodic_face_pairs,
                                                                       coarse_triangulations_const,
                                                                       coarse_periodic_face_pairs,
                                                                       data);
  }
  else if(data.triangulation_type == TriangulationType::FullyDistributed)
  {
    create_coarse_triangulations_for_fully_distributed_triangulation(fine_triangulation,
                                                                     coarse_triangulations_const,
                                                                     coarse_periodic_face_pairs,
                                                                     data,
                                                                     lambda_create_triangulation,
                                                                     vector_local_refinements);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
  }
}


template<int dim>
void
create_triangulation(
  Grid<dim> &                                                    grid,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<dim> &,
                     PeriodicFacePairs<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements)
{
  GridUtilities::create_triangulation(grid.triangulation,
                                      grid.periodic_face_pairs,
                                      mpi_comm,
                                      data,
                                      false /*construct_multigrid_hierarchy */,
                                      lambda_create_triangulation,
                                      data.n_refine_global,
                                      vector_local_refinements);
}


template<int dim>
void
create_triangulation_with_multigrid(
  Grid<dim> &                                                    grid,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  bool const                                                     involves_h_multigrid,
  std::function<void(dealii::Triangulation<dim> &,
                     PeriodicFacePairs<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements)
{
  if(involves_h_multigrid)
  {
    // Make sure that we create coarse triangulations in case of simplex meshes.
    if(data.element_type == ElementType::Simplex)
    {
      AssertThrow(data.create_coarse_triangulations == true,
                  dealii::ExcMessage(
                    "You need to set GridData::create_coarse_triangulations = true "
                    "in order to use h-multigrid for simplex meshes."));
    }

    // create fine triangulation
    GridUtilities::create_triangulation(grid.triangulation,
                                        grid.periodic_face_pairs,
                                        mpi_comm,
                                        data,
                                        not data.create_coarse_triangulations,
                                        lambda_create_triangulation,
                                        data.n_refine_global,
                                        vector_local_refinements);

    // Make sure that we create coarse triangulations in case of meshes with
    // hanging nodes.
    if(grid.triangulation->has_hanging_nodes())
    {
      AssertThrow(data.create_coarse_triangulations == true,
                  dealii::ExcMessage(
                    "You need to set GridData::create_coarse_triangulations = true "
                    "in order to use h-multigrid for meshes with hanging nodes."));
    }

    // create coarse triangulations
    if(data.create_coarse_triangulations)
    {
      GridUtilities::create_coarse_triangulations(*grid.triangulation,
                                                  grid.periodic_face_pairs,
                                                  grid.coarse_triangulations,
                                                  grid.coarse_periodic_face_pairs,
                                                  data,
                                                  lambda_create_triangulation,
                                                  vector_local_refinements);
    }
  }
  else
  {
    // If no h-multigrid is involved, simply re-direct to the other function
    // that creates the fine triangulation only.
    GridUtilities::create_triangulation<dim>(
      grid, mpi_comm, data, lambda_create_triangulation, vector_local_refinements);
  }
}


template<int dim>
void
create_coarse_triangulations_after_coarsening_and_refinement(
  dealii::Triangulation<dim> const &                               fine_triangulation,
  PeriodicFacePairs<dim> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<dim>> &                            coarse_periodic_face_pairs,
  GridData const &                                                 data,
  bool const                                                       amr_preserves_boundary_cells)
{
  if(data.triangulation_type == TriangulationType::Serial or
     data.triangulation_type == TriangulationType::Distributed)
  {
    // Update periodic face pairs if existent.
    if(fine_periodic_face_pairs.size() > 0)
    {
      AssertThrow(amr_preserves_boundary_cells,
                  dealii::ExcMessage(
                    "Combination of adaptive mesh refinement and periodic face pairs"
                    " requires boundary cells to be preserved."));
    }

    create_coarse_triangulations_automatically_from_fine_triangulation(fine_triangulation,
                                                                       fine_periodic_face_pairs,
                                                                       coarse_triangulations_const,
                                                                       coarse_periodic_face_pairs,
                                                                       data);
  }
  else if(data.triangulation_type == TriangulationType::FullyDistributed)
  {
    AssertThrow(false,
                dealii::ExcMessage("Combination of adaptive mesh refinement and "
                                   "TriangulationType::FullyDistributed not implemented."));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
  }
}


template<int dim>
void
read_external_triangulation(dealii::Triangulation<dim, dim> & tria, GridData const & data)
{
  AssertThrow(not data.file_name.empty(),
              dealii::ExcMessage(
                "You are trying to read a grid file, but the string, which is supposed to contain"
                " the file name, is empty."));

  dealii::GridIn<dim> grid_in;

  grid_in.attach_triangulation(tria);

  // find the file extension from the file name
  std::string extension = data.file_name.substr(data.file_name.find_last_of('.') + 1);

  AssertThrow(not extension.empty(),
              dealii::ExcMessage("You are trying to read a grid file, but the file extension is"
                                 " empty."));

  // decide the file format
  typename dealii::GridIn<dim>::Format format;
  if(extension == "e" || extension == "exo")
    format = dealii::GridIn<dim>::Format::exodusii;
  else
    format = grid_in.parse_format(extension);

  // TODO: check if the exodusIIData is needed
  // typename dealii::GridIn<dim>::ExodusIIData exodusIIData;

  grid_in.read(data.file_name, format);

  AssertThrow(get_element_type(tria) == data.element_type,
              dealii::ExcMessage("You are trying to read a grid file, but the element type of the"
                                 " external grid file and the element type specified in GridData"
                                 " don't match. Most likely, you forgot to change the element_type"
                                 " parameter of GridData to the desired element type."));
}


template void
create_mapping<2>(std::shared_ptr<dealii::Mapping<2>> & mapping,
                  ElementType const &                   element_type,
                  unsigned int const &                  mapping_degree);

template void
create_mapping<3>(std::shared_ptr<dealii::Mapping<3>> & mapping,
                  ElementType const &                   element_type,
                  unsigned int const &                  mapping_degree);

template void
create_mapping_with_multigrid<2, float>(
  std::shared_ptr<dealii::Mapping<2>> &          mapping,
  std::shared_ptr<MultigridMappings<2, float>> & multigrid_mappings,
  ElementType const &                            element_type,
  unsigned int const &                           mapping_degree_fine,
  unsigned int const &                           mapping_degree_coarse,
  bool const                                     involves_h_multigrid);

template void
create_mapping_with_multigrid<2, double>(
  std::shared_ptr<dealii::Mapping<2>> &           mapping,
  std::shared_ptr<MultigridMappings<2, double>> & multigrid_mappings,
  ElementType const &                             element_type,
  unsigned int const &                            mapping_degree_fine,
  unsigned int const &                            mapping_degree_coarse,
  bool const                                      involves_h_multigrid);

template void
create_mapping_with_multigrid<3, float>(
  std::shared_ptr<dealii::Mapping<3>> &          mapping,
  std::shared_ptr<MultigridMappings<3, float>> & multigrid_mappings,
  ElementType const &                            element_type,
  unsigned int const &                           mapping_degree_fine,
  unsigned int const &                           mapping_degree_coarse,
  bool const                                     involves_h_multigrid);

template void
create_mapping_with_multigrid<3, double>(
  std::shared_ptr<dealii::Mapping<3>> &           mapping,
  std::shared_ptr<MultigridMappings<3, double>> & multigrid_mappings,
  ElementType const &                             element_type,
  unsigned int const &                            mapping_degree_fine,
  unsigned int const &                            mapping_degree_coarse,
  bool const                                      involves_h_multigrid);

template std::vector<
  dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<2>::cell_iterator>>
transform_periodic_face_pairs_to_dof_cell_iterator<2>(
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<2>::cell_iterator>> const & periodic_faces,
  dealii::DoFHandler<2> const &                                dof_handler);

template std::vector<
  dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<3>::cell_iterator>>
transform_periodic_face_pairs_to_dof_cell_iterator<3>(
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<3>::cell_iterator>> const & periodic_faces,
  dealii::DoFHandler<3> const &                                dof_handler);

template void
create_triangulation<2>(
  std::shared_ptr<dealii::Triangulation<2>> &                    triangulation,
  PeriodicFacePairs<2> &                                         periodic_face_pairs,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  bool const                                                     construct_multigrid_hierarchy,
  std::function<void(dealii::Triangulation<2> &,
                     PeriodicFacePairs<2> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  unsigned int const                                             global_refinements,
  std::vector<unsigned int> const &                              vector_local_refinements);

template void
create_triangulation<3>(
  std::shared_ptr<dealii::Triangulation<3>> &                    triangulation,
  PeriodicFacePairs<3> &                                         periodic_face_pairs,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  bool const                                                     construct_multigrid_hierarchy,
  std::function<void(dealii::Triangulation<3> &,
                     PeriodicFacePairs<3> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  unsigned int const                                             global_refinements,
  std::vector<unsigned int> const &                              vector_local_refinements);

template void
remove_manifolds<2>(dealii::Triangulation<2> & tria);

template void
remove_manifolds<3>(dealii::Triangulation<3> & tria);

template void
create_coarse_triangulations_automatically_from_fine_triangulation<2>(
  dealii::Triangulation<2> const &                               fine_triangulation,
  PeriodicFacePairs<2> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<2>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data);

template void
create_coarse_triangulations_automatically_from_fine_triangulation<3>(
  dealii::Triangulation<3> const &                               fine_triangulation,
  PeriodicFacePairs<3> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<3>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data);

template void
create_coarse_triangulations_for_fully_distributed_triangulation<2>(
  dealii::Triangulation<2> const &                               fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<2>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<2> &,
                     PeriodicFacePairs<2> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_coarse_triangulations_for_fully_distributed_triangulation<3>(
  dealii::Triangulation<3> const &                               fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<3>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<3> &,
                     PeriodicFacePairs<3> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_coarse_triangulations<2>(
  dealii::Triangulation<2> const &                               fine_triangulation,
  PeriodicFacePairs<2> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<2>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<2> &,
                     PeriodicFacePairs<2> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_coarse_triangulations<3>(
  dealii::Triangulation<3> const &                               fine_triangulation,
  PeriodicFacePairs<3> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<3>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<3> &,
                     PeriodicFacePairs<3> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_triangulation<2>(
  Grid<2> &                                                      grid,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<2> &,
                     PeriodicFacePairs<2> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_triangulation<3>(
  Grid<3> &                                                      grid,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<3> &,
                     PeriodicFacePairs<3> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_triangulation_with_multigrid<2>(
  Grid<2> &                                                      grid,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  bool const                                                     involves_h_multigrid,
  std::function<void(dealii::Triangulation<2> &,
                     PeriodicFacePairs<2> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_triangulation_with_multigrid<3>(
  Grid<3> &                                                      grid,
  MPI_Comm const &                                               mpi_comm,
  GridData const &                                               data,
  bool const                                                     involves_h_multigrid,
  std::function<void(dealii::Triangulation<3> &,
                     PeriodicFacePairs<3> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const                                vector_local_refinements);

template void
create_coarse_triangulations_after_coarsening_and_refinement<2>(
  dealii::Triangulation<2> const &                               fine_triangulation,
  PeriodicFacePairs<2> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<2>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data,
  bool const                                                     amr_preserves_boundary_cells);

template void
create_coarse_triangulations_after_coarsening_and_refinement<3>(
  dealii::Triangulation<3> const &                               fine_triangulation,
  PeriodicFacePairs<3> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<3>> &                            coarse_periodic_face_pairs,
  GridData const &                                               data,
  bool const                                                     amr_preserves_boundary_cells);

template void
read_external_triangulation<2>(dealii::Triangulation<2, 2> & tria, GridData const & data);

template void
read_external_triangulation<3>(dealii::Triangulation<3, 3> & tria, GridData const & data);

} // namespace GridUtilities
} // namespace ExaDG
