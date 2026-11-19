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

#ifndef EXADG_GRID_GRID_UTILITIES_H_
#define EXADG_GRID_GRID_UTILITIES_H_

// C/C++
#include <functional>

// deal.II
#include <deal.II/grid/grid_tools.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_data.h>
#include <exadg/grid/perform_local_refinements.h>

namespace ExaDG
{
namespace GridUtilities
{
template<int dim>
using PeriodicFacePairs = std::vector<
  dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>;

/**
 * Initializes the dealii::Mapping depending on the element type
 */
template<int dim>
void
create_mapping(std::shared_ptr<dealii::Mapping<dim>> & mapping,
               ElementType const &                     element_type,
               unsigned int const &                    mapping_degree);

/**
 * This function creates mapping and multigrid_mappings (while the mapping for coarse multigrid
 * h-levels is created only if involves_h_multigrid is true and if mapping_degree_fine is
 * unequal mapping_degree_coarse). Internally, the above function is called, creating a
 * dealii::Mapping depending on the element type and the mapping_degree.
 */
template<int dim, typename Number>
void
create_mapping_with_multigrid(std::shared_ptr<dealii::Mapping<dim>> &           mapping,
                              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings,
                              ElementType const &                               element_type,
                              unsigned int const &                              mapping_degree_fine,
                              unsigned int const & mapping_degree_coarse,
                              bool const           involves_h_multigrid);

/**
 * This function can be seen as some form of "copy constructor" for periodic face pairs,
 * transforming the template argument of dealii::GridTools::PeriodicFacePair from
 * Triangulation::cell_iterator to DoFHandler::cell_iterator.
 */
template<int dim>
std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
transform_periodic_face_pairs_to_dof_cell_iterator(
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_faces,
  dealii::DoFHandler<dim> const &                                dof_handler);

/**
 * This function creates a triangulation based on a lambda function and refinement parameters for
 * global and local mesh refinements. This function is used to create the fine triangulation on the
 * one hand and the coarse triangulations required for multigrid (if needed) on the other hand.
 *
 * This function expects that the argument tria has already been constructed.
 */

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
  std::vector<unsigned int> const &                              vector_local_refinements);

/**
 * Remove all manifolds from `tria` and set the flat manifold ID on the cells of
 * all levels. Triangulation::set_all_manifold_ids() only sets the active cells,
 * such that the coarse triangulations for multigrid would keep the manifold ID
 * of the parent cells without a manifold attached.
 */
template<int dim>
void
remove_manifolds(dealii::Triangulation<dim> & tria);

/**
 * Given a fine_triangulation, this function creates all the coarse triangulations required for
 * multigrid implementations that expect a vector of triangulations.
 *
 * The vector coarse_triangulations only includes the levels coarser than the fine triangulation,
 * where the first entry of the vector corresponds to the coarsest level.
 *
 * We have to distinguish between traingulation types, since for the case of a fully distributed
 * triangulation, one cannot coarse triangulations automatically. Instead, the lambda function
 * `lambda_create_coarse_triangulation()` is used together with the `vector_local_refinemetns`.
 */
template<int dim>
void
create_coarse_triangulations_automatically_from_fine_triangulation(
  dealii::Triangulation<dim> const &                               fine_triangulation,
  PeriodicFacePairs<dim> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<dim>> &                            coarse_periodic_face_pairs,
  GridData const &                                                 data);

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
  std::vector<unsigned int> const                                  vector_local_refinements);

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
  std::vector<unsigned int> const                                  vector_local_refinements);

/**
 * This utility function initializes a Grid object by creating a triangulation and filling the
 * periodic_face_pairs. According to the settings in GridData, the corresponding constructor of
 * dealii::Triangulation (or derived classes) is called. The actual functionality "creating" the
 * triangulation needs to be provided via lambda_create_triangulation.
 */
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
  std::vector<unsigned int> const                                vector_local_refinements);

/**
 * This function creates both the fine triangulation and, if needed, the coarse triangulations
 * required for certain geometric coarsening sequences in multigrid. In addition to the functions
 * above, this function exists in order to provide a simple interface for applications.
 */
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
  std::vector<unsigned int> const                                vector_local_refinements);

/**
 * Function to create the coarse_triangulations and coarse_periodic_face_pairs given the
 * fine_triangulation and fine_periodic_face_pairs.
 */
template<int dim>
void
create_coarse_triangulations_after_coarsening_and_refinement(
  dealii::Triangulation<dim> const &                               fine_triangulation,
  PeriodicFacePairs<dim> const &                                   fine_periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> & coarse_triangulations_const,
  std::vector<PeriodicFacePairs<dim>> &                            coarse_periodic_face_pairs,
  GridData const &                                                 data,
  bool const                                                       amr_preserves_boundary_cells);

/**
 * This function reads an external triangulation. The function takes GridData as an argument
 * and stores the external triangulation in "tria".
 */
template<int dim>
void
read_external_triangulation(dealii::Triangulation<dim, dim> & tria, GridData const & data);

} // namespace GridUtilities
} // namespace ExaDG

#endif /* EXADG_GRID_GRID_UTILITIES_H_ */
