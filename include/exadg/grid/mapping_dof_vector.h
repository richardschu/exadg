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

#ifndef EXADG_GRID_MAPPING_DOF_VECTOR_H_
#define EXADG_GRID_MAPPING_DOF_VECTOR_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q_cache.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/grid/mapping_fe_cache.h>
#include <exadg/operators/finite_element.h>

namespace ExaDG
{
/**
 * A mapping class based on dealii::MappingQCache equipped with practical
 * interfaces that can be used to initialize the mapping.
 *
 * For simplex elements, dealii::MappingQCache is not available and a
 * MappingFECache object of the same degree is used instead, which stores the
 * mapping support points of all cells in the same way.
 */
template<int dim, typename Number>
class MappingDoFVector
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MappingDoFVector(unsigned int const mapping_degree)
    : hierarchic_to_lexicographic_numbering(
        dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(mapping_degree)),
      lexicographic_to_hierarchic_numbering(
        dealii::Utilities::invert_permutation(hierarchic_to_lexicographic_numbering)),
      mapping_q_cache(std::make_shared<dealii::MappingQCache<dim>>(mapping_degree))
  {
  }

  /**
   * Destructor.
   */
  virtual ~MappingDoFVector() = default;

  /**
   * returns the deal.ii mapping object that describes the deformed mapping
   */
  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const;

  /**
   * Only available for hypercube elements, use get_mapping() otherwise.
   */
  std::shared_ptr<dealii::MappingQCache<dim>>
  get_mapping_q_cache() const;

  /**
   * Extract the grid coordinates of the current mesh configuration described by
   * the `dealii::MappingQCache` object and fill a DoF vector given a
   * corresponding `dealii::DoFHandler` object.
   */
  void
  fill_grid_coordinates_vector(VectorType &                    grid_coordinates,
                               dealii::DoFHandler<dim> const & dof_handler) const;

  /**
   * Extract the grid coordinates for a given external mapping and fill a
   * DoF vector given a corresponding `dealii::DoFHandler` object.
   */
  void
  fill_grid_coordinates_vector(dealii::Mapping<dim> const &    mapping,
                               VectorType &                    grid_coordinates,
                               dealii::DoFHandler<dim> const & dof_handler) const;

  /**
   * Initializes the `dealii::MappingQCache` object by providing a `mapping`
   * that describes an undeformed reference configuration and a displacement
   * DoF vector (with a corresponding `dealii::DoFHandler` object) that
   * describes the displacement of the mesh compared to that reference
   * configuration. This function might also be used if the object of this class
   * refers to a mapping of a specific multigrid level. In both cases, the
   * `displacement_vector` and `dof_handler` need to be compatible. There are
   * two special cases:
   *
   * If the mapping pointer is invalid, this implies that the reference
   * coordinates are interpreted as zero, i.e., the displacement vector
   * describes the absolute coordinates of the grid points.
   *
   * If the displacement_vector is empty or uninitialized, this implies that no
   * displacements will be added to the grid coordinates of the reference
   * configuration described by mapping.
   *
   * For simplex elements, the displacement is evaluated in the support points
   * of the FE_SimplexP mapping. Hence, `dof_handler` may use any finite element
   * with dim components. Multigrid levels are not supported for simplex
   * elements.
   */
  void
  initialize_mapping_from_dof_vector(
    dealii::Mapping<dim> const *    mapping,
    VectorType const &              displacement_vector,
    dealii::DoFHandler<dim> const & dof_handler,
    unsigned int const              level = dealii::numbers::invalid_unsigned_int);

  /**
   * Initializes the mapping by providing a `mapping` that describes an
   * undeformed reference configuration and a `displacement_function` with dim
   * components that describes the displacement of the mesh compared to that
   * reference configuration. The displacement is evaluated in the mapping
   * support points of all cells of `triangulation`. Works for hypercube and
   * simplex elements.
   */
  void
  initialize_mapping_from_function(dealii::Mapping<dim> const &       mapping,
                                   dealii::Triangulation<dim> const & triangulation,
                                   dealii::Function<dim> const &      displacement_function);

  std::vector<unsigned int> hierarchic_to_lexicographic_numbering;
  std::vector<unsigned int> lexicographic_to_hierarchic_numbering;

protected:
  std::shared_ptr<dealii::MappingQCache<dim>> mapping_q_cache;

private:
  /**
   * Simplex variant of initialize_mapping_from_dof_vector().
   */
  void
  initialize_mapping_fe_cache(dealii::Mapping<dim> const *    mapping,
                              VectorType const &              displacement_vector,
                              dealii::DoFHandler<dim> const & dof_handler,
                              unsigned int const              level);

  /**
   * Mapping for simplex elements, created in the first call of
   * initialize_mapping_from_dof_vector() and only updated afterwards.
   */
  std::shared_ptr<MappingFECache<dim>> mapping_fe_cache;
};


namespace MappingTools
{
/**
 * Use this function to initialize the coarse mappings for use in multigrid.
 *
 * The second argument describes the mapping of the fine triangulation.
 *
 * This function only takes the grid coordinates described by the fine mapping
 * without adding displacements in order to initialize the coarse mappings for
 * all multigrid h-levels.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the
 * correct size according to the number of h-multigrid levels (excluding the
 * finest level). The first entry corresponds to the coarsest triangulation, the
 * last element to the level below the fine triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings_from_mapping_dof_vector(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> & coarse_mappings,
  unsigned int const                                            degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &  fine_mapping,
  dealii::Triangulation<dim> const &                            triangulation);

/**
 *
 * Use this function to initialize the coarse mappings for use in multigrid in
 * case the multigrid algorithm uses a separate triangulation object for each
 * multigrid h-level.
 *
 * The second argument describes the mapping of the fine triangulation.
 *
 * This function only takes the grid coordinates described by the fine mapping
 * without adding displacements in order to initialize the coarse mappings for
 * all multigrid h-levels.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the
 * correct size according to the number of h-multigrid levels (excluding the
 * finest level). The first entry corresponds to the coarsest triangulation, the
 * last element to the level below the fine triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings_from_mapping_dof_vector(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> &          coarse_mappings,
  unsigned int const                                                     degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations);

/**
 * A function to initalize mappings for coarse multigrid h-levels in case the
 * mapping on the fine level is of type `MappingDoFVector`.
 *
 * This function unifies the two functions above depending on whether the
 * multigrid algorithm uses separate triangulation objects for the coarse
 * triangulations or just a single triangulation object for all levels. In the
 * latter case, the last argument of this function remains unused.
 *
 * In all cases, a vector of coarse grid mappings is filled.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the
 * correct size according to the number of h-multigrid levels (excluding the
 * finest level). The first entry corresponds to the coarsest triangulation, the
 * last element to the level below the fine triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> &          coarse_mappings,
  unsigned int const                                                     degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations);

} // namespace MappingTools
} // namespace ExaDG

#endif /* EXADG_GRID_MAPPING_DOF_VECTOR_H_ */
