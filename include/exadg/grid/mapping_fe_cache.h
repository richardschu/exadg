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

#ifndef EXADG_GRID_MAPPING_FE_CACHE_H_
#define EXADG_GRID_MAPPING_FE_CACHE_H_

// C++
#include <functional>
#include <memory>
#include <vector>

// deal.II
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/tria.h>

namespace ExaDG
{
/**
 * A mapping class that stores the mapping support points of all cells,
 * similar to dealii::MappingQCache, but derived from dealii::MappingFE. Hence,
 * it can be used for all reference cells supported by dealii::MappingFE, in
 * particular for simplices, where dealii::MappingQCache is not available.
 *
 * The support points of all cells on all levels are computed in initialize()
 * by a user-provided function. The points of a cell are expected in the order
 * of get_unit_support_points(), i.e., the unit support points of the finite
 * element passed to the constructor, e.g., dealii::FE_SimplexP. The first
 * points are the vertices of the cell.
 *
 * As for dealii::MappingQCache, copies and clones share the cache that exists
 * at the time of copying, while a subsequent call to initialize() creates a
 * new cache. The cache is cleared upon any change of the triangulation.
 */
template<int dim>
class MappingFECache : public dealii::MappingFE<dim>
{
public:
  typedef typename dealii::Triangulation<dim>::cell_iterator CellIterator;

  /**
   * Constructor. `fe` is a scalar finite element with support points defining
   * the polynomial space of the mapping, e.g., dealii::FE_SimplexP.
   */
  explicit MappingFECache(dealii::FiniteElement<dim> const & fe) : dealii::MappingFE<dim>(fe)
  {
    AssertThrow(fe.n_components() == 1 and fe.n_dofs_per_vertex() == 1,
                dealii::ExcMessage("Expected a scalar finite element with one DoF per vertex."));
  }

  /**
   * Copy constructor, shares the cache with `mapping`.
   */
  MappingFECache(MappingFECache<dim> const & mapping)
    : dealii::MappingFE<dim>(mapping), support_point_cache(mapping.support_point_cache)
  {
  }

  /**
   * Destructor.
   */
  ~MappingFECache()
  {
    // Clear the cache before disconnecting the signal, as done in
    // dealii::MappingQCache.
    support_point_cache.reset();
    clear_signal.disconnect();
  }

  std::unique_ptr<dealii::Mapping<dim>>
  clone() const final
  {
    return std::make_unique<MappingFECache<dim>>(*this);
  }

  bool
  preserves_vertex_locations() const final
  {
    return false;
  }

  /**
   * The unit support points defining the order of the points returned by the
   * function passed to initialize().
   */
  std::vector<dealii::Point<dim>> const &
  get_unit_support_points() const
  {
    return this->fe->get_unit_support_points();
  }

  /**
   * Computes and stores the mapping support points for all cells on all levels
   * of `triangulation` by calling `compute_points_on_cell`. The function is
   * also called for artificial cells, where it may return any points of the
   * correct size.
   */
  void
  initialize(dealii::Triangulation<dim> const & triangulation,
             std::function<std::vector<dealii::Point<dim>>(CellIterator const &)> const &
               compute_points_on_cell)
  {
    clear_signal.disconnect();
    clear_signal =
      triangulation.signals.any_change.connect([this]() { this->support_point_cache.reset(); });

    support_point_cache =
      std::make_shared<std::vector<std::vector<std::vector<dealii::Point<dim>>>>>(
        triangulation.n_levels());
    for(unsigned int level = 0; level < triangulation.n_levels(); ++level)
    {
      (*support_point_cache)[level].resize(triangulation.n_raw_cells(level));
    }

    unsigned int const n_support_points = get_unit_support_points().size();
    for(auto const & cell : triangulation.cell_iterators())
    {
      std::vector<dealii::Point<dim>> & points =
        (*support_point_cache)[cell->level()][cell->index()];
      points = compute_points_on_cell(cell);
      AssertThrow(points.size() == n_support_points,
                  dealii::ExcMessage("Number of points does not match the number of support "
                                     "points of the finite element."));
    }
  }

  /**
   * For a mapping degree of one, dealii::MappingFE reuses the data of the
   * previous cell if the two cells are translations of each other. This is
   * determined from the vertices of the triangulation, which do not describe
   * the cached geometry. Hence, this reuse is disabled here (as done by
   * dealii::MappingQ via preserves_vertex_locations()).
   */
  dealii::CellSimilarity::Similarity
  fill_fe_values(CellIterator const & cell,
                 dealii::CellSimilarity::Similarity const /* cell_similarity */,
                 dealii::Quadrature<dim> const &                         quadrature,
                 typename dealii::Mapping<dim>::InternalDataBase const & internal_data,
                 dealii::internal::FEValuesImplementation::MappingRelatedData<dim, dim> &
                   output_data) const final
  {
    return dealii::MappingFE<dim>::fill_fe_values(
      cell, dealii::CellSimilarity::none, quadrature, internal_data, output_data);
  }

  boost::container::small_vector<dealii::Point<dim>, dealii::ReferenceCells::max_n_vertices<dim>()>
  get_vertices(CellIterator const & cell) const final
  {
    std::vector<dealii::Point<dim>> const & points = get_cached_points(cell);

    return boost::container::small_vector<dealii::Point<dim>,
                                          dealii::ReferenceCells::max_n_vertices<dim>()>(
      points.begin(), points.begin() + cell->n_vertices());
  }

protected:
  boost::container::small_vector<dealii::Point<dim>, 200>
  compute_mapping_support_points(CellIterator const & cell) const final
  {
    std::vector<dealii::Point<dim>> const & points = get_cached_points(cell);

    return boost::container::small_vector<dealii::Point<dim>, 200>(points.begin(), points.end());
  }

private:
  std::vector<dealii::Point<dim>> const &
  get_cached_points(CellIterator const & cell) const
  {
    Assert(support_point_cache.get() != nullptr,
           dealii::ExcMessage("Call MappingFECache::initialize() before using the mapping and "
                              "after the triangulation has changed."));
    AssertIndexRange(cell->level(), support_point_cache->size());
    AssertIndexRange(cell->index(), (*support_point_cache)[cell->level()].size());

    return (*support_point_cache)[cell->level()][cell->index()];
  }

  /**
   * Mapping support points indexed by level, cell index and support point.
   */
  std::shared_ptr<std::vector<std::vector<std::vector<dealii::Point<dim>>>>> support_point_cache;

  /**
   * Connection to the triangulation signal clearing the cache.
   */
  boost::signals2::connection clear_signal;
};

} // namespace ExaDG

#endif /* EXADG_GRID_MAPPING_FE_CACHE_H_ */
