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

// deal.II
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

// C/C++
#include <numeric>

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfer.h>

namespace ExaDG
{
namespace
{
/*
 * Sets the locally owned entries of `grid_coordinates` to the coordinates of
 * the support points of the finite element of `dof_handler` described by
 * `mapping`. In contrast to the hypercube variant using Gauss-Lobatto points,
 * this works for any finite element with support points, e.g., FE_SimplexP or
 * FE_SimplexDGP. Ghost values are not updated.
 */
template<int dim, typename VectorType>
void
fill_grid_coordinates_at_support_points(dealii::Mapping<dim> const &    mapping,
                                        VectorType &                    grid_coordinates,
                                        dealii::DoFHandler<dim> const & dof_handler)
{
  dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
  AssertThrow(fe.n_base_elements() == 1 and fe.element_multiplicity(0) == dim,
              dealii::ExcMessage("Expected finite element with dim components."));

  // Set up dealii::FEValues with FE_Nothing as we only use the geometry
  // information, see also fill_grid_coordinates_vector().
  dealii::FE_Nothing<dim> fe_nothing(fe.reference_cell());
  dealii::FEValues<dim>   fe_values(mapping,
                                  fe_nothing,
                                  dealii::Quadrature<dim>(
                                    fe.base_element(0).get_unit_support_points()),
                                  dealii::update_quadrature_points);

  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

  for(auto const & cell : dof_handler.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      fe_values.reinit(typename dealii::Triangulation<dim>::cell_iterator(cell));
      cell->get_dof_indices(dof_indices);
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      {
        auto const [d, i_scalar] = fe.system_to_component_index(i);
        if(grid_coordinates.get_partitioner()->in_local_range(dof_indices[i]))
        {
          grid_coordinates(dof_indices[i]) = fe_values.quadrature_point(i_scalar)[d];
        }
      }
    }
  }
}
} // namespace

template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
MappingDoFVector<dim, Number>::get_mapping() const
{
  if(mapping_fe_cache.get())
  {
    return mapping_fe_cache;
  }

  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

  return mapping_q_cache;
}

template<int dim, typename Number>
std::shared_ptr<dealii::MappingQCache<dim>>
MappingDoFVector<dim, Number>::get_mapping_q_cache() const
{
  AssertThrow(not mapping_fe_cache.get(),
              dealii::ExcMessage("dealii::MappingQCache is only used for hypercube elements. "
                                 "Use get_mapping() instead."));

  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

  return mapping_q_cache;
}

template<int dim, typename Number>
void
MappingDoFVector<dim, Number>::fill_grid_coordinates_vector(
  VectorType &                    grid_coordinates,
  dealii::DoFHandler<dim> const & dof_handler) const
{
  // use the deformed state described by the dealii::MappingQCache object (or
  // the MappingFECache object for simplex elements)
  fill_grid_coordinates_vector(*get_mapping(), grid_coordinates, dof_handler);
}

template<int dim, typename Number>
void
MappingDoFVector<dim, Number>::fill_grid_coordinates_vector(
  dealii::Mapping<dim> const &    mapping,
  VectorType &                    grid_coordinates,
  dealii::DoFHandler<dim> const & dof_handler) const
{
  if(grid_coordinates.size() != dof_handler.n_dofs())
  {
    dealii::IndexSet const relevant_dofs_grid =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    grid_coordinates.reinit(dof_handler.locally_owned_dofs(),
                            relevant_dofs_grid,
                            dof_handler.get_mpi_communicator());
  }
  else
  {
    grid_coordinates = 0;
  }

  if(get_element_type(dof_handler.get_triangulation()) == ElementType::Simplex)
  {
    fill_grid_coordinates_at_support_points(mapping, grid_coordinates, dof_handler);
    grid_coordinates.update_ghost_values();
    return;
  }

  AssertThrow(get_element_type(dof_handler.get_triangulation()) == ElementType::Hypercube,
              dealii::ExcMessage("Only implemented for hypercube and simplex elements."));

  dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();

  std::vector<std::array<unsigned int, dim>> component_to_system_index(
    fe.base_element(0).dofs_per_cell);

  if(fe.dofs_per_vertex > 0) // dealii::FE_Q
  {
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      component_to_system_index
        [hierarchic_to_lexicographic_numbering[fe.system_to_component_index(i).second]]
        [fe.system_to_component_index(i).first] = i;
    }
  }
  else // dealii::FE_DGQ
  {
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      component_to_system_index[fe.system_to_component_index(i).second]
                               [fe.system_to_component_index(i).first] = i;
    }
  }

  // Set up dealii::FEValues with FE_Nothing and the Gauss-Lobatto quadrature to
  // reduce setup cost, as we only use the geometry information (this means
  // we need to call fe_values.reinit(cell) with Triangulation::cell_iterator
  // rather than dealii::DoFHandler::cell_iterator).
  dealii::FE_Nothing<dim> fe_nothing;
  dealii::FEValues<dim>   fe_values(mapping,
                                  fe_nothing,
                                  dealii::QGaussLobatto<dim>(fe.degree + 1),
                                  dealii::update_quadrature_points);

  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

  for(auto const & cell : dof_handler.active_cell_iterators())
  {
    if(not cell->is_artificial())
    {
      fe_values.reinit(typename dealii::Triangulation<dim>::cell_iterator(cell));
      cell->get_dof_indices(dof_indices);
      for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
      {
        dealii::Point<dim> const point = fe_values.quadrature_point(i);
        for(unsigned int d = 0; d < dim; ++d)
        {
          if(grid_coordinates.get_partitioner()->in_local_range(
               dof_indices[component_to_system_index[i][d]]))
          {
            grid_coordinates(dof_indices[component_to_system_index[i][d]]) = point[d];
          }
        }
      }
    }
  }

  grid_coordinates.update_ghost_values();
}

template<int dim, typename Number>
void
MappingDoFVector<dim, Number>::initialize_mapping_from_dof_vector(
  dealii::Mapping<dim> const *    mapping,
  VectorType const &              displacement_vector,
  dealii::DoFHandler<dim> const & dof_handler,
  unsigned int const              level)
{
  if(get_element_type(dof_handler.get_triangulation()) == ElementType::Simplex)
  {
    initialize_mapping_fe_cache(mapping, displacement_vector, dof_handler, level);
    return;
  }

  AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

  AssertThrow(get_element_type(dof_handler.get_triangulation()) == ElementType::Hypercube,
              dealii::ExcMessage("Only implemented for hypercube and simplex elements."));

  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

  bool const mapping_valid      = mapping != nullptr;
  bool const vector_initialized = displacement_vector.size() > 0;
  AssertThrow(mapping_valid or vector_initialized,
              dealii::ExcMessage("Provide either mapping or displacement DoF vector."));

  bool const   is_mg  = level != dealii::numbers::invalid_unsigned_int;
  unsigned int n_dofs = is_mg ? dof_handler.n_dofs(level) : dof_handler.n_dofs();
  if(vector_initialized)
  {
    AssertThrow(n_dofs > 0,
                dealii::ExcMessage("DoFHandler has no DoFs distributed; this "
                                   "is required for adding a displacement."));
  }

  // Setup ghosted displacement vector to interpolate.
  VectorType displacement_vector_ghosted;
  if(vector_initialized)
  {
    if(is_mg)
    {
      dealii::IndexSet const locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);

      displacement_vector_ghosted.reinit(dof_handler.locally_owned_mg_dofs(level),
                                         locally_relevant_dofs,
                                         dof_handler.get_mpi_communicator());
    }
    else
    {
      dealii::IndexSet const locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

      displacement_vector_ghosted.reinit(dof_handler.locally_owned_dofs(),
                                         locally_relevant_dofs,
                                         dof_handler.get_mpi_communicator());
    }

    displacement_vector_ghosted.copy_locally_owned_data_from(displacement_vector);
    displacement_vector_ghosted.update_ghost_values();
  }

  // If the degree of the finite element of `dof_handler` differs from the
  // degree of the mapping, e.g., on p-multigrid levels, we evaluate the
  // displacement field in the support points of the mapping, i.e., the
  // Gauss-Lobatto points in hierarchic order.
  bool const interpolate_displacement =
    vector_initialized and
    dof_handler.get_fe().base_element(0).degree != mapping_q_cache->get_degree();
  dealii::FullMatrix<double> shape_values;
  if(interpolate_displacement)
  {
    dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
    dealii::QGaussLobatto<dim> const   quadrature(mapping_q_cache->get_degree() + 1);
    shape_values.reinit(fe.n_dofs_per_cell(), quadrature.size());
    for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
      for(unsigned int q = 0; q < quadrature.size(); ++q)
        shape_values(i, q) =
          fe.shape_value(i, quadrature.point(this->hierarchic_to_lexicographic_numbering[q]));
  }

  // Set up `dealii::FEValues` with `dealii::FE_Nothing` and the Gauss-Lobatto
  // quadrature to reduce setup cost, as we only use the geometry information
  // (this means we need to call `fe_values.reinit(cell)` with
  // `Triangulation::cell_iterator` rather than the more standard call using
  // `dealii::DoFHandler::cell_iterator`).
  dealii::FE_Nothing<dim>                fe_nothing;
  std::shared_ptr<dealii::FEValues<dim>> fe_values;
  if(mapping != nullptr)
  {
    fe_values = std::make_shared<dealii::FEValues<dim>>(*mapping,
                                                        fe_nothing,
                                                        dealii::QGaussLobatto<dim>(
                                                          mapping_q_cache->get_degree() + 1),
                                                        dealii::update_quadrature_points);
  }

  mapping_q_cache->initialize(
    dof_handler.get_triangulation(),
    [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
      -> std::vector<dealii::Point<dim>> {
      unsigned int const cell_tria_level = cell_tria->level();

      // We have to return *some* grid coordinates also for *all* levels. On the
      // other levels, we return a zero vector.
      unsigned int const scalar_dofs_per_cell =
        dealii::Utilities::pow(mapping_q_cache->get_degree() + 1, dim);

      std::vector<dealii::Point<dim>> grid_coordinates(scalar_dofs_per_cell);

      // Skip cells that are not on the target level (DoF Vector also does not
      // match).
      if(not(is_mg) or cell_tria_level == level)
      {
        // If the `mapping` is valid, write the interpolated point coordinates
        // to add displacement vector in a second step.
        if(mapping != nullptr)
        {
          fe_values->reinit(cell_tria);
          // extract displacement and add to original position
          for(unsigned int i = 0; i < scalar_dofs_per_cell; ++i)
          {
            grid_coordinates[i] =
              fe_values->quadrature_point(this->hierarchic_to_lexicographic_numbering[i]);
          }
        }

        typename dealii::DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                             cell_tria_level,
                                                             cell_tria->index(),
                                                             &dof_handler);

        // if this function is called with an empty DoF vector, this indicates
        // that the displacements are zero and the points do not have to be
        // moved
        bool const relevant_cell =
          is_mg ? vector_initialized and
                    cell->level_subdomain_id() != dealii::numbers::artificial_subdomain_id :
                  cell_tria->is_active() and not(cell_tria->is_artificial());

        if(vector_initialized and relevant_cell)
        {
          dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
          AssertThrow(fe.element_multiplicity(0) == dim,
                      dealii::ExcMessage("Expected finite element with dim components."));

          std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
          if(is_mg)
          {
            cell->get_mg_dof_indices(dof_indices);
          }
          else
          {
            cell->get_dof_indices(dof_indices);
          }

          for(unsigned int i = 0; i < dof_indices.size(); ++i)
          {
            std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

            if(interpolate_displacement)
            {
              Number const displacement = displacement_vector_ghosted(dof_indices[i]);
              for(unsigned int q = 0; q < scalar_dofs_per_cell; ++q)
                grid_coordinates[q][id.first] += shape_values(i, q) * displacement;
            }
            else if(fe.dofs_per_vertex > 0) // dealii::FE_Q
            {
              grid_coordinates[id.second][id.first] += displacement_vector_ghosted(dof_indices[i]);
            }
            else // dealii::FE_DGQ
            {
              grid_coordinates[this->lexicographic_to_hierarchic_numbering[id.second]][id.first] +=
                displacement_vector_ghosted(dof_indices[i]);
            }
          }
        }
      }

      return grid_coordinates;
    });
}

template<int dim, typename Number>
void
MappingDoFVector<dim, Number>::initialize_mapping_fe_cache(
  dealii::Mapping<dim> const *    mapping,
  VectorType const &              displacement_vector,
  dealii::DoFHandler<dim> const & dof_handler,
  unsigned int const              level)
{
  AssertThrow(level == dealii::numbers::invalid_unsigned_int,
              dealii::ExcMessage("Multigrid levels are not supported for simplex elements."));

  bool const mapping_valid      = mapping != nullptr;
  bool const vector_initialized = displacement_vector.size() > 0;
  AssertThrow(mapping_valid or vector_initialized,
              dealii::ExcMessage("Provide either mapping or displacement DoF vector."));
  if(vector_initialized)
  {
    AssertThrow(dof_handler.n_dofs() > 0,
                dealii::ExcMessage("DoFHandler has no DoFs distributed; this "
                                   "is required for adding a displacement."));
    AssertThrow(dof_handler.get_fe().n_components() == dim,
                dealii::ExcMessage("Expected finite element with dim components."));
  }

  // Create the mapping only once, such that pointers obtained via get_mapping()
  // stay valid.
  if(not mapping_fe_cache.get())
  {
    mapping_fe_cache = std::make_shared<MappingFECache<dim>>(
      dealii::FE_SimplexP<dim>(mapping_q_cache->get_degree()));
  }

  dealii::Quadrature<dim> const support_points(mapping_fe_cache->get_unit_support_points());

  // Setup ghosted displacement vector to interpolate.
  VectorType displacement_vector_ghosted;
  if(vector_initialized)
  {
    displacement_vector_ghosted.reinit(dof_handler.locally_owned_dofs(),
                                       dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                       dof_handler.get_mpi_communicator());
    displacement_vector_ghosted.copy_locally_owned_data_from(displacement_vector);
    displacement_vector_ghosted.update_ghost_values();
  }

  // Set up `dealii::FEValues` with `dealii::FE_Nothing` to evaluate the
  // undeformed grid in the support points, see
  // initialize_mapping_from_dof_vector().
  dealii::FE_Nothing<dim>                fe_nothing(dealii::ReferenceCells::get_simplex<dim>());
  std::shared_ptr<dealii::FEValues<dim>> fe_values_mapping;
  if(mapping_valid)
  {
    fe_values_mapping = std::make_shared<dealii::FEValues<dim>>(*mapping,
                                                                fe_nothing,
                                                                support_points,
                                                                dealii::update_quadrature_points);
  }

  // The displacement is evaluated in the support points of the mapping, such
  // that `dof_handler` may use any finite element with dim components.
  std::shared_ptr<dealii::FEValues<dim>> fe_values_displacement;
  if(vector_initialized)
  {
    fe_values_displacement = std::make_shared<dealii::FEValues<dim>>(dof_handler.get_fe(),
                                                                     support_points,
                                                                     dealii::update_values);
  }
  std::vector<dealii::Vector<Number>> displacement_values(support_points.size(),
                                                          dealii::Vector<Number>(dim));

  mapping_fe_cache->initialize(
    dof_handler.get_triangulation(),
    [&](typename dealii::Triangulation<dim>::cell_iterator const & cell_tria)
      -> std::vector<dealii::Point<dim>> {
      // Zero coordinates if the mapping pointer is invalid, see
      // initialize_mapping_from_dof_vector().
      std::vector<dealii::Point<dim>> grid_coordinates(support_points.size());

      if(mapping_valid)
      {
        fe_values_mapping->reinit(cell_tria);
        for(unsigned int i = 0; i < support_points.size(); ++i)
        {
          grid_coordinates[i] = fe_values_mapping->quadrature_point(i);
        }
      }

      if(vector_initialized and cell_tria->is_active() and not(cell_tria->is_artificial()))
      {
        fe_values_displacement->reinit(cell_tria->as_dof_handler_iterator(dof_handler));
        fe_values_displacement->get_function_values(displacement_vector_ghosted,
                                                    displacement_values);
        for(unsigned int i = 0; i < support_points.size(); ++i)
        {
          for(unsigned int d = 0; d < dim; ++d)
          {
            grid_coordinates[i][d] += displacement_values[i][d];
          }
        }
      }

      return grid_coordinates;
    });
}

template<int dim, typename Number>
void
MappingDoFVector<dim, Number>::initialize_mapping_from_function(
  dealii::Mapping<dim> const &       mapping,
  dealii::Triangulation<dim> const & triangulation,
  dealii::Function<dim> const &      displacement_function)
{
  AssertThrow(displacement_function.n_components == dim,
              dealii::ExcMessage("Expected displacement function with dim components."));

  ElementType const element_type = get_element_type(triangulation);
  AssertThrow(element_type == ElementType::Hypercube or element_type == ElementType::Simplex,
              dealii::ExcMessage("Only implemented for hypercube and simplex elements."));

  bool const is_simplex = element_type == ElementType::Simplex;

  // Create the mapping only once, see initialize_mapping_fe_cache().
  if(is_simplex and not mapping_fe_cache.get())
  {
    mapping_fe_cache = std::make_shared<MappingFECache<dim>>(
      dealii::FE_SimplexP<dim>(mapping_q_cache->get_degree()));
  }

  // The mapping support points are the Gauss-Lobatto points in hierarchic
  // numbering for dealii::MappingQCache and the unit support points of
  // FE_SimplexP for MappingFECache.
  dealii::Quadrature<dim>   support_points;
  std::vector<unsigned int> numbering;
  if(is_simplex)
  {
    support_points = dealii::Quadrature<dim>(mapping_fe_cache->get_unit_support_points());
    numbering.resize(support_points.size());
    std::iota(numbering.begin(), numbering.end(), 0);
  }
  else
  {
    support_points = dealii::QGaussLobatto<dim>(mapping_q_cache->get_degree() + 1);
    numbering      = hierarchic_to_lexicographic_numbering;
  }

  // dummy FE for compatibility with interface of dealii::FEValues
  dealii::FE_Nothing<dim> fe_nothing(is_simplex ? dealii::ReferenceCells::get_simplex<dim>() :
                                                  dealii::ReferenceCells::get_hypercube<dim>());
  dealii::FEValues<dim>   fe_values(mapping,
                                  fe_nothing,
                                  support_points,
                                  dealii::update_quadrature_points);

  auto const compute_points_on_cell =
    [&](typename dealii::Triangulation<dim>::cell_iterator const & cell)
    -> std::vector<dealii::Point<dim>> {
    fe_values.reinit(cell);

    // compute displacement and add to original position
    std::vector<dealii::Point<dim>> points_moved(fe_values.n_quadrature_points);
    for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
    {
      dealii::Point<dim> const point = fe_values.quadrature_point(numbering[i]);
      dealii::Point<dim>       displacement;
      for(unsigned int d = 0; d < dim; ++d)
        displacement[d] = displacement_function.value(point, d);

      points_moved[i] = point + displacement;
    }

    return points_moved;
  };

  if(is_simplex)
  {
    mapping_fe_cache->initialize(triangulation, compute_points_on_cell);
  }
  else
  {
    AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

    mapping_q_cache->initialize(triangulation, compute_points_on_cell);
  }
}

template class MappingDoFVector<2, float>;
template class MappingDoFVector<3, float>;

template class MappingDoFVector<2, double>;
template class MappingDoFVector<3, double>;

namespace MappingTools
{
template<int dim, typename Number>
void
initialize_coarse_mappings_from_mapping_dof_vector(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> & coarse_mappings,
  unsigned int const                                            degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &  fine_mapping,
  dealii::Triangulation<dim> const &                            triangulation)
{
  // This variant relies on dealii::MappingQCache and multigrid levels of a
  // single triangulation, which are not available for simplex elements.
  AssertThrow(get_element_type(triangulation) == ElementType::Hypercube,
              dealii::ExcMessage("Coarse mappings on the levels of a single triangulation are "
                                 "only implemented for hypercube elements. For simplex elements, "
                                 "use separate coarse triangulations."));

  AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

  std::shared_ptr<dealii::MappingQCache<dim>> mapping_q_cache = fine_mapping->get_mapping_q_cache();
  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Shared pointer mapping_q_cache is invalid."));

  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(degree_coarse_mappings), dim);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector_all_levels =
    std::make_shared<MappingDoFVector<dim, Number>>(degree_coarse_mappings);

  // fill a dof vector with grid coordinates of the fine level using
  // `degree_coarse_mappings`
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  VectorType                                                 grid_coordinates_fine_level;

  {
    mapping_dof_vector_all_levels->fill_grid_coordinates_vector(*fine_mapping->get_mapping(),
                                                                grid_coordinates_fine_level,
                                                                dof_handler);
  }

  // project the solution onto all coarse levels of the triangulation using
  // `degree_coarse_mappings`
  dealii::MGLevelObject<VectorType> grid_coordinates_all_levels,
    grid_coordinates_all_levels_ghosted;
  unsigned int const n_levels = triangulation.n_global_levels();
  grid_coordinates_all_levels.resize(0, n_levels - 1);
  grid_coordinates_all_levels_ghosted.resize(0, n_levels - 1);

  dealii::MGTransferMatrixFree<dim, Number> transfer;
  transfer.build(dof_handler);
  transfer.interpolate_to_mg(dof_handler, grid_coordinates_all_levels, grid_coordinates_fine_level);

  // ghosting
  for(unsigned int level = 0; level < n_levels; level++)
  {
    dealii::IndexSet const relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);

    grid_coordinates_all_levels_ghosted[level].reinit(
      dof_handler.locally_owned_mg_dofs(level),
      relevant_dofs,
      grid_coordinates_all_levels[level].get_mpi_communicator());

    grid_coordinates_all_levels_ghosted[level].copy_locally_owned_data_from(
      grid_coordinates_all_levels[level]);

    grid_coordinates_all_levels_ghosted[level].update_ghost_values();
  }

  // Call the initialize() function of dealii::MappingQCache, which initializes
  // the mapping for all levels according to
  // `grid_coordinates_all_levels_ghosted`.
  mapping_dof_vector_all_levels->get_mapping_q_cache()->initialize(
    dof_handler.get_triangulation(),
    [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
      -> std::vector<dealii::Point<dim>> {
      unsigned int const level = cell_tria->level();

      typename dealii::DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                           level,
                                                           cell_tria->index(),
                                                           &dof_handler);

      AssertThrow(fe.element_multiplicity(0) == dim,
                  dealii::ExcMessage("Expected finite element with dim components."));

      unsigned int const scalar_dofs_per_cell = dealii::Utilities::pow(fe.degree + 1, dim);

      std::vector<dealii::Point<dim>> grid_coordinates(scalar_dofs_per_cell);

      if(cell->level_subdomain_id() != dealii::numbers::artificial_subdomain_id)
      {
        std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
        cell->get_mg_dof_indices(dof_indices);

        for(unsigned int i = 0; i < dof_indices.size(); ++i)
        {
          std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

          if(fe.dofs_per_vertex > 0) // dealii::FE_Q
          {
            grid_coordinates[id.second][id.first] =
              grid_coordinates_all_levels_ghosted[level](dof_indices[i]);
          }
          else // dealii::FE_DGQ
          {
            grid_coordinates[mapping_dof_vector_all_levels
                               ->lexicographic_to_hierarchic_numbering[id.second]][id.first] =
              grid_coordinates_all_levels_ghosted[level](dof_indices[i]);
          }
        }
      }

      return grid_coordinates;
    });

  AssertThrow(
    coarse_mappings.size() == n_levels - 1,
    dealii::ExcMessage(
      "coarse_mappings does not have correct size relative to the number of levels of the triangulation."));

  // Finally, let all coarse grid mappings point to the same MappingDoFVector
  // object. Using the same Mapping object for all multigrid h-levels is some
  // form of legacy code. The class `dealii::MatrixFree` can internally extract
  // the coarse-level mapping information (provided through the fine-level
  // `dealii::Mapping` object).
  for(unsigned int h_level = 0; h_level < coarse_mappings.size(); ++h_level)
  {
    coarse_mappings[h_level] = mapping_dof_vector_all_levels;
  }
}


template<int dim, typename Number>
void
initialize_coarse_mappings_from_mapping_dof_vector(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> &          coarse_mappings,
  unsigned int const                                                     degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)
{
  AssertThrow(fine_mapping->get_mapping().get(),
              dealii::ExcMessage("Fine-level mapping is invalid."));

  // setup dof-handlers and constraints for all levels using
  // `degree_coarse_mappings`
  std::shared_ptr<dealii::FiniteElement<dim>> fe_ptr =
    create_finite_element<dim>(get_element_type(*fine_triangulation),
                               false /* is_dg */,
                               dim /* n_components */,
                               degree_coarse_mappings);
  dealii::FiniteElement<dim> const &             fe         = *fe_ptr;
  unsigned int const                             n_h_levels = coarse_triangulations.size() + 1;
  std::vector<dealii::DoFHandler<dim>>           dof_handlers_all_levels(n_h_levels);
  std::vector<dealii::AffineConstraints<Number>> constraints_all_levels(n_h_levels);
  for(unsigned int h_level = 0; h_level < n_h_levels; ++h_level)
  {
    if(h_level == n_h_levels - 1)
      dof_handlers_all_levels[h_level].reinit(*fine_triangulation);
    else
      dof_handlers_all_levels[h_level].reinit(*coarse_triangulations[h_level]);
    dof_handlers_all_levels[h_level].distribute_dofs(fe);
    // constraints are irrelevant for interpolation
    constraints_all_levels[h_level].close();
  }

  // fill a dof vector with grid coordinates of the fine level using
  // `degree_coarse_mappings`
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  VectorType                                                 grid_coordinates_fine_level;

  {
    std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector_fine_level =
      std::make_shared<MappingDoFVector<dim, Number>>(degree_coarse_mappings);

    auto const & dof_handler_fine_level = dof_handlers_all_levels[n_h_levels - 1];
    mapping_dof_vector_fine_level->fill_grid_coordinates_vector(*fine_mapping->get_mapping(),
                                                                grid_coordinates_fine_level,
                                                                dof_handler_fine_level);
  }

  // create transfer objects
  dealii::MGLevelObject<dealii::MGTwoLevelTransfer<dim, VectorType>> transfers(0, n_h_levels - 1);
  for(unsigned int h_level = 1; h_level < n_h_levels; ++h_level)
  {
    transfers[h_level].reinit_geometric_transfer(dof_handlers_all_levels[h_level],
                                                 dof_handlers_all_levels[h_level - 1],
                                                 constraints_all_levels[h_level],
                                                 constraints_all_levels[h_level - 1]);
  }

  // a function that initializes the dof-vector for a given level and
  // `dof_handler`
  const std::function<void(unsigned int const, VectorType &)> initialize_dof_vector =
    [&](unsigned int const h_level, VectorType & vector) {
      dealii::IndexSet const locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handlers_all_levels[h_level]);
      vector.reinit(dof_handlers_all_levels[h_level].locally_owned_dofs(),
                    locally_relevant_dofs,
                    dof_handlers_all_levels[h_level].get_mpi_communicator());
    };

  dealii::MGTransferMatrixFree<dim, Number> mg_transfer_global_coarsening(transfers,
                                                                          initialize_dof_vector);

  // Transfer grid coordinates to coarser h-levels.
  // The dealii::DoFHandler object will not be used for global coarsening.
  dealii::DoFHandler<dim>           dof_handler_dummy;
  dealii::MGLevelObject<VectorType> grid_coordinates_all_levels(0, n_h_levels - 1);
  mg_transfer_global_coarsening.interpolate_to_mg(dof_handler_dummy,
                                                  grid_coordinates_all_levels,
                                                  grid_coordinates_fine_level);

  // initialize mapping for all coarse h-levels using the dof-vectors with grid
  // coordinates
  AssertThrow(coarse_mappings.size() == n_h_levels - 1,
              dealii::ExcMessage(
                "coarse_mappings does not have correct size relative to coarse_triangulations."));

  for(unsigned int h_level = 0; h_level < coarse_mappings.size(); ++h_level)
  {
    coarse_mappings[h_level] =
      std::make_shared<MappingDoFVector<dim, Number>>(degree_coarse_mappings);

    // grid_coordinates_all_levels describes absolute coordinates -> use an
    // uninitialized mapping in order to interpret the grid coordinates vector
    // as absolute coordinates and not as displacements.
    coarse_mappings[h_level]->initialize_mapping_from_dof_vector(
      nullptr /* mapping */,
      grid_coordinates_all_levels[h_level],
      dof_handlers_all_levels[h_level]);
  }
}


template<int dim, typename Number>
void
initialize_coarse_mappings(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> &          coarse_mappings,
  unsigned int const                                                     degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)
{
  if(fine_mapping.get())
  {
    if(coarse_triangulations.size() > 0)
    {
      MappingTools::initialize_coarse_mappings_from_mapping_dof_vector<dim, Number>(
        coarse_mappings,
        degree_coarse_mappings,
        fine_mapping,
        fine_triangulation,
        coarse_triangulations);
    }
    else
    {
      MappingTools::initialize_coarse_mappings_from_mapping_dof_vector<dim, Number>(
        coarse_mappings, degree_coarse_mappings, fine_mapping, *fine_triangulation);
    }
  }
}


template void
initialize_coarse_mappings_from_mapping_dof_vector<2, float>(
  std::vector<std::shared_ptr<MappingDoFVector<2, float>>> & coarse_mappings,
  unsigned int const                                         degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<2, float> const> const &  fine_mapping,
  dealii::Triangulation<2> const &                           triangulation);

template void
initialize_coarse_mappings_from_mapping_dof_vector<2, double>(
  std::vector<std::shared_ptr<MappingDoFVector<2, double>>> & coarse_mappings,
  unsigned int const                                          degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<2, double> const> const &  fine_mapping,
  dealii::Triangulation<2> const &                            triangulation);

template void
initialize_coarse_mappings_from_mapping_dof_vector<3, float>(
  std::vector<std::shared_ptr<MappingDoFVector<3, float>>> & coarse_mappings,
  unsigned int const                                         degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<3, float> const> const &  fine_mapping,
  dealii::Triangulation<3> const &                           triangulation);

template void
initialize_coarse_mappings_from_mapping_dof_vector<3, double>(
  std::vector<std::shared_ptr<MappingDoFVector<3, double>>> & coarse_mappings,
  unsigned int const                                          degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<3, double> const> const &  fine_mapping,
  dealii::Triangulation<3> const &                            triangulation);

template void
initialize_coarse_mappings_from_mapping_dof_vector<2, float>(
  std::vector<std::shared_ptr<MappingDoFVector<2, float>>> &           coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<2, float> const> const &            fine_mapping,
  std::shared_ptr<dealii::Triangulation<2> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> const & coarse_triangulations);

template void
initialize_coarse_mappings_from_mapping_dof_vector<2, double>(
  std::vector<std::shared_ptr<MappingDoFVector<2, double>>> &          coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<2, double> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<2> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> const & coarse_triangulations);

template void
initialize_coarse_mappings_from_mapping_dof_vector<3, float>(
  std::vector<std::shared_ptr<MappingDoFVector<3, float>>> &           coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<3, float> const> const &            fine_mapping,
  std::shared_ptr<dealii::Triangulation<3> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> const & coarse_triangulations);

template void
initialize_coarse_mappings_from_mapping_dof_vector<3, double>(
  std::vector<std::shared_ptr<MappingDoFVector<3, double>>> &          coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<3, double> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<3> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> const & coarse_triangulations);

template void
initialize_coarse_mappings<2, float>(
  std::vector<std::shared_ptr<MappingDoFVector<2, float>>> &           coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<2, float> const> const &            fine_mapping,
  std::shared_ptr<dealii::Triangulation<2> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> const & coarse_triangulations);

template void
initialize_coarse_mappings<2, double>(
  std::vector<std::shared_ptr<MappingDoFVector<2, double>>> &          coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<2, double> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<2> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<2> const>> const & coarse_triangulations);

template void
initialize_coarse_mappings<3, float>(
  std::vector<std::shared_ptr<MappingDoFVector<3, float>>> &           coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<3, float> const> const &            fine_mapping,
  std::shared_ptr<dealii::Triangulation<3> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> const & coarse_triangulations);

template void
initialize_coarse_mappings<3, double>(
  std::vector<std::shared_ptr<MappingDoFVector<3, double>>> &          coarse_mappings,
  unsigned int const                                                   degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<3, double> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<3> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<3> const>> const & coarse_triangulations);
} // namespace MappingTools

} // namespace ExaDG
