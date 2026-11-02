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
#include <deal.II/fe/fe_values.h>

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>

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
  if(mapping_fe_field.get())
  {
    return mapping_fe_field;
  }

  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

  return mapping_q_cache;
}

template<int dim, typename Number>
std::shared_ptr<dealii::MappingQCache<dim>>
MappingDoFVector<dim, Number>::get_mapping_q_cache() const
{
  AssertThrow(not mapping_fe_field.get(),
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
  // dealii::MappingFEField object for simplex elements)
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
    initialize_mapping_fe_field(mapping, displacement_vector, dof_handler, level);
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

            if(fe.dofs_per_vertex > 0) // dealii::FE_Q
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
MappingDoFVector<dim, Number>::initialize_mapping_fe_field(
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
  }

  // The DoFHandler and the dealii::MappingFEField object are created only once,
  // see the documentation of the member variables.
  dealii::Triangulation<dim> const & triangulation = dof_handler.get_triangulation();
  if(not dof_handler_fe_field.get())
  {
    dof_handler_fe_field = std::make_shared<dealii::DoFHandler<dim>>(triangulation);
  }
  AssertThrow(&dof_handler_fe_field->get_triangulation() == &triangulation,
              dealii::ExcMessage("MappingDoFVector was initialized with another triangulation."));

  // Distribute the DoFs in every call to account for changes of the
  // triangulation, e.g., due to adaptive mesh refinement.
  dealii::FESystem<dim> const fe_fe_field(dealii::FE_SimplexP<dim>(mapping_q_cache->get_degree()),
                                          dim);
  dof_handler_fe_field->distribute_dofs(fe_fe_field);

  grid_coordinates_fe_field.reinit(dof_handler_fe_field->locally_owned_dofs(),
                                   dealii::DoFTools::extract_locally_relevant_dofs(
                                     *dof_handler_fe_field),
                                   dof_handler_fe_field->get_mpi_communicator());

  // Grid coordinates of the reference configuration, zero if no mapping is
  // provided.
  if(mapping_valid)
  {
    fill_grid_coordinates_at_support_points(*mapping,
                                            grid_coordinates_fe_field,
                                            *dof_handler_fe_field);
  }

  // Add the displacement evaluated in the support points of `fe_fe_field`. For
  // a discontinuous displacement field, the value of the last cell visited is
  // taken in support points shared by several cells.
  if(vector_initialized)
  {
    dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
    AssertThrow(fe.n_components() == dim,
                dealii::ExcMessage("Expected finite element with dim components."));

    VectorType displacement_vector_ghosted(dof_handler.locally_owned_dofs(),
                                           dealii::DoFTools::extract_locally_relevant_dofs(
                                             dof_handler),
                                           dof_handler.get_mpi_communicator());
    displacement_vector_ghosted.copy_locally_owned_data_from(displacement_vector);
    displacement_vector_ghosted.update_ghost_values();

    dealii::Quadrature<dim> const support_points(
      fe_fe_field.base_element(0).get_unit_support_points());
    dealii::FEValues<dim> fe_values(fe, support_points, dealii::update_values);

    std::vector<dealii::Vector<Number>>          displacement_values(support_points.size(),
                                                            dealii::Vector<Number>(dim));
    std::vector<dealii::types::global_dof_index> dof_indices(fe_fe_field.dofs_per_cell);

    VectorTypeFEField displacement_fe_field(dof_handler_fe_field->locally_owned_dofs(),
                                            dof_handler_fe_field->get_mpi_communicator());

    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(displacement_vector_ghosted, displacement_values);

        cell->as_dof_handler_iterator(*dof_handler_fe_field)->get_dof_indices(dof_indices);
        for(unsigned int i = 0; i < fe_fe_field.dofs_per_cell; ++i)
        {
          auto const [d, i_scalar] = fe_fe_field.system_to_component_index(i);
          if(displacement_fe_field.get_partitioner()->in_local_range(dof_indices[i]))
          {
            displacement_fe_field(dof_indices[i]) = displacement_values[i_scalar][d];
          }
        }
      }
    }

    grid_coordinates_fe_field += displacement_fe_field;
  }

  grid_coordinates_fe_field.update_ghost_values();

  if(not mapping_fe_field.get())
  {
    mapping_fe_field = std::make_shared<dealii::MappingFEField<dim, dim, VectorTypeFEField>>(
      *dof_handler_fe_field, grid_coordinates_fe_field);
  }
}

template class MappingDoFVector<2, float>;
template class MappingDoFVector<3, float>;

template class MappingDoFVector<2, double>;
template class MappingDoFVector<3, double>;

} // namespace ExaDG
