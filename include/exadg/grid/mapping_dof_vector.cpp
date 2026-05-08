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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <cstdlib> // for std::rand, std::srand
#include <ctime>   // for std::time  // ##+ remove

// deal.II
#include <deal.II/fe/fe_values.h>

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>

namespace ExaDG
{
template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
MappingDoFVector<dim, Number>::get_mapping() const
{
  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

  return mapping_q_cache;
}

template<int dim, typename Number>
std::shared_ptr<dealii::MappingQCache<dim>>
MappingDoFVector<dim, Number>::get_mapping_q_cache() const
{
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
  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

  // use the deformed state described by the dealii::MappingQCache object
  fill_grid_coordinates_vector(*mapping_q_cache, grid_coordinates, dof_handler);
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

  AssertThrow(get_element_type(dof_handler.get_triangulation()) == ElementType::Hypercube,
              dealii::ExcMessage("Only implemented for hypercube elements."));

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
  dealii::DoFHandler<dim> const & dof_handler)
{
  AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

  AssertThrow(mapping_q_cache.get(),
              dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

  std::cout << "displacement_vector.size() = " << displacement_vector.size() << "\n";
  std::cout << "dof_handler.n_dofs() = " << dof_handler.n_dofs() << "\n";

  VectorType displacement_vector_ghosted;
  if(dof_handler.n_dofs() > 0 and displacement_vector.size() == dof_handler.n_dofs())
  {
    dealii::IndexSet const locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    displacement_vector_ghosted.reinit(dof_handler.locally_owned_dofs(),
                                       locally_relevant_dofs,
                                       dof_handler.get_mpi_communicator());
    displacement_vector_ghosted.copy_locally_owned_data_from(displacement_vector);
    displacement_vector_ghosted.update_ghost_values();
  }

  AssertThrow(get_element_type(dof_handler.get_triangulation()) == ElementType::Hypercube,
              dealii::ExcMessage("Only implemented for hypercube elements."));

  std::shared_ptr<dealii::FEValues<dim>> fe_values;

  // Set up dealii::FEValues with FE_Nothing and the Gauss-Lobatto quadrature to
  // reduce setup cost, as we only use the geometry information (this means
  // we need to call fe_values.reinit(cell) with Triangulation::cell_iterator
  // rather than dealii::DoFHandler::cell_iterator).
  dealii::FE_Nothing<dim> fe_nothing;

  if(mapping != nullptr)
  {
    fe_values = std::make_shared<dealii::FEValues<dim>>(*mapping,
                                                        fe_nothing,
                                                        dealii::QGaussLobatto<dim>(
                                                          mapping_q_cache->get_degree() + 1),
                                                        dealii::update_quadrature_points);
  }

  // take the grid coordinates described by mapping and add deformation described by displacement
  // vector
  mapping_q_cache->initialize(
    dof_handler.get_triangulation(),
    [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
      -> std::vector<dealii::Point<dim>> {
      unsigned int const scalar_dofs_per_cell =
        dealii::Utilities::pow(mapping_q_cache->get_degree() + 1, dim);

      std::vector<dealii::Point<dim>> grid_coordinates(scalar_dofs_per_cell);

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

      // if this function is called with an empty dof-vector, this indicates that the
      // displacements are zero and the points do not have to be moved
      if(dof_handler.n_dofs() > 0 and displacement_vector.size() > 0 and cell_tria->is_active() and
         not(cell_tria->is_artificial()))
      {
        typename dealii::DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                             cell_tria->level(),
                                                             cell_tria->index(),
                                                             &dof_handler);

        dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
        AssertThrow(fe.element_multiplicity(0) == dim,
                    dealii::ExcMessage("Expected finite element with dim components."));

        std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        for(unsigned int i = 0; i < dof_indices.size(); ++i)
        {
          std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

          if(fe.dofs_per_vertex > 0) // dealii::FE_Q
          {
            std::srand(std::time(nullptr));
            double const val = std::rand();
            std::cout << "displacement_vector_ghosted.size() = "
                      << displacement_vector_ghosted.size() << " val = " << val << "\n";
            grid_coordinates[id.second][id.first] += displacement_vector_ghosted(dof_indices[i]);
            std::cout << "passed."
                      << " val = " << val << "\n";
          }
          else // dealii::FE_DGQ
          {
            grid_coordinates[this->lexicographic_to_hierarchic_numbering[id.second]][id.first] +=
              displacement_vector_ghosted(dof_indices[i]);
          }
        }
      }

      return grid_coordinates;
    });
}

template class MappingDoFVector<2, float>;
template class MappingDoFVector<3, float>;

template class MappingDoFVector<2, double>;
template class MappingDoFVector<3, double>;

} // namespace ExaDG
