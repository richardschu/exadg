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

#ifndef INCLUDE_EXADG_GRID_MARKED_VERTICES_H_
#define INCLUDE_EXADG_GRID_MARKED_VERTICES_H_

// deal.II
#include <deal.II/grid/tria.h>

template<typename Key, typename Data>
std::set<Key>
extract_set_of_keys_from_map(std::map<Key, Data> const & map)
{
  std::set<Key> set;
  for(auto iter : map)
  {
    set.insert(iter.first);
  }

  return set;
}

template<int dim>
std::vector<bool>
get_marked_vertices_via_boundary_ids(dealii::Triangulation<dim> const &           triangulation,
                                     std::set<dealii::types::boundary_id> const & boundary_ids,
									 bool const                                   mark_entire_cell)
{
  // mark vertices at faces on any of the boundary_ids in order to make search of active cells around point more efficient
  std::vector<bool> marked_vertices(triangulation.n_vertices(), false);

  for(auto const & cell : triangulation.active_cell_iterators())
  {
    if(not(cell->is_artificial()) and cell->at_boundary())
    {
      bool any_face_on_boundary_id = false;

      for(auto const & f : cell->face_indices())
      {
        if(cell->face(f)->at_boundary())
        {
          if(boundary_ids.find(cell->face(f)->boundary_id()) != boundary_ids.end())
          {
        	// mark the cell and break face loop
        	if(mark_entire_cell)
        	{
        	  any_face_on_boundary_id = true;
        	  break;
        	}
        	else
        	{
				for(auto const & v : cell->face(f)->vertex_indices())
				{
				  marked_vertices[cell->face(f)->vertex_index(v)] = true;
				}
        	}
          }
        }
      }

      if(mark_entire_cell and any_face_on_boundary_id)
      {
  		for(auto const & v : cell->vertex_indices())
		{
		  marked_vertices[cell->vertex_index(v)] = true;
		}
      }
    }
  }

  // To improve robustness, make sure that not all entries of marked_vertices are false.
  // If no useful information about marked vertices can be provided, an empty vector should
  // be used.
  if(std::all_of(marked_vertices.begin(), marked_vertices.end(), [](bool vertex_is_marked) {
       return vertex_is_marked == false;
     }))
  {
    marked_vertices.clear();
  }

  return marked_vertices;
}



#endif /* INCLUDE_EXADG_GRID_MARKED_VERTICES_H_ */
