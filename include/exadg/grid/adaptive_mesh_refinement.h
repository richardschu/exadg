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
 *
 * Source: https://github.com/MeltPoolDG/MeltPoolDG
 * Author: Peter Munch, Magdalena Schreter, TUM, December 2020
 */

#pragma once

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/solution_transfer.h>

namespace ExaDG
{
using namespace dealii;

struct AdaptiveMeshRefinementData
{
  bool         do_amr                       = false;
  bool         do_not_modify_boundary_cells = false;
  double       upper_perc_to_refine         = 0.0;
  double       lower_perc_to_coarsen        = 0.0;
  int          every_n_step                 = 1;
  unsigned int refine_space_max             = 10;
  int          refine_space_min             = 0;
};

inline bool
perform_update_now(const AdaptiveMeshRefinementData & amr_data, const int time_step_number)
{
  return ((time_step_number == 0) or !(time_step_number % amr_data.every_n_step));
}

template<int dim, typename VectorType>
void
refine_grid(const std::function<bool(Triangulation<dim> &)> & mark_cells_for_refinement,
            const std::function<void(
              std::vector<std::pair<const DoFHandler<dim> *,
                                    std::function<void(std::vector<VectorType *> &)>>> & data)> &
                                               attach_vectors,
            const std::function<void()> &      post,
            const std::function<void()> &      setup_dof_system,
            const AdaptiveMeshRefinementData & amr_data,
            Triangulation<dim> &               tria,
            const int                          time_step_number)
{
  AssertThrow(amr_data.do_amr,
              dealii::ExcMessage("No adaptive grid refinement requested. Check control flow."));

  if(!perform_update_now(amr_data, time_step_number))
  {
    // Current time step does not trigger adaptive refinement.
    return;
  }

  if(mark_cells_for_refinement(tria) == false)
  {
    // No flags were selected for refinement.
    return;
  }

  std::vector<std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
    data;

  attach_vectors(data);

  const unsigned int n = data.size();

  Assert(n > 0,
         dealii::ExcMessage("Vector data filled via attach_vectors() returned empty container."));

  /*
   *  Limit the maximum and minimum refinement levels of cells of the grid.
   */
  if(tria.n_levels() > amr_data.refine_space_max)
  {
    for(auto & cell : tria.active_cell_iterators_on_level(amr_data.refine_space_max))
    {
      cell->clear_refine_flag();
    }
  }

  for(auto & cell : tria.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      if(cell->level() <= amr_data.refine_space_min)
      {
        cell->clear_coarsen_flag();
      }

      /*
       *  do not coarsen/refine cells along boundary if requested
       */
      if(amr_data.do_not_modify_boundary_cells)
      {
        for(auto & face : cell->face_iterators())
        {
          if(face->at_boundary())
          {
            if(cell->refine_flag_set())
            {
              cell->clear_refine_flag();
            }
            else
            {
              cell->clear_coarsen_flag();
            }
          }
        }
      }
    }
  }

  if(dynamic_cast<parallel::distributed::Triangulation<dim> *>(&tria))
  {
    /*
     *  Initialize the triangulation change from the old grid to the new grid
     */
    tria.prepare_coarsening_and_refinement();
    /*
     *  Initialize the solution transfer from the old grid to the new grid
     */
    std::vector<std::shared_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>>
      solution_transfer(n);

    std::vector<std::vector<VectorType *>>       new_grid_solutions(n);
    std::vector<std::vector<const VectorType *>> old_grid_solutions(n);

    for(unsigned int j = 0; j < n; ++j)
    {
      data[j].second(new_grid_solutions[j]);

      for(const auto & i : new_grid_solutions[j])
      {
        i->update_ghost_values();
        old_grid_solutions[j].push_back(i);
      }
      solution_transfer[j] =
        std::make_shared<parallel::distributed::SolutionTransfer<dim, VectorType>>(*data[j].first);
      solution_transfer[j]->prepare_for_coarsening_and_refinement(old_grid_solutions[j]);
    }
    /*
     *  Execute the grid refinement
     */
    tria.execute_coarsening_and_refinement();
    /*
     *  update dof-related scratch data to match the current triangulation
     */
    setup_dof_system();
    /*
     *  interpolate the given solution to the new discretization
     *
     */
    for(unsigned int j = 0; j < n; ++j)
    {
      solution_transfer[j]->interpolate(new_grid_solutions[j]);
    }
    post();
  }
  else
  {
    /*
     *  Initialize the triangulation change from the old grid to the new grid
     */
    tria.prepare_coarsening_and_refinement();
    /*
     *  Initialize the solution transfer from the old grid to the new grid
     */
    std::vector<std::shared_ptr<SolutionTransfer<dim, VectorType>>> solution_transfer(n);

    std::vector<std::vector<VectorType *>>       new_grid_solutions(n);
    std::vector<std::vector<const VectorType *>> old_grid_solutions(n);

    std::vector<std::vector<VectorType>> new_grid_solutions_full(n);
    std::vector<std::vector<VectorType>> old_grid_solutions_full(n);

    for(unsigned int j = 0; j < n; ++j)
    {
      data[j].second(new_grid_solutions[j]);

      for(const auto & i : new_grid_solutions[j])
      {
        i->update_ghost_values();
        old_grid_solutions[j].push_back(i);
      }
      solution_transfer[j] = std::make_shared<SolutionTransfer<dim, VectorType>>(*data[j].first);

      old_grid_solutions_full[j].resize(new_grid_solutions[j].size());
      for(unsigned int i = 0; i < old_grid_solutions_full[j].size(); ++i)
      {
        const auto & distributed = *old_grid_solutions[j][i];
        IndexSet     ghost(distributed.size());
        ghost.add_range(0, distributed.size());
        old_grid_solutions_full[j][i].reinit(distributed.locally_owned_elements(),
                                             ghost,
                                             distributed.get_mpi_communicator());

        old_grid_solutions_full[j][i].copy_locally_owned_data_from(*old_grid_solutions[j][i]);
        old_grid_solutions_full[j][i].update_ghost_values();
      }
      solution_transfer[j]->prepare_for_coarsening_and_refinement(old_grid_solutions_full[j]);
    }

    /*
     *  Execute the grid refinement
     */
    tria.execute_coarsening_and_refinement();
    /*
     *  update dof-related scratch data to match the current triangulation
     */
    setup_dof_system();
    /*
     *  interpolate the given solution to the new discretization
     */
    for(unsigned int j = 0; j < n; ++j)
    {
      new_grid_solutions_full[j].resize(new_grid_solutions[j].size());
      for(unsigned int i = 0; i < new_grid_solutions_full[j].size(); ++i)
      {
        const auto & distributed = *new_grid_solutions[j][i];
        IndexSet     ghost(distributed.size());
        ghost.add_range(0, distributed.size());
        new_grid_solutions_full[j][i].reinit(distributed.locally_owned_elements(),
                                             ghost,
                                             distributed.get_mpi_communicator());
      }

      solution_transfer[j]->interpolate(old_grid_solutions_full[j], new_grid_solutions_full[j]);

      for(unsigned int i = 0; i < new_grid_solutions_full[j].size(); ++i)
      {
        new_grid_solutions[j][i]->copy_locally_owned_data_from(new_grid_solutions_full[j][i]);
      }
    }
    post();
  }
}

} // namespace ExaDG
