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

#ifndef INCLUDE_EXADG_GRID_GRID_MOTION_POISSON_H_
#define INCLUDE_EXADG_GRID_GRID_MOTION_POISSON_H_

// deal.II
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/grid/grid_motion_base.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
/**
 * Class for moving grid problems based on a Poisson-type grid motion technique.
 */
template<int dim, typename Number>
class GridMotionPoisson : public GridMotionBase<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  GridMotionPoisson(std::shared_ptr<dealii::Mapping<dim> const>          mapping_undeformed,
                    std::shared_ptr<Poisson::Operator<dim, dim, Number>> poisson_operator)
    : GridMotionBase<dim, Number>(mapping_undeformed,
                                  // extract mapping_degree_moving from Poisson operator
                                  poisson_operator->get_dof_handler().get_fe().degree,
                                  poisson_operator->get_dof_handler().get_triangulation()),
      poisson(poisson_operator),
      pcout(std::cout,
            dealii::Utilities::MPI::this_mpi_process(
              poisson_operator->get_dof_handler().get_communicator()) == 0),
      iterations({0, 0})
  {
    poisson->initialize_dof_vector(displacement);
  }

  /**
   * Updates the mapping, i.e., moves the mesh by solving a Poisson-type problem.
   */
  void
  update(double const     time,
         bool const       print_solver_info,
         types::time_step time_step_number) override
  {
    // preconditioner update has no effect
    (void)time_step_number;

    dealii::Timer timer;
    timer.restart();

    VectorType rhs;
    poisson->initialize_dof_vector(rhs);

    // compute rhs and solve mesh deformation problem
    poisson->rhs(rhs, time);

    auto const n_iter = poisson->solve(displacement, rhs, time);
    iterations.first += 1;
    iterations.second += n_iter;

    if(print_solver_info)
    {
      this->pcout << std::endl << "Solve moving mesh problem (Poisson):";
      print_solver_info_linear(pcout, n_iter, timer.wall_time());
    }

    this->moving_mapping->initialize_mapping_q_cache(this->mapping_undeformed,
                                                     displacement,
                                                     poisson->get_dof_handler());
  }

  /**
   * Prints information on iteration counts.
   */
  void
  print_iterations() const override
  {
    std::vector<std::string> names;
    std::vector<double>      iterations_avg;

    names = {"Linear iterations"};
    iterations_avg.resize(1);
    iterations_avg[0] = (double)iterations.second / std::max(1., (double)iterations.first);

    print_list_of_iterations(pcout, names, iterations_avg);
  }

private:
  std::shared_ptr<Poisson::Operator<dim, dim, Number>> poisson;

  // store solution of previous time step / iteration so that a good initial
  // guess is available in the next step, easing convergence or reducing computational
  // costs by allowing larger tolerances
  VectorType displacement;

  dealii::ConditionalOStream pcout;

  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_GRID_MOTION_POISSON_H_ */
