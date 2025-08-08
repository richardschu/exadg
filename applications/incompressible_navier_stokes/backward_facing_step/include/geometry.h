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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_GEOMETRY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_GEOMETRY_H_

namespace ExaDG
{
namespace IncNS
{
namespace Geometry
{
double const PI = dealii::numbers::PI;

// Height H
double const H = 0.0052;

// backward facing step geometry
double const LENGTH_BFS_DOWN   = 20.0 * H;
double const LENGTH_BFS_UP     = 10.0 * H;
double const HEIGHT_BFS_STEP   = 0.9423 * H;
double const HEIGHT_BFS_INFLOW = H;
double const WIDTH_BFS         = 2 * H;

double const X1_COORDINATE_INFLOW  = -LENGTH_BFS_UP;
double const X1_COORDINATE_OUTFLOW = LENGTH_BFS_DOWN;

// mesh stretching parameters
bool use_grid_stretching_in_y_direction = true;

double const GAMMA_LOWER = 60.0;
double const GAMMA_UPPER = 40.0;

double
get_inlet_height()
{
  return HEIGHT_BFS_INFLOW;
}

double
get_step_height()
{
  return HEIGHT_BFS_STEP;
}

/*
 *  maps eta in [-H, 2*H] --> y in [-H,2*H]
 */
double
grid_transform_y(double const & eta)
{
  double y = 0.0;
  double gamma, xi;
  if(eta < 0.0)
  {
    gamma = GAMMA_LOWER;
    xi    = -0.5 * H;
  }
  else
  {
    gamma = GAMMA_UPPER;
    xi    = H;
  }
  y = xi * (1.0 - (std::tanh(gamma * (xi - eta)) / std::tanh(gamma * xi)));
  return y;
}

/*
 * inverse mapping:
 *
 *  maps y in [-H,2*H] --> eta in [-H,2*H]
 */
double
inverse_grid_transform_y(double const & y)
{
  double eta = 0.0;
  double gamma, xi;
  if(y < 0.0)
  {
    gamma = GAMMA_LOWER;
    xi    = -0.5 * H;
  }
  else
  {
    gamma = GAMMA_UPPER;
    xi    = H;
  }
  eta = xi - (1.0 / gamma) * std::atanh((1.0 - y / xi) * std::tanh(gamma * xi));
  return eta;
}

template<int dim>
class MyManifold : public dealii::ChartManifold<dim, dim, dim>
{
public:
  MyManifold()
  {
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates
   *  to point x in physical coordinates
   */
  dealii::Point<dim>
  push_forward(dealii::Point<dim> const & xi) const final
  {
    dealii::Point<dim> x = xi;
    x[1]                 = grid_transform_y(xi[1]);

    return x;
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates
   */
  dealii::Point<dim>
  pull_back(dealii::Point<dim> const & x) const final
  {
    dealii::Point<dim> xi = x;
    xi[1]                 = inverse_grid_transform_y(x[1]);

    return xi;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const final
  {
    return std::make_unique<MyManifold<dim>>();
  }
};

template<int dim>
void
create_grid(dealii::Triangulation<dim> &                             triangulation,
            unsigned int const                                       target_dof_count,
            unsigned int const                                       degree_u,
            unsigned int const                                       n_refine_space,
            std::vector<dealii::GridTools::PeriodicFacePair<
              typename dealii::Triangulation<dim>::cell_iterator>> & periodic_faces)
{
  AssertThrow(dim == 3, dealii::ExcMessage("NotImplemented"));

  double       width_BFS            = WIDTH_BFS;
  unsigned int n_cells_inlet_height = 1;
  unsigned int n_cells_inlet_length = 5;
  unsigned int n_cells_width        = 1;
  unsigned int n_refine_space_base  = 1;

  // In case a target DoF count is provided, find the best possible match within a range of the grid
  // parameters.
  if(target_dof_count > 0)
  {
    // Preferred parameters come first.
    std::vector<unsigned int> n_cells_inlet_length_vec{5, 4, 6, 3, 7, 2, 8, 1, 9, 10};
    std::vector<unsigned int> n_cells_width_vec(20, 0);
    for(unsigned int i = 0; i < n_cells_width_vec.size(); ++i)
    {
      n_cells_width_vec[i] = i + 1;
    }
    std::vector<unsigned int> n_refine_space_vec{0, 1, 2, 3, 4, 5, 6, 7, 8};

    // Loop over parameter combinations, first relative tolerance match is chosen.
    bool                   match_found = false;
    unsigned long long int n_dofs      = 0;
    for(unsigned int i = 0; i < n_cells_inlet_length_vec.size(); ++i)
    {
      for(unsigned int j = 0; j < n_cells_width_vec.size(); ++j)
      {
        for(unsigned int k = 0; k < n_refine_space_vec.size(); ++k)
        {
          unsigned long long int const n_cells =
            n_cells_width_vec[j] * 5 * n_cells_inlet_height * n_cells_inlet_length_vec[i];
          unsigned long long int const n_cells_refined =
            n_cells * std::pow(2, dim * n_refine_space_vec[k]);
          unsigned int const n_dofs_per_cell =
            dealii::Utilities::fixed_power<dim>(degree_u + 1) * dim +
            dealii::Utilities::fixed_power<dim>(degree_u);
          n_dofs = n_dofs_per_cell * n_cells_refined;

          double constexpr rel_tol_dof_count_match = 0.05;
          double const n_dofs_difference =
            std::abs(static_cast<double>(n_dofs) - static_cast<double>(target_dof_count));
          if(n_dofs_difference / static_cast<double>(target_dof_count) < rel_tol_dof_count_match)
          {
            match_found          = true;
            n_cells_inlet_length = n_cells_inlet_length_vec[i];
            n_cells_width        = n_cells_width_vec[j];
            n_refine_space_base  = n_refine_space_vec[k];
            break;
          }
        }
        if(match_found)
        {
          break;
        }
      }
      if(match_found)
      {
        break;
      }
    }

    AssertThrow(match_found,
                dealii::ExcMessage("Could not find a suitable match to reach target DoFs."));

    // Adjust width of the step such that the width of the elements used in thickness direction is
    // equal their height in the inlet.
    width_BFS = static_cast<double>(n_cells_width) * HEIGHT_BFS_INFLOW /
                static_cast<double>(n_cells_inlet_height);
  }

  // Create the three triangulations and merge them.
  dealii::Triangulation<dim> tria_1, tria_2, tria_3;

  // inflow part of BFS
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria_1,
    std::vector<unsigned int>({n_cells_inlet_length, n_cells_inlet_height, n_cells_width}),
    dealii::Point<dim>(-LENGTH_BFS_UP, 0.0, -width_BFS / 2.0),
    dealii::Point<dim>(0.0, HEIGHT_BFS_INFLOW, width_BFS / 2.0));

  // downstream part of BFS (upper)
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria_2,
    std::vector<unsigned int>({2 * n_cells_inlet_length, n_cells_inlet_height, n_cells_width}),
    dealii::Point<dim>(0.0, 0.0, -width_BFS / 2.0),
    dealii::Point<dim>(LENGTH_BFS_DOWN, HEIGHT_BFS_INFLOW, width_BFS / 2.0));

  // downstream part of BFS (lower = step)
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria_3,
    std::vector<unsigned int>({2 * n_cells_inlet_length, n_cells_inlet_height, n_cells_width}),
    dealii::Point<dim>(0.0, 0.0, -width_BFS / 2.0),
    dealii::Point<dim>(LENGTH_BFS_DOWN, -HEIGHT_BFS_STEP, width_BFS / 2.0));

  dealii::Triangulation<dim> tmp1;
  dealii::GridGenerator::merge_triangulations(tria_1, tria_2, tmp1);
  dealii::GridGenerator::merge_triangulations(tmp1, tria_3, triangulation);

  // set boundary ID's
  for(auto cell : triangulation.cell_iterators())
  {
    for(auto const & f : cell->face_indices())
    {
      // outflow boundary on the right has ID = 1
      if((std::fabs(cell->face(f)->center()(0) - X1_COORDINATE_OUTFLOW) < 1.e-12))
        cell->face(f)->set_boundary_id(1);
      // inflow boundary on the left has ID = 2
      if((std::fabs(cell->face(f)->center()(0) - X1_COORDINATE_INFLOW) < 1.e-12))
        cell->face(f)->set_boundary_id(2);

      // periodicity in z-direction
      if((std::fabs(cell->face(f)->center()(2) - width_BFS / 2.0) < 1.e-12))
        cell->face(f)->set_all_boundary_ids(2 + 10);
      if((std::fabs(cell->face(f)->center()(2) + width_BFS / 2.0) < 1.e-12))
        cell->face(f)->set_all_boundary_ids(3 + 10);
    }
  }

  if(use_grid_stretching_in_y_direction == true)
  {
    // manifold
    unsigned int manifold_id = 1;
    for(auto cell : triangulation.cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const MyManifold<dim> manifold;
    triangulation.set_manifold(manifold_id, manifold);
  }

  // periodicity in z-direction
  dealii::GridTools::collect_periodic_faces(triangulation, 2 + 10, 3 + 10, 2, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  // perform global refinements
  triangulation.refine_global(n_refine_space_base + n_refine_space);
}

} // namespace Geometry
} // namespace IncNS
} // namespace ExaDG

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_GEOMETRY_H_ */
