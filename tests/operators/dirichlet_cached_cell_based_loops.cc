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

/*
 * Read cached boundary data (ContainerInterfaceData, e.g., DirichletCached
 * boundary conditions in fluid-structure interaction) with face integrators of
 * face loops and of cell-based face loops. In the latter, the lanes of a face
 * integrator belong to different face batches. The cached data are the
 * quadrature points, such that the values read have to match the quadrature
 * points of the integrator.
 */

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/container_interface_data.h>
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/categorization.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/quadrature.h>

using namespace ExaDG;

template<int dim>
void
test()
{
  using Number = double;

  MPI_Comm const             mpi_comm = MPI_COMM_WORLD;
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  // Only boundary ID 0 is cached, the other part of the boundary has ID 1.
  dealii::parallel::distributed::Triangulation<dim> tria(mpi_comm);
  dealii::GridGenerator::hyper_ball(tria);
  for(auto const & face : tria.active_face_iterators())
    if(face->at_boundary() and face->center()[0] > 0.5)
      face->set_boundary_id(1);
  tria.refine_global(1);

  unsigned int const      degree = 2;
  dealii::MappingQ<dim>   mapping(2);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(
    *create_finite_element<dim>(ElementType::Hypercube, true /* is_dg */, dim, degree));
  dealii::AffineConstraints<Number> constraints;
  constraints.close();

  MatrixFreeData<dim, Number> matrix_free_data;
  MappingFlags                mapping_flags;
  mapping_flags.boundary_faces = dealii::update_quadrature_points;
  matrix_free_data.append_mapping_flags(mapping_flags);
  matrix_free_data.insert_dof_handler(&dof_handler, "dof");
  matrix_free_data.insert_constraint(&constraints, "dof");
  matrix_free_data.insert_quadrature(*create_quadrature<dim>(ElementType::Hypercube, degree + 1),
                                     "gauss");
  Categorization::do_cell_based_loops(tria, matrix_free_data.data);

  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping,
                     matrix_free_data.get_dof_handler_vector(),
                     matrix_free_data.get_constraint_vector(),
                     matrix_free_data.get_quadrature_vector(),
                     matrix_free_data.data);

  unsigned int const dof_index  = matrix_free_data.get_dof_index("dof");
  unsigned int const quad_index = matrix_free_data.get_quad_index("gauss");

  ContainerInterfaceData<1, dim, double> cached_data;
  cached_data.setup(matrix_free, dof_index, {quad_index}, {0} /* boundary ID */);
  auto const & q_points = cached_data.get_array_q_points(quad_index);
  auto &       solution = cached_data.get_array_solution(quad_index);
  for(unsigned int i = 0; i < q_points.size(); ++i)
    solution[i] = q_points[i];

  FaceIntegrator<dim, dim, Number> integrator(matrix_free, true, dof_index, quad_index);

  // Maximum difference of the cached data and the quadrature points for the
  // first n_lanes lanes of the integrator.
  auto const max_difference = [&](unsigned int const n_lanes) {
    double difference = 0.0;
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      auto const value   = FunctionEvaluator<1, dim, Number>::value(cached_data,
                                                                  integrator.get_face_ids(),
                                                                  q,
                                                                  quad_index);
      auto const q_point = integrator.quadrature_point(q);
      for(unsigned int d = 0; d < dim; ++d)
        for(unsigned int v = 0; v < n_lanes; ++v)
          difference = std::max(difference, std::abs(value[d][v] - q_point[d][v]));
    }
    return difference;
  };

  unsigned int n_faces_face_loop    = 0;
  double       difference_face_loop = 0.0;
  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
      ++face)
  {
    if(matrix_free.get_boundary_id(face) == 0)
    {
      integrator.reinit(face);
      unsigned int const n_lanes = matrix_free.n_active_entries_per_face_batch(face);
      n_faces_face_loop += n_lanes;
      difference_face_loop = std::max(difference_face_loop, max_difference(n_lanes));
    }
  }

  unsigned int n_faces_cell_based    = 0;
  double       difference_cell_based = 0.0;
  for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
  {
    unsigned int const n_lanes = matrix_free.n_active_entries_per_cell_batch(cell);
    for(unsigned int const f : dealii::GeometryInfo<dim>::face_indices())
    {
      // The categories group the cells by the boundary IDs of their faces.
      auto const boundary_ids = matrix_free.get_faces_by_cells_boundary_id(cell, f);
      for(unsigned int v = 1; v < n_lanes; ++v)
        AssertThrow(boundary_ids[v] == boundary_ids[0],
                    dealii::ExcMessage("Different boundary IDs within a cell batch."));

      if(boundary_ids[0] == 0)
      {
        integrator.reinit(cell, f);
        n_faces_cell_based += n_lanes;
        difference_cell_based = std::max(difference_cell_based, max_difference(n_lanes));
      }
    }
  }

  n_faces_face_loop     = dealii::Utilities::MPI::sum(n_faces_face_loop, mpi_comm);
  n_faces_cell_based    = dealii::Utilities::MPI::sum(n_faces_cell_based, mpi_comm);
  difference_face_loop  = dealii::Utilities::MPI::max(difference_face_loop, mpi_comm);
  difference_cell_based = dealii::Utilities::MPI::max(difference_cell_based, mpi_comm);

  pcout << "  dim = " << dim << ": " << tria.n_global_active_cells() << " cells" << std::endl
        << "    face loops:       " << n_faces_face_loop
        << " cached faces, data at the quadrature points: " << (difference_face_loop < 1e-12)
        << std::endl
        << "    cell-based loops: " << n_faces_cell_based
        << " cached faces, data at the quadrature points: " << (difference_cell_based < 1e-12)
        << std::endl;
}

int
main(int argc, char * argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    test<2>();
    test<3>();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
