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
 * Test the injection of DirichletCached boundary values into the degrees of
 * freedom of a continuous Lagrange element, as done by the Poisson and
 * structure operators: the values are evaluated at the face quadrature points
 * of create_quadrature_at_support_points() and written to the cell-local
 * degree of freedom given by face_to_cell_index_nodal. The result is compared
 * to the nodal interpolation of the same function on the boundary, for
 * hypercube and simplex elements on a ball with a curved mapping. The vectors
 * are written to ./vtu/<dim>d_<element type>_degree_<degree>/solution.vtu, the
 * boundary values only to boundary.vtu in the same folder.
 *
 * Note that face_to_cell_index_nodal of simplices assumes the default face
 * orientation. deal.II orients each face like the first cell containing it,
 * so boundary faces are always in default orientation, which the test prints.
 */

// C/C++
#include <filesystem>
#include <fstream>

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/container_interface_data.h>
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/quadrature.h>

using namespace ExaDG;

std::string
to_string(ElementType const element_type)
{
  return element_type == ElementType::Hypercube ? "Hypercube" : "Simplex";
}

template<int dim>
class BoundaryValues : public dealii::Function<dim>
{
public:
  BoundaryValues() : dealii::Function<dim>(dim)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    return 1.0 + component + p[0] * p[0] - p[1] * p[dim - 1] + std::sin(3.0 * p[component]);
  }
};

template<int dim>
void
create_ball(dealii::Triangulation<dim> & tria, ElementType const element_type)
{
  if(element_type == ElementType::Hypercube)
  {
    dealii::GridGenerator::hyper_ball(tria);
    tria.refine_global(1);
  }
  else
  {
    // Convert the refined hypercube ball with the new vertices on the sphere.
    dealii::Triangulation<dim> tria_hypercube;
    dealii::GridGenerator::hyper_ball(tria_hypercube);
    tria_hypercube.refine_global(1);

    dealii::Triangulation<dim> tria_flat;
    dealii::GridGenerator::flatten_triangulation(tria_hypercube, tria_flat);
    tria_flat.set_all_manifold_ids(dealii::numbers::flat_manifold_id);
    tria_flat.set_all_manifold_ids_on_boundary(0);
    tria_flat.set_manifold(0, dealii::SphericalManifold<dim>());
    dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tria_flat, tria);
    tria.set_manifold(0, dealii::SphericalManifold<dim>());
  }
}

template<int dim>
void
test(ElementType const element_type, unsigned int const degree)
{
  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  dealii::Triangulation<dim> tria;
  create_ball(tria, element_type);

  std::shared_ptr<dealii::Mapping<dim> const> mapping;
  if(element_type == ElementType::Hypercube)
    mapping = std::make_shared<dealii::MappingQ<dim> const>(2);
  else
    mapping = std::make_shared<dealii::MappingFE<dim> const>(dealii::FE_SimplexP<dim>(2));

  std::shared_ptr<dealii::FiniteElement<dim>> fe =
    create_finite_element<dim>(element_type, false /* is_dg */, dim, degree);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe);

  // Boundary faces that the adjacent cell does not see in the default
  // orientation.
  unsigned int n_faces_non_default_orientation = 0;
  for(auto const & cell : tria.active_cell_iterators())
    for(unsigned int const f : cell->face_indices())
      if(cell->at_boundary(f) and
         cell->combined_face_orientation(f) != dealii::numbers::default_geometric_orientation)
        ++n_faces_non_default_orientation;

  MatrixFreeData<dim, Number> matrix_free_data;

  MappingFlags mapping_flags;
  mapping_flags.boundary_faces = dealii::update_quadrature_points;
  matrix_free_data.append_mapping_flags(mapping_flags);

  dealii::AffineConstraints<Number> empty_constraints;
  empty_constraints.close();

  matrix_free_data.insert_dof_handler(&dof_handler, "dof");
  matrix_free_data.insert_constraint(&empty_constraints, "dof");
  matrix_free_data.insert_quadrature(*create_quadrature<dim>(element_type, degree + 1), "gauss");
  matrix_free_data.insert_quadrature(*create_quadrature_at_support_points<dim>(element_type,
                                                                               degree),
                                     "support_points");

  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(*mapping,
                     matrix_free_data.get_dof_handler_vector(),
                     matrix_free_data.get_constraint_vector(),
                     matrix_free_data.get_quadrature_vector(),
                     matrix_free_data.data);

  unsigned int const dof_index  = matrix_free_data.get_dof_index("dof");
  unsigned int const quad_index = matrix_free_data.get_quad_index("support_points");

  // Fill the cached data with the function values at the face quadrature
  // points.
  BoundaryValues<dim> boundary_values;

  ContainerInterfaceData<1, dim, double> cached_data;
  cached_data.setup(matrix_free, dof_index, {quad_index}, {0} /* boundary ID */);
  auto const & q_points = cached_data.get_array_q_points(quad_index);
  auto &       solution = cached_data.get_array_solution(quad_index);
  for(unsigned int i = 0; i < q_points.size(); ++i)
    for(unsigned int d = 0; d < dim; ++d)
      solution[i][d] = boundary_values.value(q_points[i], d);

  // Inject the cached values into the degrees of freedom on the boundary.
  VectorType dst;
  matrix_free.initialize_dof_vector(dst, dof_index);

  FaceIntegrator<dim, dim, Number> integrator(matrix_free, true, dof_index, quad_index);
  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
      ++face)
  {
    integrator.reinit(face);
    integrator.read_dof_values(dst);

    unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      unsigned int const index = matrix_free.get_shape_info(dof_index, quad_index)
                                   .face_to_cell_index_nodal[local_face_number][q];

      integrator.submit_dof_value(
        FunctionEvaluator<1, dim, Number>::value(cached_data, face, q, quad_index), index);
    }

    integrator.set_dof_values_plain(dst);
  }

  // Compare to the nodal interpolation on the boundary, the other entries
  // have to remain zero.
  VectorType reference;
  matrix_free.initialize_dof_vector(reference, dof_index);
  dealii::VectorTools::interpolate(*mapping, dof_handler, boundary_values, reference);

  dealii::IndexSet const boundary_dofs = dealii::DoFTools::extract_boundary_dofs(dof_handler);

  VectorType difference;
  matrix_free.initialize_dof_vector(difference, dof_index);

  double error_boundary = 0.0, error_interior = 0.0;
  for(dealii::types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
  {
    if(boundary_dofs.is_element(i))
    {
      difference(i)  = dst(i) - reference(i);
      error_boundary = std::max(error_boundary, std::abs(difference(i)));
    }
    else
      error_interior = std::max(error_interior, std::abs(dst(i)));
  }

  // Write the vectors on the curved mesh for visual inspection. For simplices,
  // dealii::DataOut supports at most 2 subdivisions and no higher-order cells.
  {
    std::string const directory = "./vtu/" + std::to_string(dim) + "d_" + to_string(element_type) +
                                  "_degree_" + std::to_string(degree) + "/";
    std::filesystem::create_directories(directory);

    dealii::DataOut<dim>          data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = element_type == ElementType::Hypercube;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> const
      interpretation(dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(dst,
                             std::vector<std::string>(dim, "dirichlet_cached"),
                             dealii::DataOut<dim>::type_dof_data,
                             interpretation);
    data_out.add_data_vector(reference,
                             std::vector<std::string>(dim, "interpolation"),
                             dealii::DataOut<dim>::type_dof_data,
                             interpretation);
    data_out.add_data_vector(difference,
                             std::vector<std::string>(dim, "difference_on_boundary"),
                             dealii::DataOut<dim>::type_dof_data,
                             interpretation);

    unsigned int const n_subdivisions =
      element_type == ElementType::Hypercube ? degree : std::min(degree, 2u);
    data_out.build_patches(*mapping, n_subdivisions, dealii::DataOut<dim>::curved_inner_cells);

    std::ofstream file(directory + "solution.vtu");
    data_out.write_vtu(file);

    // The boundary only, where the vectors coincide.
    dealii::DataOutFaces<dim> data_out_faces(true /* surface only */);
    data_out_faces.set_flags(flags);
    data_out_faces.attach_dof_handler(dof_handler);
    data_out_faces.add_data_vector(dst,
                                   std::vector<std::string>(dim, "dirichlet_cached"),
                                   dealii::DataOutFaces<dim>::type_dof_data,
                                   interpretation);
    data_out_faces.add_data_vector(reference,
                                   std::vector<std::string>(dim, "interpolation"),
                                   dealii::DataOutFaces<dim>::type_dof_data,
                                   interpretation);
    data_out_faces.build_patches(*mapping, n_subdivisions);

    std::ofstream file_faces(directory + "boundary.vtu");
    data_out_faces.write_vtu(file_faces);
  }

  std::cout << "  dim = " << dim << ", " << to_string(element_type) << ", degree = " << degree
            << ": " << tria.n_active_cells() << " cells, " << boundary_dofs.n_elements() << " of "
            << dof_handler.n_dofs() << " DoFs on the boundary, " << n_faces_non_default_orientation
            << " boundary faces in non-default orientation" << std::endl
            << "    max error on boundary DoFs:     " << error_boundary << std::endl
            << "    max value of other DoFs:        " << error_interior << std::endl;
}

int
main(int argc, char * argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    for(ElementType const element_type : {ElementType::Hypercube, ElementType::Simplex})
      for(unsigned int degree = 1; degree <= 3; ++degree)
      {
        test<2>(element_type, degree);
        test<3>(element_type, degree);
      }
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
