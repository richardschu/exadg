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
 * Tests `MappingDoFVector` by mapping the unit square (unit cube) onto a quarter
 * annulus with radii 0.5 and 1 (extruded in z for dim = 3). The displacement
 * relative to the undeformed grid is interpolated into a `FE_Q`^dim
 * (`FE_SimplexP`^dim) space of the mapping degree, the mapping is initialized
 * via `initialize_mapping_from_dof_vector()`, and the volume, the integral of
 * x^2 and the surface are compared with the exact values. The printed rates are
 * log2(e_{n/2} / e_n) for n subdivisions per direction. For simplex elements,
 * each hypercube is split into simplices. The rate for `FE_SimplexP` of degree
 * 3 is 4 instead of 6, since its support points are equidistant, while those
 * of `FE_Q` are Gauss-Lobatto points.
 *
 * Further checks:
 *  - mapping == nullptr with the absolute grid coordinates as displacement
 *    gives the same integrals,
 *  - fill_grid_coordinates_vector() recovers the interpolated coordinates,
 *  - an empty displacement vector reproduces the undeformed grid,
 *  - a displacement in a discontinuous `FE_DGQ`^dim (`FE_SimplexDGP`^dim) space
 *    gives the same integrals,
 *  - the coarse mappings for multigrid with separate coarse triangulations
 *    reproduce an affine deformation exactly on all levels.
 */

// C++
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_utilities.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/operators/finite_element.h>

namespace ExaDG
{
using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

/*
 * Maps the unit square onto the quarter annulus r in [0.5, 1], theta in
 * [0, pi/2]. For dim = 3, the z-coordinate is kept.
 */
template<int dim>
class QuarterAnnulus : public dealii::Function<dim>
{
public:
  QuarterAnnulus() : dealii::Function<dim>(dim)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    double const r     = 0.5 + 0.5 * p[0];
    double const theta = 0.5 * dealii::numbers::PI * p[1];

    if(component == 0)
      return r * std::cos(theta);
    else if(component == 1)
      return r * std::sin(theta);
    else
      return p[component];
  }
};

/*
 * Affine map with determinant 1.375, exactly represented by mappings of any
 * degree.
 */
template<int dim>
class AffineMap : public dealii::Function<dim>
{
public:
  AffineMap() : dealii::Function<dim>(dim)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    if(component == 0)
      return 1.5 * p[0] + 0.25 * p[1];
    else if(component == 1)
      return 0.5 * p[0] + p[1];
    else
      return 0.1 * p[0] + p[component];
  }
};

/*
 * Fully-distributed triangulation of the unit square/cube with
 * `n_subdivisions` hypercubes per direction, which are split into simplices for
 * ElementType::Simplex, and refined globally `n_refinements` times.
 */
template<int dim>
std::shared_ptr<dealii::parallel::fullydistributed::Triangulation<dim>>
create_triangulation(ElementType const  element_type,
                     unsigned int const n_subdivisions,
                     unsigned int const n_refinements,
                     MPI_Comm const &   mpi_comm)
{
  dealii::Triangulation<dim> serial_triangulation;
  {
    dealii::Triangulation<dim> hypercube_triangulation;
    dealii::GridGenerator::subdivided_hyper_cube(hypercube_triangulation, n_subdivisions, 0.0, 1.0);
    if(element_type == ElementType::Simplex)
      dealii::GridGenerator::convert_hypercube_to_simplex_mesh(hypercube_triangulation,
                                                               serial_triangulation);
    else
      serial_triangulation.copy_triangulation(hypercube_triangulation);
  }
  serial_triangulation.refine_global(n_refinements);

  dealii::GridTools::partition_triangulation_zorder(
    dealii::Utilities::MPI::n_mpi_processes(mpi_comm), serial_triangulation);

  auto triangulation =
    std::make_shared<dealii::parallel::fullydistributed::Triangulation<dim>>(mpi_comm);
  triangulation->create_triangulation(
    dealii::TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_triangulation, mpi_comm));

  return triangulation;
}

/*
 * Interpolates the displacement from the grid described by `mapping` to the
 * grid described by `function` into the space given by `dof_handler`.
 */
template<int dim>
VectorType
interpolate_displacement(dealii::Mapping<dim> const &    mapping,
                         dealii::Function<dim> const &   function,
                         dealii::DoFHandler<dim> const & dof_handler,
                         MPI_Comm const &                mpi_comm)
{
  // the numbering used for hypercube elements requires the degree of the
  // finite element
  MappingDoFVector<dim, double> mapping_dof_vector(dof_handler.get_fe().degree);

  VectorType grid_coordinates;
  mapping_dof_vector.fill_grid_coordinates_vector(mapping, grid_coordinates, dof_handler);

  VectorType displacement(dof_handler.locally_owned_dofs(),
                          dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                          mpi_comm);
  dealii::VectorTools::interpolate(mapping, dof_handler, function, displacement);

  displacement -= grid_coordinates;

  return displacement;
}

struct Integrals
{
  double volume  = 0.0;
  double x2      = 0.0;
  double surface = 0.0;
};

template<int dim>
Integrals
integrate(dealii::Mapping<dim> const &       mapping,
          dealii::Triangulation<dim> const & triangulation,
          unsigned int const                 n_q_points_1d,
          MPI_Comm const &                   mpi_comm)
{
  ElementType const element_type = get_element_type(triangulation);

  std::shared_ptr<dealii::Quadrature<dim>>     quadrature;
  std::shared_ptr<dealii::Quadrature<dim - 1>> face_quadrature;
  if(element_type == ElementType::Simplex)
  {
    // dealii::QGaussSimplex is available up to 4 points per direction
    quadrature      = std::make_shared<dealii::QGaussSimplex<dim>>(std::min(n_q_points_1d, 4u));
    face_quadrature = std::make_shared<dealii::QGaussSimplex<dim - 1>>(std::min(n_q_points_1d, 4u));
  }
  else
  {
    quadrature      = std::make_shared<dealii::QGauss<dim>>(n_q_points_1d);
    face_quadrature = std::make_shared<dealii::QGauss<dim - 1>>(n_q_points_1d);
  }

  dealii::FE_Nothing<dim>   fe_nothing(triangulation.begin_active()->reference_cell());
  dealii::FEValues<dim>     fe_values(mapping,
                                  fe_nothing,
                                  *quadrature,
                                  dealii::update_JxW_values | dealii::update_quadrature_points);
  dealii::FEFaceValues<dim> fe_face_values(mapping,
                                           fe_nothing,
                                           *face_quadrature,
                                           dealii::update_JxW_values);

  Integrals integrals;
  for(auto const & cell : triangulation.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      for(unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
      {
        double const x = fe_values.quadrature_point(q)[0];
        integrals.volume += fe_values.JxW(q);
        integrals.x2 += x * x * fe_values.JxW(q);
      }

      for(unsigned int const f : cell->face_indices())
      {
        if(cell->at_boundary(f))
        {
          fe_face_values.reinit(cell, f);
          for(unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
            integrals.surface += fe_face_values.JxW(q);
        }
      }
    }
  }

  integrals.volume  = dealii::Utilities::MPI::sum(integrals.volume, mpi_comm);
  integrals.x2      = dealii::Utilities::MPI::sum(integrals.x2, mpi_comm);
  integrals.surface = dealii::Utilities::MPI::sum(integrals.surface, mpi_comm);

  return integrals;
}

double
max_difference(Integrals const & a, Integrals const & b)
{
  return std::max(
    {std::abs(a.volume - b.volume), std::abs(a.x2 - b.x2), std::abs(a.surface - b.surface)});
}

struct Result
{
  Integrals errors;

  // mapping == nullptr vs. mapping + displacement
  double difference_absolute_coordinates = 0.0;

  // fill_grid_coordinates_vector() vs. interpolated coordinates
  double difference_grid_coordinates = 0.0;

  // empty displacement vector vs. undeformed grid
  double difference_undeformed = 0.0;

  // discontinuous vs. continuous displacement
  double difference_discontinuous = 0.0;
};

template<int dim>
Result
run(ElementType const  element_type,
    unsigned int const mapping_degree,
    unsigned int const n_subdivisions,
    MPI_Comm const &   mpi_comm)
{
  std::shared_ptr<dealii::Triangulation<dim>> const triangulation_ptr =
    create_triangulation<dim>(element_type, n_subdivisions, 0 /* n_refinements */, mpi_comm);
  dealii::Triangulation<dim> const & triangulation = *triangulation_ptr;

  std::shared_ptr<dealii::Mapping<dim>> mapping_undeformed_ptr;
  GridUtilities::create_mapping(mapping_undeformed_ptr, element_type, 1 /* mapping_degree */);
  dealii::Mapping<dim> const & mapping_undeformed = *mapping_undeformed_ptr;

  std::shared_ptr<dealii::FiniteElement<dim>> fe =
    create_finite_element<dim>(element_type, false /* is_dg */, dim, mapping_degree);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(*fe);

  // undeformed and target grid coordinates, displacement = target - undeformed
  MappingDoFVector<dim, double> mapping_dof_vector(mapping_degree);

  VectorType grid_coordinates_undeformed;
  mapping_dof_vector.fill_grid_coordinates_vector(mapping_undeformed,
                                                  grid_coordinates_undeformed,
                                                  dof_handler);

  VectorType grid_coordinates_target(dof_handler.locally_owned_dofs(),
                                     dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                     mpi_comm);
  dealii::VectorTools::interpolate(mapping_undeformed,
                                   dof_handler,
                                   QuarterAnnulus<dim>(),
                                   grid_coordinates_target);

  VectorType displacement = grid_coordinates_target;
  displacement -= grid_coordinates_undeformed;

  mapping_dof_vector.initialize_mapping_from_dof_vector(&mapping_undeformed,
                                                        displacement,
                                                        dof_handler);

  unsigned int const n_q_points_1d = mapping_degree + 2;

  Integrals const integrals =
    integrate(*mapping_dof_vector.get_mapping(), triangulation, n_q_points_1d, mpi_comm);

  // exact values: annulus segment area 3 pi / 16, int x^2 = 15 pi / 256, arc
  // lengths pi / 2 and pi / 4, straight edges 0.5 each; for dim = 3 the
  // extrusion by 1 adds top and bottom to the surface
  double const pi = dealii::numbers::PI;
  Integrals    exact;
  exact.volume  = 3.0 * pi / 16.0;
  exact.x2      = 15.0 * pi / 256.0;
  exact.surface = 0.75 * pi + 1.0 + (dim == 3 ? 2.0 * exact.volume : 0.0);

  Result result;
  result.errors.volume  = std::abs(integrals.volume - exact.volume) / exact.volume;
  result.errors.x2      = std::abs(integrals.x2 - exact.x2) / exact.x2;
  result.errors.surface = std::abs(integrals.surface - exact.surface) / exact.surface;

  // mapping == nullptr: the vector holds the absolute grid coordinates
  {
    MappingDoFVector<dim, double> mapping_absolute(mapping_degree);
    mapping_absolute.initialize_mapping_from_dof_vector(nullptr,
                                                        grid_coordinates_target,
                                                        dof_handler);
    result.difference_absolute_coordinates = max_difference(
      integrals,
      integrate(*mapping_absolute.get_mapping(), triangulation, n_q_points_1d, mpi_comm));
  }

  // recover the grid coordinates from the mapping
  {
    VectorType grid_coordinates;
    mapping_dof_vector.fill_grid_coordinates_vector(grid_coordinates, dof_handler);
    grid_coordinates -= grid_coordinates_target;
    result.difference_grid_coordinates = grid_coordinates.linfty_norm();
  }

  // empty displacement vector: undeformed unit square/cube
  {
    MappingDoFVector<dim, double> mapping_empty(mapping_degree);
    mapping_empty.initialize_mapping_from_dof_vector(&mapping_undeformed,
                                                     VectorType(),
                                                     dof_handler);
    Integrals unit_cube;
    unit_cube.volume             = 1.0;
    unit_cube.x2                 = 1.0 / 3.0;
    unit_cube.surface            = 2.0 * dim;
    result.difference_undeformed = max_difference(
      unit_cube, integrate(*mapping_empty.get_mapping(), triangulation, n_q_points_1d, mpi_comm));
  }

  // discontinuous displacement field
  {
    std::shared_ptr<dealii::FiniteElement<dim>> fe_dg =
      create_finite_element<dim>(element_type, true /* is_dg */, dim, mapping_degree);
    dealii::DoFHandler<dim> dof_handler_dg(triangulation);
    dof_handler_dg.distribute_dofs(*fe_dg);

    MappingDoFVector<dim, double> mapping_dg(mapping_degree);
    mapping_dg.initialize_mapping_from_dof_vector(
      &mapping_undeformed,
      interpolate_displacement(mapping_undeformed, QuarterAnnulus<dim>(), dof_handler_dg, mpi_comm),
      dof_handler_dg);
    result.difference_discontinuous =
      max_difference(integrals,
                     integrate(*mapping_dg.get_mapping(), triangulation, n_q_points_1d, mpi_comm));
  }

  return result;
}

/*
 * Initializes the coarse mappings for multigrid with separate coarse
 * triangulations from the fine mapping for an affine deformation and returns
 * the maximum difference of the integrals on the coarse levels compared to the
 * fine level.
 */
template<int dim>
double
run_multigrid(ElementType const  element_type,
              unsigned int const mapping_degree,
              MPI_Comm const &   mpi_comm)
{
  unsigned int const n_levels = 3;

  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> coarse_triangulations;
  for(unsigned int level = 0; level < n_levels - 1; ++level)
  {
    coarse_triangulations.push_back(
      create_triangulation<dim>(element_type, 2 /* n_subdivisions */, level, mpi_comm));
  }
  std::shared_ptr<dealii::Triangulation<dim> const> const fine_triangulation =
    create_triangulation<dim>(element_type, 2 /* n_subdivisions */, n_levels - 1, mpi_comm);

  std::shared_ptr<dealii::Mapping<dim>> mapping_undeformed;
  GridUtilities::create_mapping(mapping_undeformed, element_type, 1 /* mapping_degree */);

  std::shared_ptr<dealii::FiniteElement<dim>> fe =
    create_finite_element<dim>(element_type, false /* is_dg */, dim, mapping_degree);
  dealii::DoFHandler<dim> dof_handler(*fine_triangulation);
  dof_handler.distribute_dofs(*fe);

  auto fine_mapping = std::make_shared<MappingDoFVector<dim, double>>(mapping_degree);
  fine_mapping->initialize_mapping_from_dof_vector(
    mapping_undeformed.get(),
    interpolate_displacement(*mapping_undeformed, AffineMap<dim>(), dof_handler, mpi_comm),
    dof_handler);

  std::vector<std::shared_ptr<MappingDoFVector<dim, double>>> coarse_mappings(n_levels - 1);
  MappingTools::initialize_coarse_mappings_from_mapping_dof_vector<dim, double>(
    coarse_mappings,
    mapping_degree,
    std::shared_ptr<MappingDoFVector<dim, double> const>(fine_mapping),
    fine_triangulation,
    coarse_triangulations);

  unsigned int const n_q_points_1d = mapping_degree + 2;

  Integrals const integrals_fine =
    integrate(*fine_mapping->get_mapping(), *fine_triangulation, n_q_points_1d, mpi_comm);

  double difference = std::abs(integrals_fine.volume - 1.375);
  for(unsigned int level = 0; level < n_levels - 1; ++level)
  {
    difference = std::max(difference,
                          max_difference(integrals_fine,
                                         integrate(*coarse_mappings[level]->get_mapping(),
                                                   *coarse_triangulations[level],
                                                   n_q_points_1d,
                                                   mpi_comm)));
  }

  return difference;
}

template<int dim>
void
test(ElementType const  element_type,
     unsigned int const max_mapping_degree,
     MPI_Comm const &   mpi_comm)
{
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  double const tolerance = 1e-12;

  // the integrals sum up round-off over many quadrature points, e.g., about
  // 2e-12 for the undeformed 3D simplex mesh with 12288 cells
  double const tolerance_integrals = 1e-10;

  for(unsigned int mapping_degree = 1; mapping_degree <= max_mapping_degree; ++mapping_degree)
  {
    pcout << std::endl
          << "dim = " << dim << ", " << (element_type == ElementType::Simplex ? "simplex, " : "")
          << "mapping degree = " << mapping_degree << ":" << std::endl
          << "  n_subdiv  err volume  rate  err int x^2  rate  err surface  rate" << std::endl;

    // limit the refinement for high degrees, such that the errors stay well
    // above round-off and the printed rates are reproducible
    unsigned int const max_subdivisions = mapping_degree >= 4 ? 4 : 8;

    bool      all_checks_passed = true;
    Integrals errors_previous;
    for(unsigned int n_subdivisions = 2; n_subdivisions <= max_subdivisions; n_subdivisions *= 2)
    {
      Result const result = run<dim>(element_type, mapping_degree, n_subdivisions, mpi_comm);

      pcout << "  " << std::setw(8) << n_subdivisions << std::scientific << std::setprecision(2);
      std::array<double, 3> const errors = {
        {result.errors.volume, result.errors.x2, result.errors.surface}};
      std::array<double, 3> const errors_old = {
        {errors_previous.volume, errors_previous.x2, errors_previous.surface}};
      for(unsigned int i = 0; i < errors.size(); ++i)
      {
        pcout << std::scientific << std::setprecision(2) << std::setw(i == 0 ? 12 : 13) << errors[i]
              << std::fixed << std::setprecision(2) << std::setw(6);
        if(n_subdivisions > 2)
          pcout << std::log2(errors_old[i] / errors[i]);
        else
          pcout << "-";
      }
      pcout << std::endl;

      all_checks_passed =
        all_checks_passed and result.difference_absolute_coordinates < tolerance and
        result.difference_grid_coordinates < tolerance and
        result.difference_undeformed < tolerance and result.difference_discontinuous < tolerance;

      errors_previous = result.errors;
    }

    pcout << "  absolute coordinates, recovered coordinates, empty displacement, "
          << "discontinuous displacement: " << (all_checks_passed ? "OK" : "FAILED") << std::endl;
    pcout << "  multigrid coarse mappings, affine deformation: "
          << (run_multigrid<dim>(element_type, mapping_degree, mpi_comm) < tolerance_integrals ?
                "OK" :
                "FAILED")
          << std::endl;
  }
}

} // namespace ExaDG

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    ExaDG::test<2>(ExaDG::ElementType::Hypercube, 4 /* max_mapping_degree */, MPI_COMM_WORLD);
    ExaDG::test<3>(ExaDG::ElementType::Hypercube, 3 /* max_mapping_degree */, MPI_COMM_WORLD);

    // dealii::FE_SimplexP is available up to degree 3
    ExaDG::test<2>(ExaDG::ElementType::Simplex, 3 /* max_mapping_degree */, MPI_COMM_WORLD);
    ExaDG::test<3>(ExaDG::ElementType::Simplex, 3 /* max_mapping_degree */, MPI_COMM_WORLD);
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
