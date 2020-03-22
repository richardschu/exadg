/*
 * application.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef STRUCTURE_BEAM
#define STRUCTURE_BEAM

#include "../../../include/structure/user_interface/application_base.h"

namespace Structure
{
namespace Beam
{
template<int dim>
class BendingMoment : public Function<dim>
{
public:
  BendingMoment(double force, double height, bool incremental_loading, double end_time = -1.0)
    : Function<dim>(dim),
      force_max(force / (height / 2)),
      incremental_loading(incremental_loading),
      end_time(end_time)
  {
    if(incremental_loading)
      AssertThrow(end_time > 0.0,
                  ExcMessage("End time needs to be specified in case of incremental loading."));
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    double factor = 1.0;
    if(incremental_loading)
      factor = this->get_time() / end_time;

    if(c == 0)
      return factor * force_max * p[1];
    else
      return 0.0;
  }

private:
  const double force_max;
  bool const   incremental_loading;
  double const end_time;
};

template<int dim>
class SingleForce : public Function<dim>
{
public:
  SingleForce(double force, double length, bool incremental_loading, double end_time = -1.0)
    : Function<dim>(dim),
      force_per_length(force / length),
      incremental_loading(incremental_loading),
      end_time(end_time)
  {
    if(incremental_loading)
      AssertThrow(end_time > 0.0,
                  ExcMessage("End time needs to be specified in case of incremental loading."));
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    (void)p;

    double factor = 1.0;
    if(incremental_loading)
      factor = this->get_time() / end_time;

    if(c == 1)
      return -factor * force_per_length;
    else
      return 0.0;
  }

private:
  double const force_per_length;
  bool const   incremental_loading;
  double const end_time;
};

template<int dim>
class SolutionSF : public Function<dim>
{
public:
  SolutionSF(double length, double height, double width, double singleforce)
    : Function<dim>(dim),
      length(length),
      height(height),
      width(width),
      lineforce(singleforce * width)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    (void)p;
    (void)c;

    if(c == 1)
    {
      return -(length * length * length * lineforce /
               (6 * 200e3 * width * height * height * height / 12)) *
             (-p[0] * p[0] * p[0] / (length * length * length) +
              3 * p[0] * p[0] / (length * length));
    }
    else
      return 0.0;
  }

private:
  double const length;
  double const height;
  double const width;
  double const lineforce;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
    prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
    prm.add_parameter("OutputName",       output_name,      "Name of output files.");
    prm.add_parameter("Length",           length,           "Length of domain.");
    prm.add_parameter("Height",           height,           "Height of domain.");
    prm.add_parameter("Width",            width,            "Width of domain.");
    prm.add_parameter("BoundaryType",     boundary_type,    "Type of Neumann BC at right boundary.", Patterns::Selection("SingleForce|BendingMoment"));
    prm.add_parameter("Force",            force,            "Value of force on right boundary.");
    prm.leave_subsection();
    // clang-format on
  }

  // output
  std::string output_directory = "output/beam/vtu/", output_name = "test";

  // size of geometry
  double length = 1.0, height = 1.0, width = 1.0;

  // single force or bending moment
  std::string boundary_type = "SingleForce";

  double force = 2500;

  double element_length = 1.0;

  // number of subdivisions in each direction
  unsigned int const repetitions0 = 20, repetitions1 = 4, repetitions2 = 1;

  Application() : ApplicationBase<dim, Number>("")
  {
  }

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    parse_input(input_file, prm);
  }

  void
  set_input_parameters(InputParameters & parameters)
  {
    parameters.problem_type      = ProblemType::Steady;
    parameters.right_hand_side   = false;
    parameters.large_deformation = false;

    parameters.triangulation_type = TriangulationType::Distributed;
    parameters.mapping            = MappingType::Affine;

    parameters.solver         = Solver::CG;
    parameters.preconditioner = Preconditioner::AMG;

    this->param = parameters;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    Point<dim> p1, p2;
    p1[0] = 0;
    p1[1] = -(this->height / 2);
    if(dim == 3)
      p1[2] = -(this->width / 2);

    p2[0] = this->length;
    p2[1] = +(this->height / 2);
    if(dim == 3)
      p2[2] = (this->width / 2);

    std::vector<unsigned int> repetitions(dim);
    repetitions[0] = this->repetitions0;
    repetitions[1] = this->repetitions1;
    if(dim == 3)
      repetitions[2] = this->repetitions2;

    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, p1, p2);

    element_length = this->length / (this->repetitions0 * pow(2, n_refine_space));

    double const tol = 1.e-8;
    for(auto cell : *triangulation)
    {
      for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        // left face
        if(std::fabs(cell.face(face)->center()(0) - 0) < tol)
        {
          cell.face(face)->set_all_boundary_ids(1);
        }
        // right face
        else if(std::fabs(cell.face(face)->center()(0) - this->length) < tol)
        {
          cell.face(face)->set_all_boundary_ids(2);
        }
        // top-right edge
        else if(std::fabs(cell.face(face)->center()(0) - this->length) < element_length &&
                std::fabs(cell.face(face)->center()(1) - this->height / 2) < tol)
        {
          if(boundary_type == "SingleForce")
          {
            cell.face(face)->set_all_boundary_ids(3);
          }
          else
          {
            AssertThrow(boundary_type == "BendingMoment", ExcMessage("Not implemented."));
          }
        }
      }
    }

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, ComponentMask>                  pair_mask;

    boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));

    // left side
    boundary_descriptor->dirichlet_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(1, ComponentMask()));

    // right side
    bool const incremental_loading = (this->param.problem_type == ProblemType::QuasiStatic);

    if(boundary_type == "BendingMoment")
    {
      boundary_descriptor->neumann_bc.insert(
        pair(2, new BendingMoment<dim>(force, height, incremental_loading, this->param.end_time)));
    }
    else if(boundary_type == "SingleForce")
    {
      boundary_descriptor->neumann_bc.insert(pair(2, new Functions::ZeroFunction<dim>(dim)));

      boundary_descriptor->neumann_bc.insert(pair(
        3, new SingleForce<dim>(force, element_length, incremental_loading, this->param.end_time)));
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }
  }

  void
  set_material(MaterialDescriptor & material_descriptor)
  {
    typedef std::pair<types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type = MaterialType::StVenantKirchhoff;
    // E-Modulus of Steel in unit = [N/mm^2]
    double const E = 200e3, nu = 0.3;
    Type2D const two_dim_type = Type2D::PlainStress;

    material_descriptor.insert(Pair(0, new StVenantKirchhoffData(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  construct_postprocessor(InputParameters & param, MPI_Comm const & mpi_comm)
  {
    (void)param;

    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = true;
    pp_data.output_data.output_folder      = output_directory;
    pp_data.output_data.output_name        = output_name;
    pp_data.output_data.write_higher_order = false;

    if(boundary_type == "SingleForce")
    {
      pp_data.error_data.analytical_solution_available = true;
      pp_data.error_data.calculate_relative_errors     = true;
      pp_data.error_data.analytical_solution.reset(
        new SolutionSF<dim>(length, height, width, force));
    }
    else
    {
      AssertThrow(boundary_type == "BendingMoment", ExcMessage("Not implemented."));
    }

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return post;
  }
};

} // namespace Beam
} // namespace Structure

#endif
