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

#ifndef STRUCTURE_THROUGHPUT
#define STRUCTURE_THROUGHPUT

#include <exadg/grid/deformed_cube_manifold.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
    {
      prm.add_parameter("Length",                length,                   "Length of domain.");
      prm.add_parameter("Height",                height,                   "Height of domain.");
      prm.add_parameter("Width",                 width,                    "Width of domain.");
      prm.add_parameter("SpatialIntegration",    spatial_integration,      "Use spatial integration.");
      prm.add_parameter("ForceMaterialResidual", force_material_residual,  "Use undeformed configuration to evaluate the residual.");
      prm.add_parameter("CacheLevel",            cache_level,              "Cache level: 0 none, 1 scalars, 2 tensors.");
      prm.add_parameter("CheckType",             check_type,               "Check type for deformation gradient.");
      prm.add_parameter("MappingStrength",       mapping_strength,         "Strength of the mapping applied.");
      prm.add_parameter("ProblemType",           problem_type,             "Problem type considered, QuasiStatic vs Unsteady vs. Steady");
      prm.add_parameter("MaterialType",          material_type,            "StVenantKirchhoff vs. IncompressibleNeoHookean");
      prm.add_parameter("WeakDamping",           weak_damping_coefficient, "Weak damping coefficient for unsteady problems.");
    }
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  set_parameters() final
  {
    this->param.problem_type            = problem_type;
    this->param.body_force              = true;
    this->param.large_deformation       = true;
    this->param.pull_back_body_force    = false;
    this->param.pull_back_traction      = false;
    this->param.spatial_integration     = spatial_integration;
    this->param.cache_level             = cache_level;
    this->param.check_type              = check_type;
    this->param.force_material_residual = force_material_residual;

    this->param.density = density;
    if(this->param.problem_type == ProblemType::Unsteady and weak_damping_coefficient > 0.0)
    {
      this->param.weak_damping_active      = true;
      this->param.weak_damping_coefficient = weak_damping_coefficient;
    }

    // Using a Lagrangian description, we can simplify the mapping for this box.
    if(spatial_integration or mapping_strength > 1e-12)
    {
      this->param.mapping_degree              = this->param.degree;
      this->param.mapping_degree_coarse_grids = this->param.degree;
    }
    else
    {
      this->param.mapping_degree              = 1;
      this->param.mapping_degree_coarse_grids = 1;
    }

    this->param.grid.element_type = ElementType::Hypercube; // Simplex;
    if(this->param.grid.element_type == ElementType::Simplex)
    {
      this->param.grid.triangulation_type           = TriangulationType::FullyDistributed;
      this->param.grid.create_coarse_triangulations = true;
    }
    else if(this->param.grid.element_type == ElementType::Hypercube)
    {
      this->param.grid.triangulation_type           = TriangulationType::Distributed;
      this->param.grid.create_coarse_triangulations = false; // can also be set to true if desired
    }

    // These should not be needed for the throughput applications.
    this->param.solver = Solver::FGMRES;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;

        // left-bottom-front and right-top-back point
        dealii::Point<dim> p1, p2;

        for(unsigned d = 0; d < dim; d++)
          p1[d] = 0.0;

        p2[0] = this->length;
        p2[1] = this->height;
        if(dim == 3)
          p2[2] = this->width;

        std::vector<unsigned int> repetitions(dim);
        repetitions[0] = this->repetitions0;
        repetitions[1] = this->repetitions1;
        if(dim == 3)
          repetitions[2] = this->repetitions2;

        if(this->param.grid.element_type == ElementType::Hypercube)
        {
          dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);
        }
        else if(this->param.grid.element_type == ElementType::Simplex)
        {
          dealii::Triangulation<dim, dim> tria_hypercube;
          dealii::GridGenerator::subdivided_hyper_rectangle(tria_hypercube, repetitions, p1, p2);

          dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tria_hypercube, tria);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        /*
         * illustration of 2d geometry / boundary ids:
         *
         *                  bid = 0
         *      ________________________________
         *     |                                |
         *     | bid = 1                        | bid = 2
         *     |________________________________|
         *
         *                  bid = 3
         *
         * in the 3d case: face at z = 0 has bid = 4, the other face has bid = 0 (as the top face in
         * the figure above).
         *
         */
        for(auto cell : tria)
        {
          for(auto const & face : cell.face_indices())
          {
            // left face
            if(std::fabs(cell.face(face)->center()(0) - 0.0) < 1e-8)
            {
              cell.face(face)->set_all_boundary_ids(1);
            }

            // right face
            if(std::fabs(cell.face(face)->center()(0) - this->length) < 1e-8)
            {
              cell.face(face)->set_all_boundary_ids(2);
            }

            // lower face
            if(std::fabs(cell.face(face)->center()(1) - 0.0) < 1e-8)
            {
              cell.face(face)->set_all_boundary_ids(3);
            }

            // back face
            if(dim == 3)
            {
              if(std::fabs(cell.face(face)->center()(2) - 0.0) < 1e-8)
              {
                cell.face(face)->set_all_boundary_ids(4);
              }
            }
          }
        }

        if(vector_local_refinements.size() > 0)
          refine_local(tria, vector_local_refinements);

        if(global_refinements > 0)
          tria.refine_global(global_refinements);

        // Apply manifold map on a uniform cube
        unsigned int const frequency = 1;
        if(mapping_strength > 1e-12)
          if(std::abs(this->length - this->height) < 1e-12)
            if(dim == 2 or std::abs(this->length - this->width) < 1e-12)
              apply_deformed_cube_manifold(tria, 0.0, this->length, mapping_strength, frequency);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    this->boundary_descriptor->neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // left face: Dirichlet BCs
    bool constexpr clamp_at_left_boundary = true;
    std::vector<bool> mask_left           = {true, clamp_at_left_boundary};
    if(dim == 3)
    {
      mask_left.resize(3);
      mask_left[2] = clamp_at_left_boundary;
    }
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(1, mask_left));

    // right face: Neumann BCs
    this->boundary_descriptor->neumann_bc.insert(
      pair(2, new dealii::Functions::ConstantFunction<dim>(dim)));

    // lower/upper face (3d: front/back face)
    this->boundary_descriptor->neumann_bc.insert(
      pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));

    if(dim == 3)
    {
      this->boundary_descriptor->neumann_bc.insert(
        pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));
    }
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    if(material_type == MaterialType::StVenantKirchhoff)
    {
      Type2D const two_dim_type = Type2D::PlaneStress;
      double const nu           = 0.3;
      this->material_descriptor->insert(
        Pair(0, new StVenantKirchhoffData<dim>(material_type, E_modul, nu, two_dim_type)));
    }
    else if(material_type == MaterialType::IncompressibleNeoHookean)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.49;
      double const bulk_modulus  = shear_modulus * 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));

      this->material_descriptor->insert(Pair(0,
                                             new IncompressibleNeoHookeanData<dim>(material_type,
                                                                                   shear_modulus,
                                                                                   bulk_modulus,
                                                                                   two_dim_type)));
    }
    else if(material_type == MaterialType::CompressibleNeoHookean)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.3;
      double const lambda        = shear_modulus * 2.0 * nu / (1.0 - 2.0 * nu);

      this->material_descriptor->insert(Pair(
        0,
        new CompressibleNeoHookeanData<dim>(material_type, shear_modulus, lambda, two_dim_type)));
    }
    else if(material_type == MaterialType::IncompressibleFibrousTissue)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.49;
      double const bulk_modulus  = shear_modulus * 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));

      // Parameters corresponding to aortic tissue might be found in
      // [Weisbecker et al., J Mech Behav Biomed Mater 12, 2012] or
      // [Rolf-Pissarczyk et al., Comput Methods Appl Mech Eng 373, 2021].
      // a = 3.62, b = 34.3 for medial tissue lead to the H_ii below,
      // while the k_1 coefficient is scaled relative to the shear modulus
      // (for medial tissue, e.g., 62.1 kPa) used in the other cases here.
      double const fiber_angle_phi_in_degree = 27.47;                          // [deg]
      double const fiber_H_11                = 0.9168;                         // [-]
      double const fiber_H_22                = 0.0759;                         // [-]
      double const fiber_H_33                = 0.0073;                         // [-]
      double const fiber_k_1                 = 1.4e3 / 62.1e3 * shear_modulus; // [Pa]
      double const fiber_k_2                 = 22.1;                           // [-]

      this->material_descriptor->insert(
        Pair(0,
             new IncompressibleFibrousTissueData<dim>(material_type,
                                                      shear_modulus,
                                                      bulk_modulus,
                                                      fiber_angle_phi_in_degree,
                                                      fiber_H_11,
                                                      fiber_H_22,
                                                      fiber_H_33,
                                                      fiber_k_1,
                                                      fiber_k_2,
                                                      "",
                                                      "",
                                                      0.0,
                                                      two_dim_type)));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Material type is not expected in application."));
    }
  }

  void
  set_field_functions() final
  {
    this->field_functions->right_hand_side.reset(
      new dealii::Functions::ConstantFunction<dim>(1.0, dim));
    this->field_functions->initial_displacement.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    std::shared_ptr<PostProcessor<dim, Number>> pp(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double length = 1.0, height = 1.0, width = 1.0;

  bool spatial_integration     = false;
  bool force_material_residual = false;

  unsigned int check_type  = 0;
  unsigned int cache_level = 0;

  ProblemType problem_type = ProblemType::Unsteady;

  double weak_damping_coefficient = 0.0;

  // mesh parameters
  unsigned int const repetitions0 = 1, repetitions1 = 1, repetitions2 = 1;

  MaterialType material_type = MaterialType::Undefined;
  double const E_modul       = 200.0;

  double const density = 0.001;

  double mapping_strength = 0.0;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif /* STRUCTURE_THROUGHPUT */
