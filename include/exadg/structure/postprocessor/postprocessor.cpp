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

#include <exadg/structure/postprocessor/postprocessor.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & pp_data_in,
                                          MPI_Comm const &               mpi_comm_in)
  : pp_data(pp_data_in),
    mpi_comm(mpi_comm_in),
    output_generator(OutputGenerator<dim, Number>(mpi_comm_in)),
    error_calculator(ErrorCalculator<dim, Number>(mpi_comm_in))
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(Operator<dim, Number> const & pde_operator_in)
{
  pde_operator = &pde_operator_in;

  initialize_derived_fields();

  output_generator.setup(pde_operator->get_dof_handler(),
                         pde_operator->get_mapping(),
                         pp_data.output_data);

  error_calculator.setup(pde_operator->get_dof_handler(),
                         pde_operator->get_mapping(),
                         pp_data.error_data);
}

template<int dim, typename Number>
bool
PostProcessor<dim, Number>::requires_scalar_postprocessing_field() const
{
  return (pp_data.output_data.write_displacement_magnitude or
          pp_data.output_data.write_displacement_jacobian or
          pp_data.output_data.write_max_principal_stress);
}

template<int dim, typename Number>
bool
PostProcessor<dim, Number>::requires_vector_postprocessing_field() const
{
  return (pp_data.output_data.write_E1_orientation or pp_data.output_data.write_E2_orientation or
          pp_data.output_data.write_traction_local_full or
          pp_data.output_data.write_traction_local_normal or
          pp_data.output_data.write_traction_local_inplane);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const &     solution,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  invalidate_derived_fields();

  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    std::vector<dealii::ObserverPointer<SolutionField<dim, Number>>> additional_fields_vtu;

    if(pp_data.output_data.write_displacement_magnitude)
    {
      displacement_magnitude.evaluate(solution);
      additional_fields_vtu.push_back(&displacement_magnitude);
    }

    if(pp_data.output_data.write_displacement_jacobian)
    {
      displacement_jacobian.evaluate(solution);
      additional_fields_vtu.push_back(&displacement_jacobian);
    }

    if(pp_data.output_data.write_max_principal_stress)
    {
      max_principal_stress.evaluate(solution);
      additional_fields_vtu.push_back(&max_principal_stress);
    }

    if(pp_data.output_data.write_E1_orientation)
    {
      E1_orientation.evaluate(solution);
      additional_fields_vtu.push_back(&E1_orientation);
    }

    if(pp_data.output_data.write_E2_orientation)
    {
      E2_orientation.evaluate(solution);
      additional_fields_vtu.push_back(&E2_orientation);
    }

    if(pp_data.output_data.write_traction_local_full)
    {
      traction_local_full.evaluate(solution);
      additional_fields_vtu.push_back(&traction_local_full);
    }

    if(pp_data.output_data.write_traction_local_normal)
    {
      traction_local_normal.evaluate(solution);
      additional_fields_vtu.push_back(&traction_local_normal);
    }

    if(pp_data.output_data.write_traction_local_inplane)
    {
      traction_local_inplane.evaluate(solution);
      additional_fields_vtu.push_back(&traction_local_inplane);
    }

    output_generator.evaluate(solution,
                              additional_fields_vtu,
                              time,
                              Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculate error
   */
  if(error_calculator.time_control.needs_evaluation(time, time_step_number))
    error_calculator.evaluate(solution, time, Utilities::is_unsteady_timestep(time_step_number));
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_derived_fields()
{
  // displacement magnitude
  if(pp_data.output_data.write_displacement_magnitude)
  {
    displacement_magnitude.type        = SolutionFieldType::scalar;
    displacement_magnitude.name        = "displacement_magnitude";
    displacement_magnitude.dof_handler = &pde_operator->get_dof_handler_scalar_postprocessing();
    displacement_magnitude.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar_postprocessing(dst);
    };
    displacement_magnitude.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                          const VectorType & src_vector_valued) {
      pde_operator->compute_displacement_magnitude(dst_scalar_valued, src_vector_valued);
    };

    displacement_magnitude.reinit();
  }

  // Jacobian of the displacement field
  if(pp_data.output_data.write_displacement_jacobian)
  {
    displacement_jacobian.type        = SolutionFieldType::scalar;
    displacement_jacobian.name        = "displacement_jacobian";
    displacement_jacobian.dof_handler = &pde_operator->get_dof_handler_scalar_postprocessing();
    displacement_jacobian.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar_postprocessing(dst);
    };
    displacement_jacobian.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                         const VectorType & src_vector_valued) {
      pde_operator->compute_displacement_jacobian(dst_scalar_valued, src_vector_valued);
    };

    displacement_jacobian.reinit();
  }

  // Maximum principal stress
  if(pp_data.output_data.write_max_principal_stress)
  {
    max_principal_stress.type              = SolutionFieldType::scalar;
    max_principal_stress.name              = "max_principal_stress";
    max_principal_stress.dof_handler       = &pde_operator->get_dof_handler_scalar_postprocessing();
    max_principal_stress.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar_postprocessing(dst);
    };
    max_principal_stress.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                        const VectorType & src_vector_valued) {
      pde_operator->compute_max_principal_stress(dst_scalar_valued, src_vector_valued);
    };

    max_principal_stress.reinit();
  }

  // Material orientation E1
  if(pp_data.output_data.write_E1_orientation)
  {
    E1_orientation.type              = SolutionFieldType::vector;
    E1_orientation.name              = "E1_orientation";
    E1_orientation.dof_handler       = &pde_operator->get_dof_handler_vector_postprocessing();
    E1_orientation.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_vector_postprocessing(dst);
    };
    E1_orientation.recompute_solution_field = [&](VectorType & dst_vector_postprocessing_valued,
                                                  const VectorType & src_vector_valued) {
      pde_operator->compute_E1_orientation(dst_vector_postprocessing_valued, src_vector_valued);
    };

    E1_orientation.reinit();
  }

  // Material orientation E2
  if(pp_data.output_data.write_E2_orientation)
  {
    E2_orientation.type              = SolutionFieldType::vector;
    E2_orientation.name              = "E2_orientation";
    E2_orientation.dof_handler       = &pde_operator->get_dof_handler_vector_postprocessing();
    E2_orientation.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_vector_postprocessing(dst);
    };
    E2_orientation.recompute_solution_field = [&](VectorType & dst_vector_postprocessing_valued,
                                                  const VectorType & src_vector_valued) {
      pde_operator->compute_E2_orientation(dst_vector_postprocessing_valued, src_vector_valued);
    };

    E2_orientation.reinit();
  }

  // Stress in local coordinates: full
  if(pp_data.output_data.write_traction_local_full)
  {
    traction_local_full.type              = SolutionFieldType::vector;
    traction_local_full.name              = "traction_local_full";
    traction_local_full.dof_handler       = &pde_operator->get_dof_handler_vector_postprocessing();
    traction_local_full.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_vector_postprocessing(dst);
    };
    traction_local_full.recompute_solution_field =
      [&](VectorType & dst_vector_postprocessing_valued, const VectorType & src_vector_valued) {
        pde_operator->compute_traction_local_full(dst_vector_postprocessing_valued,
                                                  src_vector_valued);
      };

    traction_local_full.reinit();
  }

  // Stress in local coordinates: normal
  if(pp_data.output_data.write_traction_local_normal)
  {
    traction_local_normal.type        = SolutionFieldType::vector;
    traction_local_normal.name        = "traction_local_normal";
    traction_local_normal.dof_handler = &pde_operator->get_dof_handler_vector_postprocessing();
    traction_local_normal.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_vector_postprocessing(dst);
    };
    traction_local_normal.recompute_solution_field =
      [&](VectorType & dst_vector_postprocessing_valued, const VectorType & src_vector_valued) {
        pde_operator->compute_traction_local_normal(dst_vector_postprocessing_valued,
                                                    src_vector_valued);
      };

    traction_local_normal.reinit();
  }

  // Stress in local coordinates: in-plane
  if(pp_data.output_data.write_traction_local_inplane)
  {
    traction_local_inplane.type        = SolutionFieldType::vector;
    traction_local_inplane.name        = "traction_local_inplane";
    traction_local_inplane.dof_handler = &pde_operator->get_dof_handler_vector_postprocessing();
    traction_local_inplane.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_vector_postprocessing(dst);
    };
    traction_local_inplane.recompute_solution_field =
      [&](VectorType & dst_vector_postprocessing_valued, const VectorType & src_vector_valued) {
        pde_operator->compute_traction_local_inplane(dst_vector_postprocessing_valued,
                                                     src_vector_valued);
      };

    traction_local_inplane.reinit();
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::invalidate_derived_fields()
{
  displacement_magnitude.invalidate();
  displacement_jacobian.invalidate();
  max_principal_stress.invalidate();
  E1_orientation.invalidate();
  E2_orientation.invalidate();
  traction_local_full.invalidate();
  traction_local_normal.invalidate();
  traction_local_inplane.invalidate();
}

template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace Structure
} // namespace ExaDG
