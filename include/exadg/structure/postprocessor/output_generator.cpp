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

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/postprocessor/write_output.h>
#include <exadg/structure/postprocessor/output_generator.h>
#include <exadg/time_integration/restart.h>
#include <exadg/utilities/create_directories.h>
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
OutputGenerator<dim, Number>::OutputGenerator(MPI_Comm const & comm) : mpi_comm(comm)
{
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::setup(dealii::DoFHandler<dim> const & dof_handler_in,
                                    dealii::Mapping<dim> const &    mapping_in,
                                    OutputData const &              output_data_in)
{
  dof_handler = &dof_handler_in;
  mapping     = &mapping_in;
  output_data = output_data_in;

  time_control.setup(output_data_in.time_control_data);


  if(output_data_in.time_control_data.is_active)
  {
    create_directories(output_data.directory, mpi_comm);

    // Visualize boundary IDs:
    // since boundary IDs typically do not change during the simulation, we only do this
    // once at the beginning of the simulation (i.e., in the setup function).
    if(output_data.write_boundary_IDs)
    {
      write_boundary_IDs(dof_handler->get_triangulation(),
                         output_data.directory,
                         output_data.filename,
                         mpi_comm);
    }

    // write surface mesh
    if(output_data.write_surface_mesh)
    {
      write_surface_mesh(dof_handler->get_triangulation(),
                         *mapping,
                         output_data.degree,
                         output_data.directory,
                         output_data.filename,
                         0,
                         mpi_comm);
    }

    // write grid
    if(output_data.write_grid)
    {
      write_grid(dof_handler->get_triangulation(),
                 *mapping,
                 output_data.degree,
                 output_data.directory,
                 output_data.filename,
                 0,
                 mpi_comm);
    }

    // processor_id
    if(output_data.write_processor_id)
    {
      dealii::GridOut grid_out;

      grid_out.write_mesh_per_processor_as_vtu(dof_handler->get_triangulation(),
                                               output_data.directory + output_data.filename +
                                                 "_processor_id");
    }

    // Write solution in triangulation to file.
    if(output_data.restart_data.write_restart)
    {
      RestartData const &         restart_data = output_data.restart_data;
      DeserializationParameters & deserialization_parameters =
        output_data.deserialization_parameters;

      AssertThrow(deserialization_parameters.degree != dealii::numbers::invalid_unsigned_int,
                  dealii::ExcMessage("Set degree in output_data."
                                     "deserialization_parameters to write the solution."));

      AssertThrow(deserialization_parameters.consider_mapping_write ==
                    restart_data.consider_mapping_write,
                  dealii::ExcMessage(
                    "Conflicting information in deserialization_parameters and restart_data"));

      AssertThrow(not deserialization_parameters.consider_mapping_write or
                    deserialization_parameters.mapping_degree !=
                      dealii::numbers::invalid_unsigned_int,
                  dealii::ExcMessage("Set mapping_degree in output_data."
                                     "deserialization_parameters to write the mapping."));

      AssertThrow(not deserialization_parameters.consider_mapping_write or
                    deserialization_parameters.mapping_degree == deserialization_parameters.degree,
                  dealii::ExcMessage("Writing the mapping is only possible if the "
                                     "DoFHandler the one used for the simulation."));

      deserialization_parameters.triangulation_type =
        get_triangulation_type(dof_handler->get_triangulation());

      // Serialization can only be triggered by `time`.
      AssertThrow(restart_data.interval_wall_time == std::numeric_limits<double>::max(),
                  dealii::ExcMessage("Serialization cannot be triggered by wall time."));

      bool const stress_qois_available = output_data.write_max_principal_stress and
                                         output_data.write_traction_local_normal and
                                         output_data.write_traction_local_inplane;

      AssertThrow((not output_data.deserialize_stress_qois) or stress_qois_available,
                  dealii::ExcMessage("Stress QoIs were not requested in output."));
    }
  }
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::evaluate(
  VectorType const &                                                       solution,
  std::vector<dealii::ObserverPointer<SolutionField<dim, Number>>> const & additional_fields,
  double const                                                             time,
  unsigned int const                                                       time_step_number)
{
  bool const unsteady = Utilities::is_unsteady_timestep(time_step_number);
  print_write_output_time(time, time_control.get_counter(), unsteady, mpi_comm);

  VectorWriter<dim, Number> vector_writer(output_data, time_control.get_counter(), mpi_comm);

  std::vector<std::string> component_names(dim, "displacement");
  std::vector<bool>        component_is_part_of_vector(dim, true);
  vector_writer.add_data_vector(solution,
                                *dof_handler,
                                component_names,
                                component_is_part_of_vector);

  vector_writer.write_aspect_ratio(*dof_handler, *mapping);

  vector_writer.add_fields(additional_fields);

  vector_writer.write_pvtu(&(*mapping));

  // Store the solution in the triangulation and serialize with serialization files.
  if(output_data.restart_data.write_restart)
  {
    // Restart can only be triggered by the `time` or `time_step_number` for the unsteady case.
    bool const trigger_unsteady = output_data.restart_data.do_restart(0.0 /* wall_time */,
                                                                      time,
                                                                      time_step_number,
                                                                      false /* reset_counter */);
    // For the steady case, we always write the restart data.
    bool const trigger_steady = Utilities::is_steady_timestep(time_step_number);

    if(trigger_unsteady or trigger_steady)
    {
      dealii::ConditionalOStream pcout(std::cout,
                                       dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);
      std::string const add_string = trigger_steady ? "" : "at time t = " + std::to_string(time);
      pcout << std::endl
            << print_horizontal_line() << std::endl
            << std::endl
            << " Writing restart file " << add_string << std::endl;

      RestartData const & restart_data = output_data.restart_data;
      std::string const   filename =
        restart_data.directory + generate_restart_filename(restart_data.filename);

      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        rename_old_restart_files(filename, restart_data.n_snapshots_keep);
      }

      // The restart header archive file is written using a single thread only.
      if(dealii::Utilities::MPI::this_mpi_process(this->mpi_comm) == 0)
      {
        std::ostringstream     oss;
        BoostOutputArchiveType oa(oss);

        // 1. time
        oa & time;

        write_restart_file(oss, filename);
      }

      // Write deserialization parameters. These do not change during the simulation, but the data
      // are small and we want to make sure to overwrite them.
      DeserializationParameters const & deserialization_parameters =
        output_data.deserialization_parameters;
      write_deserialization_parameters(mpi_comm, restart_data, deserialization_parameters);

      // Attach vectors to triangulation and serialize.
      std::vector<dealii::DoFHandler<dim> const *> dof_handlers{dof_handler.get()};
      std::vector<std::vector<VectorType const *>> vectors_per_dof_handler{{&solution}};

      // Attach the stress QoIs and `DoFHandler`s if needed.
      if(output_data.deserialize_stress_qois)
      {
        std::vector<std::string> const names_stress_qois = {"max_principal_stress",
                                                            "traction_local_normal",
                                                            "traction_local_inplane"};
        for(unsigned int i = 0; i < names_stress_qois.size(); ++i)
        {
          bool found = false;
          for(unsigned int j = 0; j < additional_fields.size(); ++j)
          {
            if(names_stress_qois[i] == additional_fields[j]->name)
            {
              // Check if `dealii::DoFHandler`s is already in list. Note that the vector space might
              // be discontinuous, leading to a new `DoFHandler`.
              found                        = false;
              unsigned int dof_handler_idx = 0;
              for(unsigned int k = 0; k < dof_handlers.size(); ++k)
              {
                if(dof_handlers[k] == &additional_fields[j]->get_dof_handler())
                {
                  found           = true;
                  dof_handler_idx = k;
                  break;
                }
              }

              if(found)
              {
                vectors_per_dof_handler[dof_handler_idx].push_back(&additional_fields[j]->get());
              }
              else
              {
                dof_handlers.push_back(&additional_fields[j]->get_dof_handler());
                vectors_per_dof_handler.push_back({&additional_fields[j]->get()});
              }

              // Name match found.
              found = true;
              if(found)
              {
                break;
              }
            }
          }
          AssertThrow(found,
                      dealii::ExcMessage("The field with name " + names_stress_qois[i] +
                                         " was not found in the additional fields."));
        }
      }

      // Optionally attach the mapping as a DoF vector.
      if(restart_data.consider_mapping_write)
      {
        store_vectors_in_triangulation_and_serialize(restart_data,
                                                     dof_handlers,
                                                     vectors_per_dof_handler,
                                                     *mapping,
                                                     dof_handler.get(),
                                                     deserialization_parameters.mapping_degree);
      }
      else
      {
        store_vectors_in_triangulation_and_serialize(restart_data,
                                                     dof_handlers,
                                                     vectors_per_dof_handler);
      }

      pcout << std::endl << " ... done!" << std::endl << print_horizontal_line() << std::endl;
    }
  }
}

template class OutputGenerator<2, float>;
template class OutputGenerator<3, float>;

template class OutputGenerator<2, double>;
template class OutputGenerator<3, double>;

} // namespace Structure
} // namespace ExaDG
