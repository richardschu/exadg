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

// ExaDG
#include <exadg/time_integration/restart_data.h>

namespace ExaDG
{
void
DeserializationParameters::print(dealii::ConditionalOStream const & pcout) const
{
  pcout << "  Deserialization parameters:" << std::endl;
  print_parameter(pcout, "Polynomial degree `degree`", degree);
  print_parameter(pcout, "Polynomial degree `degree_u`", degree_u);
  print_parameter(pcout, "Polynomial degree `degree_p`", degree_p);
  print_parameter(pcout, "Mapping degree", mapping_degree);
  print_parameter(pcout, "Consider mapping", consider_mapping_write);
  print_parameter(pcout, "Triangulation type", triangulation_type);
  print_parameter(pcout, "Spatial discretization", spatial_discretization);
  print_parameter(pcout, "Number of vectors `n_vectors_u`", n_vectors_u);
  print_parameter(pcout, "Number of vectors `n_vectors_p`", n_vectors_p);
}

void
RestartData::print(dealii::ConditionalOStream const & pcout) const
{
  pcout << "  Restart:" << std::endl;
  print_parameter(pcout, "Write restart", write_restart);

  if(write_restart == true)
  {
    print_parameter(pcout, "Interval physical time", interval_time);
    print_parameter(pcout, "Interval physical time window start", interval_time_start);
    print_parameter(pcout, "Interval physical time window end", interval_time_end);
    print_parameter(pcout, "Interval wall time", interval_wall_time);
    print_parameter(pcout, "Interval time steps", interval_time_steps);
    print_parameter(pcout, "Directory coarse triangulation", directory_coarse_triangulation);
    print_parameter(pcout, "Directory", directory);
    print_parameter(pcout, "Filename", filename);
  }
}

bool
RestartData::do_restart(double const           wall_time,
                        double const           time,
                        types::time_step const time_step_number,
                        bool const             reset_counter) const
{
  // After a restart, the counter is reset to 1, but time = current_time - start time != 0 after a
  // restart. Hence, we have to explicitly reset the counter in that case. There is nothing to do
  // if the restart is controlled by the wall time or the time_step_number because these
  // variables are reinitialized after a restart anyway.
  if(reset_counter)
  {
    counter += int((time + 1.e-10) / interval_time);
  }

  bool const trigger_write_base = wall_time > interval_wall_time * counter or
                                  time > interval_time * counter or
                                  time_step_number > interval_time_steps * counter;

  // Additionally use the physical time window.
  bool const trigger_write =
    trigger_write_base and time >= interval_time_start and time <= interval_time_end;

  if(trigger_write)
  {
    ++counter;
  }

  return trigger_write;
}

} // namespace ExaDG
