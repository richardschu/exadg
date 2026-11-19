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
#include <exadg/grid/grid_data.h>

namespace ExaDG
{
void
GridData::print(dealii::ConditionalOStream const & pcout) const
{
  print_parameter(pcout, "Triangulation type", triangulation_type);

  print_parameter(pcout, "Element type", element_type);

  if(triangulation_type == TriangulationType::FullyDistributed)
    print_parameter(pcout, "Partitioning type (fully-distributed)", partitioning_type);

  print_parameter(pcout, "Number of global refinements", n_refine_global);

  if(not file_name.empty())
    print_parameter(pcout, "Grid file name", file_name);

  print_parameter(pcout, "Create coarse triangulations", create_coarse_triangulations);
}

} // namespace ExaDG
