/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_OPERATORS_BOUNDARY_MASS_KERNEL_H_
#define INCLUDE_OPERATORS_BOUNDARY_MASS_KERNEL_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
template<int dim, typename Number>
class BoundaryMassKernel
{
public:
  BoundaryMassKernel()
  {
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::nothing;
    flags.cell_integrate = dealii::EvaluationFlags::nothing;

    flags.face_evaluate  = dealii::EvaluationFlags::values;
    flags.face_integrate = dealii::EvaluationFlags::values;

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_default;

    flags.inner_faces = dealii::update_default;

    flags.boundary_faces = dealii::update_values | dealii::update_JxW_values;

    return flags;
  }

  /*
   * Boundary face integral including scaling factor
   */
  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
	get_boundary_mass_value(double scaling_factor, T const & value) const
  {
    return scaling_factor * value;
  }
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_BOUNDARY_MASS_KERNEL_H_ */
