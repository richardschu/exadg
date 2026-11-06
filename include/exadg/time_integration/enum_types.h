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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_TIME_INTEGRATION_ENUM_TYPES_H_
#define EXADG_TIME_INTEGRATION_ENUM_TYPES_H_

// C/C++
#include <string>

namespace ExaDG
{
enum class CFLConditionType
{
  VelocityNorm,
  VelocityComponents
};

enum class GenAlphaType
{
  Newmark,
  GenAlpha,
  HHTAlpha,
  BossakAlpha
};

/*
 * Initial guess for the iterative solvers within a time step, computed from the solutions at
 * previous time steps:
 *
 *  - `Zero`: start from zero, ignoring the previous time steps
 *  - `LastTimeStep`: start from the solution at the last time step
 *  - `Extrapolation`: extrapolate the solutions at previous time steps to the new time
 *  - `Projection`: linear combination of the solutions at previous time steps minimizing the
 *    residual of the linear system (least-squares projection, see Loehner, Projective prediction
 *    of pressure increments, https://doi.org/10.1002/cnm.743)
 */
enum class InitialGuessType
{
  Zero,
  LastTimeStep,
  Extrapolation,
  Projection
};

} // namespace ExaDG

#endif /* EXADG_TIME_INTEGRATION_ENUM_TYPES_H_ */
