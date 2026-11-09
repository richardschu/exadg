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

#ifndef EXADG_STRUCTURE_USER_INTERFACE_ENUM_TYPES_H_
#define EXADG_STRUCTURE_USER_INTERFACE_ENUM_TYPES_H_

// C/C++
#include <string>

namespace ExaDG
{
namespace Structure
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  ProblemType describes which type of problem has to be solved:
 *  .) Steady: steady-state problem, i.e., no time dependence
 *  .) QuasiStatic: quasi-static problem, i.e., time dependence in loading, but no inertial terms
 *  .) Unsteady: unsteady problem, i.e., time dependence through inertial terms and loading
 *  .) InverseAnalysis: inverse analysis, i.e., given loaded state, find the material configuration
 */
enum class ProblemType
{
  Undefined,
  Steady,
  QuasiStatic,
  Unsteady,
  InverseAnalysis
};

enum class Type2D
{
  Undefined,
  PlaneStress,
  PlaneStrain
};

enum class MaterialType
{
  Undefined,
  StVenantKirchhoff,
  IncompressibleNeoHookean,
  CompressibleNeoHookean,
  IncompressibleFibrousTissue
};

/**************************************************************************************/
/*                                                                                    */
/*                                 PHYSICAL QUANTITIES                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                             TEMPORAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Treatment of inadmissible deformation states with J = det(F) <= tol (see
 *  get_J_tol()) in the nonlinear material models:
 *  .) None: J is neither checked nor modified.
 *  .) GlobalQuasiNewton: the linearization vector is only updated if the
 *     deformation field is admissible in all integration points, see
 *     NonLinearOperator::set_solution_linearization().
 *  .) LocalQuasiNewton: in integration points with an inadmissible
 *     deformation, the stored F and J of the last admissible linearization
 *     are used.
 *  .) ClampJacobian: J = tol is enforced, while F is not modified.
 *  .) IdentityShiftExact: F is modified by adding a scaled identity such that
 *     J = tol, the scaling factor is the positive root of the quadratic
 *     (dim = 2) or cubic (dim = 3) polynomial computed via a closed formula.
 *  .) IdentityShiftNewton: as IdentityShiftExact, but the root is computed via
 *     Newton's method.
 *  The last three variants modify F and J in place in the integration points,
 *  see reconstruct_admissible_F_Jm1().
 *
 *  Possible extension: enforce J = tol via an eigenvalue decomposition of F.
 */
enum class InvalidJacobianTreatment
{
  None,
  GlobalQuasiNewton,
  LocalQuasiNewton,
  ClampJacobian,
  IdentityShiftExact,
  IdentityShiftNewton
};

/*
 *  Returns true if F and J are modified in place in the integration points.
 */
constexpr bool
modify_F_and_J_in_place(InvalidJacobianTreatment const invalid_jacobian_treatment)
{
  return invalid_jacobian_treatment == InvalidJacobianTreatment::ClampJacobian or
         invalid_jacobian_treatment == InvalidJacobianTreatment::IdentityShiftExact or
         invalid_jacobian_treatment == InvalidJacobianTreatment::IdentityShiftNewton;
}

/*
 *  Data of the nonlinear material models stored in the integration points to
 *  avoid recomputing it in each evaluation, the implementation depends on the
 *  material model:
 *  .) None: nothing is stored, all quantities are computed on the fly.
 *  .) Scalars: scalar quantities are stored, e.g., J - 1, powers of J and
 *     scalar coefficients of the stress.
 *  .) ScalarsAndTensors: additionally to the scalars, tensorial quantities are
 *     stored, e.g., F, its inverse, C, its inverse, and the stress tensors.
 */
enum class CacheLevel
{
  None,
  Scalars,
  ScalarsAndTensors
};

/*
 *  Returns true if scalar quantities are stored in the integration points.
 */
constexpr bool
store_scalars(CacheLevel const cache_level)
{
  return cache_level == CacheLevel::Scalars or cache_level == CacheLevel::ScalarsAndTensors;
}

/*
 *  Returns true if tensorial quantities are stored in the integration points.
 */
constexpr bool
store_tensors(CacheLevel const cache_level)
{
  return cache_level == CacheLevel::ScalarsAndTensors;
}



/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Preconditioner type for solution of linear system of equations
 */
enum class Preconditioner
{
  None,
  PointJacobi,
  AdditiveSchwarz,
  Multigrid,
  AMG
};

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_USER_INTERFACE_ENUM_TYPES_H_ */
