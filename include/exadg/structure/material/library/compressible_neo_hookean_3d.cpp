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
#include <exadg/structure/material/library/compressible_neo_hookean.templates.h>

namespace ExaDG
{
namespace Structure
{
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::None,
                                      true,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::None,
                                      true,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::None,
                                      true,
                                      CacheLevel::ScalarsAndTensors>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::None,
                                      false,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::None,
                                      false,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::None,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;

template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::None,
                                      true,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::None,
                                      true,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::None,
                                      true,
                                      CacheLevel::ScalarsAndTensors>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::None,
                                      false,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::None,
                                      false,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::None,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;

template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      true,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      true,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      true,
                                      CacheLevel::ScalarsAndTensors>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      false,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      false,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;

template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      true,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      true,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      true,
                                      CacheLevel::ScalarsAndTensors>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      false,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      false,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::ClampJacobian,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;

template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::IdentityShift,
                                      true,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::IdentityShift,
                                      true,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::IdentityShift,
                                      true,
                                      CacheLevel::ScalarsAndTensors>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::IdentityShift,
                                      false,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::IdentityShift,
                                      false,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::IdentityShift,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;

template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::IdentityShift,
                                      true,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::IdentityShift,
                                      true,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::IdentityShift,
                                      true,
                                      CacheLevel::ScalarsAndTensors>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::IdentityShift,
                                      false,
                                      CacheLevel::None>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::IdentityShift,
                                      false,
                                      CacheLevel::Scalars>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::IdentityShift,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;

template class CompressibleNeoHookean<3,
                                      float,
                                      InvalidJacobianTreatment::LocalQuasiNewton,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;
template class CompressibleNeoHookean<3,
                                      double,
                                      InvalidJacobianTreatment::LocalQuasiNewton,
                                      false,
                                      CacheLevel::ScalarsAndTensors>;
// clang-format on

} // namespace Structure
} // namespace ExaDG
