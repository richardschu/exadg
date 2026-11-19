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
#include <exadg/structure/material/library/incompressible_fibrous_tissue.templates.h>

namespace ExaDG
{
namespace Structure
{
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::None,
                                           true,
                                           CacheLevel::None>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::None,
                                           true,
                                           CacheLevel::Scalars>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::None,
                                           true,
                                           CacheLevel::ScalarsAndTensors>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::None,
                                           false,
                                           CacheLevel::None>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::None,
                                           false,
                                           CacheLevel::Scalars>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::None,
                                           false,
                                           CacheLevel::ScalarsAndTensors>;

template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::ClampJacobian,
                                           true,
                                           CacheLevel::None>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::ClampJacobian,
                                           true,
                                           CacheLevel::Scalars>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::ClampJacobian,
                                           true,
                                           CacheLevel::ScalarsAndTensors>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::ClampJacobian,
                                           false,
                                           CacheLevel::None>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::ClampJacobian,
                                           false,
                                           CacheLevel::Scalars>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::ClampJacobian,
                                           false,
                                           CacheLevel::ScalarsAndTensors>;

template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::IdentityShift,
                                           true,
                                           CacheLevel::None>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::IdentityShift,
                                           true,
                                           CacheLevel::Scalars>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::IdentityShift,
                                           true,
                                           CacheLevel::ScalarsAndTensors>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::IdentityShift,
                                           false,
                                           CacheLevel::None>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::IdentityShift,
                                           false,
                                           CacheLevel::Scalars>;
template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::IdentityShift,
                                           false,
                                           CacheLevel::ScalarsAndTensors>;

template class IncompressibleFibrousTissue<2,
                                           float,
                                           InvalidJacobianTreatment::LocalQuasiNewton,
                                           false,
                                           CacheLevel::ScalarsAndTensors>;
// clang-format on

} // namespace Structure
} // namespace ExaDG
