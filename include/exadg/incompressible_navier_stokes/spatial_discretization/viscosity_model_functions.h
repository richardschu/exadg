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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_FUNCTIONS_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_FUNCTIONS_H_

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/user_interface/viscosity_model_data.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Point-wise evaluation of the variable viscosity models, i.e., turbulence
 * models and generalized Newtonian models. These functions are used by the
 * classes TurbulenceModel and GeneralizedNewtonianModel and can be called
 * elsewhere, e.g., in postprocessing, to obtain the identical viscosity.
 * Scalar values can be evaluated by copying them into all lanes of a
 * dealii::VectorizedArray.
 */
namespace ViscosityModels
{
/**
 * This function adds the turbulent eddy-viscosity to the laminar viscosity
 * by using one of the implemented models, see turbulence_model_data.
 */
template<int dim, typename Number>
void
add_turbulent_viscosity(
  dealii::VectorizedArray<Number> &                               viscosity,
  dealii::VectorizedArray<Number> const &                         filter_width,
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
  TurbulenceModelData const &                                     turbulence_model_data);

/**
 * This function computes the filter width of a cell as
 * h = V^{1/dim} / (degree + 1),
 * with the cell volume V computed via the JxW values of `fe_values`, which
 * has to be reinitialized on the cell of interest.
 */
template<int dim>
double
compute_filter_width(dealii::FEValues<dim> const & fe_values, unsigned int const degree);

/**
 * This function adds the generalized Newtonian part of the kinematic
 * viscosity, i.e., based on the shear rate. Such models can be found in,
 * e.g., Galdi et al., 2008 ("Hemodynamical Flows: Modeling, Analysis and
 * Simulation").
 * With
 * y    = sqrt(2*sym_grad_velocity : sym_grad_velocity)
 * e_oo = lower viscosity limit
 * e_0  = upper viscosity limit
 * and fitting parameters
 * k    = kappa
 * l    = lambda
 * a
 * n,
 * we have the apparent viscosity
 * viscosity = e_oo + (e_0 - e_oo) * [k + (l * y)^a]^[(n - 1) / a],
 * where the term added to `viscosity` is
 * (e_0 - e_oo) * [k + (l * y)^a]^[(n - 1) / a].
 */
template<int dim, typename Number>
void
add_generalized_newtonian_viscosity(
  dealii::VectorizedArray<Number> &                               viscosity,
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
  GeneralizedNewtonianModelData const &                           generalized_newtonian_model_data);

} // namespace ViscosityModels
} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_FUNCTIONS_H_ \
        */
