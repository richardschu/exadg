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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_GENERALIZED_NEWTONIAN_MODEL_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_GENERALIZED_NEWTONIAN_MODEL_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/viscosity_model_base.h>

namespace ExaDG
{
namespace IncNS
{
/*
 *  Generalized Newtonian model.
 */
template<int dim, typename Number>
class GeneralizedNewtonianModel : public ViscosityModelBase<dim, Number>
{
private:
  typedef GeneralizedNewtonianModel<dim, Number> This;
  typedef ViscosityModelBase<dim, Number>        Base;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

public:
  /**
   * Constructor.
   */
  GeneralizedNewtonianModel();

  /**
   * Destructor.
   */
  virtual ~GeneralizedNewtonianModel();

  /**
   * Initialization function.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &                matrix_free_in,
             std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
             GeneralizedNewtonianModelData const & generalized_newtonian_model_data_in,
             unsigned int const                    dof_index_velocity_in);

  /**
   * Function for *setting* the viscosity taking the viscosity stored in the viscous kernel's data
   * as a basis.
   */
  void
  set_viscosity(VectorType const & velocity) const final;

  /**
   * Function for *adding to* the viscosity taking the currently stored viscosity as a basis.
   */
  void
  add_viscosity(VectorType const & velocity) const final;

private:
  void
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const;

  void
  face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      face_range) const;

  void
  boundary_face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                                      VectorType &,
                                      VectorType const & src,
                                      Range const &      face_range) const;

  GeneralizedNewtonianModelData data;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_GENERALIZED_NEWTONIAN_MODEL_H_ \
        */
