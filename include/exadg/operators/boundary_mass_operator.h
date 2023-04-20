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

#ifndef INCLUDE_OPERATORS_BOUNDARY_MASS_OPERATOR_H_
#define INCLUDE_OPERATORS_BOUNDARY_MASS_OPERATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/boundary_mass_kernel.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
template<int dim, typename Number>
struct BoundaryMassOperatorData : public OperatorBaseData
{
  BoundaryMassOperatorData() : OperatorBaseData()
  {
  }

  std::map<dealii::types::boundary_id, std::pair<bool, Number>> ids_normal_coefficients;
};

template<int dim, typename Number, int n_components>
class BoundaryMassOperator : public OperatorBase<dim, Number, n_components>
{
public:
  typedef OperatorBase<dim, Number, n_components> Base;

  typedef typename Base::VectorType     VectorType;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  BoundaryMassOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             BoundaryMassOperatorData<dim, Number> const &     data);

  void
  set_scaling_factor(Number const & number) const;

  void
  set_ids_normal_coefficients(std::map<dealii::types::boundary_id, std::pair<bool, Number>> const & ids_normal_coefficients_in) const;

private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_boundary_integral(IntegratorFace &                   integrator_m,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const;

  void
  do_boundary_integral_continuous(IntegratorFace &                   integrator_m,
                                  OperatorType const &               operator_type,
                                  dealii::types::boundary_id const & boundary_id) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  BoundaryMassKernel<dim, Number> kernel;

  mutable double                                                        scaling_factor;
  mutable std::map<dealii::types::boundary_id, std::pair<bool, Number>> ids_normal_coefficients;
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_BOUNDARY_MASS_OPERATOR_H_ */
