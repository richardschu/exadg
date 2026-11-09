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

#ifndef EXADG_OPERATORS_QUADRATURE_H_
#define EXADG_OPERATORS_QUADRATURE_H_

// deal.II
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_simplex_p.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
/**
 * This function creates a Gauss-type quadrature rule depending on the ElementType and the number of
 * one-dimensional quadrature points.
 */
template<int dim>
std::shared_ptr<dealii::Quadrature<dim>>
create_quadrature(ElementType const & element_type, unsigned int const n_q_points_1d)
{
  std::shared_ptr<dealii::Quadrature<dim>> quadrature;
  if(element_type == ElementType::Hypercube)
  {
    quadrature = std::make_shared<dealii::QGauss<dim>>(n_q_points_1d);
  }
  else if(element_type == ElementType::Simplex)
  {
    quadrature = std::make_shared<dealii::QGaussSimplex<dim>>(n_q_points_1d);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  return quadrature;
}

/**
 * This function creates a quadrature rule with the quadrature points at the support points of the
 * continuous Lagrange element of the given degree, i.e., Gauss-Lobatto points for hypercubes and
 * the support points of FE_SimplexP for simplices. The weights of the latter are not meant for
 * integration. The face quadrature points then coincide with the support points on the face,
 * which allows to set values of constrained degrees of freedom on the boundary.
 */
template<int dim>
std::shared_ptr<dealii::Quadrature<dim>>
create_quadrature_at_support_points(ElementType const & element_type, unsigned int const degree)
{
  std::shared_ptr<dealii::Quadrature<dim>> quadrature;
  if(element_type == ElementType::Hypercube)
  {
    quadrature = std::make_shared<dealii::QGaussLobatto<dim>>(degree + 1);
  }
  else if(element_type == ElementType::Simplex)
  {
    AssertThrow(degree <= 3,
                dealii::ExcMessage("The quadrature at the support points of simplex elements is "
                                   "only supported by dealii::MatrixFree up to degree 3."));
    quadrature = std::make_shared<dealii::Quadrature<dim>>(
      dealii::FE_SimplexP<dim>(degree).get_unit_support_points());
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  return quadrature;
}

} // namespace ExaDG

#endif /* EXADG_OPERATORS_QUADRATURE_H_ */
