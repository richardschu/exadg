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

/*
 * Tests the treatment of inadmissible deformation states with J = det(F) <= tol
 * in the integration points, see InvalidJacobianTreatment. For admissible and
 * inadmissible deformation gradients F, compute_modified_F_Jm1(),
 * compute_modified_F() and compute_modified_Jm1() are evaluated for all
 * variants. Printed are the returned J, the determinant of the returned F and
 * the shift c if the returned F equals F + c I. The variants modifying F and J
 * in place return J = tol, the identity shift variants additionally det(F) =
 * tol up to the tolerance of the root finding, the others return F and J
 * unmodified.
 */

// C/C++
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

// ExaDG
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/user_interface/enum_types.h>
#include <exadg/utilities/enum_utilities.h>

namespace ExaDG
{
using Number = double;
using scalar = dealii::VectorizedArray<Number>;

template<int dim>
using tensor = dealii::Tensor<2, dim, scalar>;

template<int dim>
using matrix = dealii::Tensor<2, dim, Number>;

// Deformation gradient F with the given entries in all lanes.
template<int dim>
tensor<dim>
vectorize(matrix<dim> const & F_in)
{
  tensor<dim> F;
  for(unsigned int i = 0; i < dim; ++i)
    for(unsigned int j = 0; j < dim; ++j)
      F[i][j] = dealii::make_vectorized_array(F_in[i][j]);
  return F;
}

// Returns the shift c if F_out = F_in + c I, and NaN otherwise.
template<int dim>
Number
identity_shift(tensor<dim> const & F_out, tensor<dim> const & F_in)
{
  Number const c = F_out[0][0][0] - F_in[0][0][0];
  for(unsigned int i = 0; i < dim; ++i)
    for(unsigned int j = 0; j < dim; ++j)
    {
      Number const expected = (i == j) ? c : 0.0;
      if(std::abs(F_out[i][j][0] - F_in[i][j][0] - expected) > 1e-12)
        return std::numeric_limits<Number>::quiet_NaN();
    }
  // Avoid printing round-off from F - I + I.
  return std::abs(c) < 1e-12 ? 0.0 : c;
}

template<int dim, Structure::InvalidJacobianTreatment invalid_jacobian_treatment>
void
test_variant(tensor<dim> const & F_in)
{
  tensor<dim> gradient_displacement = F_in;
  for(unsigned int d = 0; d < dim; ++d)
    gradient_displacement[d][d] -= 1.0;

  auto const [F, Jm1] = Structure::
    compute_modified_F_Jm1<dim, Number, invalid_jacobian_treatment, false /* stable_formulation */>(
      gradient_displacement);

  // The variants returning only F or only J have to be consistent.
  tensor<dim> const F_only = Structure::
    compute_modified_F<dim, Number, invalid_jacobian_treatment, false /* stable_formulation */>(
      gradient_displacement);
  scalar const Jm1_only = Structure::
    compute_modified_Jm1<dim, Number, invalid_jacobian_treatment, false /* stable_formulation */>(
      gradient_displacement);

  bool consistent = std::abs(Jm1_only[0] - Jm1[0]) < 1e-12;
  for(unsigned int i = 0; i < dim; ++i)
    for(unsigned int j = 0; j < dim; ++j)
      consistent = consistent and std::abs(F_only[i][j][0] - F[i][j][0]) < 1e-12;

  std::cout << "    " << std::left << std::setw(20)
            << Utilities::enum_to_string(invalid_jacobian_treatment) << std::right
            << " J = " << std::setw(11) << Jm1[0] + 1.0 << ", det(F) = " << std::setw(11)
            << determinant(F)[0] << ", c = " << std::setw(11) << identity_shift(F, F_in)
            << ", consistent: " << (consistent ? "yes" : "no") << "\n";
}

template<int dim>
void
test(std::string const & name, matrix<dim> const & F_in)
{
  tensor<dim> const F = vectorize(F_in);

  std::cout << "  " << name << ", J = " << determinant(F)[0] << ":\n";

  test_variant<dim, Structure::InvalidJacobianTreatment::None>(F);
  test_variant<dim, Structure::InvalidJacobianTreatment::GlobalQuasiNewton>(F);
  test_variant<dim, Structure::InvalidJacobianTreatment::LocalQuasiNewton>(F);
  test_variant<dim, Structure::InvalidJacobianTreatment::ClampJacobian>(F);
  test_variant<dim, Structure::InvalidJacobianTreatment::IdentityShiftExact>(F);
  test_variant<dim, Structure::InvalidJacobianTreatment::IdentityShiftNewton>(F);
}

template<int dim>
void
test_all()
{
  std::cout << "dim = " << dim << ", tol = " << Structure::get_J_tol<Number>() << ":\n";

  // Admissible state with shear.
  matrix<dim> F = dealii::unit_symmetric_tensor<dim, Number>();
  F[0][1]       = 0.3;
  F[1][0]       = -0.2;
  test<dim>("admissible", F);

  // 0 < J < tol, diagonal.
  F       = dealii::unit_symmetric_tensor<dim, Number>();
  F[0][0] = 0.0005;
  test<dim>("0 < J < tol, diagonal", F);

  // J < 0, diagonal.
  F       = dealii::unit_symmetric_tensor<dim, Number>();
  F[0][0] = -0.5;
  test<dim>("J < 0, diagonal", F);

  // J < 0, with shear.
  F       = dealii::unit_symmetric_tensor<dim, Number>();
  F[0][0] = 0.2;
  F[0][1] = 0.5;
  F[1][0] = 0.9;
  F[1][1] = 0.1;
  test<dim>("J < 0, with shear", F);

  // J < 0 with d det(F + x I) / dx < 0 at x = 0.
  F       = dealii::unit_symmetric_tensor<dim, Number>();
  F[0][0] = -0.5;
  F[1][1] = 0.2;
  test<dim>("J < 0, decreasing det(F + x I)", F);

  std::cout << "\n";
}

} // namespace ExaDG

int
main()
{
  std::cout << std::scientific << std::setprecision(4);

  ExaDG::test_all<2>();
  ExaDG::test_all<3>();

  return 0;
}
