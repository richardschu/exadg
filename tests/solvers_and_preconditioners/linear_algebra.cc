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

/*
 * Tests the small dense linear algebra in linear_algebra.h. Each check computes a relative norm by
 * inserting the result back into its definition, e.g., |Q R - V| / |V| for the QR decomposition
 * or |C x - d| / |d| for the Cholesky solver. The norms are printed directly, numdiff regards
 * values below 1e-6 as zero.
 */

// C++
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/vector.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/utilities/linear_algebra.h>

namespace ExaDG
{
template<typename Number>
using Vector = dealii::Vector<Number>;

template<typename Number>
using Matrix = LinearAlgebra::Matrix<Number>;

/**************************************************************************************/
/*                                                                                    */
/*                                      HELPERS                                       */
/*                                                                                    */
/**************************************************************************************/

template<typename Number>
std::string
type_name()
{
  return std::is_same<Number, float>::value ? "float" : "double";
}

void
print_header(std::string const & name)
{
  std::cout << std::endl << name << ":" << std::endl;
}

void
print_norm(std::string const & name, double const value)
{
  std::cout << "  " << std::left << std::setw(56) << name << " = " << std::scientific
            << std::setprecision(2) << value << std::endl;
}

template<typename Number>
void
print_values(std::string const & name, std::vector<Number> const & values)
{
  std::cout << "  " << std::left << std::setw(56) << name << (name.empty() ? "   " : " = ")
            << std::fixed << std::setprecision(4);
  for(Number const & value : values)
    std::cout << " " << std::setw(9) << std::right << value;
  std::cout << std::endl;
}

template<typename Number>
void
print_matrix(std::string const & name, Matrix<Number> const & matrix, unsigned int const size)
{
  for(unsigned int i = 0; i < size; ++i)
  {
    std::vector<Number> row(size);
    for(unsigned int j = 0; j < size; ++j)
      row[j] = matrix.get(i, j);
    print_values(i == 0 ? name : "", row);
  }
}

template<typename Number>
Vector<Number>
make_vector(std::vector<double> const & values)
{
  Vector<Number> vector(values.size());
  for(unsigned int i = 0; i < values.size(); ++i)
    vector[i] = Number(values[i]);
  return vector;
}

// sum_i coefficients[i] vectors[i]
template<typename Number>
Vector<Number>
linear_combination(std::vector<Vector<Number>> const & vectors,
                   std::vector<Number> const &         coefficients)
{
  Vector<Number> result(vectors[0].size());
  for(unsigned int i = 0; i < vectors.size(); ++i)
    result.add(coefficients[i], vectors[i]);
  return result;
}

// Frobenius norm of the matrix whose columns are the vectors
template<typename Number>
double
frobenius_norm(std::vector<Vector<Number>> const & vectors)
{
  double norm_squared = 0.0;
  for(Vector<Number> const & vector : vectors)
    norm_squared += std::pow(double(vector.l2_norm()), 2);
  return std::sqrt(norm_squared);
}

template<typename Number>
double
relative_error(std::vector<Number> const & values, std::vector<Number> const & reference)
{
  double error = 0.0, norm = 0.0;
  for(unsigned int i = 0; i < values.size(); ++i)
  {
    error += std::pow(double(values[i]) - double(reference[i]), 2);
    norm += std::pow(double(reference[i]), 2);
  }
  return std::sqrt(error / norm);
}

// |Q^T Q - I|_F over the first `n` columns
template<typename Number>
double
orthogonality_error(std::vector<Vector<Number>> const & Q, unsigned int const n)
{
  double error = 0.0;
  for(unsigned int i = 0; i < n; ++i)
    for(unsigned int j = 0; j < n; ++j)
      error += std::pow(double(Q[i] * Q[j]) - (i == j ? 1.0 : 0.0), 2);
  return std::sqrt(error);
}

// |Q R - V|_F / |V|_F over the first `n` columns
template<typename Number>
double
reconstruction_error_QR(std::vector<Vector<Number>> const & V,
                        std::vector<Vector<Number>> const & Q,
                        Matrix<Number> const &              R,
                        unsigned int const                  n)
{
  std::vector<Vector<Number>> difference(V.begin(), V.begin() + n);
  for(unsigned int i = 0; i < n; ++i)
    for(unsigned int j = 0; j <= i; ++j)
      difference[i].add(-R.get(j, i), Q[j]);
  return frobenius_norm(difference) /
         frobenius_norm(std::vector<Vector<Number>>(V.begin(), V.begin() + n));
}

// max |R(i,j)| for i > j
template<typename Number>
double
lower_triangle(Matrix<Number> const & R, unsigned int const size)
{
  double max = 0.0;
  for(unsigned int i = 0; i < size; ++i)
    for(unsigned int j = 0; j < i; ++j)
      max = std::max(max, std::abs(double(R.get(i, j))));
  return max;
}

// |C x - d| / |d| with the full symmetric matrix `C` given by its lower triangle
double
residual_symmetric(Matrix<double> const &      C,
                   std::vector<double> const & x,
                   std::vector<double> const & d)
{
  std::vector<double> Cx(x.size(), 0.0);
  for(unsigned int i = 0; i < x.size(); ++i)
    for(unsigned int j = 0; j < x.size(); ++j)
      Cx[i] += (j <= i ? C.get(i, j) : C.get(j, i)) * x[j];
  return relative_error(Cx, d);
}

// |L L^T - C|_F / |C|_F, with `L` and `C` given by their lower triangles
double
reconstruction_error_cholesky(Matrix<double> const & L,
                              Matrix<double> const & C,
                              unsigned int const     size)
{
  double error = 0.0, norm = 0.0;
  for(unsigned int i = 0; i < size; ++i)
    for(unsigned int j = 0; j < size; ++j)
    {
      double LLT = 0.0;
      for(unsigned int k = 0; k <= std::min(i, j); ++k)
        LLT += L.get(i, k) * L.get(j, k);
      double const C_ij = j <= i ? C.get(i, j) : C.get(j, i);
      error += std::pow(LLT - C_ij, 2);
      norm += std::pow(C_ij, 2);
    }
  return std::sqrt(error / norm);
}

// vectors of size 5, the first three are linearly independent
std::vector<std::vector<double>> const small_vectors = {{1.0, 2.0, 0.0, 1.0, 1.0},
                                                        {0.0, 1.0, 3.0, 1.0, -1.0},
                                                        {2.0, 0.0, 1.0, -1.0, 1.0},
                                                        {1.0, 0.0, 0.0, 0.0, 0.0}};

/**************************************************************************************/
/*                                                                                    */
/*                                       TESTS                                        */
/*                                                                                    */
/**************************************************************************************/

void
test_matrix()
{
  print_header("Matrix");

  Matrix<double> matrix(3);
  print_matrix("after construction", matrix, 3);

  matrix.set(2.5, 1, 2);
  print_values("get(1,2) after set(2.5,1,2)", std::vector<double>{matrix.get(1, 2)});

  try
  {
    matrix.get(3, 0);
    std::cout << "  get(3,0): no exception" << std::endl;
  }
  catch(dealii::ExceptionBase const &)
  {
    std::cout << "  get(3,0): exception caught" << std::endl;
  }
}

template<typename Number>
void
test_QR_decomposition()
{
  print_header("compute_QR_decomposition (" + type_name<Number>() + ")");

  // linearly independent vectors
  {
    std::vector<Vector<Number>> V;
    for(unsigned int i = 0; i < 3; ++i)
      V.push_back(make_vector<Number>(small_vectors[i]));
    std::vector<Vector<Number>> Q = V;
    Matrix<Number>              R(3);
    LinearAlgebra::compute_QR_decomposition(Q, R);

    std::cout << "  linearly independent vectors:" << std::endl;
    print_matrix("R", R, 3);
    print_norm("|Q^T Q - I|_F", orthogonality_error(Q, 3));
    print_norm("|Q R - V|_F / |V|_F", reconstruction_error_QR(V, Q, R, 3));
    print_norm("max |R(i,j)|, i > j", lower_triangle(R, 3));
  }

  // v_2 = v_0 + 2 v_1 is dropped: Q_2 = 0 and R(:,2) = e_2
  {
    std::vector<Vector<Number>> V;
    for(unsigned int i = 0; i < 2; ++i)
      V.push_back(make_vector<Number>(small_vectors[i]));
    V.push_back(V[0]);
    V[2].add(Number(2.0), V[1]);
    std::vector<Vector<Number>> Q = V;
    Matrix<Number>              R(3);
    LinearAlgebra::compute_QR_decomposition(Q, R);

    std::cout << "  linearly dependent vectors, v_2 = v_0 + 2 v_1:" << std::endl;
    print_matrix("R", R, 3);
    print_norm("|Q_2|", Q[2].l2_norm());
    print_norm("|Q^T Q - I|_F, first 2 columns", orthogonality_error(Q, 2));
    print_norm("|Q R - V|_F / |V|_F, first 2 columns", reconstruction_error_QR(V, Q, R, 2));
  }

  // v_2 = v_0 + 2 v_1 + delta w with w orthogonal to v_0, v_1, |w| = 1: v_2 is dropped if
  // delta / |v_0 + 2 v_1| is below eps = 1e-2
  for(double const ratio : {1e-3, 1e-1})
  {
    std::vector<Vector<Number>> W;
    for(unsigned int i : {0, 1, 3})
      W.push_back(make_vector<Number>(small_vectors[i]));
    Matrix<Number> R_W(3);
    LinearAlgebra::compute_QR_decomposition(W, R_W);

    std::vector<Vector<Number>> V;
    for(unsigned int i = 0; i < 2; ++i)
      V.push_back(make_vector<Number>(small_vectors[i]));
    V.push_back(V[0]);
    V[2].add(Number(2.0), V[1]);
    V[2].add(Number(ratio * V[2].l2_norm()), W[2]);
    std::vector<Vector<Number>> Q = V;
    Matrix<Number>              R(3);
    LinearAlgebra::compute_QR_decomposition(Q, R);

    std::cout << "  v_2 = v_0 + 2 v_1 + delta w, delta / |v_0 + 2 v_1| = " << std::scientific
              << std::setprecision(0) << ratio << ":" << std::endl;
    print_values("R(:,2)", std::vector<Number>{R.get(0, 2), R.get(1, 2), R.get(2, 2)});
    print_norm("|Q_2|", Q[2].l2_norm());
  }
}

template<typename Number>
void
test_backward_substitution()
{
  print_header("backward_substitution (" + type_name<Number>() + ")");

  // upper triangular matrix, the lower triangle is filled with values that must not be accessed
  Matrix<Number>                         U(3);
  std::vector<std::vector<double>> const entries = {{2.0, 1.0, -1.0},
                                                    {99.0, 3.0, 2.0},
                                                    {99.0, 99.0, 4.0}};
  for(unsigned int i = 0; i < 3; ++i)
    for(unsigned int j = 0; j < 3; ++j)
      U.set(Number(entries[i][j]), i, j);

  // x = (1, -2, 3), rhs = U x
  {
    std::vector<Number> const x_exact = {1.0, -2.0, 3.0};
    std::vector<Number> const rhs     = {-3.0, 0.0, 12.0};
    std::vector<Number>       x(3);
    LinearAlgebra::backward_substitution(U, x, rhs);
    print_values("3x3: x", x);
    print_norm("3x3: |x - x_exact| / |x_exact|", relative_error(x, x_exact));
  }

  // the size of `x` determines the number of unknowns: leading 2x2 block, x = (1, -2)
  {
    std::vector<Number> const x_exact = {1.0, -2.0};
    std::vector<Number> const rhs     = {0.0, -6.0};
    std::vector<Number>       x(2);
    LinearAlgebra::backward_substitution(U, x, rhs);
    print_values("2x2 block: x", x);
    print_norm("2x2 block: |x - x_exact| / |x_exact|", relative_error(x, x_exact));
  }
}

template<typename Number>
void
test_backward_substitution_multiple_rhs()
{
  print_header("backward_substitution_multiple_rhs (" + type_name<Number>() + ")");

  std::vector<Vector<Number>> V;
  for(unsigned int i = 0; i < 3; ++i)
    V.push_back(make_vector<Number>(small_vectors[i]));
  std::vector<Vector<Number>> Q = V;
  Matrix<Number>              R(3);
  LinearAlgebra::compute_QR_decomposition(Q, R);

  std::vector<Vector<Number>> Z(3, Vector<Number>(V[0].size()));
  LinearAlgebra::backward_substitution_multiple_rhs(R, Z, Q);

  // sum_{j >= i} R(i,j) Z_j = Q_i
  std::vector<Vector<Number>> difference = Q;
  for(unsigned int i = 0; i < 3; ++i)
    for(unsigned int j = i; j < 3; ++j)
      difference[i].add(-R.get(i, j), Z[j]);
  print_norm("|sum_j R(i,j) Z_j - Q_i|_F / |Q|_F", frobenius_norm(difference) / frobenius_norm(Q));

  // Z = Q R^{-T} and V = Q R give Z^T V = I
  double error = 0.0;
  for(unsigned int i = 0; i < 3; ++i)
    for(unsigned int j = 0; j < 3; ++j)
      error += std::pow(double(Z[i] * V[j]) - (i == j ? 1.0 : 0.0), 2);
  print_norm("|Z^T V - I|_F", std::sqrt(error));
}

/*
 * One time step of the IQN-IMVLS history: differences of iterates `D` and of residuals `R`, and
 * `Z = Q U^{-T}` from the QR decomposition `R = Q U`.
 */
struct TimeStepHistory
{
  TimeStepHistory(std::vector<std::vector<double>> const & D_entries,
                  std::vector<std::vector<double>> const & R_entries)
    : D(std::make_shared<std::vector<Vector<double>>>()),
      R(std::make_shared<std::vector<Vector<double>>>()),
      Z(std::make_shared<std::vector<Vector<double>>>())
  {
    for(unsigned int i = 0; i < D_entries.size(); ++i)
    {
      D->push_back(make_vector<double>(D_entries[i]));
      R->push_back(make_vector<double>(R_entries[i]));
    }

    std::vector<Vector<double>> Q = *R;
    Matrix<double>              U(Q.size());
    LinearAlgebra::compute_QR_decomposition(Q, U);
    Z->resize(Q.size(), Vector<double>(Q[0].size()));
    LinearAlgebra::backward_substitution_multiple_rhs(U, *Z, Q);
  }

  std::shared_ptr<std::vector<Vector<double>>> D, R, Z;
};

void
test_inv_jacobian_times_residual()
{
  print_header("inv_jacobian_times_residual");

  typedef std::vector<std::shared_ptr<std::vector<Vector<double>>>> History;

  // the residual differences of the two time steps are orthogonal to each other
  TimeStepHistory const     old_step({{2.0, 1.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0, 1.0}},
                                 {{0.0, 0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 2.0}});
  TimeStepHistory const     new_step({{1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0, 0.0}},
                                 {{1.0, 2.0, 0.0, 0.0, 0.0}, {1.0, -1.0, 0.0, 0.0, 0.0}});
  std::vector<double> const c = {2.0, -1.0};

  Vector<double> b(5);

  // empty history: b = 0
  {
    History const  empty;
    Vector<double> residual = make_vector<double>(small_vectors[0]);
    LinearAlgebra::inv_jacobian_times_residual(b, empty, empty, empty, residual);
    print_norm("empty history: |b|", b.l2_norm());
  }

  // one time step, residual = R c: b = D c
  {
    Vector<double> const residual = linear_combination(*new_step.R, c);
    Vector<double>       b_exact  = linear_combination(*new_step.D, c);
    LinearAlgebra::inv_jacobian_times_residual(
      b, History{new_step.D}, History{new_step.R}, History{new_step.Z}, residual);
    b_exact.add(-1.0, b);
    print_norm("one time step: |b - D c| / |D c|",
               b_exact.l2_norm() / linear_combination(*new_step.D, c).l2_norm());
  }

  // two time steps, the newest one is the last entry of the history
  for(bool const newest : {true, false})
  {
    TimeStepHistory const & step     = newest ? new_step : old_step;
    Vector<double> const    residual = linear_combination(*step.R, c);
    Vector<double>          b_exact  = linear_combination(*step.D, c);
    LinearAlgebra::inv_jacobian_times_residual(b,
                                               History{old_step.D, new_step.D},
                                               History{old_step.R, new_step.R},
                                               History{old_step.Z, new_step.Z},
                                               residual);
    b_exact.add(-1.0, b);
    print_norm(newest ? "two steps, r = R_new c: |b - D_new c| / |D_new c|" :
                        "two steps, r = R_old c: |b - D_old c| / |D_old c|",
               b_exact.l2_norm() / linear_combination(*step.D, c).l2_norm());
  }
}

void
test_solve_cholesky()
{
  print_header("solve_cholesky");

  double const tol = 1e-14;

  // SPD matrix with x = (1, -1, 2), the upper triangle is filled with values that must not be
  // accessed
  {
    std::vector<std::vector<double>> const entries = {{4.0, 99.0, 99.0},
                                                      {2.0, 5.0, 99.0},
                                                      {0.0, 1.0, 3.0}};
    Matrix<double>                         C(3);
    for(unsigned int i = 0; i < 3; ++i)
      for(unsigned int j = 0; j < 3; ++j)
        C.set(entries[i][j], i, j);
    Matrix<double> const      C_original = C;
    std::vector<double> const x_exact    = {1.0, -1.0, 2.0};
    std::vector<double> const d          = {2.0, -1.0, 5.0};
    std::vector<double>       x          = d;
    unsigned int const        n          = LinearAlgebra::solve_cholesky(C, x, tol);

    std::cout << "  SPD 3x3: n = " << n << std::endl;
    print_values("SPD 3x3: x", x);
    print_norm("SPD 3x3: |x - x_exact| / |x_exact|", relative_error(x, x_exact));
    print_norm("SPD 3x3: |C x - d| / |d|", residual_symmetric(C_original, x, d));
    print_norm("SPD 3x3: |L L^T - C|_F / |C|_F", reconstruction_error_cholesky(C, C_original, 3));
  }

  // Gram matrix of v_0, v_1, v_2 = v_0 + 2 v_1: v_2 is dropped, x = (1, -1, 0)
  {
    std::vector<Vector<double>> V;
    for(unsigned int i = 0; i < 2; ++i)
      V.push_back(make_vector<double>(small_vectors[i]));
    V.push_back(V[0]);
    V[2].add(2.0, V[1]);

    Matrix<double> C(3);
    for(unsigned int i = 0; i < 3; ++i)
      for(unsigned int j = 0; j <= i; ++j)
        C.set(V[i] * V[j], i, j);
    Matrix<double> const      C_original = C;
    std::vector<double> const x_exact    = {1.0, -1.0, 0.0};
    std::vector<double>       d(3, 0.0);
    for(unsigned int i = 0; i < 3; ++i)
      for(unsigned int j = 0; j < 3; ++j)
        d[i] += (j <= i ? C.get(i, j) : C.get(j, i)) * x_exact[j];
    std::vector<double> x = d;
    unsigned int const  n = LinearAlgebra::solve_cholesky(C, x, tol);

    std::cout << "  Gram matrix, v_2 = v_0 + 2 v_1: n = " << n << std::endl;
    print_values("Gram matrix: x", x);
    print_norm("Gram matrix: |x - x_exact| / |x_exact|", relative_error(x, x_exact));
    print_norm("Gram matrix: |C x - d| / |d|", residual_symmetric(C_original, x, d));
    print_norm("Gram matrix: |L L^T - C|_F / |C|_F, 2x2",
               reconstruction_error_cholesky(C, C_original, 2));
  }

  // empty system
  {
    Matrix<double>      C(0);
    std::vector<double> x;
    std::cout << "  empty system: n = " << LinearAlgebra::solve_cholesky(C, x, tol) << std::endl;
  }

  // NaN pivot in row 1: stops at n = 1, x_0 = d_0 / C(0,0)
  {
    Matrix<double> C(3);
    C.set(4.0, 0, 0);
    C.set(2.0, 1, 0);
    C.set(std::numeric_limits<double>::quiet_NaN(), 1, 1);
    C.set(1.0, 2, 1);
    C.set(3.0, 2, 2);
    std::vector<double> x = {2.0, -1.0, 5.0};
    unsigned int const  n = LinearAlgebra::solve_cholesky(C, x, tol);

    std::cout << "  NaN in C(1,1): n = " << n << std::endl;
    print_values("NaN in C(1,1): x", x);
  }
}

/*
 * Least-squares problem of the projection initial guess (see TimeIntBDF::
 * compute_initial_guess_projection()): minimize |A sum_k gamma_k x_k - b| for m = 6 vectors x_k
 * of size 1000 and the matrix A of a 1D finite difference stencil, solved via the QR decomposition
 * of the products A x_k and via the Cholesky decomposition of the normal equations.
 */
void
test_least_squares_problem()
{
  unsigned int const size = 1000, m = 6;

  auto const vmult = [&](Vector<double> & dst, Vector<double> const & src) {
    for(unsigned int i = 0; i < size; ++i)
      dst[i] = 2.5 * src[i] - (i > 0 ? src[i - 1] : 0.0) - (i + 1 < size ? src[i + 1] : 0.0);
  };

  std::vector<Vector<double>> x(m, Vector<double>(size)), products(m, Vector<double>(size));
  for(unsigned int k = 0; k < m; ++k)
  {
    for(unsigned int i = 0; i < size; ++i)
      x[k][i] = std::sin(0.01 * (i + 1) * (k + 1)) + 0.1 * std::cos(0.003 * (i + 1) * (k + 2));
    vmult(products[k], x[k]);
  }

  std::vector<double> const gamma_exact = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};

  // QR decomposition of the products, used for both right-hand sides
  std::vector<Vector<double>> Q = products;
  Matrix<double>              R(m);
  LinearAlgebra::compute_QR_decomposition(Q, R);

  for(bool const consistent : {true, false})
  {
    // b = A sum_k gamma_exact_k x_k, plus a perturbation orthogonal to the products, which keeps
    // gamma_exact as the minimizer
    Vector<double> b = linear_combination(products, gamma_exact);
    if(not consistent)
    {
      Vector<double> perturbation(size);
      for(unsigned int i = 0; i < size; ++i)
        perturbation[i] = std::cos(0.05 * i * i);
      for(unsigned int k = 0; k < m; ++k)
        perturbation.add(-(Q[k] * perturbation), Q[k]);
      b.add(0.5 * b.l2_norm() / perturbation.l2_norm(), perturbation);
    }

    print_header(std::string("least-squares problem, size 1000, m = 6, ") +
                 (consistent ? "b in range of A X" : "b = A X gamma + r, r orthogonal to A X"));

    // |A X gamma - b| / |b| and the normal equations |(A X)^T (A X gamma - b)| / (|A X|_F |b|)
    auto const print_residuals = [&](std::string const & name, std::vector<double> const & gamma) {
      Vector<double> residual = linear_combination(products, gamma);
      residual.add(-1.0, b);
      print_norm(name + ": |gamma - gamma_exact| / |gamma_exact|",
                 relative_error(gamma, gamma_exact));
      if(consistent)
        print_norm(name + ": |A X gamma - b| / |b|", residual.l2_norm() / b.l2_norm());
      double normal_equations = 0.0;
      for(unsigned int k = 0; k < m; ++k)
        normal_equations += std::pow(products[k] * residual, 2);
      print_norm(name + ": |(A X)^T (A X gamma - b)| / (|A X|_F |b|)",
                 std::sqrt(normal_equations) / (frobenius_norm(products) * b.l2_norm()));
    };

    // QR: R gamma = Q^T b
    {
      std::vector<double> rhs(m), gamma(m);
      for(unsigned int k = 0; k < m; ++k)
        rhs[k] = Q[k] * b;
      LinearAlgebra::backward_substitution(R, gamma, rhs);

      if(consistent)
      {
        print_norm("QR: |Q^T Q - I|_F", orthogonality_error(Q, m));
        print_norm("QR: |Q R - A X|_F / |A X|_F", reconstruction_error_QR(products, Q, R, m));
      }
      print_residuals("QR", gamma);
    }

    // Cholesky: C gamma = d with C = (A X)^T A X and d = (A X)^T b
    {
      Matrix<double>      C(m);
      std::vector<double> d(m);
      for(unsigned int i = 0; i < m; ++i)
      {
        for(unsigned int j = 0; j <= i; ++j)
          C.set(products[i] * products[j], i, j);
        d[i] = products[i] * b;
      }
      Matrix<double> const C_original = C;
      std::vector<double>  gamma      = d;
      unsigned int const   n          = LinearAlgebra::solve_cholesky(C, gamma, 1e-14);

      std::cout << "  Cholesky: n = " << n << std::endl;
      if(consistent)
        print_norm("Cholesky: |L L^T - C|_F / |C|_F",
                   reconstruction_error_cholesky(C, C_original, m));
      print_norm("Cholesky: |C gamma - d| / |d|", residual_symmetric(C_original, gamma, d));
      print_residuals("Cholesky", gamma);
    }
  }
}

} // namespace ExaDG

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    dealii::deallog.depth_console(0);

    ExaDG::test_matrix();

    ExaDG::test_QR_decomposition<double>();
    ExaDG::test_QR_decomposition<float>();

    ExaDG::test_backward_substitution<double>();
    ExaDG::test_backward_substitution<float>();

    ExaDG::test_backward_substitution_multiple_rhs<double>();
    ExaDG::test_backward_substitution_multiple_rhs<float>();

    ExaDG::test_inv_jacobian_times_residual();

    ExaDG::test_solve_cholesky();

    ExaDG::test_least_squares_problem();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
