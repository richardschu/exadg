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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_LINEAR_ALGEBRA_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_LINEAR_ALGEBRA_H_

// C/C++
#include <cmath>
#include <memory>
#include <vector>

// deal.II
#include <deal.II/base/exceptions.h>

namespace ExaDG
{
namespace LinearAlgebra
{
/*
 * Small dense linear algebra for problems whose size is the number of stored vectors (e.g.,
 * previous iterates or time steps), not the number of unknowns.
 */

/*
 * Own implementation of a dense square matrix of size M x M, stored row-wise and initialized with
 * zeros. The accessors `get()` and `set()` check the indices.
 */
template<typename Number>
class Matrix
{
public:
  // Constructor.
  Matrix(unsigned int const size) : M(size)
  {
    data.resize(M * M);

    init();
  }

  void
  init()
  {
    for(unsigned int i = 0; i < M; ++i)
      for(unsigned int j = 0; j < M; ++j)
        data[i * M + j] = Number(0.0);
  }

  Number
  get(unsigned int const i, unsigned int const j) const
  {
    AssertThrow(i < M and j < M, dealii::ExcMessage("Index exceeds matrix dimensions."));

    return data[i * M + j];
  }

  void
  set(Number const value, unsigned int const i, unsigned int const j)
  {
    AssertThrow(i < M and j < M, dealii::ExcMessage("Index exceeds matrix dimensions."));

    data[i * M + j] = value;
  }

private:
  // number of rows and columns of matrix
  unsigned int const  M;
  std::vector<Number> data;
};

/*
 * QR decomposition of the matrix whose columns are the vectors `Q` via modified Gram-Schmidt. On
 * output, `Q` contains the orthonormal columns and `R` the upper triangular factor, such that
 * `Q_in = Q R`. A column is dropped if its norm after orthogonalization is smaller than `eps` times
 * its initial norm, i.e., if it is (numerically) linearly dependent on the previous columns: the
 * column is set to zero and the respective column of `R` to the unit vector, which keeps `R`
 * invertible.
 */
template<typename VectorType, typename Number>
void
compute_QR_decomposition(std::vector<VectorType> & Q, Matrix<Number> & R, Number const eps = 1.e-2)
{
  for(unsigned int i = 0; i < Q.size(); ++i)
  {
    Number const norm_initial = Number(Q[i].l2_norm());

    // orthogonalize
    for(unsigned int j = 0; j < i; ++j)
    {
      Number r_ji = Q[j] * Q[i];
      R.set(r_ji, j, i);
      Q[i].add(-r_ji, Q[j]);
    }

    // normalize or drop if linear dependent
    Number r_ii = Number(Q[i].l2_norm());
    if(r_ii < eps * norm_initial)
    {
      Q[i] = 0.0;
      for(unsigned int j = 0; j < i; ++j)
        R.set(0.0, j, i);
      R.set(1.0, i, i);
    }
    else
    {
      R.set(r_ii, i, i);
      Q[i] *= 1. / r_ii;
    }
  }
}

/*
 * Solves `matrix * dst = rhs` by backward substitution, with the number of unknowns given by the
 * size of `dst`. Only the upper triangle of `matrix` is accessed, it has to have nonzero diagonal
 * entries.
 */
template<typename Number>
void
backward_substitution(Matrix<Number> const &      matrix,
                      std::vector<Number> &       dst,
                      std::vector<Number> const & rhs)
{
  int const n = dst.size();

  for(int i = n - 1; i >= 0; --i)
  {
    Number value = rhs[i];
    for(int j = i + 1; j < n; ++j)
    {
      value -= matrix.get(i, j) * dst[j];
    }

    dst[i] = value / matrix.get(i, i);
  }
}

/*
 * Same as `backward_substitution()`, but the entries of `dst` and `rhs` are vectors, i.e., solves
 * `sum_j matrix(i,j) dst[j] = rhs[i]` for all `i`. Given the QR decomposition `R = Q U`, this
 * computes the columns of `Z = Q U^{-T}` from `rhs = Q`.
 */
template<typename Number, typename VectorType>
void
backward_substitution_multiple_rhs(Matrix<Number> const &          matrix,
                                   std::vector<VectorType> &       dst,
                                   std::vector<VectorType> const & rhs)
{
  int const n = dst.size();

  for(int i = n - 1; i >= 0; --i)
  {
    VectorType value = rhs[i];
    for(int j = i + 1; j < n; ++j)
    {
      value.add(-matrix.get(i, j), dst[j]);
    }

    dst[i].equ(1.0 / matrix.get(i, i), value);
  }
}

/*
 * Applies the inverse Jacobian approximation of previous time steps in the IQN-IMVLS method to the
 * `residual`. Starting with `a = residual`, each stored time step `k`, from the newest to the
 * oldest, contributes `b += D_k Z_k^T a` and updates `a -= R_k Z_k^T a`, where `D_k` and `R_k`
 * contain the differences of iterates and residuals, and `Z_k = Q_k U_k^{-T}` stems from the QR
 * decomposition `R_k = Q_k U_k`.
 */
template<typename VectorType>
void
inv_jacobian_times_residual(VectorType &                                                  b,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & D_history,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & R_history,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & Z_history,
                            VectorType const &                                            residual)
{
  VectorType a = residual;

  // reset
  b = 0.0;

  for(int idx = Z_history.size() - 1; idx >= 0; --idx)
  {
    std::shared_ptr<std::vector<VectorType>> D = D_history[idx];
    std::shared_ptr<std::vector<VectorType>> R = R_history[idx];
    std::shared_ptr<std::vector<VectorType>> Z = Z_history[idx];

    int const                                    k = Z->size();
    std::vector<typename VectorType::value_type> Z_times_a(k, 0.0);
    for(int i = 0; i < k; ++i)
      Z_times_a[i] = (*Z)[i] * a;

    // add to b
    for(int i = 0; i < k; ++i)
      b.add(Z_times_a[i], (*D)[i]);

    // add to a
    for(int i = 0; i < k; ++i)
      a.add(-Z_times_a[i], (*R)[i]);
  }
}

/*
 * Solves the symmetric positive semi-definite system `C x = d` via the Cholesky decomposition
 * `C = L L^T`. Only the lower triangle of `C` is accessed and overwritten by `L`. The decomposition
 * stops at the first row whose pivot is not larger than `tol` times the first diagonal entry of
 * `C`, i.e., at the first unknown that is (numerically) linearly dependent on the previous ones.
 * This and all following unknowns are dropped and set to zero, such that the system is solved for
 * the first `n` unknowns only.
 *
 * On input, `x` contains the right-hand side `d`, on output the solution. Returns `n`.
 */
template<typename Number>
unsigned int
solve_cholesky(Matrix<Number> & C, std::vector<Number> & x, Number const tol)
{
  unsigned int const m = x.size();
  if(m == 0)
    return 0;

  // Cholesky decomposition, stopping at the first (numerically) linearly dependent unknown. The
  // negated comparison also stops for a NaN pivot.
  Number const tolerance_dependence = tol * C.get(0, 0);
  unsigned int n                    = 0;
  for(; n < m; ++n)
  {
    Number pivot = C.get(n, n);
    for(unsigned int k = 0; k < n; ++k)
      pivot -= C.get(n, k) * C.get(n, k);
    if(not(pivot > tolerance_dependence))
      break;
    C.set(std::sqrt(pivot), n, n);
    for(unsigned int i = n + 1; i < m; ++i)
    {
      Number value = C.get(i, n);
      for(unsigned int k = 0; k < n; ++k)
        value -= C.get(i, k) * C.get(n, k);
      C.set(value / C.get(n, n), i, n);
    }
  }

  // forward substitution L y = d and backward substitution L^T x = y for the first n unknowns
  for(unsigned int i = 0; i < n; ++i)
  {
    for(unsigned int k = 0; k < i; ++k)
      x[i] -= C.get(i, k) * x[k];
    x[i] /= C.get(i, i);
  }
  for(unsigned int i = n; i-- > 0;)
  {
    for(unsigned int k = i + 1; k < n; ++k)
      x[i] -= C.get(k, i) * x[k];
    x[i] /= C.get(i, i);
  }

  // dropped unknowns
  for(unsigned int i = n; i < m; ++i)
    x[i] = Number(0.0);

  return n;
}

} // namespace LinearAlgebra
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_LINEAR_ALGEBRA_H_ */
