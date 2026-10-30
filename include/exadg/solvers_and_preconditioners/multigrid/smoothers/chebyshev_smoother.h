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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_SMOOTHERS_CHEBYSHEV_SMOOTHER_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_SMOOTHERS_CHEBYSHEV_SMOOTHER_H_

// deal.II
#include <deal.II/lac/precondition.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/smoother_base.h>
#include <exadg/solvers_and_preconditioners/preconditioners/additive_schwarz_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
/*
 * Wrapper around an operator providing only the standard matrix-vector product
 * vmult(dst, src), i.e., hiding vmult(dst, src, before, after) with fused
 * vector operations. With this wrapper, `dealii::PreconditionChebyshev` and its
 * eigenvalue estimation (`dealii::SolverCG`) use their non-fused algorithms.
 */
template<typename Operator, typename VectorType>
class OperatorWithoutFusedOperations : public dealii::EnableObserverPointer
{
public:
  OperatorWithoutFusedOperations(Operator const & operator_in) : underlying_operator(operator_in)
  {
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    underlying_operator.vmult(dst, src);
  }

  dealii::types::global_dof_index
  m() const
  {
    return underlying_operator.m();
  }

  dealii::types::global_dof_index
  n() const
  {
    return underlying_operator.n();
  }

  typename VectorType::value_type
  el(unsigned int const i, unsigned int const j) const
  {
    return underlying_operator.el(i, j);
  }

private:
  Operator const & underlying_operator;
};

template<typename Operator, typename VectorType>
class ChebyshevSmoother : public SmootherBase<VectorType>
{
public:
  typedef dealii::PreconditionChebyshev<Operator, VectorType, dealii::DiagonalMatrix<VectorType>>
    ChebyshevPointJacobi;
  typedef dealii::PreconditionChebyshev<OperatorWithoutFusedOperations<Operator, VectorType>,
                                        VectorType,
                                        dealii::DiagonalMatrix<VectorType>>
    ChebyshevPointJacobiWithoutFusedOperations;
  typedef dealii::PreconditionChebyshev<Operator, VectorType, BlockJacobiPreconditioner<Operator>>
    ChebyshevBlockJacobi;
  typedef dealii::
    PreconditionChebyshev<Operator, VectorType, AdditiveSchwarzPreconditioner<Operator>>
      ChebyshevAdditiveSchwarz;

  ChebyshevSmoother() : underlying_operator(nullptr)
  {
  }

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
      : preconditioner(PreconditionerSmoother::PointJacobi),
        smoothing_range(20),
        degree(5),
        iterations_eigenvalue_estimation(20),
        use_fused_operations(true)
    {
    }

    // preconditioner
    PreconditionerSmoother preconditioner;

    // sets the smoothing range (range of eigenvalues to be smoothed)
    double smoothing_range;

    // degree of Chebyshev smoother
    unsigned int degree;

    // number of CG iterations for estimation of eigenvalues
    unsigned int iterations_eigenvalue_estimation;

    // use `vmult(dst, src, before, after)` of the operator with vector operations
    // fused into the matrix-vector product, when available (point Jacobi only).
    // In single precision, the fused `dealii::SolverCG` of the eigenvalue
    // estimation can break down (NaN) on small continuous multigrid levels with
    // clustered spectra, where the fused residual norm suffers from
    // cancellation and the absolute tolerance is not reached before the Krylov
    // space is exhausted. Continuous levels do not profit from fusion anyway,
    // see `OperatorBase::vmult()`.
    bool use_fused_operations;
  };

  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      if(data.use_fused_operations)
        chebyshev_point_jacobi->vmult(dst, src);
      else
        chebyshev_point_jacobi_without_fused_operations->vmult(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      chebyshev_block_jacobi->vmult(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      chebyshev_additive_schwarz->vmult(dst, src);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  step(VectorType & dst, VectorType const & src) const final
  {
    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      if(data.use_fused_operations)
        chebyshev_point_jacobi->step(dst, src);
      else
        chebyshev_point_jacobi_without_fused_operations->step(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      chebyshev_block_jacobi->step(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      chebyshev_additive_schwarz->step(dst, src);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  update() final
  {
    AssertThrow(underlying_operator != nullptr,
                dealii::ExcMessage("Pointer underlying_operator is uninitialized."));

    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      underlying_operator->calculate_inverse_diagonal(preconditioner_point_jacobi->get_vector());

      initialize_chebyshev_point_jacobi();
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      preconditioner_block_jacobi->update();
      chebyshev_block_jacobi->initialize(*underlying_operator, additional_data_block);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      preconditioner_additive_schwarz->update();
      chebyshev_additive_schwarz->initialize(*underlying_operator,
                                             additional_data_additive_schwarz);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  setup(Operator const &       operator_in,
        bool const             initialize_preconditioner,
        AdditionalData const & additional_data)
  {
    underlying_operator = &operator_in;
    data                = additional_data;

    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      preconditioner_point_jacobi = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
      if(initialize_preconditioner)
      {
        underlying_operator->calculate_inverse_diagonal(preconditioner_point_jacobi->get_vector());
      }

      additional_data_point.preconditioner      = preconditioner_point_jacobi;
      additional_data_point.smoothing_range     = data.smoothing_range;
      additional_data_point.degree              = data.degree;
      additional_data_point.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      if(data.use_fused_operations)
      {
        chebyshev_point_jacobi = std::make_shared<ChebyshevPointJacobi>();
      }
      else
      {
        operator_without_fused_operations =
          std::make_shared<OperatorWithoutFusedOperations<Operator, VectorType>>(
            *underlying_operator);

        additional_data_point_without_fused_operations.preconditioner = preconditioner_point_jacobi;
        additional_data_point_without_fused_operations.smoothing_range = data.smoothing_range;
        additional_data_point_without_fused_operations.degree          = data.degree;
        additional_data_point_without_fused_operations.eig_cg_n_iterations =
          data.iterations_eigenvalue_estimation;

        chebyshev_point_jacobi_without_fused_operations =
          std::make_shared<ChebyshevPointJacobiWithoutFusedOperations>();
      }

      if(initialize_preconditioner)
      {
        initialize_chebyshev_point_jacobi();
      }
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      preconditioner_block_jacobi =
        std::make_shared<BlockJacobiPreconditioner<Operator>>(*underlying_operator,
                                                              initialize_preconditioner);

      additional_data_block.preconditioner      = preconditioner_block_jacobi;
      additional_data_block.smoothing_range     = data.smoothing_range;
      additional_data_block.degree              = data.degree;
      additional_data_block.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      chebyshev_block_jacobi = std::make_shared<ChebyshevBlockJacobi>();

      if(initialize_preconditioner)
        chebyshev_block_jacobi->initialize(*underlying_operator, additional_data_block);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      preconditioner_additive_schwarz =
        std::make_shared<AdditiveSchwarzPreconditioner<Operator>>(*underlying_operator,
                                                                  initialize_preconditioner);

      additional_data_additive_schwarz.preconditioner      = preconditioner_additive_schwarz;
      additional_data_additive_schwarz.smoothing_range     = data.smoothing_range;
      additional_data_additive_schwarz.degree              = data.degree;
      additional_data_additive_schwarz.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      chebyshev_additive_schwarz = std::make_shared<ChebyshevAdditiveSchwarz>();

      if(initialize_preconditioner)
        chebyshev_additive_schwarz->initialize(*underlying_operator,
                                               additional_data_additive_schwarz);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

private:
  void
  initialize_chebyshev_point_jacobi()
  {
    if(data.use_fused_operations)
      chebyshev_point_jacobi->initialize(*underlying_operator, additional_data_point);
    else
      chebyshev_point_jacobi_without_fused_operations->initialize(
        *operator_without_fused_operations, additional_data_point_without_fused_operations);
  }

  Operator const * underlying_operator;
  AdditionalData   data;

  std::shared_ptr<OperatorWithoutFusedOperations<Operator, VectorType>>
    operator_without_fused_operations;

  std::shared_ptr<ChebyshevPointJacobi> chebyshev_point_jacobi;
  std::shared_ptr<ChebyshevPointJacobiWithoutFusedOperations>
                                            chebyshev_point_jacobi_without_fused_operations;
  std::shared_ptr<ChebyshevBlockJacobi>     chebyshev_block_jacobi;
  std::shared_ptr<ChebyshevAdditiveSchwarz> chebyshev_additive_schwarz;

  std::shared_ptr<dealii::DiagonalMatrix<VectorType>>      preconditioner_point_jacobi;
  std::shared_ptr<BlockJacobiPreconditioner<Operator>>     preconditioner_block_jacobi;
  std::shared_ptr<AdditiveSchwarzPreconditioner<Operator>> preconditioner_additive_schwarz;

  typename ChebyshevPointJacobi::AdditionalData additional_data_point;
  typename ChebyshevPointJacobiWithoutFusedOperations::AdditionalData
                                                    additional_data_point_without_fused_operations;
  typename ChebyshevBlockJacobi::AdditionalData     additional_data_block;
  typename ChebyshevAdditiveSchwarz::AdditionalData additional_data_additive_schwarz;
};

} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_SMOOTHERS_CHEBYSHEV_SMOOTHER_H_ */
