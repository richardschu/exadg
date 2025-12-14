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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>
#include <exadg/incompressible_navier_stokes/time_integration/poisson_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting_extruded.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/restart.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/time_integration/vector_handling.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
TimeIntBDFDualSplittingExtruded<dim, Number>::TimeIntBDFDualSplittingExtruded(
  std::shared_ptr<Operator>                       operator_in,
  std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in)
  : Base(operator_in, helpers_ale_in, postprocessor_in, param_in, mpi_comm_in, is_test_in),
    pde_operator(operator_in),
    velocity(this->order),
    pressure(4),
    pressure_matvec(4),
    velocity_red(4),
    velocity_matvec(4 * 2),
    iterations_pressure({0, 0}),
    iterations_projection({0, 0}),
    iterations_viscous({0, {0, 0}}),
    iterations_penalty({0, 0}),
    iterations_mass({0, 0}),
    extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                       this->param.start_with_low_order)
{
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  Base::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure Neumann bc
  extra_pressure_nbc.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

  // use this function to check the correctness of the time integrator constants
  //    std::cout << "Coefficients extrapolation scheme pressure NBC:" << std::endl;
  //    extra_pressure_nbc.print(this->pcout);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::setup_derived()
{
  Base::setup_derived();
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::get_vectors_serialization(
  std::vector<VectorType const *> & vectors_velocity,
  std::vector<VectorType const *> & vectors_pressure) const
{
  (void)vectors_pressure;
  (void)vectors_velocity;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::set_vectors_deserialization(
  std::vector<VectorType> const & vectors_velocity,
  std::vector<VectorType> const & vectors_pressure)
{
  (void)vectors_pressure;
  (void)vectors_velocity;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // velocity
  for(unsigned int i = 0; i < velocity.size(); ++i)
    pde_operator->initialize_vector_velocity(velocity[i]);
  pde_operator->initialize_vector_velocity(velocity_np);

  // pressure
  pde_operator->initialize_vector_pressure(pressure_np);

  // do test for matrix-free operator of optimized kind
  std::vector<unsigned int>          cell_vectorization_category;
  const dealii::Triangulation<dim> & tria = pde_operator->get_dof_handler_u().get_triangulation();
  if(tria.n_levels() > 2)
  {
    cell_vectorization_category.resize(tria.n_active_cells(),
                                       dealii::numbers::invalid_unsigned_int);
    unsigned int                             next_category = 0;
    std::array<std::vector<unsigned int>, 8> next_cells;
    std::vector<unsigned int>                sorted_next_cells;
    for(const auto & grandparent : tria.cell_iterators_on_level(tria.n_levels() - 3))
      if(grandparent->has_children())
      {
        for(auto & entry : next_cells)
          entry.clear();
        for(unsigned int c0 = 0; c0 < grandparent->n_children(); ++c0)
        {
          const auto parent = grandparent->child(c0);
          if(parent->has_children())
            for(unsigned int c = 0; c < parent->n_children(); ++c)
            {
              const auto cell = parent->child(c);
              Assert(cell->is_active(), dealii::ExcInternalError());
              if(cell->is_locally_owned())
              {
                unsigned int boundary_type = 0;
                for(unsigned int d = 0; d < dim; ++d)
                  if(cell->at_boundary(2 * d) ||
                     cell->neighbor(2 * d)->subdomain_id() != cell->subdomain_id())
                  {
                    boundary_type += dealii::Utilities::pow(2, d);
                  }
                next_cells[boundary_type].push_back(cell->active_cell_index());
              }
            }
        }
        unsigned int n_cells = 0;
        for(const std::vector<unsigned int> & cells : next_cells)
          n_cells += cells.size();
        sorted_next_cells.clear();
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[7].begin(),
                                 next_cells[7].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[6].begin(),
                                 next_cells[6].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[5].begin(),
                                 next_cells[5].end());
        unsigned int count_4 = 0, count_2 = 0, count_1 = 0;
        while(sorted_next_cells.size() % 8 != 0 && count_4 < next_cells[4].size())
        {
          sorted_next_cells.push_back(next_cells[4][count_4]);
          ++count_4;
        }
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[3].begin(),
                                 next_cells[3].end());
        // avoid having 9 cells of one kind that spills two 8-sized lanes,
        // so peel off one of each here
        if(sorted_next_cells.size() % 8 != 0)
          if(count_2 < next_cells[2].size())
          {
            sorted_next_cells.push_back(next_cells[2][count_2]);
            ++count_2;
          }
        if(sorted_next_cells.size() % 8 != 0)
          if(count_1 < next_cells[1].size())
          {
            sorted_next_cells.push_back(next_cells[1][count_1]);
            ++count_1;
          }
        while(sorted_next_cells.size() % 8 != 0 && count_1 < next_cells[1].size())
        {
          sorted_next_cells.push_back(next_cells[1][count_1]);
          ++count_1;
        }
        while(sorted_next_cells.size() % 8 != 0 && count_2 < next_cells[2].size())
        {
          sorted_next_cells.push_back(next_cells[2][count_2]);
          ++count_2;
        }
        // possibly move some insertions of category 0 in between to fill
        // up and avoid further exchange steps
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[2].begin() + count_2,
                                 next_cells[2].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[4].begin() + count_4,
                                 next_cells[4].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[1].begin() + count_1,
                                 next_cells[1].end());
        sorted_next_cells.insert(sorted_next_cells.end(),
                                 next_cells[0].begin(),
                                 next_cells[0].end());
        AssertThrow(sorted_next_cells.size() == n_cells,
                    dealii::ExcDimensionMismatch(sorted_next_cells.size(), n_cells));

        for(unsigned int i = 0; i < n_cells; ++next_category)
          for(unsigned int j = 0; j < 16 && i < n_cells; ++j, ++i)
            cell_vectorization_category[sorted_next_cells[i]] = next_category;
      }
  }
  op_rt = std::make_shared<RTOperator::RaviartThomasOperatorBase<dim, Number>>();
  op_rt->reinit(*pde_operator->get_mapping(),
                pde_operator->get_dof_handler_u(),
                pde_operator->get_constraint_u(),
                cell_vectorization_category,
                dealii::QGauss<1>(pde_operator->get_dof_handler_u().get_fe().degree + 1));

  op_rt->set_penalty_parameters(pde_operator->get_viscous_kernel_data().IP_factor);
  op_rt->initialize_dof_vector(solution_rt);
  op_rt->initialize_dof_vector(rhs_rt);

  op_rt_float = std::make_shared<RTOperator::RaviartThomasOperatorBase<dim, float>>();
  op_rt_float->reinit(*pde_operator->get_mapping(),
                      pde_operator->get_dof_handler_u(),
                      pde_operator->get_constraint_u(),
                      cell_vectorization_category,
                      dealii::QGauss<1>(pde_operator->get_dof_handler_u().get_fe().degree + 1));

  op_rt_float->set_penalty_parameters(pde_operator->get_viscous_kernel_data().IP_factor);

  if constexpr(dim == 3)
  {
    op_rt_float->set_parameters(1.0, 0.0);
    op_rt_float->compute_diagonal(diagonal_mass);

    preconditioner_mass.get_vector().reinit(diagonal_mass, true);
    const unsigned int local_size = diagonal_mass.locally_owned_size();
    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < local_size; ++i)
    {
      AssertThrow(diagonal_mass.local_element(i) > 1e-30, dealii::ExcInternalError());
      preconditioner_mass.get_vector().local_element(i) = 1. / diagonal_mass.local_element(i);
    }
    op_rt_float->set_parameters(0.0, 1.0);
    op_rt_float->compute_diagonal(diagonal_laplace);

    solutions_convective.resize(velocity.size());
    for(unsigned int i = 0; i < solutions_convective.size(); ++i)
      op_rt->initialize_dof_vector(solutions_convective[i]);

    for(auto & vector : velocity_red)
      op_rt_float->initialize_dof_vector(vector);
    for(auto & vector : velocity_matvec)
      op_rt_float->initialize_dof_vector(vector);
    op_rt_float->initialize_dof_vector(rhs_float);
  }

  laplace_op = std::make_shared<LaplaceOperator::LaplaceOperatorDG<dim, Number>>();
  laplace_op->reinit(*pde_operator->get_mapping(),
                     pde_operator->get_dof_handler_p(),
                     pde_operator->get_constraint_p(),
                     cell_vectorization_category,
                     dealii::QGauss<1>(pde_operator->get_dof_handler_p().get_fe().degree + 1));
  laplace_op->set_penalty_parameters(
    pde_operator->laplace_operator.get_data().kernel_data.IP_factor);

  op_rt->verify_other_cell_level_index(laplace_op->get_cell_level_index());
  op_rt->initialize_coupling_pressure(pde_operator->get_dof_handler_p().get_fe(),
                                      laplace_op->get_dof_indices());

  poisson_preconditioner = std::make_shared<LaplaceOperator::PoissonPreconditionerMG<dim, float>>(
    *pde_operator->get_mapping(),
    pde_operator->get_dof_handler_p(),
    cell_vectorization_category,
    pde_operator->get_grid().mapping_function,
    pde_operator->laplace_operator.get_data().kernel_data.IP_factor);

  for(unsigned int i = 0; i < pressure.size(); ++i)
    poisson_preconditioner->get_dg_matrix().initialize_dof_vector(pressure[i]);
  for(unsigned int i = 0; i < pressure_matvec.size(); ++i)
    poisson_preconditioner->get_dg_matrix().initialize_dof_vector(pressure_matvec[i]);
}



template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->helpers_ale->move_grid(this->get_time());

  pde_operator->prescribe_initial_conditions(velocity[0], pressure_np, this->get_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::initialize_former_multistep_dof_vectors()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
    if(this->param.ale_formulation)
      this->helpers_ale->move_grid(this->get_previous_time(i));

    pde_operator->prescribe_initial_conditions(velocity[i],
                                               pressure_np,
                                               this->get_previous_time(i));
  }
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_velocity() const
{
  return velocity[0];
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_velocity_np() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_pressure() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_pressure(const unsigned int i) const
{
  AssertThrow(i == 0, dealii::ExcNotImplemented());
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_pressure_np() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_velocity(unsigned int i) const
{
  return velocity[i];
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::set_velocity(VectorType const & velocity_in,
                                                           unsigned int const i)
{
  velocity[i] = velocity_in;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::set_pressure(VectorType const & pressure_in,
                                                           unsigned int const i)
{
  pressure[i] = pressure_in;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::do_timestep_solve()
{
  // perform the sub-steps of the dual-splitting method
  convective_step();

  pressure_step();

  if(this->param.apply_penalty_terms_in_postprocessing_step == false and
     (this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true))
    projection_step();

  viscous_step();

  if(this->param.apply_penalty_terms_in_postprocessing_step)
    penalty_step();

  // evaluate convective term once the final solution at time
  // t_{n+1} is known
  evaluate_convective_term();
}

template<typename Number, typename Number2, typename Number3>
void
extrapolate_vectors(std::vector<Number3> const &                                            factors,
                    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>> const & vectors,
                    dealii::LinearAlgebra::distributed::Vector<Number2> &                   result)
{
  unsigned int const locally_owned_size = result.locally_owned_size();
  if(factors.size() == 1)
  {
    Number const * vec_0  = vectors[0].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < locally_owned_size; ++i)
      res[i] = beta_0 * vec_0[i];
  }
  else if(factors.size() == 2)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];
    Number2 const  beta_1 = factors[1];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < locally_owned_size; ++i)
      res[i] = beta_0 * vec_0[i] + beta_1 * vec_1[i];
  }
  else if(factors.size() == 3)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number const * vec_2  = vectors[2].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];
    Number2 const  beta_1 = factors[1];
    Number2 const  beta_2 = factors[2];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < locally_owned_size; ++i)
      res[i] = beta_0 * vec_0[i] + beta_1 * vec_1[i] + beta_2 * vec_2[i];
  }
  else if(factors.size() == 4)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number const * vec_2  = vectors[2].begin();
    Number const * vec_3  = vectors[3].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];
    Number2 const  beta_1 = factors[1];
    Number2 const  beta_2 = factors[2];
    Number2 const  beta_3 = factors[3];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < locally_owned_size; ++i)
      res[i] = beta_0 * vec_0[i] + beta_1 * vec_1[i] + beta_2 * vec_2[i] + beta_3 * vec_3[i];
  }
  else
    for(unsigned int i = 0; i < locally_owned_size; ++i)
    {
      Number2 entry = factors[0] * vectors[0].local_element(i);
      for(unsigned int j = 1; j < factors.size(); ++j)
        entry += factors[j] * vectors[j].local_element(i);
      result.local_element(i) = entry;
    }
}



template<typename OperatorType, typename VectorType>
void
compute_least_squares_fit(OperatorType const &            op,
                          std::vector<VectorType> const & vectors,
                          VectorType const &              rhs,
                          VectorType &                    result)
{
  using Number = typename VectorType::value_type;
  std::vector<VectorType>    tmp(vectors.size());
  dealii::FullMatrix<double> matrix(vectors.size(), vectors.size());
  std::vector<Number>        small_vector(vectors.size());

  // This algorithm performs a Cholesky (LDLT) factorization of the normal
  // equations for the minimization problem
  // min_{alpha_i} | sum(alpha_i A x_i) - b |
  // which eventually gives the linear combination sum (alpha_i x_i)
  // minimizing the residual among the given search vectors
  unsigned int i = 0;
  for(; i < vectors.size(); ++i)
  {
    tmp[i].reinit(vectors[0], true);
    op.vmult(tmp[i], vectors[i]);

    std::array<Number *, 11> vec_ptrs = {};
    for(unsigned int j = 0; j <= i; ++j)
      vec_ptrs[j] = tmp[j].begin();
    Number const * rhs_ptr = rhs.begin();

    unsigned int constexpr n_lanes    = dealii::VectorizedArray<Number>::size();
    unsigned int constexpr n_lanes_4  = 4 * n_lanes;
    unsigned int const regular_size_4 = (vectors[0].locally_owned_size()) / n_lanes_4 * n_lanes_4;
    unsigned int const regular_size   = (vectors[0].locally_owned_size()) / n_lanes * n_lanes;

    // compute inner products in normal equations (all at once)
    std::array<dealii::VectorizedArray<Number>, 12> local_sums = {};

    unsigned int k = 0;
    for(; k < regular_size_4; k += n_lanes_4)
    {
      dealii::VectorizedArray<Number> v_k_0, v_k_1, v_k_2, v_k_3;
      v_k_0.load(vec_ptrs[i] + k);
      v_k_1.load(vec_ptrs[i] + k + n_lanes);
      v_k_2.load(vec_ptrs[i] + k + 2 * n_lanes);
      v_k_3.load(vec_ptrs[i] + k + 3 * n_lanes);
      for(unsigned int j = 0; j < i; ++j)
      {
        dealii::VectorizedArray<Number> v_j_k, tmp0;
        v_j_k.load(vec_ptrs[j] + k);
        tmp0 = v_k_0 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + n_lanes);
        tmp0 += v_k_1 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + 2 * n_lanes);
        tmp0 += v_k_2 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + 3 * n_lanes);
        tmp0 += v_k_3 * v_j_k;
        local_sums[j] += tmp0;
      }
      local_sums[i] += v_k_0 * v_k_0 + v_k_1 * v_k_1 + v_k_2 * v_k_2 + v_k_3 * v_k_3;

      dealii::VectorizedArray<Number> rhs_k, tmp0;
      rhs_k.load(rhs_ptr + k);
      tmp0 = rhs_k * v_k_0;
      rhs_k.load(rhs_ptr + k + n_lanes);
      tmp0 += rhs_k * v_k_1;
      rhs_k.load(rhs_ptr + k + 2 * n_lanes);
      tmp0 += rhs_k * v_k_2;
      rhs_k.load(rhs_ptr + k + 3 * n_lanes);
      tmp0 += rhs_k * v_k_3;
      local_sums[i + 1] += tmp0;
    }
    for(; k < regular_size; k += n_lanes)
    {
      dealii::VectorizedArray<Number> v_k;
      v_k.load(vec_ptrs[i] + k);
      for(unsigned int j = 0; j < i; ++j)
      {
        dealii::VectorizedArray<Number> v_j_k;
        v_j_k.load(vec_ptrs[j] + k);
        local_sums[j] += v_k * v_j_k;
      }
      local_sums[i] += v_k * v_k;
      dealii::VectorizedArray<Number> rhs_k;
      rhs_k.load(rhs_ptr + k);
      local_sums[i + 1] += v_k * rhs_k;
    }
    for(; k < vectors[0].locally_owned_size(); ++k)
    {
      for(unsigned int j = 0; j <= i; ++j)
        local_sums[j][k - regular_size] += vec_ptrs[i][k] * vec_ptrs[j][k];
      local_sums[i + 1][k - regular_size] += vec_ptrs[i][k] * rhs_ptr[k];
    }
    std::array<Number, 12> scalar_sums;
    for(unsigned int j = 0; j < i + 2; ++j)
      scalar_sums[j] = local_sums[j].sum();

    dealii::Utilities::MPI::sum(dealii::ArrayView<Number const>(scalar_sums.data(), i + 2),
                                vectors[0].get_mpi_communicator(),
                                dealii::ArrayView<Number>(scalar_sums.data(), i + 2));

    for(unsigned int j = 0; j <= i; ++j)
      matrix(i, j) = scalar_sums[j];

    // update row in Cholesky factorization associated to matrix of normal
    // equations using the diagonal entry D
    for(unsigned int j = 0; j < i; ++j)
    {
      double const inv_entry = matrix(i, j) / matrix(j, j);
      for(unsigned int k = j + 1; k <= i; ++k)
        matrix(i, k) -= matrix(k, j) * inv_entry;
    }
    if(matrix(i, i) < 1e-12 * matrix(0, 0) or matrix(0, 0) < 1e-30)
      break;

    // update for the right hand side (forward substitution)
    small_vector[i] = scalar_sums[i + 1];
    for(unsigned int j = 0; j < i; ++j)
      small_vector[i] -= matrix(i, j) / matrix(j, j) * small_vector[j];
  }

  // backward substitution of Cholesky factorization
  for(unsigned int s = i; s < small_vector.size(); ++s)
    small_vector[s] = 0.;
  for(int s = i - 1; s >= 0; --s)
  {
    double sum = small_vector[s];
    for(unsigned int j = s + 1; j < i; ++j)
      sum -= small_vector[j] * matrix(j, s);
    small_vector[s] = sum / matrix(s, s);
  }
  // if(dealii::Utilities::MPI::this_mpi_process(vectors[0].get_mpi_communicator()) == 0)
  //{
  //  std::cout << "extrapolate " << std::defaultfloat << std::setprecision(3) << result.size()
  //            << ": ";
  //  for(const double a : small_vector)
  //    std::cout << a << " ";
  //  if(i > 0)
  //    std::cout << "i=" << i << " " << matrix(i - 1, i - 1) / matrix(0, 0) << "   ";
  //}
  extrapolate_vectors(small_vector, vectors, result);
}

// Compute a least squares fit and return the norm of the right-hand side as
// well as the achieved residual
template<typename VectorType1, typename VectorType2, bool combine_two = false>
std::pair<double, double>
compute_least_squares_fit(std::vector<VectorType1> const & vectors_matvec,
                          VectorType2 const &              rhs,
                          std::vector<VectorType1> const & vectors,
                          VectorType2 &                    result,
                          double const                     factor_second = 1.0)
{
  AssertDimension((combine_two ? 2 : 1) * vectors.size(), vectors_matvec.size());
  const unsigned int n_vectors = vectors.size();
  using Number                 = typename VectorType1::value_type;
  using Number2                = typename VectorType2::value_type;
  dealii::FullMatrix<double> matrix(n_vectors, n_vectors);
  std::vector<double>        small_vector(n_vectors);

  // Solve the normal equations for the minimization problem
  // min_{alpha_i} | sum(alpha_i A x_i) - b |
  // for which we compute the matrix (A x_i)^T (A x_j) and rhs (A x_i)^T b
  AssertThrow(vectors.size() <= 5, dealii::ExcNotImplemented());
  std::array<const Number *, (combine_two ? 10 : 5)> vec_ptrs = {};
  for(unsigned int j = 0; j < (combine_two ? 2 * n_vectors : n_vectors); ++j)
    vec_ptrs[j] = vectors_matvec[j].begin();
  Number2 const * rhs_ptr = rhs.begin();

  unsigned int constexpr n_lanes    = dealii::VectorizedArray<double>::size();
  unsigned int constexpr n_lanes_4  = 4 * n_lanes;
  unsigned int const regular_size_4 = (vectors[0].locally_owned_size()) / n_lanes_4 * n_lanes_4;
  unsigned int const regular_size   = (vectors[0].locally_owned_size()) / n_lanes * n_lanes;

  // compute inner products in normal equations (all at once)
  dealii::ndarray<dealii::VectorizedArray<double>, 5, 6> local_sums = {};

  unsigned int k = 0;
  for(; k < regular_size_4; k += n_lanes_4)
  {
    for(unsigned int i = 0; i < n_vectors; ++i)
    {
      dealii::VectorizedArray<double> v_k_0, v_k_1, v_k_2, v_k_3, tmp;
      if(combine_two)
      {
        v_k_0.load(vec_ptrs[2 * i] + k);
        v_k_1.load(vec_ptrs[2 * i] + k + n_lanes);
        v_k_2.load(vec_ptrs[2 * i] + k + 2 * n_lanes);
        v_k_3.load(vec_ptrs[2 * i] + k + 3 * n_lanes);
        tmp.load(vec_ptrs[2 * i + 1] + k);
        v_k_0 += factor_second * tmp;
        tmp.load(vec_ptrs[2 * i + 1] + k + n_lanes);
        v_k_1 += factor_second * tmp;
        tmp.load(vec_ptrs[2 * i + 1] + k + 2 * n_lanes);
        v_k_2 += factor_second * tmp;
        tmp.load(vec_ptrs[2 * i + 1] + k + 3 * n_lanes);
        v_k_3 += factor_second * tmp;

        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k, tmp0;
          v_j_k.load(vec_ptrs[2 * j] + k);
          tmp.load(vec_ptrs[2 * j + 1] + k);
          v_j_k += factor_second * tmp;
          tmp0 = v_k_0 * v_j_k;

          v_j_k.load(vec_ptrs[2 * j] + k + n_lanes);
          tmp.load(vec_ptrs[2 * j + 1] + k + n_lanes);
          v_j_k += factor_second * tmp;
          tmp0 += v_k_1 * v_j_k;

          v_j_k.load(vec_ptrs[2 * j] + k + 2 * n_lanes);
          tmp.load(vec_ptrs[2 * j + 1] + k + 2 * n_lanes);
          v_j_k += factor_second * tmp;
          tmp0 += v_k_2 * v_j_k;

          v_j_k.load(vec_ptrs[2 * j] + k + 3 * n_lanes);
          tmp.load(vec_ptrs[2 * j + 1] + k + 3 * n_lanes);
          v_j_k += factor_second * tmp;
          tmp0 += v_k_3 * v_j_k;

          local_sums[i][j] += tmp0;
        }
      }
      else
      {
        v_k_0.load(vec_ptrs[i] + k);
        v_k_1.load(vec_ptrs[i] + k + n_lanes);
        v_k_2.load(vec_ptrs[i] + k + 2 * n_lanes);
        v_k_3.load(vec_ptrs[i] + k + 3 * n_lanes);
        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k, tmp0;
          v_j_k.load(vec_ptrs[j] + k);
          tmp0 = v_k_0 * v_j_k;
          v_j_k.load(vec_ptrs[j] + k + n_lanes);
          tmp0 += v_k_1 * v_j_k;
          v_j_k.load(vec_ptrs[j] + k + 2 * n_lanes);
          tmp0 += v_k_2 * v_j_k;
          v_j_k.load(vec_ptrs[j] + k + 3 * n_lanes);
          tmp0 += v_k_3 * v_j_k;
          local_sums[i][j] += tmp0;
        }
      }
      local_sums[i][i] += v_k_0 * v_k_0 + v_k_1 * v_k_1 + v_k_2 * v_k_2 + v_k_3 * v_k_3;

      dealii::VectorizedArray<double> rhs_k, tmp0, tmp1;
      rhs_k.load(rhs_ptr + k);
      if(i == 0)
      {
        tmp0 = rhs_k * v_k_0;
        tmp1 = rhs_k * rhs_k;
        rhs_k.load(rhs_ptr + k + n_lanes);
        tmp0 += rhs_k * v_k_1;
        tmp1 += rhs_k * rhs_k;
        rhs_k.load(rhs_ptr + k + 2 * n_lanes);
        tmp0 += rhs_k * v_k_2;
        tmp1 += rhs_k * rhs_k;
        rhs_k.load(rhs_ptr + k + 3 * n_lanes);
        tmp0 += rhs_k * v_k_3;
        tmp1 += rhs_k * rhs_k;
        local_sums[i][i + 1] += tmp0;
        local_sums[0][5] += tmp1;
      }
      else
      {
        tmp0 = rhs_k * v_k_0;
        rhs_k.load(rhs_ptr + k + n_lanes);
        tmp0 += rhs_k * v_k_1;
        rhs_k.load(rhs_ptr + k + 2 * n_lanes);
        tmp0 += rhs_k * v_k_2;
        rhs_k.load(rhs_ptr + k + 3 * n_lanes);
        tmp0 += rhs_k * v_k_3;
        local_sums[i][i + 1] += tmp0;
      }
    }
  }
  for(; k < regular_size; k += n_lanes)
  {
    dealii::VectorizedArray<double> rhs_k;
    for(unsigned int i = 0; i < n_vectors; ++i)
    {
      dealii::VectorizedArray<double> v_k, tmp;
      if(combine_two)
      {
        v_k.load(vec_ptrs[2 * i] + k);
        tmp.load(vec_ptrs[2 * i + 1] + k);
        v_k += factor_second * tmp;
        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k;
          v_j_k.load(vec_ptrs[2 * j] + k);
          tmp.load(vec_ptrs[2 * j + 1] + k);
          v_j_k += factor_second * tmp;
          local_sums[i][j] += v_k * v_j_k;
        }
      }
      else
      {
        v_k.load(vec_ptrs[i] + k);
        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k;
          v_j_k.load(vec_ptrs[j] + k);
          local_sums[i][j] += v_k * v_j_k;
        }
      }
      local_sums[i][i] += v_k * v_k;
      rhs_k.load(rhs_ptr + k);
      local_sums[i][i + 1] += v_k * rhs_k;
    }
    local_sums[0][5] += rhs_k * rhs_k;
  }
  for(; k < vectors[0].locally_owned_size(); ++k)
  {
    for(unsigned int i = 0; i < n_vectors; ++i)
    {
      if(combine_two)
      {
        const double v_i_k = vec_ptrs[2 * i][k] + factor_second * vec_ptrs[2 * i + 1][k];
        for(unsigned int j = 0; j < i; ++j)
        {
          local_sums[i][j][k - regular_size] +=
            v_i_k * (vec_ptrs[2 * j][k] + factor_second * vec_ptrs[2 * j + 1][k]);
        }
        local_sums[i][i][k - regular_size] += v_i_k * v_i_k;
        local_sums[i][i + 1][k - regular_size] += v_i_k * rhs_ptr[k];
      }
      else
      {
        for(unsigned int j = 0; j <= i; ++j)
          local_sums[i][j][k - regular_size] += vec_ptrs[i][k] * vec_ptrs[j][k];
        local_sums[i][i + 1][k - regular_size] += vec_ptrs[i][k] * rhs_ptr[k];
      }
    }
    local_sums[0][5][k - regular_size] += rhs_ptr[k] * rhs_ptr[k];
  }
  std::array<double, 21> scalar_sums;
  unsigned int           count = 0;
  for(unsigned int i = 0; i < n_vectors; ++i)
    for(unsigned int j = 0; j < i + 2; ++j, ++count)
      scalar_sums[count] = local_sums[i][j].sum();
  scalar_sums[count] = local_sums[0][5].sum();

  dealii::Utilities::MPI::sum(dealii::ArrayView<double const>(scalar_sums.data(), count + 1),
                              vectors[0].get_mpi_communicator(),
                              dealii::ArrayView<double>(scalar_sums.data(), count + 1));

  // This algorithm performs a Cholesky (LDLT) factorization of
  // which eventually gives the linear combination sum (alpha_i x_i)
  // minimizing the residual among the given search vectors
  unsigned int i = 0;
  for(unsigned int c = 0; i < n_vectors; ++i, ++c)
  {
    for(unsigned int j = 0; j <= i; ++j, ++c)
      matrix(i, j) = scalar_sums[c];

    // update row in Cholesky factorization associated to matrix of normal
    // equations using the diagonal entry D
    for(unsigned int j = 0; j < i; ++j)
    {
      double const inv_entry = matrix(i, j) / matrix(j, j);
      for(unsigned int k = j + 1; k <= i; ++k)
        matrix(i, k) -= matrix(k, j) * inv_entry;
    }
    if(matrix(i, i) < 1e-12 * matrix(0, 0) or matrix(0, 0) < 1e-30)
      break;

    // update for the right hand side (forward substitution)
    small_vector[i] = scalar_sums[c];
    for(unsigned int j = 0; j < i; ++j)
      small_vector[i] -= matrix(i, j) / matrix(j, j) * small_vector[j];
  }

  // backward substitution of Cholesky factorization
  for(unsigned int s = i; s < small_vector.size(); ++s)
    small_vector[s] = 0.;
  for(int s = i - 1; s >= 0; --s)
  {
    double sum = small_vector[s];
    for(unsigned int j = s + 1; j < i; ++j)
      sum -= small_vector[j] * matrix(j, s);
    small_vector[s] = sum / matrix(s, s);
  }

  // compute residual norm of resulting minimization problem
  double residual_norm_sqr = scalar_sums[count];
  for(unsigned int i = 0, c = 0; i < n_vectors; ++i, c += 2)
  {
    for(unsigned int j = 0; j < i; ++j, ++c)
      residual_norm_sqr += 2. * scalar_sums[c] * small_vector[i] * small_vector[j];
    residual_norm_sqr += scalar_sums[c] * small_vector[i] * small_vector[i];
    residual_norm_sqr -= 2 * scalar_sums[c + 1] * small_vector[i];
  }

  // if(dealii::Utilities::MPI::this_mpi_process(vectors[0].get_mpi_communicator()) == 0)
  //{
  //  std::cout << "extrapolate " << std::defaultfloat << std::setprecision(3) << result.size()
  //            << ": ";
  //  for(const double a : small_vector)
  //    std::cout << a << " ";
  //  if(i > 0)
  //    std::cout << "i=" << i << " " << matrix(i - 1, i - 1) / matrix(0, 0) << " "
  //              << std::sqrt(residual_norm_sqr) << "   ";
  //}
  extrapolate_vectors(small_vector, vectors, result);

  return std::make_pair(std::sqrt(scalar_sums[count]), std::sqrt(std::abs(residual_norm_sqr)));
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::convective_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute convective term and extrapolate convective term (if not Stokes equations)
  if(this->param.convective_problem())
  {
    if(this->param.ale_formulation)
    {
      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      {
        // in a general setting, we only know the boundary conditions at time t_{n+1}
        pde_operator->evaluate_convective_term(this->vec_convective_term[i],
                                               velocity[i],
                                               this->get_next_time());
      }
    }

    velocity_np.equ(-this->extra.get_beta(0), this->vec_convective_term[0]);
    for(unsigned int i = 1; i < this->vec_convective_term.size(); ++i)
      velocity_np.add(-this->extra.get_beta(i), this->vec_convective_term[i]);
  }
  else
    velocity_np = 0.0;

  // compute body force vector
  if(this->param.right_hand_side == true)
  {
    pde_operator->evaluate_add_body_force_term(velocity_np, this->get_next_time());
  }

  // apply inverse mass operator
  unsigned int n_iter_mass = 0;
  if constexpr(dim == 3)
  {
    op_rt->copy_mf_to_this_vector(velocity_np, rhs_rt);

    dealii::ReductionControl control(this->param.inverse_mass_operator.solver_data.max_iter,
                                     this->param.inverse_mass_operator.solver_data.abs_tol,
                                     this->param.inverse_mass_operator.solver_data.rel_tol);

    dealii::SolverCG<VectorType> solver_cg(control);
    op_rt->set_parameters(1.0, 0.0);
    std::vector<Number> beta(velocity.size());
    for(unsigned int i = 0; i < velocity.size(); ++i)
      beta[i] = this->extra.get_beta(i);
    extrapolate_vectors(beta, solutions_convective, solution_rt);
    solver_cg.solve(*op_rt, solution_rt, rhs_rt, preconditioner_mass);
    n_iter_mass = control.last_step();
    op_rt->copy_this_to_mf_vector(solution_rt, velocity_np);
    for(unsigned int i = solutions_convective.size() - 1; i > 0; --i)
      solutions_convective[i].swap(solutions_convective[i - 1]);
    solutions_convective[0].copy_locally_owned_data_from(solution_rt);
  }
  else
  {
    n_iter_mass = pde_operator->apply_inverse_mass_operator(velocity_np, velocity_np);
  }
  iterations_mass.first += 1;
  iterations_mass.second += n_iter_mass;

  // calculate sum (alpha_i/dt * u_i) and add to velocity_np
  for(unsigned int i = 0; i < velocity.size() - 1; ++i)
  {
    velocity_np.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
  }
  velocity_np.sadd(this->get_time_step_size() / this->bdf.get_gamma0(),
                   this->bdf.get_alpha(velocity.size() - 1) / this->bdf.get_gamma0(),
                   velocity.back());

  if(this->print_solver_info() and not(this->is_test))
  {
    if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
    {
      this->pcout << std::endl << "Convective step:";
      print_solver_info_linear(this->pcout, n_iter_mass, timer.wall_time());
    }
    else if(this->param.spatial_discretization == SpatialDiscretization::L2)
    {
      this->pcout << std::endl << "Explicit convective step:";
      print_wall_time(this->pcout, timer.wall_time());
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::evaluate_convective_term()
{
  dealii::Timer timer;
  timer.restart();

  if(this->param.convective_problem())
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      pde_operator->evaluate_convective_term(this->convective_term_np,
                                             velocity_np,
                                             this->get_next_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}



template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::pressure_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(pressure_np);
  rhs_pressure(rhs);

  const double t_rhs = timer.wall_time();

  if(false)
  {
    VectorType vec1, vec2;
    laplace_op->initialize_dof_vector(vec1);
    laplace_op->initialize_dof_vector(vec2);
    if(this->print_solver_info() and not(this->is_test))
    {
      vec1.copy_locally_owned_data_from(rhs);
      for(unsigned int t = 0; t < 2; ++t)
      {
        dealii::Timer timer2;
        for(unsigned int i = 0; i < 20; ++i)
          laplace_op->vmult(vec2, vec1);
        std::cout << std::defaultfloat << std::setprecision(4);
        Helper::print_time(2e-5 * vec2.size() / timer2.wall_time(),
                           "Mat-vec DG optim [MDoF/s]",
                           vec1.get_mpi_communicator());
      }
      for(unsigned int t = 0; t < 2; ++t)
      {
        dealii::Timer timer2;
        for(unsigned int i = 0; i < 20; ++i)
          pde_operator->laplace_operator.vmult(pressure_np, rhs);
        Helper::print_time(2e-5 * vec2.size() / timer2.wall_time(),
                           "Mat-vec DG basic [MDoF/s]",
                           vec1.get_mpi_communicator());
      }
      this->pcout << "Norm vec2: " << vec2.l2_norm() << " " << pressure_np.l2_norm() << " ";
      pressure_np -= vec2;
      this->pcout << " diff " << pressure_np.l2_norm() << std::endl;
    }
  }

  // extrapolate old solution to get a good initial estimate for the solver
  dealii::Timer             timer2;
  std::pair<double, double> extrapolate_accuracy(0., 0.);
  if(this->use_extrapolation)
  {
    poisson_preconditioner->get_dg_matrix().vmult(pressure_matvec[0], pressure[0]);
    extrapolate_accuracy = compute_least_squares_fit(pressure_matvec, rhs, pressure, pressure_np);
  }
  else
  {
    pressure_np = pressure_last_iter;
  }
  const double t_extrapol = timer2.wall_time();
  timer2.restart();

  // solve linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner_pressure_poisson and
    ((this->time_step_number - 1) %
       this->param.update_preconditioner_pressure_poisson_every_time_steps ==
     0);

  // const double rhs_norm = rhs.l2_norm();
  // VectorType tmp, tmp2;
  // tmp.reinit(pressure_np, true);
  // tmp2.reinit(pressure_np, true);
  // pde_operator->apply_laplace_operator(tmp, pressure_np);
  // const double res_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
  // if (pressure.size() > 0)
  // pde_operator->apply_laplace_operator(tmp, pressure[0]);
  // const double res2_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));

  unsigned int n_iter = 0;
  if(false)
  {
    n_iter = pde_operator->solve_pressure(pressure_np, rhs, update_preconditioner);
  }
  else
  {
    dealii::ReductionControl     control(this->param.solver_data_pressure_poisson.max_iter,
                                     this->param.solver_data_pressure_poisson.abs_tol,
                                     this->param.solver_data_pressure_poisson.rel_tol);
    dealii::SolverCG<VectorType> solver(control);
    solver.solve(*laplace_op, pressure_np, rhs, *poisson_preconditioner);
    n_iter = control.last_step();
  }
  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;
  const double t_sol = timer2.wall_time();

  // pde_operator->apply_laplace_operator(tmp, pressure_np);
  // const double res_norm4 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
  // this->pcout << "Residual norms:   " << std::setprecision(3) << rhs_norm << " " << res_norm << "
  // "
  //             << res2_norm << " " << res_norm4 << " (" << n_iter << ")" << std::endl;

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  if(this->store_solution)
    pressure_last_iter = pressure_np;

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl
                << "Pressure step prepare: " << t_rhs << "/" << t_extrapol << " s, solve " << t_sol
                << std::endl
                << "Solve pressure step (projection reduced residual from "
                << extrapolate_accuracy.first << " to " << extrapolate_accuracy.second << "):";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pressure step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::rhs_pressure(VectorType & rhs) const
{
  /*
   *  I. calculate divergence term
   */
  // homogeneous part of velocity divergence operator
  pde_operator->apply_velocity_divergence_term(rhs, velocity_np);

  rhs *= -this->bdf.get_gamma0() / this->get_time_step_size();

  // inhomogeneous parts of boundary face integrals of velocity divergence operator
  if(this->param.divu_integrated_by_parts == true and this->param.divu_use_boundary_data == true)
  {
    VectorType temp;
    temp.reinit(rhs);

    // sum alpha_i * u_i term
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      // note that the minus sign related to this term is already taken into account
      // in the function rhs() of the divergence operator
      rhs.add(this->bdf.get_alpha(i) / this->get_time_step_size(), temp);
    }

    // convective term
    if(this->param.convective_problem())
    {
      for(unsigned int i = 0; i < velocity.size(); ++i)
      {
        temp = 0.0;
        pde_operator->rhs_ppe_div_term_convective_term_add(temp, velocity[i]);
        rhs.add(this->extra.get_beta(i), temp);
      }
    }

    // body force term
    if(this->param.right_hand_side)
      pde_operator->rhs_ppe_div_term_body_forces_add(rhs, this->get_next_time());
  }

  /*
   *  II. calculate terms originating from inhomogeneous parts of boundary face integrals of Laplace
   * operator
   */

  // II.1. pressure Dirichlet boundary conditions
  pde_operator->rhs_ppe_laplace_add(rhs, this->get_next_time());

  // II.2. pressure Neumann boundary condition: body force vector
  if(this->param.right_hand_side)
  {
    pde_operator->rhs_ppe_nbc_body_force_term_add(rhs, this->get_next_time());
  }

  // II.4. viscous term of pressure Neumann boundary condition on Gamma_D:
  //       extrapolate velocity, evaluate vorticity, and subsequently evaluate boundary
  //       face integral (this is possible since pressure Neumann BC is linear in vorticity)
  if(this->param.viscous_problem())
  {
    if(this->param.order_extrapolation_pressure_nbc > 0)
    {
      VectorType velocity_extra;
      velocity_extra.reinit(velocity[0], true);
      velocity_extra.equ(this->extra_pressure_nbc.get_beta(0), velocity[0]);
      for(unsigned int i = 1; i < extra_pressure_nbc.get_order(); ++i)
      {
        velocity_extra.add(this->extra_pressure_nbc.get_beta(i), velocity[i]);
      }

      pde_operator->rhs_ppe_nbc_viscous_add(rhs, velocity_extra);
    }
  }

  // II.5. convective term of pressure Neumann boundary condition on Gamma_D:
  //       evaluate convective term and subsequently extrapolate rhs vectors
  //       (the convective term is nonlinear!)
  if(this->param.convective_problem())
  {
    if(this->param.order_extrapolation_pressure_nbc > 0)
    {
      VectorType temp(rhs);
      for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
      {
        temp = 0.0;
        pde_operator->rhs_ppe_nbc_convective_add(temp, velocity[i]);
        rhs.add(this->extra_pressure_nbc.get_beta(i), temp);
      }
    }
  }

  // special case: pressure level is undefined
  // Set mean value of rhs to zero in order to obtain a consistent linear system of equations.
  // This is really necessary for the dual-splitting scheme in contrast to the pressure-correction
  // scheme and coupled solution approach due to the Dirichlet BC prescribed for the intermediate
  // velocity field and the pressure Neumann BC in case of the dual-splitting scheme.
  if(pde_operator->is_pressure_level_undefined())
    dealii::VectorTools::subtract_mean_value(rhs);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::projection_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(velocity_np);
  rhs_projection(rhs);

  // apply inverse mass operator: this is the solution if no penalty terms are applied
  // and serves as a good initial guess for the case with penalty terms
  unsigned int const n_iter_mass = pde_operator->apply_inverse_mass_operator(velocity_np, rhs);
  iterations_mass.first += 1;
  iterations_mass.second += n_iter_mass;

  // penalty terms
  if(this->param.apply_penalty_terms_in_postprocessing_step == false and
     (this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true))
  {
    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated;
    if(this->use_extrapolation)
    {
      velocity_extrapolated.reinit(velocity[0]);
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);
    }
    else
    {
      velocity_extrapolated = velocity_projection_last_iter;
    }

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection and
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np = velocity_projection_last_iter;

    unsigned int n_iter = pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);
    iterations_projection.first += 1;
    iterations_projection.second += n_iter;

    if(this->store_solution)
      velocity_projection_last_iter = velocity_np;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve projection step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
  }
  else // no penalty terms
  {
    if(this->print_solver_info() and not(this->is_test))
    {
      if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
      {
        this->pcout << std::endl << "Projection step:";
        print_solver_info_linear(this->pcout, n_iter_mass, timer.wall_time());
      }
      else if(this->param.spatial_discretization == SpatialDiscretization::L2)
      {
        this->pcout << std::endl << "Explicit projection step:";
        print_wall_time(this->pcout, timer.wall_time());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }
  }

  this->timer_tree->insert({"Timeloop", "Projection step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::rhs_projection(VectorType & rhs) const
{
  /*
   *  I. calculate pressure gradient term
   */
  pde_operator->evaluate_pressure_gradient_term(rhs, pressure_np, this->get_next_time());

  rhs *= -this->get_time_step_size() / this->bdf.get_gamma0();

  /*
   *  II. add mass operator term
   */
  pde_operator->apply_mass_operator_add(rhs, velocity_np);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::viscous_step()
{
  dealii::Timer timer;
  timer.restart();

  // in case we need to iteratively solve a linear or nonlinear system of equations
  if(this->param.viscous_problem() or this->param.non_explicit_convective_problem())
  {
    /*
     *  Calculate the right-hand side of the linear system of equations.
     */
    op_rt->copy_mf_to_this_vector(velocity_np, solution_rt);
    op_rt->evaluate_momentum_rhs(pressure_np,
                                 solution_rt,
                                 this->bdf.get_gamma0() / this->get_time_step_size(),
                                 rhs_rt);

    const double t_rhs = timer.wall_time();

    dealii::Timer timer2;

    //   const double rhs_norm = rhs.l2_norm();
    // VectorType tmp, tmp2;
    // tmp.reinit(velocity_np, true);
    // tmp2.reinit(velocity_np, true);
    // pde_operator->momentum_operator.vmult(tmp, velocity_np);
    // const double res_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // if (velocity.size() > 0)
    //  pde_operator->momentum_operator.vmult(tmp, velocity[0]);
    // const double res2_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // compute_least_squares_fit(pde_operator->momentum_operator, velocity, rhs, tmp2);
    // pde_operator->momentum_operator.vmult(tmp, tmp2);
    // const double res_norm3 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // if (velocity_viscous_last_iter.size() > 0)
    //  pde_operator->momentum_operator.vmult(tmp, velocity_viscous_last_iter);
    // const double res_norm5 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // this->pcout << std::setprecision(6) << "Residual norms momentum:   " << rhs_norm << " " <<
    // res_norm << " " << res2_norm << " " << res_norm3 << " " << res_norm5;

    // solve linear system of equations
    std::pair<double, double> extrapolate_accuracy(0., 0.);
    unsigned int              n_iter = 0;
    if constexpr(dim == 3)
    {
      const Number factor_mass = this->get_scaling_factor_time_derivative_term();
      const Number factor_lapl = this->pde_operator->get_viscous_kernel_data().viscosity;

      op_rt_float->set_parameters(0.0, factor_lapl);
      op_rt_float->vmult(velocity_matvec[0], velocity_red[0]);
      op_rt_float->set_parameters(1.0, 0.0);
      op_rt_float->vmult(velocity_matvec[1], velocity_red[0]);
      extrapolate_accuracy = compute_least_squares_fit<VectorTypeFloat, VectorType, true>(
        velocity_matvec, rhs_rt, velocity_red, solution_rt, factor_mass);
      const double t_proj = timer2.wall_time();
      timer2.restart();

      preconditioner_viscous.get_vector().reinit(diagonal_mass, true);

      const unsigned int owned_size = diagonal_mass.locally_owned_size();
      DEAL_II_OPENMP_SIMD_PRAGMA
      for(unsigned int i = 0; i < owned_size; ++i)
        preconditioner_viscous.get_vector().local_element(i) =
          1.0 / (factor_mass * diagonal_mass.local_element(i) +
                 factor_lapl * diagonal_laplace.local_element(i));
      const double t_prec = timer2.wall_time();
      timer2.restart();

      op_rt->set_parameters(-factor_mass, -factor_lapl);
      op_rt->vmult_add(rhs_rt, solution_rt);
      rhs_float.copy_locally_owned_data_from(rhs_rt);
      const double t_residual = timer2.wall_time();
      timer2.restart();

      dealii::ReductionControl          control(this->param.solver_data_momentum.max_iter,
                                       this->param.solver_data_momentum.abs_tol,
                                       this->param.solver_data_momentum.rel_tol);
      dealii::SolverCG<VectorTypeFloat> solver_cg(control);
      velocity_red.back() = 0;
      op_rt_float->set_parameters(factor_mass, factor_lapl);
      solver_cg.solve(*op_rt_float, velocity_red.back(), rhs_float, preconditioner_viscous);
      n_iter = control.last_step();
      DEAL_II_OPENMP_SIMD_PRAGMA
      for(unsigned int i = 0; i < owned_size; ++i)
      {
        const Number u_i = solution_rt.local_element(i) + velocity_red.back().local_element(i);
        velocity_np.local_element(i)         = u_i;
        velocity_red.back().local_element(i) = u_i;
      }

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl
                    << "Viscous step prepare: " << t_rhs << "/" << t_proj << "/" << t_prec << "/"
                    << t_residual << " s, solve " << timer2.wall_time() << " s";
      }
    }
    else
      AssertThrow(false, dealii::ExcNotImplemented());

    iterations_viscous.first += 1;
    std::get<1>(iterations_viscous.second) += n_iter;

    // pde_operator->apply_momentum_operator(tmp, velocity_np);
    // const double res_norm4 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // this->pcout << " " << res_norm4 << " (" << n_iter << ")" << std::endl;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl
                  << "Solve viscous step (projection reduced residual from "
                  << extrapolate_accuracy.first << " to " << extrapolate_accuracy.second << "):";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }

    if(this->store_solution)
      velocity_viscous_last_iter = velocity_np;
  }
  else // no viscous term and no (linearly) implicit convective term, i.e. there is nothing to do in
       // this step of the dual splitting scheme
  {
    // nothing to do
    AssertThrow(this->param.equation_type == EquationType::Euler and
                  this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit,
                dealii::ExcMessage("Logical error."));
  }

  this->timer_tree->insert({"Timeloop", "Viscous step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::rhs_viscous(
  VectorType &       rhs,
  VectorType const & velocity,
  VectorType const & transport_velocity) const
{
  /*
   *  apply mass operator
   */
  pde_operator->apply_scaled_mass_operator(rhs,
                                           this->bdf.get_gamma0() / this->get_time_step_size(),
                                           velocity);
  if(this->param.apply_penalty_terms_in_postprocessing_step == true or
     (this->param.use_divergence_penalty == false and this->param.use_continuity_penalty == false))
  {
    VectorType tmp;
    tmp.reinit(rhs);
    pde_operator->evaluate_pressure_gradient_term(tmp, pressure_np, this->get_next_time());
    rhs -= tmp;
  }

  // compensate for explicit convective term taken into account in the first sub-step of the
  // dual-splitting scheme
  if(this->param.non_explicit_convective_problem())
  {
    for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      rhs.add(this->extra.get_beta(i), this->vec_convective_term[i]);
  }

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    // for a nonlinear problem, inhomogeneous contributions are taken into account when evaluating
    // the nonlinear residual
  }
  else // linear problem
  {
    // compute inhomogeneous contributions of linearly implicit convective term
    if(this->param.convective_problem() and
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::LinearlyImplicit)
    {
      pde_operator->rhs_add_convective_term(rhs, transport_velocity, this->get_next_time());
    }

    // inhomogeneous parts of boundary face integrals of viscous operator
    pde_operator->rhs_add_viscous_term(rhs, this->get_next_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::penalty_step()
{
  if(this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true)
  {
    dealii::Timer timer;
    timer.restart();

    // compute right-hand-side vector
    VectorType rhs;
    rhs.reinit(velocity_np, true);
    pde_operator->apply_mass_operator(rhs, velocity_np);

    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated;
    velocity_extrapolated.reinit(velocity_np, true);
    std::vector<Number> beta(velocity.size());
    for(unsigned int i = 0; i < velocity.size(); ++i)
      beta[i] = this->extra.get_beta(i);
    extrapolate_vectors(beta, velocity, velocity_extrapolated);

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // right-hand side term: add inhomogeneous contributions of continuity penalty operator to
    // rhs-vector if desired
    if(this->param.use_continuity_penalty and this->param.continuity_penalty_use_boundary_data)
      pde_operator->rhs_add_projection_operator(rhs, this->get_next_time());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection and
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np = velocity_projection_last_iter;
    else
      compute_least_squares_fit(*pde_operator->projection_operator, velocity, rhs, velocity_np);

    VectorType tmp;
    tmp.reinit(velocity_np, true);
    pde_operator->projection_operator->vmult(tmp, velocity_np);
    const double res_norm3 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    this->pcout << "Residual norms projection: " << rhs.l2_norm() << " " << res_norm3;

    unsigned int const n_iter =
      pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);
    pde_operator->projection_operator->vmult(tmp, velocity_np);
    const double res_norm4 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    this->pcout << " " << res_norm4 << " (" << n_iter << ")" << std::endl;

    iterations_penalty.first += 1;
    iterations_penalty.second += n_iter;

    if(this->store_solution)
      velocity_projection_last_iter = velocity_np;

    // write output
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve penalty step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }

    this->timer_tree->insert({"Timeloop", "Penalty step"}, timer.wall_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  swap_back_one_step(velocity);
  velocity[0].swap(velocity_np);

  swap_back_one_step(pressure);
  pressure[0] = pressure_np;

  swap_back_one_step(pressure_matvec);
  swap_back_one_step(velocity_red);

  // swap two steps because we keep viscous and mass vectors for viscosity
  swap_back_one_step(velocity_matvec);
  swap_back_one_step(velocity_matvec);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::solve_steady_problem()
{
  AssertThrow(false, dealii::ExcMessage("Steady solver not implemented yet."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    names = {"Convective step",
             "Pressure step",
             "Projection step",
             "Viscous step (nonlinear)",
             "Viscous step (accumulated)",
             "Viscous step (linear per nonlinear)"};

    iterations_avg.resize(6);
    iterations_avg[0] = 0.0; // explicit convective step
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<0>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
    iterations_avg[4] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);

    if(iterations_avg[3] > std::numeric_limits<double>::min())
      iterations_avg[5] = iterations_avg[4] / iterations_avg[3];
    else
      iterations_avg[5] = iterations_avg[4];
  }
  else
  {
    names = {"Convective step", "Pressure step", "Projection step", "Viscous step"};

    iterations_avg.resize(4);
    iterations_avg[0] = 0.0; // explicit convective step
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
  }

  if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    names.push_back("Mass solver");
    iterations_avg.push_back(iterations_mass.second / std::max(1., (double)iterations_mass.first));
  }

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty step");
    iterations_avg.push_back((double)iterations_penalty.second /
                             std::max(1., (double)iterations_penalty.first));
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

// instantiations

template class TimeIntBDFDualSplittingExtruded<2, float>;
template class TimeIntBDFDualSplittingExtruded<2, double>;

template class TimeIntBDFDualSplittingExtruded<3, float>;
template class TimeIntBDFDualSplittingExtruded<3, double>;

} // namespace IncNS
} // namespace ExaDG
