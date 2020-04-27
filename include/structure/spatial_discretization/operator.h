/*
 * operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_
#define INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_

// deal.II
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

// user interface
#include "../user_interface/boundary_descriptor.h"
#include "../user_interface/field_functions.h"
#include "../user_interface/input_parameters.h"

// operators
#include "operators/body_force_operator.h"
#include "operators/linear_operator.h"
#include "operators/nonlinear_operator.h"

#include "../../operators/inverse_mass_matrix.h"
#include "../../operators/mass_matrix_operator.h"

// solvers
#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/preconditioner/preconditioner_base.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"
#include "../preconditioners/amg_preconditioner.h"

using namespace dealii;

namespace Structure
{
// forward declaration
template<int dim, typename Number>
class Operator;

/*
 * This operator provides the interface required by the non-linear Newton solver.
 * Requests to evaluate the residual for example are simply handed over to the
 * operators that implement the physics.
 */
template<int dim, typename Number>
class ResidualOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Operator<dim, Number> PDEOperator;

public:
  ResidualOperator()
    : pde_operator(nullptr), const_vector(nullptr), scaling_factor_mass(0.0), time(0.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  void
  update(VectorType const & const_vector, double const factor, double const time)
  {
    this->const_vector        = &const_vector;
    this->scaling_factor_mass = factor;
    this->time                = time;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_residual'.
   */
  void
  evaluate_residual(VectorType & dst, VectorType const & src) const
  {
    pde_operator->evaluate_nonlinear_residual(dst, src, *const_vector, scaling_factor_mass, time);
  }

private:
  PDEOperator const * pde_operator;

  VectorType const * const_vector;

  double scaling_factor_mass;
  double time;
};

/*
 * This operator implements the interface required by the non-linear Newton solver.
 * Requests to apply the linearized operator are simply handed over to the
 * operators that implement the physics.
 */
template<int dim, typename Number>
class LinearizedOperator : public dealii::Subscriptor
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Operator<dim, Number> PDEOperator;

public:
  LinearizedOperator()
    : dealii::Subscriptor(), pde_operator(nullptr), scaling_factor_mass(0.0), time(0.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'set_solution_linearization'.
   */
  void
  set_solution_linearization(VectorType const & solution_linearization) const
  {
    pde_operator->set_solution_linearization(solution_linearization);
  }

  void
  update(double const factor, double const time)
  {
    this->scaling_factor_mass = factor;
    this->time                = time;
  }

  /*
   * The implementation of linear solvers in deal.ii requires that a function called 'vmult' is
   * provided.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    pde_operator->apply_linearized_operator(dst, src, scaling_factor_mass, time);
  }

private:
  PDEOperator const * pde_operator;

  double scaling_factor_mass;
  double time;
};

template<int dim, typename Number>
class Operator : public dealii::Subscriptor
{
private:
  typedef float MultigridNumber;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

public:
  /*
   * Constructor.
   */
  Operator(parallel::TriangulationBase<dim> &             triangulation_in,
           Mapping<dim> const &                           mapping_in,
           unsigned int const &                           degree_in,
           PeriodicFaces const &                          periodic_face_pairs_in,
           std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
           std::shared_ptr<FieldFunctions<dim>> const     field_functions_in,
           std::shared_ptr<MaterialDescriptor> const      material_descriptor_in,
           InputParameters const &                        param_in,
           MPI_Comm const &                               mpi_comm_in);

  /*
   * Setup function. Initializes basic finite element components, matrix-free object, and basic
   * operators. This function does not perform the setup related to the solution of linear systems
   * of equations.
   */
  void
  setup();

  /*
   * This function initializes operators, preconditioners, and solvers related to the solution of
   * linear systems of equation required for implicit formulations.
   */
  void
  setup_solver();

  /*
   * Initialization of dof-vector.
   */
  void
  initialize_dof_vector(VectorType & src) const;

  /*
   * Prescribe initial conditions using a specified initial solution function.
   */
  void
  prescribe_initial_displacement(VectorType & displacement, double const time) const;

  void
  prescribe_initial_velocity(VectorType & velocity, double const time) const;

  void
  compute_initial_acceleration(VectorType &       acceleration,
                               VectorType const & displacement,
                               double const       time) const;

  void
  apply_mass_matrix(VectorType & dst, VectorType const & src) const;

  /*
   * This function calculates the right-hand side of the linear system
   * of equations for linear elasticity problems.
   */
  void
  compute_rhs_linear(VectorType & dst, double const time) const;

  /*
   * This function evaluates the nonlinear residual which is required by
   * the Newton solver.
   */
  void
  evaluate_nonlinear_residual(VectorType &       dst,
                              VectorType const & src,
                              VectorType const & const_vector,
                              double const       factor,
                              double const       time) const;

  void
  set_solution_linearization(VectorType const & vector) const;

  void
  apply_linearized_operator(VectorType &       dst,
                            VectorType const & src,
                            double const       factor,
                            double const       time) const;

  void
  set_constrained_values_to_zero(VectorType & vector) const;

  /*
   * This function solves the (non-)linear system of equations.
   */
  std::tuple<unsigned int, unsigned int>
  solve_nonlinear(VectorType &       sol,
                  VectorType const & rhs,
                  double const       factor,
                  double const       time,
                  bool const         update_preconditioner);

  unsigned int
  solve_linear(VectorType & sol, VectorType const & rhs, double const factor, double const time);

  /*
   * Setters and getters.
   */
  MatrixFree<dim, Number> const &
  get_matrix_free() const;

  Mapping<dim> const &
  get_mapping() const;

  DoFHandler<dim> const &
  get_dof_handler() const;

  types::global_dof_index
  get_number_of_dofs() const;

private:
  /*
   * Initializes DoFHandler.
   */
  void
  distribute_dofs();

  /*
   * Initializes MatrixFree-object.
   */
  void
  initialize_matrix_free();

  /*
   * Initializes operators.
   */
  void
  setup_operators();

  /*
   * Initializes preconditioner.
   */
  void
  initialize_preconditioner();

  /*
   * Initializes solver.
   */
  void
  initialize_solver();

  /*
   * Mapping
   */
  Mapping<dim> const & mapping;

  /*
   * Periodic boundaries.
   */
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  /*
   * User interface.
   */
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<MaterialDescriptor>      material_descriptor;

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  /*
   * MPI communicator
   */
  MPI_Comm const & mpi_comm;

  /*
   * Basic finite element ingredients.
   */
  unsigned int const degree;

  FESystem<dim>             fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraint_matrix;

  MatrixFree<dim, Number> matrix_free;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;

  /*
   * Basic operators.
   */
  BodyForceOperator<dim, Number> body_force_operator;

  LinearOperator<dim, Number>    elasticity_operator_linear;
  NonLinearOperator<dim, Number> elasticity_operator_nonlinear;
  OperatorData<dim>              operator_data;

  MassMatrixOperator<dim, dim, Number> mass;

  /*
   * Solution of nonlinear systems of equations
   */

  // operators required for Newton solver
  ResidualOperator<dim, Number>   residual_operator;
  LinearizedOperator<dim, Number> linearized_operator;

  typedef Newton::Solver<VectorType,
                         ResidualOperator<dim, Number>,
                         LinearizedOperator<dim, Number>,
                         IterativeSolverBase<VectorType>>
    NewtonSolver;

  std::shared_ptr<NewtonSolver> newton_solver;

  /*
   * Solution of linear systems of equations
   */
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> linear_solver;

  // mass matrix inversion
  std::shared_ptr<PreconditionerBase<Number>>      mass_preconditioner;
  std::shared_ptr<IterativeSolverBase<VectorType>> mass_solver;
};

} // namespace Structure

#endif
