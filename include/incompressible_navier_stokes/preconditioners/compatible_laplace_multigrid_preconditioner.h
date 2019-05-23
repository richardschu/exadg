/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_

#include <vector>

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "compatible_laplace_operator.h"

namespace IncNS
{
/*
 *  Multigrid preconditioner for compatible Laplace operator.
 */
template<int dim, typename Number, typename MultigridNumber>
class CompatibleLaplaceMultigridPreconditioner
  : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef CompatibleLaplaceOperator<dim, MultigridNumber> PDEOperator;

  typedef MultigridOperatorBase<dim, MultigridNumber>          MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperator> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map          Map;
  typedef typename Base::VectorType   VectorType;
  typedef typename Base::VectorTypeMG VectorTypeMG;

  void
  initialize(MultigridData const &                      mg_data,
             const parallel::Triangulation<dim> *       tria,
             const FiniteElement<dim> &                 fe,
             Mapping<dim> const &                       mapping,
             CompatibleLaplaceOperatorData<dim> const & operator_data_in,
             Map const *                                dirichlet_bc = nullptr,
             std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
               periodic_face_pairs = nullptr)
  {
    operator_data                    = operator_data_in;
    operator_data.dof_index_velocity = 1;
    operator_data.dof_index_pressure = 0;

    operator_data.gradient_operator_data.dof_index_velocity = operator_data.dof_index_velocity;
    operator_data.gradient_operator_data.dof_index_pressure = operator_data.dof_index_pressure;
    operator_data.gradient_operator_data.quad_index         = 0;

    operator_data.divergence_operator_data.dof_index_velocity = operator_data.dof_index_velocity;
    operator_data.divergence_operator_data.dof_index_pressure = operator_data.dof_index_pressure;
    operator_data.divergence_operator_data.quad_index         = 0;

    Base::initialize(mg_data,
                     tria,
                     fe,
                     mapping,
                     operator_data.operator_is_singular,
                     dirichlet_bc,
                     periodic_face_pairs);
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    auto & dof_handler_p = *this->mg_dofhandler[level];
    auto & dof_handler_u = *this->mg_dofhandler_vel[level];

    // dof_handler
    std::vector<const DoFHandler<dim> *> dof_handler_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    dof_handler_vec.resize(2);
    dof_handler_vec[operator_data.dof_index_velocity] = &dof_handler_u;
    dof_handler_vec[operator_data.dof_index_pressure] = &dof_handler_p;

    // constraint matrix
    std::vector<AffineConstraints<double> const *> constraint_matrix_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    constraint_matrix_vec.resize(2);
    constraint_matrix_vec[operator_data.dof_index_velocity] = &*this->mg_constraints_vel[level];
    constraint_matrix_vec[operator_data.dof_index_pressure] = &*this->mg_constraints[level];

    // quadratures
    // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature
    // formula that is used for the gradient operator and the divergence operator (and the inverse
    // velocity mass matrix operator)
    std::vector<Quadrature<1>> quadrature_vec;
    quadrature_vec.resize(2);
    quadrature_vec[0] = QGauss<1>(this->global_levels[level].degree + 1 +
                                  (operator_data.degree_u - operator_data.degree_p));
    // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature is
    // needed for p-transfer
    quadrature_vec[1] = QGauss<1>(this->global_levels[level].degree + 1);

    // additional data
    typename MatrixFree<dim, MultigridNumber>::AdditionalData addit_data;

    addit_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    if(this->global_levels[level].is_dg)
    {
      addit_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);

      addit_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    }

    addit_data.level_mg_handler = this->global_levels[level].level;

    // if(operator_data.use_cell_based_loops)
    //{
    //  auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
    //    &dof_handler_p.get_triangulation());
    //  Categorization::do_cell_based_loops(*tria, additional_data,
    //  this->global_levels[level].level);
    //}

    matrix_free->reinit(
      mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, addit_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperator> pde_operator(new PDEOperator());
    pde_operator->reinit_multigrid(*this->mg_matrixfree[level],
                                   *this->mg_constraints[level],
                                   operator_data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
  }

  void
  initialize_dof_handler_and_constraints(
    bool const operator_is_singular,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                                                                         periodic_face_pairs,
    FiniteElement<dim> const &                                           fe,
    parallel::Triangulation<dim> const *                                 tria,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc)
  {
    Base::initialize_dof_handler_and_constraints(
      operator_is_singular, periodic_face_pairs, fe, tria, dirichlet_bc);

    // do setup required for derived class

    std::vector<MGLevelInfo>            global_levels_vel;
    std::vector<MGDofHandlerIdentifier> p_levels_vel;

    // setup global velocity levels
    for(auto & i : this->global_levels)
      global_levels_vel.push_back(
        {i.level, i.degree + operator_data.degree_u - operator_data.degree_p, i.is_dg});

    // setup p velocity levels
    for(auto i : global_levels_vel)
      p_levels_vel.push_back(i.id);

    sort(p_levels_vel.begin(), p_levels_vel.end());
    p_levels_vel.erase(unique(p_levels_vel.begin(), p_levels_vel.end()), p_levels_vel.end());
    std::reverse(std::begin(p_levels_vel), std::end(p_levels_vel));

    // setup dofhandler and constraint matrices
    FE_DGQ<dim>   temp(operator_data.degree_u);
    FESystem<dim> fe_vel(temp, dim);

    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc_vel;
    this->initialize_mg_dof_handler_and_constraints(false,
                                                    periodic_face_pairs,
                                                    fe_vel,
                                                    tria,
                                                    dirichlet_bc_vel,
                                                    global_levels_vel,
                                                    p_levels_vel,
                                                    this->mg_dofhandler_vel,
                                                    this->mg_constrained_dofs_vel,
                                                    this->mg_constraints_vel);
  }

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>     mg_dofhandler_vel;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>         mg_constrained_dofs_vel;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> mg_constraints_vel;

  CompatibleLaplaceOperatorData<dim> operator_data;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_ \
        */
