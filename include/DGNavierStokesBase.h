/*
 * DGNavierStokesBase.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESBASE_H_
#define INCLUDE_DGNAVIERSTOKESBASE_H_

#include <deal.II/matrix_free/operators.h>

#include "FEEvaluationWrapper.h"
//#include "XWall.h"
#include "FE_Parameters.h"

#include "InputParameters.h"

#include "InverseMassMatrix.h"
#include "NavierStokesOperators.h"

using namespace dealii;

//forward declarations
template<int dim> class AnalyticalSolution;
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesDualSplittingXWall;

enum class DofHandlerSelector{
  velocity = 0,
  pressure = 1,
  wdist_tauw = 2
};

enum class QuadratureSelector{
  velocity = 0,
  pressure = 1,
  velocity_nonlinear = 2,
  enriched = 3
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesBase
{
public:
  typedef double value_type;
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
  static const bool is_xwall = (n_q_points_1d_xwall>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+1;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  // constructor
  DGNavierStokesBase(parallel::distributed::Triangulation<dim> const &triangulation,
                     InputParameters const                           &parameter)
    :
    // fe_u(FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree+1)),dim),
    fe_u(new FESystem<dim>(FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree+1)),dim)),
    fe_p(QGaussLobatto<1>(fe_degree_p+1)),
    mapping(fe_degree),
    dof_handler_u(triangulation),
    dof_handler_p(triangulation),
    time(0.0),
    time_step(1.0),
    scaling_factor_time_derivative_term(1.0),
    viscosity(parameter.viscosity),
    dof_index_first_point(0),
    param(parameter),
    element_volume(0),
    fe_param(param)
  {}

  // destructor
  virtual ~DGNavierStokesBase()
  {
    data.clear();
  }

  virtual void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
              std::set<types::boundary_id> dirichlet_bc_indicator,
              std::set<types::boundary_id> neumann_bc_indicator);

  virtual void setup_solvers () = 0;

  void prescribe_initial_conditions(parallel::distributed::Vector<value_type> &velocity,
                                    parallel::distributed::Vector<value_type> &pressure,
                                    double const                              evaluation_time) const;

  // getters
  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }

  unsigned int get_dof_index_velocity() const
  {
    return static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  }

  unsigned int get_quad_index_velocity_linear() const
  {
    return static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity);
  }

  unsigned int get_dof_index_pressure() const
  {
    return static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  }

  unsigned int get_quad_index_pressure() const
  {
    return static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure);
  }

  MappingQ<dim> const & get_mapping() const
  {
    return mapping;
  }

  FESystem<dim> const & get_fe_u() const
  {
    return *fe_u;
  }

  FE_DGQArbitraryNodes<dim> const & get_fe_p() const
  {
    return fe_p;
  }

  FESystem<dim> const & get_fe_xwall() const
  {
    DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> * dg_ns_xwall
    = dynamic_cast<DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>* >(&this);
    AssertThrow(dg_ns_xwall, ExcMessage("wrong call"));
    return (*dg_ns_xwall).fe_xwall;
  }

  DoFHandler<dim> const & get_dof_handler_u() const
  {
    return dof_handler_u;
  }

  DoFHandler<dim> const & get_dof_handler_p() const
  {
    return dof_handler_p;
  }

  DoFHandler<dim> const & get_dof_handler_xwall() const
  {
    DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> * dg_ns_xwall
    = dynamic_cast<DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>* >(&this);
    AssertThrow(dg_ns_xwall, ExcMessage("wrong call"));
    return (*dg_ns_xwall).dof_handler_xwall;
  }

  double get_viscosity() const
  {
    return viscosity;
  }

  FEParameters const & get_fe_parameters() const
  {
    return fe_param;
  }

  value_type get_scaling_factor_time_derivative_term() const
  {
    return scaling_factor_time_derivative_term;
  }

  std::set<types::boundary_id> get_dirichlet_boundary() const
  {
    return dirichlet_boundary;
  }

  std::set<types::boundary_id> get_neumann_boundary() const
  {
    return neumann_boundary;
  }

  const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > get_periodic_face_pairs() const
  {
    return periodic_face_pairs;
  }

  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator_data;
  }

  ViscousOperatorData<dim> const & get_viscous_operator_data() const
  {
    return viscous_operator_data;
  }

  // setters
  void set_scaling_factor_time_derivative_term(double const value)
  {
    scaling_factor_time_derivative_term = value;
  }

  void set_time(double const current_time)
  {
    time = current_time;
  }

  void set_time_step(double const time_step_in)
  {
    time_step = time_step_in;
  }

  // initialization of vectors
  void initialize_vector_velocity(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
  }

  void initialize_vector_vorticity(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
  }

  void initialize_vector_pressure(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));
  }

  //shift pressure (pure Dirichlet BC case)
  void  shift_pressure (parallel::distributed::Vector<value_type> &pressure) const;

  // vorticity
  void compute_vorticity (parallel::distributed::Vector<value_type>       &dst,
                          const parallel::distributed::Vector<value_type> &src) const;

  // divergence
  void compute_divergence (parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src) const;

  void evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                 parallel::distributed::Vector<value_type> const &src,
                                 value_type const                                evaluation_time) const;

protected:
  MatrixFree<dim,value_type> data;

  std_cxx11::shared_ptr< FESystem<dim> > fe_u;
  FE_DGQArbitraryNodes<dim> fe_p;

  MappingQ<dim> mapping;

  DoFHandler<dim>  dof_handler_u;
  DoFHandler<dim>  dof_handler_p;

  double time, time_step;
  double scaling_factor_time_derivative_term;

  const double viscosity;

  Point<dim> first_point;
  types::global_dof_index dof_index_first_point;

  std::set<types::boundary_id> dirichlet_boundary;
  std::set<types::boundary_id> neumann_boundary;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs;

  InputParameters const &param;

  AlignedVector<VectorizedArray<value_type> > element_volume;
  FEParameters fe_param;

  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;

  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> mass_matrix_operator;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> convective_operator;
  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> viscous_operator;
  BodyForceOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> body_force_operator;
  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> gradient_operator;
  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> divergence_operator;

private:
  virtual void create_dofs();

  virtual void data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data);

  // compute vorticity
  void local_compute_vorticity (const MatrixFree<dim,value_type>                 &data,
                                parallel::distributed::Vector<value_type>        &dst,
                                const parallel::distributed::Vector<value_type>  &src,
                                const std::pair<unsigned int,unsigned int>       &cell_range) const;

  // divergence
  void local_compute_divergence (const MatrixFree<dim,value_type>                &data,
                                 parallel::distributed::Vector<value_type>       &dst,
                                 const parallel::distributed::Vector<value_type> &src,
                                 const std::pair<unsigned int,unsigned int>      &cell_range) const;

};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
       std::set<types::boundary_id> dirichlet_bc_indicator,
       std::set<types::boundary_id> neumann_bc_indicator)
{
  this->dirichlet_boundary = dirichlet_bc_indicator;
  this->neumann_boundary = neumann_bc_indicator;
  this->periodic_face_pairs = periodic_face_pairs;

  create_dofs();

  // initialize matrix_free_data
  typename MatrixFree<dim,value_type>::AdditionalData additional_data;
  additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme = MatrixFree<dim,value_type>::AdditionalData::partition_partition;
  additional_data.build_face_info = true;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points | update_normal_vectors |
                                          update_values);
  additional_data.periodic_face_pairs_level_0 = periodic_face_pairs;

  data_reinit(additional_data);

  // mass matrix operator
//  MassMatrixOperatorData mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  mass_matrix_operator.initialize(data,fe_param,mass_matrix_operator_data);

  // inverse mass matrix operator
  inverse_mass_matrix_operator.initialize(data,
          static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
          static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity));

  // body force operator
  BodyForceOperatorData body_force_operator_data;
  body_force_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  body_force_operator.initialize(data,fe_param,body_force_operator_data);

  // gradient operator
  GradientOperatorData gradient_operator_data;
  gradient_operator_data.dof_index_velocity = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  gradient_operator_data.dof_index_pressure = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  gradient_operator_data.integration_by_parts_of_gradP = param.gradp_integrated_by_parts;
  gradient_operator_data.use_boundary_data = param.gradp_use_boundary_data;
  gradient_operator_data.dirichlet_boundaries = dirichlet_boundary;
  gradient_operator_data.neumann_boundaries = neumann_boundary;
  gradient_operator.initialize(data,fe_param,gradient_operator_data);

  // divergence operator
  DivergenceOperatorData divergence_operator_data;
  divergence_operator_data.dof_index_velocity = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  divergence_operator_data.dof_index_pressure = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  divergence_operator_data.integration_by_parts_of_divU = param.divu_integrated_by_parts;
  divergence_operator_data.use_boundary_data = param.divu_use_boundary_data;
  divergence_operator_data.dirichlet_boundaries = dirichlet_boundary;
  divergence_operator_data.neumann_boundaries = neumann_boundary;
  divergence_operator.initialize(data,fe_param,divergence_operator_data);

  // convective operator
  ConvectiveOperatorData convective_operator_data;
  convective_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  convective_operator_data.dirichlet_boundaries = dirichlet_boundary;
  convective_operator_data.neumann_boundaries = neumann_boundary;
  convective_operator.initialize(data,fe_param,convective_operator_data);

  // viscous operator
//  ViscousOperatorData viscous_operator_data;
  viscous_operator_data.formulation_viscous_term = param.formulation_viscous_term;
  viscous_operator_data.IP_formulation_viscous = param.IP_formulation_viscous;
  viscous_operator_data.IP_factor_viscous = param.IP_factor_viscous;
  viscous_operator_data.dirichlet_boundaries = dirichlet_boundary;
  viscous_operator_data.neumann_boundaries = neumann_boundary;
  viscous_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  viscous_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;
  viscous_operator_data.viscosity = param.viscosity;
  viscous_operator.initialize(mapping,data,fe_param,viscous_operator_data);
  // viscous_operator.set_constant_viscosity(viscosity);
  // viscous_operator.set_variable_viscosity(viscosity);

  dof_index_first_point = 0;
  for(unsigned int d=0;d<dim;++d)
    first_point[d] = 0.0;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    typename DoFHandler<dim>::active_cell_iterator first_cell;
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(), endc = dof_handler_p.end();
    for(;cell!=endc;++cell)
    {
      if (cell->is_locally_owned())
      {
        first_cell = cell;
        break;
      }
    }
    FEValues<dim> fe_values(dof_handler_p.get_fe(),
                Quadrature<dim>(dof_handler_p.get_fe().get_unit_support_points()),
                update_quadrature_points);
    fe_values.reinit(first_cell);
    first_point = fe_values.quadrature_point(0);
    std::vector<types::global_dof_index>
    dof_indices(dof_handler_p.get_fe().dofs_per_cell);
    first_cell->get_dof_indices(dof_indices);
    dof_index_first_point = dof_indices[0];
  }
  dof_index_first_point = Utilities::MPI::sum(dof_index_first_point,MPI_COMM_WORLD);
  for(unsigned int d=0;d<dim;++d)
  {
    first_point[d] = Utilities::MPI::sum(first_point[d],MPI_COMM_WORLD);
  }

  QGauss<dim> quadrature(fe_degree+1);
  FEValues<dim> fe_values(mapping, dof_handler_u.get_fe(), quadrature, update_JxW_values);
  element_volume.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  for (unsigned int i=0; i<data.n_macro_cells()+data.n_macro_ghost_cells(); ++i)
  {
    for (unsigned int v=0; v<data.n_components_filled(i); ++v)
    {
      typename DoFHandler<dim>::cell_iterator cell = data.get_cell_iterator(i,v);
      fe_values.reinit(cell);
      double volume = 0.;
      for (unsigned int q=0; q<quadrature.size(); ++q)
        volume += fe_values.JxW(q);
      element_volume[i][v] = volume;
      //pcout << "surface to volume ratio: " << pressure_poisson_solver.get_matrix().get_array_penalty_parameter()[i][v] << std::endl;
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
create_dofs()
{
  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_p.distribute_dofs(fe_p);
  dof_handler_p.distribute_mg_dofs(fe_p);
  dof_handler_u.distribute_mg_dofs(*fe_u);

  unsigned int ndofs_per_cell_velocity    = Utilities::fixed_int_power<fe_degree+1,dim>::value*dim;
  unsigned int ndofs_per_cell_pressure    = Utilities::fixed_int_power<fe_degree_p+1,dim>::value;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
    << "Velocity:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_velocity << std::endl
    << "  number of dofs (velocity):\t" << std::fixed << std::setw(10) << std::right << dof_handler_u.n_dofs() << std::endl
    << "Pressure:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree_p << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_pressure << std::endl
    << "  number of dofs (pressure):\t" << std::fixed << std::setw(10) << std::right << dof_handler_p.n_dofs() << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data)
{
  std::vector<const DoFHandler<dim> * >  dof_handler_vec;

  dof_handler_vec.push_back(&dof_handler_u);
  dof_handler_vec.push_back(&dof_handler_p);

  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  ConstraintMatrix constraint, constraint_p;
  constraint.close();
  constraint_p.close();
  constraint_matrix_vec.push_back(&constraint);
  constraint_matrix_vec.push_back(&constraint_p);

  std::vector<Quadrature<1> > quadratures;

  // velocity
  quadratures.push_back(QGauss<1>(fe_degree+1));
  // pressure
  quadratures.push_back(QGauss<1>(fe_degree_p+1));
  // exact integration of nonlinear convective term
  quadratures.push_back(QGauss<1>(fe_degree + (fe_degree+2)/2));

  data.reinit (mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
prescribe_initial_conditions(parallel::distributed::Vector<value_type> &velocity,
                             parallel::distributed::Vector<value_type> &pressure,
                             double const                              evaluation_time) const
{
  VectorTools::interpolate(mapping, dof_handler_u, AnalyticalSolution<dim>(true,evaluation_time), velocity);
  VectorTools::interpolate(mapping, dof_handler_p, AnalyticalSolution<dim>(false,evaluation_time), pressure);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
shift_pressure (parallel::distributed::Vector<value_type>  &pressure) const
{
  parallel::distributed::Vector<value_type> vec1(pressure);
  for(unsigned int i=0;i<vec1.local_size();++i)
    vec1.local_element(i) = 1.;
  AnalyticalSolution<dim> analytical_solution(false,time+time_step);
  double exact = analytical_solution.value(first_point);
  double current = 0.;
  if (pressure.locally_owned_elements().is_element(dof_index_first_point))
    current = pressure(dof_index_first_point);
  current = Utilities::MPI::sum(current, MPI_COMM_WORLD);
  pressure.add(exact-current,vec1);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
compute_vorticity (parallel::distributed::Vector<value_type>       &dst,
                   const parallel::distributed::Vector<value_type> &src) const
{
  dst = 0;

  data.cell_loop (&DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_vorticity,this, dst, src);

  inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_compute_vorticity(const MatrixFree<dim,value_type>                 &data,
                        parallel::distributed::Vector<value_type>        &dst,
                        const parallel::distributed::Vector<value_type>  &src,
                        const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEval_Velocity_Velocity_linear velocity(data,fe_param,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit(cell);
    velocity.read_dof_values(src);
    velocity.evaluate (false,true,false);
    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
      // omega_vector is a vector with dim components
      // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
      // for dim=2: omega_vector[0] = omega,
      //            omega_vector[1] = 0
      Tensor<1,dim,VectorizedArray<value_type> > omega_vector;
      for (unsigned int d=0; d<number_vorticity_components; ++d)
        omega_vector[d] = omega[d];
      velocity.submit_value (omega_vector, q);
    }
    velocity.integrate (true,false);
    velocity.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
compute_divergence (parallel::distributed::Vector<value_type>       &dst,
                    const parallel::distributed::Vector<value_type> &src) const
{
  dst = 0;

  data.cell_loop(&DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divergence,
                             this, dst, src);

  inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_compute_divergence (const MatrixFree<dim,value_type>                 &data,
                          parallel::distributed::Vector<value_type>        &dst,
                          const parallel::distributed::Vector<value_type>  &src,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_velocity.reinit(cell);
    fe_eval_velocity.read_dof_values(src);
    fe_eval_velocity.evaluate(false,true);

    for (unsigned int q=0; q<fe_eval_velocity.n_q_points; q++)
    {
      Tensor<1,dim,VectorizedArray<value_type> > div_vector;
        div_vector[0] = fe_eval_velocity.get_divergence(q);
      fe_eval_velocity.submit_value(div_vector,q);
    }
    fe_eval_velocity.integrate(true,false);
    fe_eval_velocity.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                          parallel::distributed::Vector<value_type> const &src,
                          value_type const                                evaluation_time) const
{
  convective_operator.evaluate(dst,src,evaluation_time);
}


#endif /* INCLUDE_DGNAVIERSTOKESBASE_H_ */
