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

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/acoustic_conservation_equations/spatial_discretization/spatial_operator.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/grid_related_time_step_restrictions.h>
#include <exadg/operators/quadrature.h>
#include <exadg/time_integration/restart.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim, typename Number>
SpatialOperator<dim, Number>::SpatialOperator(
  std::shared_ptr<Grid<dim> const>               grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>    mapping_in,
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>     field_functions_in,
  Parameters const &                             parameters_in,
  std::string const &                            field_in,
  MPI_Comm const &                               mpi_comm_in)
  : Interface::SpatialOperator<dim, Number>(),
    grid(grid_in),
    mapping(mapping_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(parameters_in),
    field(field_in),
    dof_handler_p(*grid_in->triangulation),
    dof_handler_u(*grid_in->triangulation),
    aero_acoustic_source_term(nullptr),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl
        << "Construct acoustic conservation equations operator ..." << std::endl
        << std::flush;

  initialize_dof_handler_and_constraints();

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, Number> & matrix_free_data) const
{
  // append mapping flags
  matrix_free_data.append_mapping_flags(Operators::Kernel<dim, Number>::get_mapping_flags());

  if(param.right_hand_side)
    matrix_free_data.append_mapping_flags(
      ExaDG::Operators::RHSKernel<dim, Number>::get_mapping_flags());

  // mapping flags required for CFL condition
  if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    MappingFlags flags_cfl;
    flags_cfl.cells = dealii::update_quadrature_points;
    matrix_free_data.append_mapping_flags(flags_cfl);
  }

  // dof handler
  matrix_free_data.insert_dof_handler(&dof_handler_p, field + dof_index_p);
  matrix_free_data.insert_dof_handler(&dof_handler_u, field + dof_index_u);

  // constraint
  matrix_free_data.insert_constraint(&constraint_p, field + dof_index_p);
  matrix_free_data.insert_constraint(&constraint_u, field + dof_index_u);

  // quadrature for pressure
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_p =
    create_quadrature<dim>(param.grid.element_type, param.degree_p + 1);
  matrix_free_data.insert_quadrature(*quadrature_p, field + quad_index_p);

  // quadrature for velocity
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_u =
    create_quadrature<dim>(param.grid.element_type, param.degree_u + 1);
  matrix_free_data.insert_quadrature(*quadrature_u, field + quad_index_u);

  // quadrature that works for pressure and velocity
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_p_u =
    create_quadrature<dim>(param.grid.element_type, std::max(param.degree_p, param.degree_u) + 1);
  matrix_free_data.insert_quadrature(*quadrature_p_u, field + quad_index_p_u);
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::setup()
{
  // initialize MatrixFree and MatrixFreeData
  std::shared_ptr<dealii::MatrixFree<dim, Number>> mf =
    std::make_shared<dealii::MatrixFree<dim, Number>>();
  std::shared_ptr<MatrixFreeData<dim, Number>> mf_data =
    std::make_shared<MatrixFreeData<dim, Number>>();

  fill_matrix_free_data(*mf_data);

  mf->reinit(*get_mapping(),
             mf_data->get_dof_handler_vector(),
             mf_data->get_constraint_vector(),
             mf_data->get_quadrature_vector(),
             mf_data->data);

  // Subsequently, call the other setup function with MatrixFree/MatrixFreeData objects as
  // arguments.
  this->setup(mf, mf_data);
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::setup(
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in)
{
  pcout << std::endl
        << "Setup acoustic conservation equations operator ..." << std::endl
        << std::flush;

  // MatrixFree
  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  initialize_operators();

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
dealii::MatrixFree<dim, Number> const &
SpatialOperator<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, typename Number>
std::string
SpatialOperator<dim, Number>::get_dof_name_pressure() const
{
  return field + dof_index_p;
}

template<int dim, typename Number>
unsigned int
SpatialOperator<dim, Number>::get_dof_index_pressure() const
{
  return matrix_free_data->get_dof_index(get_dof_name_pressure());
}

template<int dim, typename Number>
std::string
SpatialOperator<dim, Number>::get_dof_name_velocity() const
{
  return field + dof_index_u;
}

template<int dim, typename Number>
unsigned int
SpatialOperator<dim, Number>::get_dof_index_velocity() const
{
  return matrix_free_data->get_dof_index(get_dof_name_velocity());
}

template<int dim, typename Number>
unsigned int
SpatialOperator<dim, Number>::get_quad_index_pressure_velocity() const
{
  return matrix_free_data->get_quad_index(field + quad_index_p_u);
}

template<int dim, typename Number>
unsigned int
SpatialOperator<dim, Number>::get_quad_index_pressure() const
{
  return matrix_free_data->get_quad_index(field + quad_index_p);
}

template<int dim, typename Number>
unsigned int
SpatialOperator<dim, Number>::get_quad_index_velocity() const
{
  return matrix_free_data->get_quad_index(field + quad_index_u);
}

template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
SpatialOperator<dim, Number>::get_mapping() const
{
  return mapping;
}

template<int dim, typename Number>
dealii::FiniteElement<dim> const &
SpatialOperator<dim, Number>::get_fe_p() const
{
  return *fe_p;
}

template<int dim, typename Number>
dealii::FiniteElement<dim> const &
SpatialOperator<dim, Number>::get_fe_u() const
{
  return *fe_u;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperator<dim, Number>::get_dof_handler_p() const
{
  return dof_handler_p;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperator<dim, Number>::get_dof_handler_u() const
{
  return dof_handler_u;
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::serialize_vectors(
  std::vector<BlockVectorType const *> block_vectors) const
{
  std::vector<dealii::DoFHandler<dim> const *> dof_handlers;
  dof_handlers.push_back(&this->get_dof_handler_u());
  dof_handlers.push_back(&this->get_dof_handler_p());

  std::vector<std::vector<VectorType const *>> vectors_per_dof_handler =
    get_vectors_per_block<VectorType const, BlockVectorType const>(block_vectors);

  if(param.restart_data.consider_mapping)
  {
    store_vectors_in_triangulation_and_serialize(
      param.restart_data.filename,
      vectors_per_dof_handler,
      dof_handlers,
      *this->get_mapping(),
      &this->get_dof_handler_u() /* dof_handler_mapping */);
  }
  else
  {
    store_vectors_in_triangulation_and_serialize(param.restart_data.filename,
                                                 vectors_per_dof_handler,
                                                 dof_handlers);
  }
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::deserialize_vectors(
  std::vector<BlockVectorType *> block_vectors) const
{
  // Load potentially unfitting checkpoint triangulation of identical TriangulationType.
  // We assume the coarse grid was stored to disk during the call to
  // `ApplicationBase::create_grid()`.
  std::shared_ptr<dealii::Triangulation<dim>> checkpoint_triangulation =
    deserialize_triangulation<dim>(param.restart_data.filename,
                                    param.restart_data.triangulation_type,
                                    mpi_comm);

  // Setup DoFHandlers *as checkpointed*, sequence matches `this->serialize_vectors()`.
  dealii::DoFHandler<dim> checkpoint_dof_handler_u(*checkpoint_triangulation);
  dealii::DoFHandler<dim> checkpoint_dof_handler_p(*checkpoint_triangulation);
  ElementType             checkpoint_element_type = get_element_type(*checkpoint_triangulation);
  std::shared_ptr<dealii::FiniteElement<dim>> checkpoint_fe_u =
    create_finite_element<dim>(checkpoint_element_type, true, dim, param.restart_data.degree_u);
  std::shared_ptr<dealii::FiniteElement<dim>> checkpoint_fe_p =
    create_finite_element<dim>(checkpoint_element_type, true, 1, param.restart_data.degree_p);
  checkpoint_dof_handler_u.distribute_dofs(*checkpoint_fe_u);
  checkpoint_dof_handler_p.distribute_dofs(*checkpoint_fe_p);
  std::vector<dealii::DoFHandler<dim> const *> checkpoint_dof_handlers{&checkpoint_dof_handler_u,
                                                                       &checkpoint_dof_handler_p};

  // Deserialize vectors stored in triangulation, sequence matches `this->serialize_vectors()`.
  std::vector<BlockVectorType> checkpoint_block_vectors =
    get_block_vectors_from_dof_handlers<dim, BlockVectorType>(2 /* n_vectors */,
                                                              checkpoint_dof_handlers);

  std::vector<std::vector<VectorType *>> checkpoint_vectors =
    get_vectors_per_block<VectorType, BlockVectorType>(checkpoint_block_vectors);

  std::shared_ptr<dealii::Mapping<dim>> checkpoint_mapping;
  if(param.restart_data.consider_mapping)
  {
    checkpoint_mapping = load_vectors(checkpoint_vectors,
                                      checkpoint_dof_handlers,
                                      &checkpoint_dof_handler_u /* dof_handler_mapping */);
  }
  else
  {
    load_vectors(checkpoint_vectors, checkpoint_dof_handlers);
    unsigned int const mapping_degree = checkpoint_dof_handler_u.get_fe().degree;
    GridUtilities::create_mapping(checkpoint_mapping, get_element_type(*checkpoint_triangulation), mapping_degree);
  }

  if(param.restart_data.discretization_identical)
  {
    // DoFHandlers need to be setup with `checkpoint_triangulation`, otherwise
    // they are identical. We can simply copy the vector contents.
    for(unsigned int i = 0; i < block_vectors.size(); ++i)
    {
      block_vectors[i]->copy_locally_owned_data_from(checkpoint_block_vectors[i]);
      block_vectors[i]->update_ghost_values();
    }
  }
  else
  {
    // Perform global projection in case of a non-matching discretization.
    std::vector<dealii::DoFHandler<dim> const *> dof_handlers;
    dof_handlers.push_back(&this->get_dof_handler_u());
    dof_handlers.push_back(&this->get_dof_handler_p());

    std::vector<std::vector<VectorType *>> vectors_per_dof_handler =
      get_vectors_per_block<VectorType, BlockVectorType>(block_vectors);

    grid_to_grid_projection(checkpoint_vectors,
                                            checkpoint_dof_handlers,
                                            checkpoint_mapping,
                                            vectors_per_dof_handler,
                                            dof_handlers,
                                            this->get_mapping());
  }
}

template<int dim, typename Number>
dealii::AffineConstraints<Number> const &
SpatialOperator<dim, Number>::get_constraint_p() const
{
  return constraint_p;
}

template<int dim, typename Number>
dealii::AffineConstraints<Number> const &
SpatialOperator<dim, Number>::get_constraint_u() const
{
  return constraint_u;
}

template<int dim, typename Number>
dealii::types::global_dof_index
SpatialOperator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler_u.n_dofs() + dof_handler_p.n_dofs();
}

/*
 * Initialization of vectors.
 */
template<int dim, typename Number>
void
SpatialOperator<dim, Number>::initialize_dof_vector(BlockVectorType & dst) const
{
  dst.reinit(2);

  matrix_free->initialize_dof_vector(dst.block(block_index_pressure), get_dof_index_pressure());
  matrix_free->initialize_dof_vector(dst.block(block_index_velocity), get_dof_index_velocity());

  dst.collect_sizes();
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::initialize_dof_vector_pressure(VectorType & dst) const
{
  matrix_free->initialize_dof_vector(dst, get_dof_index_pressure());
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::prescribe_initial_conditions(BlockVectorType & dst,
                                                           double const      time) const
{
  field_functions->initial_solution_pressure->set_time(time);
  field_functions->initial_solution_velocity->set_time(time);

  // This is necessary if Number == float
  using VectorTypeDouble = dealii::LinearAlgebra::distributed::Vector<double>;

  VectorTypeDouble pressure_double;
  VectorTypeDouble velocity_double;
  pressure_double = dst.block(block_index_pressure);
  velocity_double = dst.block(block_index_velocity);

  dealii::VectorTools::interpolate(*get_mapping(),
                                   dof_handler_p,
                                   *(field_functions->initial_solution_pressure),
                                   pressure_double);

  dealii::VectorTools::interpolate(*get_mapping(),
                                   dof_handler_u,
                                   *(field_functions->initial_solution_velocity),
                                   velocity_double);

  dst.block(block_index_pressure) = pressure_double;
  dst.block(block_index_velocity) = velocity_double;
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::set_aero_acoustic_source_term(
  VectorType const & aero_acoustic_source_term_in)
{
  aero_acoustic_source_term = &aero_acoustic_source_term_in;
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::evaluate(BlockVectorType &       dst,
                                       BlockVectorType const & src,
                                       double const            time) const
{
  evaluate_acoustic_operator(dst, src, time);

  // shift to the right-hand side of the equation
  dst *= -1.0;

  if(param.right_hand_side)
    rhs_operator.evaluate_add(dst.block(block_index_pressure), time);

  if(param.aero_acoustic_source_term)
  {
    AssertThrow(aero_acoustic_source_term,
                dealii::ExcMessage("Aero-acoustic source term not valid."));
    dst.block(block_index_pressure) += *aero_acoustic_source_term;
  }

  apply_scaled_inverse_mass_operator(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::evaluate_acoustic_operator(BlockVectorType &       dst,
                                                         BlockVectorType const & src,
                                                         double const            time) const
{
  acoustic_operator.evaluate(dst, src, time);
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::apply_scaled_inverse_mass_operator(BlockVectorType &       dst,
                                                                 BlockVectorType const & src) const
{
  inverse_mass_pressure.apply_scale(dst.block(block_index_pressure),
                                    param.speed_of_sound * param.speed_of_sound,
                                    src.block(block_index_pressure));
  inverse_mass_velocity.apply(dst.block(block_index_velocity), src.block(block_index_velocity));
}

template<int dim, typename Number>
double
SpatialOperator<dim, Number>::calculate_time_step_cfl() const
{
  // In case of mixed-orders use the maximum fe_degree and the corresponding
  // quadrature rule.

  // The time-step size is not adapted every time-step. Thus, we are using
  // a constant function to pass in the speed of sound, even though it is
  // possible to optimize calculate_time_step_cfl_local() for this case.

  return calculate_time_step_cfl_local<dim, Number>(
    get_matrix_free(),
    get_dof_index_velocity(),
    get_quad_index_pressure_velocity(),
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(param.speed_of_sound, dim),
    param.start_time /* will not be used (ConstantFunction) */,
    std::max(param.degree_p, param.degree_u),
    param.cfl_exponent_fe_degree,
    CFLConditionType::VelocityNorm,
    mpi_comm);
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::initialize_dof_handler_and_constraints()
{
  fe_p = create_finite_element<dim>(param.grid.element_type, true, 1, param.degree_p);
  fe_u = create_finite_element<dim>(param.grid.element_type, true, dim, param.degree_u);

  // enumerate degrees of freedom
  dof_handler_p.distribute_dofs(*fe_p);
  dof_handler_u.distribute_dofs(*fe_u);

  // close constraints
  constraint_u.close();
  constraint_p.close();

  // Output DoF information
  pcout << "Pressure:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.degree_p);
  print_parameter(pcout, "number of dofs per cell", fe_p->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler_p.n_dofs());

  pcout << "Velocity:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.degree_u);
  print_parameter(pcout, "number of dofs per cell", fe_u->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler_u.n_dofs());

  pcout << "Pressure and velocity:" << std::endl;
  print_parameter(pcout,
                  "number of dofs per cell",
                  fe_p->n_dofs_per_cell() + fe_u->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", get_number_of_dofs());

  pcout << std::flush;
}

template<int dim, typename Number>
void
SpatialOperator<dim, Number>::initialize_operators()
{
  // inverse mass operator pressure
  {
    InverseMassOperatorData data;
    data.dof_index  = get_dof_index_pressure();
    data.quad_index = get_quad_index_pressure();
    inverse_mass_pressure.initialize(*matrix_free, data);
  }

  // inverse mass operator velocity
  {
    InverseMassOperatorData data;
    data.dof_index  = get_dof_index_velocity();
    data.quad_index = get_quad_index_velocity();
    inverse_mass_velocity.initialize(*matrix_free, data);
  }

  // acoustic operator
  {
    OperatorData<dim> data;
    data.dof_index_pressure   = get_dof_index_pressure();
    data.dof_index_velocity   = get_dof_index_velocity();
    data.quad_index           = get_quad_index_pressure_velocity();
    data.block_index_pressure = block_index_pressure;
    data.block_index_velocity = block_index_velocity;
    data.speed_of_sound       = param.speed_of_sound;
    data.formulation          = param.formulation;
    data.bc                   = boundary_descriptor;
    acoustic_operator.initialize(*matrix_free, data);
  }

  // rhs operator
  if(param.right_hand_side)
  {
    RHSOperatorData<dim> data;
    data.dof_index     = get_dof_index_pressure();
    data.quad_index    = get_quad_index_pressure();
    data.kernel_data.f = field_functions->right_hand_side;
    rhs_operator.initialize(*matrix_free, data);
  }
}

template class SpatialOperator<2, float>;
template class SpatialOperator<3, float>;

template class SpatialOperator<2, double>;
template class SpatialOperator<3, double>;

} // namespace Acoustics
} // namespace ExaDG
