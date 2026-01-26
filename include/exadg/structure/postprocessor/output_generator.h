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

#ifndef EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/solution_field.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace Structure
{
struct OutputData : public OutputDataBase
{
  OutputData()
    : write_displacement_magnitude(false),
      write_displacement_jacobian(false),
      write_max_principal_stress(false),
      write_E1_orientation(false),
      write_E2_orientation(false),
      write_traction_local_full(false),
      write_traction_local_normal(false),
      write_traction_local_inplane(false)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write displacement magnitude", write_displacement_magnitude);
    print_parameter(pcout, "Write displacement Jacobian", write_displacement_jacobian);
    print_parameter(pcout, "Write maximum principal stress", write_max_principal_stress);
    print_parameter(pcout, "Write E1 orientation", write_E1_orientation);
    print_parameter(pcout, "Write E2 orientation", write_E2_orientation);
    print_parameter(pcout, "Write traction: full", write_traction_local_full);
    print_parameter(pcout, "Write traction: normal", write_traction_local_normal);
    print_parameter(pcout, "Write traction: in-plane", write_traction_local_inplane);
  }

  // write displacement magnitude
  bool write_displacement_magnitude;

  // write Jacobian of the displacement field
  bool write_displacement_jacobian;

  // write maximum principal stress
  bool write_max_principal_stress;

  // write the material orientation vector
  bool write_E1_orientation;
  bool write_E2_orientation;

  // write traction vector components relative to the material coordinates:
  // full vector, normal or in-plane components.
  bool write_traction_local_full;
  bool write_traction_local_normal;
  bool write_traction_local_inplane;
};

template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler,
        dealii::Mapping<dim> const &    mapping,
        OutputData const &              output_data);

  void
  evaluate(
    VectorType const &                                                       solution,
    std::vector<dealii::ObserverPointer<SolutionField<dim, Number>>> const & additional_fields,
    double const                                                             time,
    bool const                                                               unsteady);

  TimeControl time_control;

private:
  MPI_Comm const mpi_comm;

  OutputData output_data;

  dealii::ObserverPointer<dealii::DoFHandler<dim> const> dof_handler;
  dealii::ObserverPointer<dealii::Mapping<dim> const>    mapping;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
