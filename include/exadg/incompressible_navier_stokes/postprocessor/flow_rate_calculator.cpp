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

// C/C++
#include <fstream>

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/flow_rate_calculator.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
FlowRateCalculator<dim, Number>::FlowRateCalculator(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in,
  FlowRateCalculatorData<dim> const &     data_in,
  MPI_Comm const &                        comm)
  : data(data_in),
    matrix_free(matrix_free_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    clear_files(true),
    mpi_comm(comm)
{
  if(data.calculate)
    create_directories(data.directory, mpi_comm);
}

template<int dim, typename Number>
void
FlowRateCalculator<dim, Number>::calculate_flow_rates(
  VectorType const &                             velocity,
  double const &                                 time,
  std::map<dealii::types::boundary_id, Number> & flow_rates)
{
  if(data.calculate == true)
  {
    do_calculate_flow_rates(velocity, flow_rates);

    // copy flow rates and boundary IDs
    std::vector<Number> values;
    std::vector<dealii::types::boundary_id> boundary_ids;
    for(auto const & it : flow_rates)
    {
      boundary_ids.push_back(it.first);
      values.push_back(it.second);
    }

    write_output(values, boundary_ids, time);
  }
}

template<int dim, typename Number>
void
FlowRateCalculator<dim, Number>::write_output(std::vector<Number> const & flow_rates,
		                                      std::vector<dealii::types::boundary_id> const & boundary_ids,
                                              double const &      time)
{
  // write output file
  if(data.write_to_file == true and dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::string filename = data.directory + data.filename;

    std::ofstream f;
    if(clear_files == true)
    {
      f.open(filename.c_str(), std::ios::trunc);
      f << std::endl << "  Time.  Q on IDs:";
      for (unsigned int i = 0; i<boundary_ids.size(); ++i)
        f << std::setw(5) << boundary_ids[i] << "               ";
      f << std::endl;

      clear_files = false;
    }
    else
    {
      f.open(filename.c_str(), std::ios::app);
    }

    unsigned int precision = 12;
    f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time;
    for (unsigned int i = 0; i<flow_rates.size(); ++i)
      f << std::setw(precision + 8) << flow_rates[i];
    f << std::endl;
  }
}

template<int dim, typename Number>
void
FlowRateCalculator<dim, Number>::do_calculate_flow_rates(
  VectorType const &                             velocity,
  std::map<dealii::types::boundary_id, Number> & flow_rates)
{
  // zero flow rates since we sum into these variables
  for(auto iterator = flow_rates.begin(); iterator != flow_rates.end(); ++iterator)
  {
    iterator->second = 0.0;
  }

  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::map<dealii::types::boundary_id, Number>::iterator it;
    dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    it = flow_rates.find(boundary_id);
    if(it != flow_rates.end())
    {
      integrator.reinit(face);
      integrator.read_dof_values(velocity);
      integrator.evaluate(dealii::EvaluationFlags::values);

      scalar flow_rate_face = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        flow_rate_face +=
          integrator.JxW(q) * integrator.get_value(q) * integrator.get_normal_vector(q);
      }

      // sum over all entries of dealii::VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
        flow_rates.at(boundary_id) += flow_rate_face[n];
    }
  }

  // copy, sum with communication, and store flow rates
  std::vector<double> flow_rates_vector(flow_rates.size());
  auto                iterator = flow_rates.begin();
  for(unsigned int counter = 0; counter < flow_rates.size(); ++counter)
  {
    flow_rates_vector[counter] = (iterator++)->second;
  }

  dealii::Utilities::MPI::sum(
    dealii::ArrayView<double const>(&(*flow_rates_vector.begin()), flow_rates_vector.size()),
    mpi_comm,
    dealii::ArrayView<double>(&(*flow_rates_vector.begin()), flow_rates_vector.size()));

  iterator = flow_rates.begin();
  for(unsigned int counter = 0; counter < flow_rates.size(); ++counter)
  {
    (iterator++)->second = flow_rates_vector[counter];
  }
}

template class FlowRateCalculator<2, float>;
template class FlowRateCalculator<2, double>;

template class FlowRateCalculator<3, float>;
template class FlowRateCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
