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

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics_homogeneous.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::LinePlotCalculatorStatisticsHomogeneous(
  dealii::DoFHandler<dim> const & dof_handler_velocity_in,
  dealii::DoFHandler<dim> const & dof_handler_pressure_in,
  dealii::Mapping<dim> const &    mapping_in,
  MPI_Comm const &                mpi_comm_in)
  : clear_files(true),
    dof_handler_velocity(dof_handler_velocity_in),
    dof_handler_pressure(dof_handler_pressure_in),
    mapping(mapping_in),
    mpi_comm(mpi_comm_in),
    number_of_samples(0),
    averaging_direction(2),
    write_final_output(false),
    evaluator_tensor_product(mapping,
                             dof_handler_velocity.get_fe(),
                             dealii::QGaussLobatto<1>(dof_handler_velocity.get_fe().degree + 2),
                             dealii::update_values)

{
  AssertThrow(get_element_type(dof_handler_velocity.get_triangulation()) == ElementType::Hypercube,
              dealii::ExcMessage("Only implemented for hypercube elements."));
}

template<int dim>
dealii::Point<dim>
find_unit_point(dealii::Mapping<dim, dim> const &                               mapping,
                typename dealii::Triangulation<dim, dim>::cell_iterator const & cell,
                dealii::Point<dim> const &                                      point)
{
  // coarse search first that identifies candidates by affine approximation
  auto const vertices = mapping.get_vertices(cell);
  auto const A_b      = dealii::GridTools::affine_cell_approximation<dim, dim>(vertices);
  dealii::DerivativeForm<1, dim, dim> const A_inv = A_b.first.covariant_form().transpose();
  dealii::Point<dim>                        p_unit(apply_transformation(A_inv, point - A_b.second));

  // if point is far away in affine approximation of unit cell, do not try
  // more accurate search to limit computational costs
  if(dealii::GeometryInfo<dim>::is_inside_unit_cell(p_unit, 1.0))
  {
    try
    {
      p_unit = mapping.transform_real_to_unit_cell(cell, point);
    }
    catch(...)
    {
      // A point that does not lie on the reference cell.
      p_unit[0] = 2.0;
    }
  }
  return p_unit;
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::setup(
  LinePlotDataStatistics<dim> const & data_in)
{
  time_all = 0;
  data     = data_in;

  AssertThrow(Utilities::is_valid_timestep(
                data_in.time_control_data_statistics.write_preliminary_results_every_nth_time_step),
              dealii::ExcMessage("write_preliminary_results_every_nth_time_step has to be set."));
  time_control_statistics.setup(data_in.time_control_data_statistics);

  if(data_in.time_control_data_statistics.time_control_data.is_active)
  {
    AssertThrow(dim == 3, dealii::ExcMessage("Not implemented."));

    AssertThrow(data.lines.size() > 0, dealii::ExcMessage("Empty data"));

    global_points.resize(data.lines.size());
    cells_and_ref_points_velocity.resize(data.lines.size());
    cells_and_ref_points_pressure.resize(data.lines.size());
    cells_and_ref_points_ref_pressure.resize(data.lines.size());

    velocity_global.resize(data.lines.size());
    wall_shear_global.resize(data.lines.size());
    reynolds_global.resize(data.lines.size());
    pressure_global.resize(data.lines.size());
    reference_pressure_global.resize(data.lines.size());

    // make sure that line type is correct
    std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
      std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(data.lines[0]);
    AssertThrow(line_hom.get() != 0,
                dealii::ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));
    averaging_direction = line_hom->averaging_direction;

    AssertThrow(averaging_direction == 0 or averaging_direction == 1 or averaging_direction == 2,
                dealii::ExcMessage("Take the average either in x, y or z-direction"));

    unsigned int line_iterator = 0;
    for(const std::shared_ptr<Line<dim>> & line : data.lines)
    {
      // make sure that line type is correct
      std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
        std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(line);

      AssertThrow(line_hom.get() != 0,
                  dealii::ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));

      AssertThrow(averaging_direction == line_hom->averaging_direction,
                  dealii::ExcMessage("All lines must use the same averaging direction."));

      // Resize global variables for # of points on line
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        velocity_global[line_iterator].resize(line->n_points);
        pressure_global[line_iterator].resize(line->n_points);
        wall_shear_global[line_iterator].resize(line->n_points);
        reynolds_global[line_iterator].resize(line->n_points);
      }

      // initialize global_points: use equidistant points along line
      global_points[line_iterator].reserve(line->n_points);
      for(unsigned int i = 0; i < line->n_points; ++i)
      {
        dealii::Point<dim> point =
          line->begin + double(i) / double(line->n_points - 1) * (line->end - line->begin);
        global_points[line_iterator].push_back(point);
      }
      ++line_iterator;
    }

    // Save all cells and corresponding points on unit cell
    // that are relevant for a given point along the line.

    // use a tolerance to check whether a point is inside the unit cell
    double const tolerance = 1.e-10;

    // For velocity quantities:
    for(auto const & cell : dof_handler_velocity.active_cell_iterators())
    {
      if(cell->is_locally_owned())
      {
        unsigned int line_iterator = 0;
        for(const std::shared_ptr<Line<dim>> & line : data.lines)
        {
          AssertThrow(line->quantities.size() > 0,
                      dealii::ExcMessage("No quantities specified for line."));

          bool velocity_has_to_be_evaluated = false;
          for(const std::shared_ptr<Quantity> & quantity : line->quantities)
          {
            if(quantity->type == QuantityType::Velocity or
               quantity->type == QuantityType::SkinFriction or
               quantity->type == QuantityType::ReynoldsStresses)
            {
              velocity_has_to_be_evaluated = true;
            }
          }

          if(velocity_has_to_be_evaluated == true)
          {
            bool found_a_point_on_this_cell = false;
            // cells and reference points for all points along a line
            for(unsigned int p = 0; p < line->n_points; ++p)
            {
              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              dealii::Point<dim> translated_point   = global_points[line_iterator][p];
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into
              // account
              dealii::Point<dim> const p_unit = find_unit_point(mapping, cell, translated_point);
              if(dealii::GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
              {
                if(not found_a_point_on_this_cell)
                {
                  cells_and_ref_points_velocity[line_iterator].emplace_back(
                    cell, std::vector<std::pair<unsigned int, dealii::Point<dim>>>());
                  found_a_point_on_this_cell = true;
                }
                cells_and_ref_points_velocity[line_iterator].back().second.emplace_back(p, p_unit);
              }
            }
          }
          ++line_iterator;
        }
      }
    }

    // Save all cells and corresponding points on unit cell that are relevant for a given point
    // along the line. We have to do the same for the pressure because the dealii::DoFHandlers for
    // velocity and pressure are different.
    for(auto const & cell : dof_handler_pressure.active_cell_iterators())
    {
      if(cell->is_locally_owned())
      {
        unsigned int line_iterator = 0;
        for(const std::shared_ptr<Line<dim>> & line : data.lines)
        {
          AssertThrow(line->quantities.size() > 0,
                      dealii::ExcMessage("No quantities specified for line."));
          for(const std::shared_ptr<Quantity> & quantity : line->quantities)
          {
            // evaluate quantities that involve pressure
            bool found_a_point_on_this_cell = false;
            if(quantity->type == QuantityType::Pressure)
            {
              // cells and reference points for all points along a line
              for(unsigned int p = 0; p < line->n_points; ++p)
              {
                // First, we move the line to the position of the current cell (vertex 0) in
                // averaging direction and check whether this new point is inside the current cell
                dealii::Point<dim> translated_point   = global_points[line_iterator][p];
                translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

                // If the new point lies in the current cell, we have to take the current cell into
                // account
                dealii::Point<dim> const p_unit = find_unit_point(mapping, cell, translated_point);
                if(dealii::GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
                {
                  if(not found_a_point_on_this_cell)
                  {
                    cells_and_ref_points_pressure[line_iterator].emplace_back(
                      cell, std::vector<std::pair<unsigned int, dealii::Point<dim>>>());
                    found_a_point_on_this_cell = true;
                  }
                  cells_and_ref_points_pressure[line_iterator].back().second.emplace_back(p,
                                                                                          p_unit);
                }
              }
            }
          }

          // cells and reference points for reference pressure (only one point for each line)
          for(const std::shared_ptr<Quantity> & quantity : line->quantities)
          {
            AssertThrow(line->quantities.size() > 0,
                        dealii::ExcMessage("No quantities specified for line."));

            // evaluate quantities that involve pressure
            if(quantity->type == QuantityType::PressureCoefficient)
            {
              std::shared_ptr<QuantityPressureCoefficient<dim>> quantity_ref_pressure =
                std::dynamic_pointer_cast<QuantityPressureCoefficient<dim>>(quantity);

              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              dealii::Point<dim> translated_point   = quantity_ref_pressure->reference_point;
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into
              // account
              dealii::Point<dim> const p_unit = find_unit_point(mapping, cell, translated_point);
              if(dealii::GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
              {
                cells_and_ref_points_ref_pressure[line_iterator].push_back(
                  std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,
                            dealii::Point<dim>>(cell, p_unit));
              }
            }
          }
          ++line_iterator;
        }
      }
    }

    // Initialize non-matching mapping info
    nonmatching_mapping_info = std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, double>>(
      mapping, dealii::update_values | dealii::update_jacobians | dealii::update_gradients);
    std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> cells;
    std::vector<std::vector<dealii::Point<dim>>>                        unit_points;
    dealii::QGauss<1> gauss_1d(dof_handler_velocity.get_fe().degree + 1);
    for(const auto & line_points : cells_and_ref_points_velocity)
      for(const auto & cell_and_pts : line_points)
      {
        cells.push_back(cell_and_pts.first);
        unit_points.emplace_back();
        unit_points.back().resize(cell_and_pts.second.size() * gauss_1d.size());
        for(unsigned int p = 0, idx = 0; p < cell_and_pts.second.size(); ++p)
          for(unsigned int q = 0; q < gauss_1d.size(); ++q, ++idx)
            for(unsigned int d = 0; d < dim; ++d)
              unit_points.back()[idx][d] = (d == averaging_direction) ?
                                             gauss_1d.point(q)[0] :
                                             cell_and_pts.second[p].second[d];
      }
    nonmatching_mapping_info->reinit_cells(cells, unit_points);

    create_directories(data.directory, mpi_comm);
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::evaluate(VectorType const & velocity,
                                                               VectorType const & pressure)
{
  do_evaluate(velocity, pressure);
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::write_output() const
{
  do_write_output();
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::print_headline(
  std::ofstream &    f,
  unsigned int const number_of_samples) const
{
  f << "number of samples: N = " << number_of_samples << std::endl;
}

void
mpi_sum_at_root(double * data_ptr, const unsigned int size, const MPI_Comm mpi_comm)
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    int ierr = MPI_Reduce(MPI_IN_PLACE, data_ptr, size, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    AssertThrowMPI(ierr);
  }
  else
  {
    int ierr = MPI_Reduce(data_ptr, data_ptr, size, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    AssertThrowMPI(ierr);
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_evaluate(VectorType const & velocity,
                                                                  VectorType const & pressure)
{
  dealii::Timer time;
  // increment number of samples
  number_of_samples++;

  std::vector<std::vector<double>>                         length_local(data.lines.size());
  std::vector<std::vector<dealii::Tensor<1, dim, double>>> velocity_local(data.lines.size());
  std::vector<std::vector<double>>                         wall_shear_local(data.lines.size());
  std::vector<std::vector<dealii::Tensor<2, dim, double>>> reynolds_local(data.lines.size());
  std::vector<std::vector<double>>                         pressure_local(data.lines.size());
  std::vector<double> reference_pressure_local(data.lines.size());

  dealii::FiniteElement<dim> const & fe_u = dof_handler_velocity.get_fe();

  // use quadrature for averaging in homogeneous direction
  const unsigned int                  n_q_points_1d = fe_u.degree + 1;
  dealii::QGauss<1>                   gauss_1d(n_q_points_1d);
  std::vector<dealii::Point<dim>>     points; // 1D points combined for several line points
  dealii::FE_DGQ<dim>                 fe_dgq(fe_u.degree + 1);
  std::vector<double>                 velocity_dgq_on_cell(fe_dgq.dofs_per_cell * dim);
  dealii::FEPointEvaluation<dim, dim> evaluator_u(*nonmatching_mapping_info, fe_dgq);

  dealii::FiniteElement<dim> const &           fe_p = dof_handler_pressure.get_fe();
  std::vector<double>                          pressure_on_cell(fe_p.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices_p(fe_p.dofs_per_cell);
  dealii::FEPointEvaluation<1, dim>            evaluator_p(mapping,
                                                fe_p,
                                                dealii::update_values | dealii::update_jacobians |
                                                  dealii::update_quadrature_points |
                                                  dealii::update_gradients);
  velocity.update_ghost_values();
  pressure.update_ghost_values();

  unsigned int counter_all_cells = 0;
  for(unsigned int index = 0; index < data.lines.size(); ++index)
  {
    Line<dim> & line = *data.lines[index];

    bool evaluate_velocity = false;
    for(const std::shared_ptr<Quantity> & quantity : line.quantities)
    {
      // evaluate quantities that involve velocity
      if(quantity->type == QuantityType::Velocity or quantity->type == QuantityType::SkinFriction or
         quantity->type == QuantityType::ReynoldsStresses)
      {
        evaluate_velocity = true;
      }
    }

    if(evaluate_velocity == true)
    {
      length_local[index].resize(line.n_points);
      velocity_local[index].resize(line.n_points);
      reynolds_local[index].resize(line.n_points);
      wall_shear_local[index].resize(line.n_points);

      for(auto const & [cell, point_list] : cells_and_ref_points_velocity[index])
      {
        evaluator_tensor_product.reinit(cell);
        evaluator_tensor_product.read_dof_values(velocity);
        evaluator_tensor_product.evaluate(dealii::EvaluationFlags::values);
        for(unsigned int q : evaluator_tensor_product.quadrature_point_indices())
        {
          const dealii::Tensor<1, dim, dealii::VectorizedArray<Number, 1>> vel =
            evaluator_tensor_product.get_value(q);
          for(unsigned int d = 0; d < dim; ++d)
            velocity_dgq_on_cell[q + d * fe_dgq.dofs_per_cell] = vel[d][0];
        }

        evaluator_u.reinit(counter_all_cells);
        ++counter_all_cells;
        evaluator_u.evaluate(velocity_dgq_on_cell,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

        // perform averaging in homogeneous direction
        for(unsigned int p1 = 0, q = 0; p1 < point_list.size(); ++p1)
          for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1, ++q)
          {
            unsigned int const p = point_list[p1].first;
            double const       det =
              std::abs(evaluator_u.jacobian(q)[averaging_direction][averaging_direction]);
            double const JxW = det * gauss_1d.weight(q1);

            // calculate integrals in homogeneous direction
            length_local[index][p] += JxW;

            for(const std::shared_ptr<Quantity> & quantity : line.quantities)
            {
              dealii::Tensor<1, dim> const velocity = evaluator_u.get_value(q);

              if(quantity->type == QuantityType::Velocity)
              {
                for(unsigned int d = 0; d < dim; ++d)
                  velocity_local[index][p][d] += velocity[d] * JxW;
              }
              else if(quantity->type == QuantityType::ReynoldsStresses)
              {
                for(unsigned int d = 0; d < dim; ++d)
                  for(unsigned int e = 0; e < dim; ++e)
                    reynolds_local[index][p][d][e] += velocity[d] * velocity[e] * JxW;
              }
              else if(quantity->type == QuantityType::SkinFriction)
              {
                dealii::Tensor<2, dim> const velocity_gradient = evaluator_u.get_gradient(q);

                std::shared_ptr<QuantitySkinFriction<dim>> quantity_skin_friction =
                  std::dynamic_pointer_cast<QuantitySkinFriction<dim>>(quantity);

                dealii::Tensor<1, dim, double> normal  = quantity_skin_friction->normal_vector;
                dealii::Tensor<1, dim, double> tangent = quantity_skin_friction->tangent_vector;

                for(unsigned int d = 0; d < dim; ++d)
                  for(unsigned int e = 0; e < dim; ++e)
                    wall_shear_local[index][p] +=
                      tangent[d] * velocity_gradient[d][e] * normal[e] * JxW;
              }
            }
          }
      }
    }

    for(const std::shared_ptr<Quantity> & quantity : line.quantities)
    {
      // evaluate quantities that involve velocity
      if(quantity->type == QuantityType::Pressure)
      {
        const bool must_collect_line = length_local[index].empty();
        if(must_collect_line)
          length_local[index].resize(line.n_points);
        pressure_local[index].resize(line.n_points);

        for(auto const & [cell, point_list] : cells_and_ref_points_pressure[index])
        {
          points.resize(point_list.size() * gauss_1d.size());
          for(unsigned int p = 0, idx = 0; p < point_list.size(); ++p)
            for(unsigned int q = 0; q < gauss_1d.size(); ++q, ++idx)
              for(unsigned int d = 0; d < dim; ++d)
                points[idx][d] =
                  (d == averaging_direction) ? gauss_1d.point(q)[0] : point_list[p].second[d];

          evaluator_p.reinit(cell, points);
          cell->get_dof_indices(dof_indices_p);

          for(unsigned int j = 0; j < dof_indices_p.size(); ++j)
            pressure_on_cell[j] = pressure(dof_indices_p[j]);
          evaluator_p.evaluate(pressure_on_cell, dealii::EvaluationFlags::values);

          for(unsigned int p1 = 0, q = 0; p1 < point_list.size(); ++p1)
            for(unsigned int q1 = 0; q1 < gauss_1d.size(); ++q1, ++q)
            {
              unsigned int const p = point_list[p1].first;

              double det =
                std::abs(evaluator_p.jacobian(q)[averaging_direction][averaging_direction]);
              double JxW = det * gauss_1d.weight(q);

              if(must_collect_line)
                length_local[index][p] += JxW;
              pressure_local[index][p] += evaluator_p.get_value(q) * JxW;
            }
        }
      }
      if(quantity->type == QuantityType::PressureCoefficient)
      {
        double length_local   = 0.0;
        double pressure_local = 0.0;

        TYPE vector_cells_and_ref_points = cells_and_ref_points_ref_pressure[index];

        average_pressure_for_given_point(pressure,
                                         vector_cells_and_ref_points,
                                         length_local,
                                         pressure_local);

        // MPI communication
        length_local   = dealii::Utilities::MPI::sum(length_local, mpi_comm);
        pressure_local = dealii::Utilities::MPI::sum(pressure_local, mpi_comm);

        // averaging in space (over homogeneous direction)
        reference_pressure_global[index] += pressure_local / length_local;
      }
    }
  }

  velocity.zero_out_ghost_values();
  pressure.zero_out_ghost_values();
  for(unsigned int line = 0; line < data.lines.size(); ++line)
  {
    mpi_sum_at_root(length_local[line].data(), length_local[line].size(), mpi_comm);
    for(const std::shared_ptr<Quantity> & quantity : data.lines[line]->quantities)
    {
      // Cells are distributed over processors, therefore we need
      // to sum the contributions of every single processor.
      if(quantity->type == QuantityType::Velocity)
      {
        mpi_sum_at_root(&velocity_local[line][0][0], velocity_local[line].size() * dim, mpi_comm);

        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < data.lines[line]->n_points; ++p)
          {
            velocity_global[line][p] += velocity_local[line][p] / length_local[line][p];
          }
      }
      else if(quantity->type == QuantityType::ReynoldsStresses)
      {
        mpi_sum_at_root(&reynolds_local[line][0][0][0],
                        reynolds_local[line].size() * dim * dim,
                        mpi_comm);

        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < data.lines[line]->n_points; ++p)
          {
            reynolds_global[line][p] += reynolds_local[line][p] / length_local[line][p];
          }
      }
      else if(quantity->type == QuantityType::SkinFriction)
      {
        mpi_sum_at_root(wall_shear_local[line].data(), wall_shear_local[line].size(), mpi_comm);

        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < data.lines[line]->n_points; ++p)
          {
            wall_shear_global[line][p] += wall_shear_local[line][p] / length_local[line][p];
          }
      }
      else if(quantity->type == QuantityType::Pressure)
      {
        mpi_sum_at_root(pressure_local[line].data(), pressure_local[line].size(), mpi_comm);

        // averaging in space (over homogeneous direction)
        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < data.lines[line]->n_points; ++p)
            pressure_global[line][p] += pressure_local[line][p] / length_local[line][p];
      }
    }
  }

  time_all += time.wall_time();
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::average_pressure_for_given_point(
  VectorType const & pressure,
  TYPE const &       vector_cells_and_ref_points,
  double &           length_local,
  double &           pressure_local)
{
  dealii::FiniteElement<dim> const &           fe = dof_handler_pressure.get_fe();
  std::vector<double>                          pressure_on_cell(fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

  // use quadrature for averaging in homogeneous direction
  dealii::QGauss<1>                 gauss_1d(fe.degree + 1);
  std::vector<dealii::Point<dim>>   points(gauss_1d.size());  // 1D points
  std::vector<double>               weights(gauss_1d.size()); // 1D weights
  dealii::FEPointEvaluation<1, dim> evaluator(mapping,
                                              fe,
                                              dealii::update_values | dealii::update_jacobians |
                                                dealii::update_quadrature_points |
                                                dealii::update_gradients);

  for(typename TYPE::const_iterator cell_and_ref_point = vector_cells_and_ref_points.begin();
      cell_and_ref_point != vector_cells_and_ref_points.end();
      ++cell_and_ref_point)
  {
    dealii::Point<dim> const p_unit = cell_and_ref_point->second;

    // Find points for Gauss quadrature
    for(unsigned int i = 0; i < gauss_1d.size(); ++i)
      for(unsigned int d = 0; d < dim; ++d)
        points[i][d] = (d == averaging_direction) ? gauss_1d.point(i)[0] : p_unit[d];

    typename dealii::DoFHandler<dim>::active_cell_iterator const cell = cell_and_ref_point->first;
    evaluator.reinit(cell, points);
    cell->get_dof_indices(dof_indices);

    for(unsigned int j = 0; j < dof_indices.size(); ++j)
      pressure_on_cell[j] = pressure(dof_indices[j]);
    evaluator.evaluate(pressure_on_cell,
                       dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    for(unsigned int q = 0; q < points.size(); ++q)
    {
      double const p = evaluator.get_value(q);

      double det = std::abs(evaluator.jacobian(q)[averaging_direction][averaging_direction]);
      double JxW = det * gauss_1d.weight(q);

      length_local += JxW;
      pressure_local += p * JxW;
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_write_output() const
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    dealii::Timer      time;
    unsigned int const precision = data.precision;

    // Iterator for lines
    unsigned int line_iterator = 0;
    for(const std::shared_ptr<Line<dim>> & line : data.lines)
    {
      std::string filename_prefix = data.directory + line->name;

      for(const std::shared_ptr<Quantity> & quantity : line->quantities)
      {
        if(quantity->type == QuantityType::Velocity)
        {
          std::string   filename = filename_prefix + "_velocity" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "u_" + dealii::Utilities::int_to_string(d + 1);

          f << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            // write velocity and average over time
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left
                << velocity_global[line_iterator][p][d] / number_of_samples;

            f << std::endl;
          }
          f.close();
        }

        if(quantity->type == QuantityType::ReynoldsStresses)
        {
          std::string   filename = filename_prefix + "_reynoldsstresses" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          for(unsigned int i = 0; i < dim; ++i)
          {
            for(unsigned int j = 0; j < dim; ++j)
            {
              f << std::setw(precision + 8) << std::left
                << "u_" + dealii::Utilities::int_to_string(i + 1) + "u_" +
                     dealii::Utilities::int_to_string(j + 1);
            }
          }
          f << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            for(unsigned int i = 0; i < dim; ++i)
            {
              for(unsigned int j = 0; j < dim; ++j)
              {
                // equation <u_i' u_j'> = <u_i*u_j> - <u_i> * <u_j>
                f << std::setw(precision + 8) << std::left
                  << reynolds_global[line_iterator][p][i][j] / number_of_samples -
                       (velocity_global[line_iterator][p][i] / number_of_samples) *
                         (velocity_global[line_iterator][p][j] / number_of_samples);
              }
            }

            f << std::endl;
          }
          f.close();
        }

        if(quantity->type == QuantityType::SkinFriction)
        {
          std::shared_ptr<QuantitySkinFriction<dim>> averaging_quantity =
            std::dynamic_pointer_cast<QuantitySkinFriction<dim>>(quantity);

          std::string   filename = filename_prefix + "_wall_shear_stress" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          f << std::setw(precision + 8) << std::left << "tau_w" << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            // tau_w -> C_f = tau_w / (1/2 rho u²)
            double const viscosity = averaging_quantity->viscosity;
            f << std::setw(precision + 8) << std::left
              << viscosity * wall_shear_global[line_iterator][p] / number_of_samples;

            f << std::endl;
          }
          f.close();
        }

        if(quantity->type == QuantityType::Pressure or
           quantity->type == QuantityType::PressureCoefficient)
        {
          std::string   filename = filename_prefix + "_pressure" + ".txt";
          std::ofstream f;

          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          f << std::setw(precision + 8) << std::left << "p";

          if(quantity->type == QuantityType::PressureCoefficient)
            f << std::setw(precision + 8) << std::left << "p-p_ref";

          f << std::endl;

          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            f << std::setw(precision + 8) << std::left
              << pressure_global[line_iterator][p] / number_of_samples;

            if(quantity->type == QuantityType::PressureCoefficient)
            {
              // p - p_ref -> C_p = (p - p_ref) / (1/2 rho u²)
              f << std::left
                << (pressure_global[line_iterator][p] - reference_pressure_global[line_iterator]) /
                     number_of_samples;
            }
            f << std::endl;
          }
          f.close();
        }
      }
      ++line_iterator;
    }
    const_cast<double &>(time_all) += time.wall_time();
    std::cout << "Accumulated times for " << number_of_samples
              << " output line stats t = " << time_all << " s" << std::endl;
  }
}

template class LinePlotCalculatorStatisticsHomogeneous<2, float>;
template class LinePlotCalculatorStatisticsHomogeneous<3, float>;

template class LinePlotCalculatorStatisticsHomogeneous<2, double>;
template class LinePlotCalculatorStatisticsHomogeneous<3, double>;

} // namespace IncNS
} // namespace ExaDG
