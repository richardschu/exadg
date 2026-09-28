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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_

// C/C++
#include <string>
#include <vector>

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace IncNS
{
template<typename Number>
class PostProcessorInterface
{
protected:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  virtual ~PostProcessorInterface()
  {
  }

  /*
   * This function has to be called to apply the postprocessing tools.
   */
  virtual void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) = 0;

  /*
   * Restart support for postprocessors with a state evolving in time, e.g., time integrals. The
   * time integrator writes the restart before the postprocessing of the same time step, so the
   * state stored is the one after the previous time step, and the postprocessing of the restart
   * time is repeated in the restarted simulation.
   *
   * Vectors with the layout of the velocity or pressure DoFHandler to be serialized. The same
   * vectors have to be provided when writing and reading the restart.
   */
  virtual void
  get_vectors_serialization(std::vector<VectorType const *> & vectors_velocity,
                            std::vector<VectorType const *> & vectors_pressure) const
  {
    (void)vectors_velocity;
    (void)vectors_pressure;
  }

  /*
   * Receive the deserialized vectors, in the sequence of `get_vectors_serialization()`.
   */
  virtual void
  set_vectors_deserialization(std::vector<VectorType> const & vectors_velocity,
                              std::vector<VectorType> const & vectors_pressure)
  {
    (void)vectors_velocity;
    (void)vectors_pressure;
  }

  /*
   * Additional state that is identical on all processes, e.g., scalars, stored in the restart
   * header. `set_restart_state()` is called on all processes with the string returned by
   * `get_restart_state()` on process 0, or with an empty string for restart files without it.
   */
  virtual std::string
  get_restart_state() const
  {
    return std::string();
  }

  virtual void
  set_restart_state(std::string const & state)
  {
    (void)state;
  }
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_ */
