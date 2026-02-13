/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
 *  Richard Schussnig
 *
 *  This file is dual-licensed under the Apache-2.0 with LLVM Exception (see
 *  https://spdx.org/licenses/Apache-2.0.html and
 *  https://spdx.org/licenses/LLVM-exception.html) and the GNU General Public
 *  License as published by the Free Software Foundation, either version 3 of
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License in the top-level LICENSE file for
 *  more details.
 *  ______________________________________________________________________
 */

#pragma once

// deal.II
#include <deal.II/base/function.h>

namespace ExaDG
{
template<int dim, typename Number = double>
class SerializableFunction : public dealii::Function<dim, Number>
{
public:
  SerializableFunction(const unsigned int n_components = 1)
    : dealii::Function<dim, Number>(n_components)
  {
  }

  SerializableFunction(const dealii::Function<dim> & function)
    : dealii::Function<dim, Number>(function)
  {
  }

  virtual void
  write_restart_data(boost::archive::binary_oarchive &) const
  {
  }

  // The information is always read on rank zero and then broadcast to all
  // ranks by the additional function synchronize_function_parameters()
  virtual void
  read_restart_data(boost::archive::binary_iarchive &)
  {
  }


  virtual void
  broadcast_function_parameters(const MPI_Comm, const unsigned int = 0)
  {
  }
};
} // namespace ExaDG
