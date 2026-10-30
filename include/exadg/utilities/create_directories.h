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

#ifndef EXADG_UTILITIES_CREATE_DIRECTORIES_H_
#define EXADG_UTILITIES_CREATE_DIRECTORIES_H_

// C/C++
#include <filesystem>

// deal.II
#include <deal.II/base/exceptions.h>

namespace ExaDG
{
/**
 * Creates directories if not already existing. An MPI barrier ensures that
 * directories have been created for all processes when completing this function.
 *
 * A relative directory in `output` (e.g., `output/`, `./output/` or
 * `output/poiseuille/`, but not `output_test/`) is not allowed: the deal.II test
 * suite writes the screen output of a test to a file `output` in the working
 * directory, which collides with a directory of the same name.
 */
inline void
create_directories(std::string const & directory, MPI_Comm const & mpi_comm)
{
  std::filesystem::path const path = std::filesystem::path(directory).lexically_normal();
  AssertThrow(path.empty() or path.is_absolute() or *path.begin() != "output",
              dealii::ExcMessage("The directory \"" + directory +
                                 "\" is in \"output/\", which collides with the file "
                                 "\"output\" of the deal.II test suite. Use, e.g., "
                                 "\"results/\" instead."));

  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    std::filesystem::create_directories(directory);

  MPI_Barrier(mpi_comm);
}

} // namespace ExaDG

#endif /* EXADG_UTILITIES_CREATE_DIRECTORIES_H_ */
