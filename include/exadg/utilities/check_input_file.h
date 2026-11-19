/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
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

#ifndef EXADG_UTILITIES_CHECK_INPUT_FILE_H_
#define EXADG_UTILITIES_CHECK_INPUT_FILE_H_

// C/C++
#include <string>

// deal.II
#include <deal.II/base/parameter_handler.h>

namespace ExaDG
{
/**
 * Throws an exception listing all entries of the JSON `input_file` that are
 * not declared in `prm`, e.g., misspelled ones. The parameter classes parse
 * the input file separately with `skip_undefined = true`, which silently
 * ignores such entries. Hence, `prm` has to contain the parameters of all
 * classes reading `input_file`. Other formats than JSON are not checked.
 */
void
check_input_file(std::string const & input_file, dealii::ParameterHandler const & prm);

} // namespace ExaDG

#endif /* EXADG_UTILITIES_CHECK_INPUT_FILE_H_ */
