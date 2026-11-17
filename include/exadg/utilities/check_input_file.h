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
#include <sstream>
#include <string>
#include <vector>

// boost
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

namespace ExaDG
{
namespace internal
{
/**
 * Appends the paths of all entries and subsections in `input` that are not
 * declared in `declared` to `undeclared`.
 */
inline void
collect_undeclared_entries(boost::property_tree::ptree const & input,
                           boost::property_tree::ptree const & declared,
                           std::string const &                 path,
                           std::vector<std::string> &          undeclared)
{
  for(auto const & [name, node] : input)
  {
    std::string const node_path = path.empty() ? name : path + "." + name;

    auto const declared_node = declared.find(name);
    if(declared_node == declared.not_found())
    {
      // An empty subsection, e.g., `"Application": {}`, contains nothing to
      // ignore. The property tree does not distinguish it from an entry with
      // an empty value.
      if(not(node.empty() and node.data().empty()))
        undeclared.push_back(node_path);
    }
    else if(not declared_node->second.empty())
      collect_undeclared_entries(node, declared_node->second, node_path, undeclared);
  }
}
} // namespace internal

/**
 * Throws an exception listing all entries of the JSON `input_file` that are
 * not declared in `prm`, e.g., misspelled ones. The parameter classes parse
 * the input file separately with `skip_undefined = true`, which silently
 * ignores such entries. Hence, `prm` has to contain the parameters of all
 * classes reading `input_file`. Other formats than JSON are not checked.
 */
inline void
check_input_file(std::string const & input_file, dealii::ParameterHandler const & prm)
{
  std::string const extension = ".json";
  if(input_file.size() < extension.size() or
     input_file.compare(input_file.size() - extension.size(), extension.size(), extension) != 0)
    return;

  // Compare with the declared parameters as printed, since printing does not
  // call the actions of parameters added via `add_parameter()`.
  std::stringstream declared_stream;
  prm.print_parameters(declared_stream,
                       dealii::ParameterHandler::JSON | dealii::ParameterHandler::Short);
  boost::property_tree::ptree declared;
  boost::property_tree::read_json(declared_stream, declared);

  boost::property_tree::ptree input;
  boost::property_tree::read_json(input_file, input);

  std::vector<std::string> undeclared;
  internal::collect_undeclared_entries(input, declared, "", undeclared);

  std::string undeclared_list;
  for(std::string const & entry : undeclared)
    undeclared_list += "\n  " + entry;

  AssertThrow(undeclared.empty(),
              dealii::ExcMessage("The input file " + input_file +
                                 " contains entries that are not declared:" + undeclared_list +
                                 "\nCheck the spelling or remove these entries."));
}

} // namespace ExaDG

#endif /* EXADG_UTILITIES_CHECK_INPUT_FILE_H_ */
