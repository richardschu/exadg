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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_

namespace ExaDG
{
namespace FSI
{
enum class AccelerationMethod
{
  Undefined,
  Aitken,
  IQN_ILS,
  IQN_IMVLS
};

enum class CouplingScheme
{
  Undefined,
  DirichletNeumann,
  DirichletRobinFixedParam,
  DirichletRobinAdaptiveParam
};

enum class FieldUpdateVariant
{
  Undefined,
  Implicit,
  GeometricExplicit,
  ImplicitPressureStructure
};

struct Parameters
{
  Parameters()
    : acceleration_method(AccelerationMethod::Aitken),
      coupling_scheme(CouplingScheme::DirichletNeumann),
      field_update_variant(FieldUpdateVariant::Implicit),
      robin_parameter_scale(0.0),
      abs_tol(1.e-12),
      rel_tol(1.e-3),
      omega_init(0.1),
      reused_time_steps(0),
      partitioned_iter_max(100),
      geometric_tolerance(1.e-10)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "FSI")
  {
    // clang-format off
    prm.enter_subsection(subsection_name);
      prm.add_parameter("AccelerationMethod",
                        acceleration_method_string,
                        "Acceleration method.",
                        dealii::Patterns::Anything(),
                        true);
      prm.add_parameter("CouplingScheme",
    		            coupling_scheme_string,
						"Partitioned coupling scheme.",
						dealii::Patterns::Anything(),
						false);
      prm.add_parameter("FieldUpdateVariant",
          		        field_update_variant_string,
      					"Update variants in partitioned coupling.",
      					dealii::Patterns::Anything(),
      					false);
      prm.add_parameter("RobinParameterScale",
    		            robin_parameter_scale,
						"Additional scaling for Robin parameter.",
						dealii::Patterns::Double(),
						false);
      prm.add_parameter("AbsTol",
                        abs_tol,
                        "Absolute solver tolerance.",
                        dealii::Patterns::Double(0.0,1.0),
                        true);
      prm.add_parameter("RelTol",
                        rel_tol,
                        "Relative solver tolerance.",
                        dealii::Patterns::Double(0.0,1.0),
                        true);
      prm.add_parameter("OmegaInit",
                        omega_init,
                        "Initial relaxation parameter.",
                        dealii::Patterns::Double(0.0,1.0),
                        true);
      prm.add_parameter("ReusedTimeSteps",
                        reused_time_steps,
                        "Number of time steps reused for acceleration.",
                        dealii::Patterns::Integer(0, 100),
                        false);
      prm.add_parameter("PartitionedIterMax",
                        partitioned_iter_max,
                        "Maximum number of fixed-point iterations.",
                        dealii::Patterns::Integer(1,1000),
                        true);
      prm.add_parameter("GeometricTolerance",
                        geometric_tolerance,
                        "Tolerance used to locate points at FSI interface.",
                        dealii::Patterns::Double(0.0, 1.0),
                        false);
    prm.leave_subsection();
    // clang-format on
  }

  void
  parse_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "FSI")
  {
    prm.enter_subsection(subsection_name);

    Utilities::string_to_enum(acceleration_method, prm.get("AccelerationMethod"));

    std::string const coupling_scheme_string = prm.get("CouplingScheme");
    Utilities::string_to_enum(coupling_scheme, coupling_scheme_string);

    std::string const field_update_variant_string = prm.get("FieldUpdateVariant");
    Utilities::string_to_enum(field_update_variant, field_update_variant_string);

    prm.leave_subsection();
  }

  AccelerationMethod acceleration_method;
  CouplingScheme     coupling_scheme;
  FieldUpdateVariant field_update_variant;

  double       robin_parameter_scale;
  double       abs_tol;
  double       rel_tol;
  double       omega_init;
  unsigned int reused_time_steps;
  unsigned int partitioned_iter_max;

  // tolerance used to locate points at the fluid-structure interface
  double geometric_tolerance;

private:
  std::string acceleration_method_string;
  std::string coupling_scheme_string;
  std::string field_update_variant_string;
};
} // namespace FSI
} // namespace ExaDG



#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_ */
