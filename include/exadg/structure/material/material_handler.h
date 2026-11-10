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

#ifndef EXADG_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_
#define EXADG_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_

// C/C++
#include <memory>
#include <type_traits>

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/structure/material/library/compressible_neo_hookean.h>
#include <exadg/structure/material/library/incompressible_fibrous_tissue.h>
#include <exadg/structure/material/library/incompressible_neo_hookean.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/material/material.h>
#include <exadg/structure/user_interface/material_descriptor.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class MaterialHandler
{
public:
  typedef std::pair<dealii::types::material_id, std::shared_ptr<Material<dim, Number>>> Pair;
  typedef std::map<dealii::types::material_id, std::shared_ptr<Material<dim, Number>>>  Materials;

  MaterialHandler() : dof_index(0)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             unsigned int const                        dof_index,
             unsigned int const                        quad_index,
             std::shared_ptr<MaterialDescriptor const> material_descriptor,
             bool const                                large_deformation,
             bool const                                spatial_integration,
             bool const                                force_material_residual,
             InvalidJacobianTreatment const            invalid_jacobian_treatment,
             bool const                                stable_formulation,
             CacheLevel const                          cache_level)
  {
    this->dof_index           = dof_index;
    this->material_descriptor = material_descriptor;

    for(auto iter = material_descriptor->begin(); iter != material_descriptor->end(); ++iter)
    {
      dealii::types::material_id    id   = iter->first;
      std::shared_ptr<MaterialData> data = iter->second;
      MaterialType                  type = data->type;

      switch(type)
      {
        case MaterialType::Undefined:
        {
          AssertThrow(false, dealii::ExcMessage("Material type is undefined."));
          break;
        }
        case MaterialType::StVenantKirchhoff:
        {
          std::shared_ptr<StVenantKirchhoffData<dim>> data_StVenantKirchhoff =
            std::static_pointer_cast<StVenantKirchhoffData<dim>>(data);
          material_map.insert(Pair(id,
                                   new StVenantKirchhoff<dim, Number>(matrix_free,
                                                                      dof_index,
                                                                      quad_index,
                                                                      *data_StVenantKirchhoff,
                                                                      large_deformation)));
          break;
        }
        case MaterialType::IncompressibleNeoHookean:
        {
          AssertThrow(
            large_deformation == true,
            dealii::ExcMessage(
              "Incompressible Neo-Hookean material model defined for finite strain theory."));

          std::shared_ptr<IncompressibleNeoHookeanData<dim>> data_IncompressibleNeoHookean =
            std::static_pointer_cast<IncompressibleNeoHookeanData<dim>>(data);

          material_map.insert(
            Pair(id,
                 create_nonlinear_material<IncompressibleNeoHookean>(matrix_free,
                                                                     dof_index,
                                                                     quad_index,
                                                                     *data_IncompressibleNeoHookean,
                                                                     spatial_integration,
                                                                     force_material_residual,
                                                                     invalid_jacobian_treatment,
                                                                     stable_formulation,
                                                                     cache_level)));
          break;
        }
        case MaterialType::IncompressibleFibrousTissue:
        {
          AssertThrow(large_deformation == true,
                      dealii::ExcMessage(
                        "Incompressible fiber model defined for finite strain theory."));

          std::shared_ptr<IncompressibleFibrousTissueData<dim>> data_IncompressibleFibrousTissue =
            std::static_pointer_cast<IncompressibleFibrousTissueData<dim>>(data);

          material_map.insert(Pair(id,
                                   create_nonlinear_material<IncompressibleFibrousTissue>(
                                     matrix_free,
                                     dof_index,
                                     quad_index,
                                     *data_IncompressibleFibrousTissue,
                                     spatial_integration,
                                     force_material_residual,
                                     invalid_jacobian_treatment,
                                     stable_formulation,
                                     cache_level)));

          break;
        }
        case MaterialType::CompressibleNeoHookean:
        {
          AssertThrow(
            large_deformation == true,
            dealii::ExcMessage(
              "Compressible Neo-Hookean material model defined for finite strain theory."));

          std::shared_ptr<CompressibleNeoHookeanData<dim>> data_CompressibleNeoHookean =
            std::static_pointer_cast<CompressibleNeoHookeanData<dim>>(data);

          material_map.insert(
            Pair(id,
                 create_nonlinear_material<CompressibleNeoHookean>(matrix_free,
                                                                   dof_index,
                                                                   quad_index,
                                                                   *data_CompressibleNeoHookean,
                                                                   spatial_integration,
                                                                   force_material_residual,
                                                                   invalid_jacobian_treatment,
                                                                   stable_formulation,
                                                                   cache_level)));
          break;
        }
        default:
        {
          AssertThrow(false, dealii::ExcMessage("Specified material type is not implemented."));
          break;
        }
      }
    }
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const cell)
  {
    auto mid = matrix_free.get_cell_iterator(cell, 0, dof_index)->material_id();

#ifdef DEBUG
    for(unsigned int v = 1; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
      AssertThrow(mid == matrix_free.get_cell_iterator(cell, v)->material_id(),
                  dealii::ExcMessage("You have to categorize cells according to their materials!"));
#endif

    material = material_map[mid];
  }

  std::shared_ptr<Material<dim, Number>>
  get_material() const
  {
    return material;
  }

private:
  /*
   * Creates the nonlinear material model `MaterialModel`, converting the
   * run-time parameters to the template parameters of the material model.
   */
  template<template<int, typename, InvalidJacobianTreatment, bool, CacheLevel> class MaterialModel,
           typename MaterialModelData>
  static std::shared_ptr<Material<dim, Number>>
  create_nonlinear_material(dealii::MatrixFree<dim, Number> const & matrix_free,
                            unsigned int const                      dof_index,
                            unsigned int const                      quad_index,
                            MaterialModelData const &               data,
                            bool const                              spatial_integration,
                            bool const                              force_material_residual,
                            InvalidJacobianTreatment const          invalid_jacobian_treatment,
                            bool const                              stable_formulation,
                            CacheLevel const                        cache_level)
  {
    // Create the material model for template arguments given as
    // std::integral_constant.
    auto const create = [&](auto treatment, auto stable, auto level) {
      return std::static_pointer_cast<Material<dim, Number>>(
        std::make_shared<MaterialModel<dim,
                                       Number,
                                       decltype(treatment)::value,
                                       decltype(stable)::value,
                                       decltype(level)::value>>(
          matrix_free, dof_index, quad_index, data, spatial_integration, force_material_residual));
    };

    // Convert `stable_formulation` and `cache_level`.
    auto const create_any_stable_and_level = [&](auto treatment) {
      auto const create_any_level = [&](auto stable) {
        switch(cache_level)
        {
          case CacheLevel::None:
            return create(treatment,
                          stable,
                          std::integral_constant<CacheLevel, CacheLevel::None>());
          case CacheLevel::Scalars:
            return create(treatment,
                          stable,
                          std::integral_constant<CacheLevel, CacheLevel::Scalars>());
          case CacheLevel::ScalarsAndTensors:
            return create(treatment,
                          stable,
                          std::integral_constant<CacheLevel, CacheLevel::ScalarsAndTensors>());
        }
        AssertThrow(false, dealii::ExcMessage("This CacheLevel is not implemented."));
        return std::shared_ptr<Material<dim, Number>>();
      };

      if(stable_formulation)
        return create_any_level(std::true_type());
      else
        return create_any_level(std::false_type());
    };

    switch(invalid_jacobian_treatment)
    {
      // The global quasi-Newton variant is implemented in the NonLinearOperator,
      // the integration points are treated as without modification.
      case InvalidJacobianTreatment::None:
      case InvalidJacobianTreatment::GlobalQuasiNewton:
        return create_any_stable_and_level(
          std::integral_constant<InvalidJacobianTreatment, InvalidJacobianTreatment::None>());
      case InvalidJacobianTreatment::LocalQuasiNewton:
      {
        // The stored gradient of the last admissible linearization is required
        // and J has to be recomputed from F.
        AssertThrow(not stable_formulation,
                    dealii::ExcMessage("InvalidJacobianTreatment::LocalQuasiNewton requires "
                                       "stable_formulation == false."));
        AssertThrow(cache_level == CacheLevel::ScalarsAndTensors and
                      ((not spatial_integration) or force_material_residual),
                    dealii::ExcMessage("InvalidJacobianTreatment::LocalQuasiNewton requires "
                                       "CacheLevel::ScalarsAndTensors and the residual "
                                       "evaluated in the material configuration, i.e., "
                                       "spatial_integration == false or "
                                       "force_material_residual == true."));
        return create(std::integral_constant<InvalidJacobianTreatment,
                                             InvalidJacobianTreatment::LocalQuasiNewton>(),
                      std::false_type(),
                      std::integral_constant<CacheLevel, CacheLevel::ScalarsAndTensors>());
      }
      case InvalidJacobianTreatment::ClampJacobian:
        return create_any_stable_and_level(
          std::integral_constant<InvalidJacobianTreatment,
                                 InvalidJacobianTreatment::ClampJacobian>());
      case InvalidJacobianTreatment::IdentityShiftExact:
        return create_any_stable_and_level(
          std::integral_constant<InvalidJacobianTreatment,
                                 InvalidJacobianTreatment::IdentityShiftExact>());
      case InvalidJacobianTreatment::IdentityShiftNewton:
        return create_any_stable_and_level(
          std::integral_constant<InvalidJacobianTreatment,
                                 InvalidJacobianTreatment::IdentityShiftNewton>());
    }

    AssertThrow(false, dealii::ExcMessage("This InvalidJacobianTreatment is not implemented."));
    return std::shared_ptr<Material<dim, Number>>();
  }

  unsigned int dof_index;

  std::shared_ptr<MaterialDescriptor const> material_descriptor;
  Materials                                 material_map;

  // pointer to material of current cell
  std::shared_ptr<Material<dim, Number>> material;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_ */
