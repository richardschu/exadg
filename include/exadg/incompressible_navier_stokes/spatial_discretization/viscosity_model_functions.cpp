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

#include <exadg/incompressible_navier_stokes/spatial_discretization/viscosity_model_functions.h>

namespace ExaDG
{
namespace IncNS
{
namespace ViscosityModels
{
namespace
{
/**
 *  Smagorinsky model (1963):
 *
 *    nu_SGS = (C * filter_width)^{2} * sqrt(2 * S:S)
 *
 *    where S is the symmetric part of the velocity gradient
 *
 *      S = 1/2 * (grad(u) + grad(u)^T) and S:S = S_ij * S_ij
 *
 *    and the model constant is
 *
 *      C = 0.165 (Nicoud et al. (2011))
 *      C = 0.18  (Toda et al. (2010))
 */
template<int dim, typename Number>
void
smagorinsky_model(dealii::VectorizedArray<Number> const &                         filter_width,
                  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
                  double const &                                                  C,
                  dealii::VectorizedArray<Number> &                               viscosity)
{
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<2, dim, scalar>  tensor;

  tensor symmetric_gradient =
    dealii::make_vectorized_array<Number>(0.5) * (velocity_gradient + transpose(velocity_gradient));

  scalar rate_of_strain = 2.0 * scalar_product(symmetric_gradient, symmetric_gradient);
  rate_of_strain        = std::exp(0.5 * std::log(rate_of_strain));

  scalar factor = C * filter_width;

  viscosity += factor * factor * rate_of_strain;
}

/**
 *  Vreman model (2004): Note that we only consider the isotropic variant of the Vreman model:
 *
 *    nu_SGS = (C * filter_width)^{2} * D
 *
 *  where the differential operator D is defined as
 *
 *    D = sqrt(B_gamma / ||grad(u)||^{2})
 *
 *  with
 *
 *    ||grad(u)||^{2} = grad(u) : grad(u) and grad(u) = d(u_i)/d(x_j) ,
 *
 *    gamma = grad(u) * grad(u)^T ,
 *
 *  and
 *
 *    B_gamma = gamma_11 * gamma_22 - gamma_12^{2}
 *             +gamma_11 * gamma_33 - gamma_13^{2}
 *             +gamma_22 * gamma_33 - gamma_23^{2}
 *
 *  Note that if ||grad(u)||^{2} = 0, nu_SGS is consistently defined as zero.
 *
 */
template<int dim, typename Number>
void
vreman_model(dealii::VectorizedArray<Number> const &                         filter_width,
             dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
             double const &                                                  C,
             dealii::VectorizedArray<Number> &                               viscosity)
{
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<2, dim, scalar>  tensor;

  scalar       velocity_gradient_norm_square = scalar_product(velocity_gradient, velocity_gradient);
  Number const tolerance                     = 1.0e-12;

  tensor gamma_tensor = velocity_gradient * transpose(velocity_gradient);

  AssertThrow(dim == 3,
              dealii::ExcMessage(
                "Number of dimensions has to be dim==3 to evaluate Vreman turbulence model."));

  scalar B_gamma =
    +gamma_tensor[0][0] * gamma_tensor[1][1] - gamma_tensor[0][1] * gamma_tensor[0][1] +
    gamma_tensor[0][0] * gamma_tensor[2][2] - gamma_tensor[0][2] * gamma_tensor[0][2] +
    gamma_tensor[1][1] * gamma_tensor[2][2] - gamma_tensor[1][2] * gamma_tensor[1][2];

  scalar factor = C * filter_width;

  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); i++)
  {
    // If the norm of the velocity gradient tensor is zero, the subgrid-scale
    // viscosity is defined as zero, so we do nothing in that case.
    // Make sure that B_gamma[i] is larger than zero since we calculate
    // the square root of B_gamma[i].
    if(velocity_gradient_norm_square[i] > tolerance and B_gamma[i] > tolerance)
    {
      viscosity[i] += factor[i] * factor[i] *
                      std::exp(0.5 * std::log(B_gamma[i] / velocity_gradient_norm_square[i]));
    }
  }
}

/**
 *  WALE (wall-adapting local eddy-viscosity) model (Nicoud & Ducros 1999):
 *
 *    nu_SGS = (C * filter_width)^{2} * D ,
 *
 *  where the differential operator D is defined as
 *
 *    D = (S^{d}:S^{d})^{3/2} / ( (S:S)^{5/2} + (S^{d}:S^{d})^{5/4} )
 *
 *    where S is the symmetric part of the velocity gradient
 *
 *      S = 1/2 * (grad(u) + grad(u)^T) and S:S = S_ij * S_ij
 *
 *    and S^{d} the traceless symmetric part of the square of the velocity
 *    gradient tensor
 *
 *      S^{d} = 1/2 * (g^{2} + (g^{2})^T) - 1/3 * trace(g^{2}) * I
 *
 *    with the square of the velocity gradient tensor
 *
 *      g^{2} = grad(u) * grad(u)
 *
 *    and the identity tensor I.
 *
 */
template<int dim, typename Number>
void
wale_model(dealii::VectorizedArray<Number> const &                         filter_width,
           dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
           double const &                                                  C,
           dealii::VectorizedArray<Number> &                               viscosity)
{
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<2, dim, scalar>  tensor;

  tensor S =
    dealii::make_vectorized_array<Number>(0.5) * (velocity_gradient + transpose(velocity_gradient));
  scalar S_norm_square = scalar_product(S, S);

  tensor square_gradient       = velocity_gradient * velocity_gradient;
  scalar trace_square_gradient = trace(square_gradient);

  tensor isotropic_tensor;
  for(unsigned int i = 0; i < dim; ++i)
  {
    isotropic_tensor[i][i] = 1.0 / 3.0 * trace_square_gradient;
  }

  tensor S_d =
    dealii::make_vectorized_array<Number>(0.5) * (square_gradient + transpose(square_gradient)) -
    isotropic_tensor;

  scalar S_d_norm_square = scalar_product(S_d, S_d);

  scalar D = dealii::make_vectorized_array<Number>(0.0);

  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); i++)
  {
    Number const tolerance = 1.e-12;
    if(S_d_norm_square[i] > tolerance)
    {
      D[i] = std::pow(S_d_norm_square[i], 1.5) /
             (std::pow(S_norm_square[i], 2.5) + std::pow(S_d_norm_square[i], 1.25));
    }
  }

  scalar factor = C * filter_width;

  viscosity += factor * factor * D;
}

/**
 *  Sigma model (Toda et al. 2010, Nicoud et al. 2011):
 *
 *    nu_SGS = (C * filter_width)^{2} * D
 *
 *    where the differential operator D is defined as
 *
 *      D = s3 * (s1 - s2) * (s2 - s3) / s1^{2}
 *
 *    where s1 >= s2 >= s3 >= 0 are the singular values of
 *    the velocity gradient tensor g = grad(u).
 *
 *    The model constant is
 *
 *      C = 1.35 (Nicoud et al. (2011)) ,
 *      C = 1.5  (Toda et al. (2010)) .
 */
template<int dim, typename Number>
void
sigma_model(dealii::VectorizedArray<Number> const &                         filter_width,
            dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
            double const &                                                  C,
            dealii::VectorizedArray<Number> &                               viscosity)
{
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<2, dim, scalar>  tensor;

  AssertThrow(dim == 3,
              dealii::ExcMessage(
                "Number of dimensions has to be dim==3 to evaluate Sigma turbulence model."));

  /*
   *  Compute singular values manually using a self-contained method
   *  (see appendix in Nicoud et al. (2011)). This approach is more efficient
   *  than calculating eigenvalues or singular values using LAPACK routines.
   */
  scalar D = dealii::make_vectorized_array<Number>(0.0);

  tensor G = transpose(velocity_gradient) * velocity_gradient;

  scalar invariant1 = trace(G);
  scalar invariant2 = 0.5 * (invariant1 * invariant1 - trace(G * G));
  scalar invariant3 = determinant(G);

  for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  {
    // if trace(G) = 0, all eigenvalues (and all singular values) have to be zero
    // and hence G is also zero. Set D[n]=0 in that case.
    if(invariant1[n] > 1.0e-12)
    {
      Number alpha1 = invariant1[n] * invariant1[n] / 9.0 - invariant2[n] / 3.0;
      Number alpha2 = invariant1[n] * invariant1[n] * invariant1[n] / 27.0 -
                      invariant1[n] * invariant2[n] / 6.0 + invariant3[n] / 2.0;

      AssertThrow(alpha1 >= std::numeric_limits<double>::denorm_min() /*smallest positive value*/,
                  dealii::ExcMessage("alpha1 has to be larger than zero."));

      Number factor = alpha2 / std::pow(alpha1, 1.5);

      AssertThrow(std::abs(factor) <=
                    1.0 + 1.0e-12, /* we found that a larger tolerance (1e-8,1e-6,1e-4) might be
                                      necessary in some cases */
                  dealii::ExcMessage("Cannot compute arccos(value) if abs(value)>1.0."));

      // Ensure that the argument of arccos() is in the interval [-1,1].
      if(factor > 1.0)
        factor = 1.0;
      else if(factor < -1.0)
        factor = -1.0;

      Number alpha3 = 1.0 / 3.0 * std::acos(factor);

      dealii::Vector<Number> sv = dealii::Vector<Number>(dim);

      sv[0] = invariant1[n] / 3.0 + 2 * std::sqrt(alpha1) * std::cos(alpha3);
      sv[1] =
        invariant1[n] / 3.0 - 2 * std::sqrt(alpha1) * std::cos(dealii::numbers::PI / 3.0 + alpha3);
      sv[2] =
        invariant1[n] / 3.0 - 2 * std::sqrt(alpha1) * std::cos(dealii::numbers::PI / 3.0 - alpha3);

      // Calculate the square root only if the value is larger than zero.
      // Otherwise set sv to zero (this is reasonable since negative values will
      // only occur due to numerical errors).
      for(unsigned int d = 0; d < dim; ++d)
      {
        if(sv[d] > 0.0)
          sv[d] = std::sqrt(sv[d]);
        else
          sv[d] = 0.0;
      }

      Number const tolerance = 1.e-12;
      if(sv[0] > tolerance)
      {
        D[n] = (sv[2] * (sv[0] - sv[1]) * (sv[1] - sv[2])) / (sv[0] * sv[0]);
      }
    }
  }

  /*
   * The singular values of the velocity gradient g = grad(u) are
   * the square root of the eigenvalues of G = g^T * g.
   */

  //    scalar D_copy = D; // save a copy in order to verify the correctness of
  //    the computation for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size();
  //    n++)
  //    {
  //      LAPACKFullMatrix<Number> G_local = LAPACKFullMatrix<Number>(dim);
  //
  //      for(unsigned int i = 0; i < dim; i++)
  //      {
  //        for(unsigned int j = 0; j < dim; j++)
  //        {
  //          G_local(i,j) = G[i][j][n];
  //        }
  //      }
  //
  //      G_local.compute_eigenvalues();
  //
  //      std::list<Number> ev_list;
  //
  //      for(unsigned int l = 0; l < dim; l++)
  //      {
  //        ev_list.push_back(std::abs(G_local.eigenvalue(l)));
  //      }
  //
  //      // This sorts the list in ascending order, beginning with the smallest eigenvalue.
  //      ev_list.sort();
  //
  //      dealii::Vector<Number> ev = dealii::Vector<Number>(dim);
  //      typename std::list<Number>::reverse_iterator it;
  //      unsigned int k;
  //
  //      // Write values in vector "ev" and reverse the order so that we
  //      // ev[0] corresponds to the largest eigenvalue.
  //      for(it = ev_list.rbegin(), k=0; it != ev_list.rend() and k<dim; ++it, ++k)
  //      {
  //        ev[k] = std::sqrt(*it);
  //      }
  //
  //      Number const tolerance = 1.e-12;
  //      if(ev[0] > tolerance)
  //      {
  //        D[n] = (ev[2]*(ev[0]-ev[1])*(ev[1]-ev[2]))/(ev[0]*ev[0]);
  //      }
  //    }
  //
  //    // make sure that both variants yield the same result
  //    for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  //    {
  //      AssertThrow(std::abs(D[n]-D_copy[n])<1.e-5,dealii::ExcMessage("Calculation of singular
  //      values is incorrect."));
  //    }


  /*
   *  Alternatively, compute singular values directly using SVD.
   */
  //    scalar D_copy2 = D; // save a copy in order to verify the correctness of
  //    the computation D = dealii::make_vectorized_array<Number>(0.0);
  //
  //    for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  //    {
  //      LAPACKFullMatrix<Number> gradient = LAPACKFullMatrix<Number>(dim);
  //      for(unsigned int i = 0; i < dim; i++)
  //      {
  //        for(unsigned int j = 0; j < dim; j++)
  //        {
  //          gradient(i,j) = velocity_gradient[i][j][n];
  //        }
  //      }
  //      gradient.compute_svd();
  //
  //      dealii::Vector<Number> sv = dealii::Vector<Number>(dim);
  //      for(unsigned int i=0;i<dim;++i)
  //      {
  //        sv[i] = gradient.singular_value(i);
  //      }
  //
  //      Number const tolerance = 1.e-12;
  //      if(sv[0] > tolerance)
  //      {
  //        D[n] = (sv[2]*(sv[0]-sv[1])*(sv[1]-sv[2]))/(sv[0]*sv[0]);
  //      }
  //    }
  //
  //    // make sure that both variants yield the same result
  //    for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  //    {
  //      AssertThrow(std::abs(D[n]-D_copy2[n])<1.e-5,dealii::ExcMessage("Calculation of singular
  //      values is incorrect."));
  //    }

  // add turbulent eddy-viscosity to laminar viscosity
  scalar factor = C * filter_width;
  viscosity += factor * factor * D;
}

template<typename Number>
void
generalized_carreau_yasuda_model(dealii::VectorizedArray<Number> &       viscosity_factor,
                                 dealii::VectorizedArray<Number> const & shear_rate,
                                 GeneralizedNewtonianModelData const &   data)
{
  viscosity_factor = pow(data.kappa + pow(shear_rate * data.lambda, static_cast<Number>(data.a)),
                         static_cast<Number>((data.n - 1.0) / data.a));
}

/**
 * Many rheological models differ in the factor scaling the margin
 * between upper and lower viscosity limits
 * e_0 - e_oo,
 * which we denote here as `viscosity_margin`. This margin is multiplied
 * with a factor [k + (l * y)^a]^[(n - 1) / a], which we denote as `viscosity_factor`.
 */
template<typename Number>
dealii::VectorizedArray<Number>
compute_viscosity_factor(dealii::VectorizedArray<Number> const & shear_rate,
                         GeneralizedNewtonianModelData const &   data)
{
  typedef dealii::VectorizedArray<Number> scalar;

  // compute the factor multiplying the `viscosity_margin`, i.e., (eta_0 - eta_00), in
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  scalar viscosity_factor;

  switch(data.generalized_newtonian_model)
  {
    case GeneralizedNewtonianViscosityModel::Undefined:
      AssertThrow(data.generalized_newtonian_model != GeneralizedNewtonianViscosityModel::Undefined,
                  dealii::ExcMessage("parameter must be defined"));
      break;
    case GeneralizedNewtonianViscosityModel::GeneralizedCarreauYasuda:
      generalized_carreau_yasuda_model(viscosity_factor, shear_rate, data);
      break;
    default:
      AssertThrow(
        false, dealii::ExcMessage("This GeneralizedNewtonianViscosityModel is not implemented."));
  }

  return viscosity_factor;
}

} // namespace

template<int dim, typename Number>
void
add_turbulent_viscosity(
  dealii::VectorizedArray<Number> &                               viscosity,
  dealii::VectorizedArray<Number> const &                         filter_width,
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
  TurbulenceModelData const &                                     turbulence_model_data)
{
  switch(turbulence_model_data.turbulence_model)
  {
    case TurbulenceEddyViscosityModel::Undefined:
      AssertThrow(turbulence_model_data.turbulence_model != TurbulenceEddyViscosityModel::Undefined,
                  dealii::ExcMessage("Parameter must be defined."));
      break;
    case TurbulenceEddyViscosityModel::Smagorinsky:
      smagorinsky_model(filter_width, velocity_gradient, turbulence_model_data.constant, viscosity);
      break;
    case TurbulenceEddyViscosityModel::Vreman:
      vreman_model(filter_width, velocity_gradient, turbulence_model_data.constant, viscosity);
      break;
    case TurbulenceEddyViscosityModel::WALE:
      wale_model(filter_width, velocity_gradient, turbulence_model_data.constant, viscosity);
      break;
    case TurbulenceEddyViscosityModel::Sigma:
      sigma_model(filter_width, velocity_gradient, turbulence_model_data.constant, viscosity);
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage("This TurbulenceEddyViscosityModel is not implemented."));
  }
}

template<int dim>
double
compute_filter_width(dealii::FEValues<dim> const & fe_values, unsigned int const degree)
{
  // calculate cell volume
  double volume = 0.0;
  for(unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
  {
    volume += fe_values.JxW(q);
  }

  // h = V^{1/dim}
  double h = std::exp(std::log(volume) / (double)dim);

  // take polynomial degree of shape functions into account:
  // h/(k_u + 1)
  h /= (double)(degree + 1);

  return h;
}

template<int dim, typename Number>
void
add_generalized_newtonian_viscosity(
  dealii::VectorizedArray<Number> &                               viscosity,
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & velocity_gradient,
  GeneralizedNewtonianModelData const &                           data)
{
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<2, dim, scalar>  tensor;

  tensor symmetric_velocity_gradient = 0.5 * (velocity_gradient + transpose(velocity_gradient));

  scalar shear_rate =
    std::sqrt(2.0 * scalar_product(symmetric_velocity_gradient, symmetric_velocity_gradient));

  scalar viscosity_factor = compute_viscosity_factor(shear_rate, data);

  // add (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  viscosity += data.viscosity_margin * viscosity_factor;
}

template void
add_turbulent_viscosity<2, float>(dealii::VectorizedArray<float> &,
                                  dealii::VectorizedArray<float> const &,
                                  dealii::Tensor<2, 2, dealii::VectorizedArray<float>> const &,
                                  TurbulenceModelData const &);
template void
add_generalized_newtonian_viscosity<2, float>(
  dealii::VectorizedArray<float> &,
  dealii::Tensor<2, 2, dealii::VectorizedArray<float>> const &,
  GeneralizedNewtonianModelData const &);
template void
add_turbulent_viscosity<2, double>(dealii::VectorizedArray<double> &,
                                   dealii::VectorizedArray<double> const &,
                                   dealii::Tensor<2, 2, dealii::VectorizedArray<double>> const &,
                                   TurbulenceModelData const &);
template void
add_generalized_newtonian_viscosity<2, double>(
  dealii::VectorizedArray<double> &,
  dealii::Tensor<2, 2, dealii::VectorizedArray<double>> const &,
  GeneralizedNewtonianModelData const &);
template double
compute_filter_width<2>(dealii::FEValues<2> const &, unsigned int const);

template void
add_turbulent_viscosity<3, float>(dealii::VectorizedArray<float> &,
                                  dealii::VectorizedArray<float> const &,
                                  dealii::Tensor<2, 3, dealii::VectorizedArray<float>> const &,
                                  TurbulenceModelData const &);
template void
add_generalized_newtonian_viscosity<3, float>(
  dealii::VectorizedArray<float> &,
  dealii::Tensor<2, 3, dealii::VectorizedArray<float>> const &,
  GeneralizedNewtonianModelData const &);
template void
add_turbulent_viscosity<3, double>(dealii::VectorizedArray<double> &,
                                   dealii::VectorizedArray<double> const &,
                                   dealii::Tensor<2, 3, dealii::VectorizedArray<double>> const &,
                                   TurbulenceModelData const &);
template void
add_generalized_newtonian_viscosity<3, double>(
  dealii::VectorizedArray<double> &,
  dealii::Tensor<2, 3, dealii::VectorizedArray<double>> const &,
  GeneralizedNewtonianModelData const &);
template double
compute_filter_width<3>(dealii::FEValues<3> const &, unsigned int const);

} // namespace ViscosityModels
} // namespace IncNS
} // namespace ExaDG
