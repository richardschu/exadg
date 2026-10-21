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

#ifndef APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_
#define APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_

namespace ExaDG
{
double
m_to_mm(double coordinate)
{
  return 1000.0 * coordinate;
}

double
mm_to_m(double coordinate)
{
  return 0.001 * coordinate;
}

/*
 * This function returns the distance by which points are shifted in y-direction due to the hill,
 * i.e., a value of 0 is returned at x=0H, 9H, and a value of -H at x=4.5H.
 */
double
f(double x_m, double const H, double const LENGTH)
{
  if(x_m > LENGTH / 2.0)
    x_m = LENGTH - x_m;

  double x = m_to_mm(x_m);
  double y = 0.0;

  AssertThrow(x_m > -1e-12 and x_m <= LENGTH / 2.0 + 1.e-12,
              dealii::ExcMessage("Parameter out of bounds."));

  if(x <= 9.0)
    y =
      -m_to_mm(H) +
      std::min(m_to_mm(H), m_to_mm(H) + 6.775070969851e-3 * x * x - 2.124527775800e-3 * x * x * x);
  else if(x > 9.0 and x <= 14.0)
    y = -m_to_mm(H) + 2.507355893131e1 + 9.754803562315e-1 * x - 1.016116352781e-1 * x * x +
        1.889794677828e-3 * x * x * x;
  else if(x > 14.0 and x <= 20.0)
    y = -m_to_mm(H) + 2.579601052357e1 + 8.206693007457e-1 * x - 9.055370274339e-2 * x * x +
        1.626510569859e-3 * x * x * x;
  else if(x > 20.0 and x <= 30.0)
    y = -m_to_mm(H) + 4.046435022819e1 - 1.379581654948 * x + 1.945884504128e-2 * x * x -
        2.070318932190e-4 * x * x * x;
  else if(x > 30.0 and x <= 40.0)
    y = -m_to_mm(H) + 1.792461334664e1 + 8.743920332081e-1 * x - 5.567361123058e-2 * x * x +
        6.277731764683e-4 * x * x * x;
  else if(x > 40.0 and x <= 54.0)
    y = -m_to_mm(H) + std::max(0.0,
                               5.639011190988e1 - 2.010520359035 * x + 1.644919857549e-2 * x * x +
                                 2.674976141766e-5 * x * x * x);
  else if(x > 54.0)
    y = -m_to_mm(H);
  else
    AssertThrow(false, dealii::ExcMessage("Not implemented."));

  return mm_to_m(y);
}



/**
 * Manifold to describe the periodic hill geometry, using two distinctive
 * features for the mesh:
 * (1) The mesh is graded towards the boundary with a tanh profile with
 * scaling determined by @p GRID_STRETCH_FACTOR
 * (2) It is possible to manually deform the mesh to improve the density of
 * mesh elements in regions with higher dissipation. This is purely
 * hand-written and might not optimally place points. As an alternative that
 * has been systematically optimized for the actual flow, use the class
 * PeriodicHillManifoldOptimizedMesh instead.
 */
template<int dim>
class PeriodicHillManifoldTanh : public dealii::ChartManifold<dim>
{
public:
  PeriodicHillManifoldTanh(const double H,
                           const double LENGTH,
                           const double HEIGHT,
                           const double GRID_STRETCH_FAC,
                           const bool   apply_manual_deformation)
    : dealii::ChartManifold<dim>(),
      H(H),
      LENGTH(LENGTH),
      HEIGHT(HEIGHT),
      GRID_STRETCH_FAC(GRID_STRETCH_FAC),
      apply_manual_deformation(apply_manual_deformation)
  {
    // Assume we have a slope of -2 (the actual slope extremum is around
    // -0.86, but due to resolution requirements near the hill top choose a
    // higher value), and then transition from a steeper path of curve length
    // to the more flat part. Start by defining a piecewise linear function (a
    // more complicated function to check the actual path was tried but found
    // to not be better but just more expensive) and then transition over a
    // length corresponding to the hill height to the flat part; the actual
    // evaluation will use a quintic Hermite interpolating polynomial.
    const double slope_assumed        = -2;
    const double scaling_curved_start = 0.7 * H;
    const double scaling_curved_end   = 1.8 * H;
    const double curve_length_sloped  = std::sqrt(1. + slope_assumed * slope_assumed);
    const double curve_length =
      0.5 * (scaling_curved_start + scaling_curved_end) * (curve_length_sloped - 1.) + LENGTH / 2;
    x_transition_points = {
      {0.0,
       curve_length_sloped * scaling_curved_start * LENGTH * 0.5 / curve_length,
       LENGTH * 0.5 - (LENGTH * 0.5 - scaling_curved_end) * LENGTH * 0.5 / curve_length,
       LENGTH * 0.5}};
    x_transition_values = {{0.0, scaling_curved_start, scaling_curved_end, LENGTH * 0.5}};
  }

  dealii::Point<dim>
  push_forward(const dealii::Point<dim> & xi_in) const final
  {
    dealii::Point<dim> xi = xi_in;

    const double gamma = GRID_STRETCH_FAC;
    const double y     = 2.0 * (xi[1] - H) / HEIGHT - 1.0;

    if(apply_manual_deformation)
    {
      // Shift elements towards the lower part of the domain
      const double xi_1_hat =
        std::tanh(gamma * (y - 0.08 * std::sin(dealii::numbers::PI * y))) / std::tanh(gamma);
      xi[1] = xi_1_hat * 0.5 + 0.5;

      // Create shift of mesh elements towards the left, where dissipation is
      // higher in the wake of the hill. The idea is to have the element
      // spacing between 5/32 and 14/32 to be 0.7 the size of a uniform
      // distribution. Along the x coordinate, we transition with C^3
      // continuity between the initial size of 1 and then back to a remainder
      // size, which is 19/15 the size of the initial elements. The obtained x
      // shift is further weighted in y direction to not act near the
      // boundaries with 'y_scaling'
      const double x_unit  = xi_in[0] / LENGTH;
      double       x_shift = 0;
      if(x_unit > 2. / 32)
      {
        if(x_unit < 5. / 32.)
        {
          const double t = 32. / 3. * (x_unit - 2. / 32);
          x_shift        = -9. / 320 * t * t * t * t * (2.5 + t * (-3. + t));
        }
        else if(x_unit < 14. / 32.)
        {
          x_shift = 21. / 640 - 0.3 * x_unit;
        }
        else if(x_unit < 17. / 32)
        {
          const double t = 32. / 3. * (x_unit - 14. / 32.);
          x_shift = 21. / 640 - 0.3 * x_unit + 9. / 320 * t * t * t * t * (2.5 + t * (-3. + t));
        }
        else if(x_unit < 20. / 32)
        {
          const double t = 32. / 3. * (x_unit - 17. / 32.);
          x_shift        = -18. / 160 + 1. / 40 * t * t * t * t * (2.5 + t * (-3. + t));
        }
        else
          x_shift = -128. / 480 + 4. / 15 * x_unit;
      }
      double y_scaling = 0;
      if(xi_1_hat < -0.24)
      {
        const double t = (xi_1_hat + 1) * (1. / 0.76);
        y_scaling      = t * t * (3. - 2 * t);
      }
      else
      {
        const double t = (1. - xi_1_hat) * (1. / 1.24);
        y_scaling      = t * t * (3 - 2 * t);
      }
      xi[0] += 0.8 * LENGTH * x_shift * y_scaling;
    }
    else
    {
      const double xi_1_hat = std::tanh(gamma * y) / std::tanh(gamma);
      xi[1]                 = xi_1_hat * 0.5 + 0.5;
    }

    // Finally move the points near the boundary to ensure approximately
    // equal-length elements (apart from at the hill top, where we want finer
    // element distribution). At this point, xi[1] holds the weight in [0, 1]
    // for blending between the bottom and top walls.
    const double xi_0 =
      xi[0] < LENGTH / 2 ? get_scaled_x_point(xi[0]) : LENGTH - get_scaled_x_point(LENGTH - xi[0]);

    dealii::Point<dim> xi_bottom;
    xi_bottom[0] = xi_0;
    xi_bottom[1] = H + f(xi_0, H, LENGTH);
    if(dim > 2)
      xi_bottom[2] = xi[2];

    dealii::Point<dim> xi_top = xi;
    xi_top[1]                 = H + HEIGHT;
    return xi_top * xi[1] + xi_bottom * (1.0 - xi[1]);
  }

  dealii::Point<dim>
  pull_back(const dealii::Point<dim> & x) const final
  {
    AssertThrow(false, ExcNotImplemented());
    return x;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const final
  {
    return std::make_unique<PeriodicHillManifoldTanh<dim>>(
      H, LENGTH, HEIGHT, GRID_STRETCH_FAC, apply_manual_deformation);
  }

  double
  get_scaled_x_point(const double x_in) const
  {
    if(x_in >= x_transition_points[2])
      return (x_transition_values[2] * (x_transition_points[3] - x_in) +
              x_transition_values[3] * (x_in - x_transition_points[2])) /
             (x_transition_points[3] - x_transition_points[2]);
    else if(x_in < x_transition_points[1])
      return (x_transition_values[1] * x_in) / x_transition_points[1];
    else
    {
      // evaluate Hermite interpolating polynomial between
      // transition_points[1] and transition_points[2]
      const double x0  = x_transition_points[1];
      const double x1  = x_transition_points[2];
      const double y0  = x_transition_values[1];
      const double y1  = x_transition_values[2];
      const double yp0 = (x_transition_values[1] - x_transition_values[0]) /
                         (x_transition_points[1] - x_transition_points[0]);
      const double yp1 = (x_transition_values[3] - x_transition_values[2]) /
                         (x_transition_points[3] - x_transition_points[2]);
      const double t = (x_in - x0) / (x1 - x0);

      // terms for cubic polynomial
      // const double h00 = (1 + 2*t) * (1 - t) * (1 - t);
      // const double h10 = t * (1 - t) * (1 - t);
      // const double h01 = t * t * (3 - 2*t);
      // const double h11 = t * t * (t - 1);

      // define a quintic polynomial with second derivative zero at the end
      // points, so ignore the part involving second derivatives, which
      // leads to four contributions similar to the cubic Hermite polynomial
      const double h00 = 1. + t * t * t * (-10. + t * (15. - 6. * t));
      const double h01 = t + t * t * t * (-6. + t * (8. - 3. * t));
      const double h10 = t * t * t * (10. + t * (-15. + 6. * t));
      const double h11 = t * t * t * (-4. + t * (7. - 3. * t));
      return h00 * y0 + h10 * y1 + (x1 - x0) * (h01 * yp0 + h11 * yp1);
    }
  }

private:
  const double          H, LENGTH, HEIGHT, GRID_STRETCH_FAC;
  const bool            apply_manual_deformation;
  std::array<double, 4> x_transition_points;
  std::array<double, 4> x_transition_values;
};



/**
 * Alternative to the basic PeriodicHillManifoldTanh class, where the mesh
 * displacement is based on an optimization of the mesh. The actual
 * computation was done for a 64 x 48 x 32 mesh of the periodic hill
 * at Reynolds number 10595 with element FE_RaviartThomasNodal(6)
 * (degree 7/6 in normal/tangential part) using statistics of 10
 * flow-through times in the developed phase, and aims to balance (h / eta)^2
 * with the Kolmogorov length eta and mesh size h = vol(K)^(1/3)
 * with the fe_values.jacobian(q).determinant() quantity in the
 * 2D x-y slice of the mesh using a linear elasticity problem with `mu = (h
 * / eta)^2 * fe_values.jacobian(q).determinant()` and `lambda = -0.33 * mu`
 * (almost pure shear problem).
 */
template<int dim>
class PeriodicHillManifoldOptimizedMesh : public dealii::ChartManifold<dim>
{
  // This is the degree of the optimization computation performed, which uses
  // two panels for the left and right half of the hill, respectively (the
  // final distribution is non-symmetric, so use two separate functions).
  static constexpr unsigned int n_shapes = 6;

public:
  PeriodicHillManifoldOptimizedMesh(const double H,
                                    const double LENGTH,
                                    const double HEIGHT,
                                    const double GRID_STRETCH_FAC)
    : dealii::ChartManifold<dim>(),
      H(H),
      LENGTH(LENGTH),
      HEIGHT(HEIGHT),
      GRID_STRETCH_FAC(GRID_STRETCH_FAC),
      INVERSE_LENGTH(1.0 / LENGTH),
      INVERSE_HEIGHT(1.0 / HEIGHT),
      tanh_gamma(std::tanh(GRID_STRETCH_FAC)),
      deriv_tanh_gamma(GRID_STRETCH_FAC /
                       (tanh_gamma * std::cosh(GRID_STRETCH_FAC) * std::cosh(GRID_STRETCH_FAC)))
  {
    // These coefficients were determined by the offline computation
    // underlying this class.
    const dealii::ndarray<double, n_shapes * n_shapes, 2> panel_left{
      {{{0, 0}},
       {{0.101091, 0}},
       {{0.327903, 0}},
       {{0.613958, 0}},
       {{0.843835, 0}},
       {{0.96, 0}},
       {{0, 0.09783}},
       {{0.1217769177288, 0.1023455664107}},
       {{0.3528949637918, 0.1383954118446}},
       {{0.6129618830114, 0.1077863683778}},
       {{0.8361865192995, 0.1097237562196}},
       {{0.9509881659537, 0.1106382325862}},
       {{0, 0.309909}},
       {{0.1322720685086, 0.300544644577}},
       {{0.3697106341703, 0.3009992195921}},
       {{0.6169278986294, 0.2815407241844}},
       {{0.8356585159495, 0.2848430347859}},
       {{0.9478323679665, 0.2877317390483}},
       {{0, 0.595338}},
       {{0.1308721940147, 0.5628075053081}},
       {{0.3742168957085, 0.5556230610245}},
       {{0.6279644838771, 0.5381058207746}},
       {{0.8479552104111, 0.5322273617283}},
       {{0.9598836688846, 0.5313307123757}},
       {{0, 0.86404}},
       {{0.1222861067047, 0.8690634245057}},
       {{0.3638139743724, 0.8616613973007}},
       {{0.6374117326923, 0.8600403934544}},
       {{0.8706583011288, 0.8610240071519}},
       {{0.98611946227, 0.8614710910516}},
       {{0, 1}},
       {{0.1174723380353, 1}},
       {{0.3573842417597, 1}},
       {{0.6426157582403, 1}},
       {{0.8825276619647, 1}},
       {{1, 1}}}};
    const dealii::ndarray<double, n_shapes * n_shapes, 2> panel_right{
      {{{-0.04, 0}},
       {{0.092967, 0}},
       {{0.391264, 0}},
       {{0.703962, 0}},
       {{0.905966, 0}},
       {{1, 0}},
       {{-0.04901183404628, 0.1106382325862}},
       {{0.07848806491164, 0.110441354272}},
       {{0.3642743760869, 0.1113399770794}},
       {{0.6752339480096, 0.1281137828057}},
       {{0.8893651054201, 0.1101837911437}},
       {{1, 0.09783}},
       {{-0.05216763203353, 0.2877317390483}},
       {{0.06891342910551, 0.2910750764155}},
       {{0.3340310895566, 0.3002825602888}},
       {{0.6354260954103, 0.3321069889958}},
       {{0.8760828997625, 0.3064530793761}},
       {{1, 0.309909}},
       {{-0.04011633111541, 0.5313307123757}},
       {{0.07548328763114, 0.532366956156}},
       {{0.3229018370881, 0.5451417584881}},
       {{0.6160930122147, 0.5703201025913}},
       {{0.8731655717461, 0.562987339441}},
       {{1, 0.595338}},
       {{-0.01388053773, 0.8614710910516}},
       {{0.1030003648129, 0.8613257415966}},
       {{0.3449502543886, 0.8610955914297}},
       {{0.6323324578336, 0.864692122045}},
       {{0.8784200407932, 0.8678662740671}},
       {{1, 0.86404}},
       {{0, 1}},
       {{0.1174723380353, 1}},
       {{0.3573842417597, 1}},
       {{0.6426157582403, 1}},
       {{0.8825276619647, 1}},
       {{1, 1}}}};

    interpolation_points[0].resize(panel_left.size());
    for(unsigned int i = 0; i < panel_left.size(); ++i)
      for(unsigned int d = 0; d < 2; ++d)
        interpolation_points[0][i][d] = panel_left[i][d];
    interpolation_points[1].resize(panel_right.size());
    for(unsigned int i = 0; i < panel_right.size(); ++i)
      for(unsigned int d = 0; d < 2; ++d)
        interpolation_points[1][i][d] = panel_right[i][d];
    basis = dealii::Polynomials::generate_complete_Lagrange_basis(
      dealii::QGaussLobatto<1>(n_shapes).get_points());
  }

  dealii::Point<dim>
  push_forward(const dealii::Point<dim> & xi_in) const final
  {
    dealii::Point<dim> xi = xi_in;
    const double       y  = 2.0 * (xi[1] - H) * INVERSE_HEIGHT - 1.0;

    // Cubic Hermite polynomial with same value and slope as
    // tanh(gamma*y)/tanh(gamma) at +1 and -1, with gamma = GRID_STRETCH_FAC.
    // The base mesh used for the optimization had gamma = 2.
    const double xi_1_hat = y * 0.5 * ((3. - deriv_tanh_gamma) + (deriv_tanh_gamma - 1.) * y * y);

    // Option that deforms mesh based on polynomial interpolation from a given
    // set of interpolation points, applied after the cubic Hermite
    // approximation. Note that the two panels are only C^0 continuous at
    // x = LENGTH/2, so this line must coincide with element faces.
    const bool   left_panel = xi[0] < LENGTH / 2;
    const double r0 =
      (left_panel) ? (xi[0] * INVERSE_LENGTH * 2) : (2 * xi[0] - LENGTH) * INVERSE_LENGTH;
    const double r1 = 0.5 * xi_1_hat + 0.5;

    // evaluate polynomial basis in x and y direction with deal.II function
    // of polynomial class
    dealii::ndarray<double, n_shapes, 2, 2> shapes;
    for(unsigned int i = 0; i < n_shapes; ++i)
      basis[i].values_of_array(std::array<double, 2>{{r0, r1}}, 0, shapes[i].data());
    const std::vector<dealii::Tensor<1, 2>> & coeffs = interpolation_points[left_panel ? 0 : 1];

    int                  dummy = 0;
    dealii::Tensor<1, 2> val =
      dealii::internal::do_interpolate_xy_value<2, n_shapes, double, dealii::Tensor<1, 2>, false>(
        coeffs.data(), {}, shapes.data(), 0, dummy);

    xi[0] = LENGTH * 0.5 * val[0] + (left_panel ? 0. : LENGTH * 0.5);
    xi[1] = (1.0 - val[1]) * (H + f(xi[0], H, LENGTH)) + val[1] * (H + HEIGHT);
    return xi;
  }

  dealii::Point<dim>
  pull_back(const dealii::Point<dim> & x) const final
  {
    AssertThrow(false, ExcNotImplemented());
    return x;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const final
  {
    return std::make_unique<PeriodicHillManifoldOptimizedMesh<dim>>(H,
                                                                    LENGTH,
                                                                    HEIGHT,
                                                                    GRID_STRETCH_FAC);
  }

  double
  get_scaled_x_point(const double x_in) const
  {
    // 7-th order Hermite polynomial, which has 4 continuous derivatives at
    // x=0.5, the position where the halves are stitched together; the
    // coefficients are set up in the constructor; several terms are zero
    // because f(0) = 0, f''(0) = 0.
    const double t = x_in * INVERSE_LENGTH * 2;
    return t * (coefficients_hermite[0] +
                t * t *
                  (coefficients_hermite[1] +
                   t * (coefficients_hermite[2] +
                        t * (coefficients_hermite[3] +
                             t * (coefficients_hermite[4] + t * coefficients_hermite[5])))));
  }

private:
  const double H, LENGTH, HEIGHT, GRID_STRETCH_FAC, INVERSE_LENGTH, INVERSE_HEIGHT;
  const double tanh_gamma;
  const double deriv_tanh_gamma;
  std::array<std::vector<dealii::Tensor<1, 2>>, 2>     interpolation_points;
  std::vector<dealii::Polynomials::Polynomial<double>> basis;

  // description of shift along lower wall to place points more densely near
  // the hill top, described as Hermite polynomial of degree 7 (but with some
  // coefficients zero, meaning that we store only 6 out of 8 coefficients).
  std::array<double, 6> coefficients_hermite;
};



enum class GridDeformationType
{
  GradedMeshUniform,
  GradedMeshManualShift,
  GradedMeshOptimizationShift
};


/**
 * A manifold class used when constructing anisotropically coarsened
 * meshes. We realize this by independent meshes with different numbers of
 * cells in one of the directions (currently hardcoded as the y
 * direction). We then want to match cells in the two triangulations, for
 * which we want to shift the cells to create a nested space. This shift is
 * implemented by the present manifold class.
 */
template<int dim>
class PiecewiseLinearManifold : public dealii::ChartManifold<dim>
{
public:
  PiecewiseLinearManifold(const std::vector<std::array<double, 2>> & pieces) : pieces(pieces)
  {
    AssertThrow(pieces.size() >= 2,
                dealii::ExcMessage("Can only construct a transformation "
                                   "manifold when given at least two pieces."));
    for(unsigned int i = 1; i < pieces.size(); ++i)
    {
      AssertThrow(pieces[i][0] > pieces[i - 1][0],
                  dealii::ExcMessage("Pieces need to be strictly increasing"));
      AssertThrow(pieces[i][1] > pieces[i - 1][1],
                  dealii::ExcMessage("Pieces need to be strictly increasing"));
    }
  }

  virtual std::unique_ptr<dealii::Manifold<dim, dim>>
  clone() const override
  {
    return std::make_unique<PiecewiseLinearManifold<dim>>(pieces);
  }

  virtual dealii::Point<dim>
  pull_back(const dealii::Point<dim> & space_point) const override
  {
    dealii::Point<dim> chart_point = space_point;
    for(unsigned int i = 0; i < pieces.size() - 1; ++i)
      if(space_point[1] < pieces[i + 1][1])
      {
        chart_point[1] = pieces[i][0] + (space_point[1] - pieces[i][1]) *
                                          (pieces[i + 1][0] - pieces[i][0]) /
                                          (pieces[i + 1][1] - pieces[i][1]);
        return chart_point;
      }

    return chart_point;
  }

  virtual dealii::Point<dim>
  push_forward(const dealii::Point<dim> & chart_point) const override
  {
    dealii::Point<dim> space_point = chart_point;

    for(unsigned int i = 0; i < pieces.size() - 1; ++i)
      if(chart_point[1] < pieces[i + 1][0])
      {
        space_point[1] = pieces[i][1] + (chart_point[1] - pieces[i][0]) *
                                          (pieces[i + 1][1] - pieces[i][1]) /
                                          (pieces[i + 1][0] - pieces[i][0]);
        return space_point;
      }

    return space_point;
  }

private:
  const std::vector<std::array<double, 2>> pieces;
};

} // namespace ExaDG


#endif /* APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_ */
