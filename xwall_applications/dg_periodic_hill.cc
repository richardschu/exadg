
// Navier-Stokes splitting program
// authors: Niklas Fehn, Benjamin Krank, Martin Kronbichler, LNM
// years: 2015-2016
// application to turbulent flow past periodic hills

#include <deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/integrators/laplace.h>

#include <fstream>
#include <sstream>

#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../include/incompressible_navier_stokes/postprocessor/statistics_manager_ph.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/driver_steady_problems.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/xwall/DGNavierStokesDualSplittingXWall.h"
#include "../include/incompressible_navier_stokes/xwall/PostProcessorXWall.h"
#include "../include/incompressible_navier_stokes/xwall/TimeIntBDFDualSplittingXWall.h"
#include "../include/incompressible_navier_stokes/xwall/TimeIntBDFDualSplittingXWallSpalartAllmaras.h"
#include "../include/incompressible_navier_stokes_equations/spatial_discretization/DGNavierStokesDualSplitting.h"
#include "../include/InputParametersNavierStokes.h"
#include "PrintInputParameters.h"


const unsigned int FE_DEGREE = 4;//3
const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
const unsigned int FE_DEGREE_XWALL = 1;
const unsigned int N_Q_POINTS_1D_XWALL = 1;
const unsigned int DIMENSION = 3; // DIMENSION >= 2
const unsigned int REFINE_STEPS_SPACE = 3;//4
const double GRID_STRETCH_FAC = 1.8;
const bool USE_SOURCE_TERM_CONTROLLER = true;

template<int dim>
void InputParametersNavierStokes<dim>::set_input_parameters()
{

  output_data.output_prefix = "ph10595_l3_k4_gt18mod";
  cfl = 0.14;
  //Re = 19000: 0.8284788e-5
  //Re = 10595: 1.48571e-5
  //Re = 5600:  2.8109185e-5
  //Re = 700:   2.2487348e-4
  //Re = 1400:  1.1243674e-4
  viscosity = 1.48571e-5;

  if(N_Q_POINTS_1D_XWALL>1) //enriched
  {
    xwall_turb = XWallTurbulenceApproach::RANSSpalartAllmaras;
    max_wdist_xwall = 0.25;
    diffusion_number = 0.02;
    spatial_discretization = SpatialDiscretization::DGXWall;
    IP_formulation_viscous = InteriorPenaltyFormulation::NIPG;
    solver_viscous = SolverViscous::GMRES;
    penalty_factor_divergence = 1.0e1;
    turb_stat_data.statistics_start_time = 0.95;
    end_time = 1.;
    variabletauw = true;
    dtauw = 1.;
  }
  else //LES
  {
    max_wdist_xwall = -0.25;
    spatial_discretization = SpatialDiscretization::DG;
    IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
    solver_viscous = SolverViscous::PCG;
    penalty_factor_divergence = 1.0e0;
    turb_stat_data.statistics_start_time = 0.00001;
    end_time = 2.;
  }

  //dual splitting scheme
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  projection_type = ProjectionType::DivergencePenalty;
  order_time_integrator = 2;
  start_with_low_order = true;

  //xwall specific
  IP_factor_viscous = 1.;

  calculation_of_time_step_size = TimeStepCalculation::AdaptiveTimeStepCFL;//AdaptiveTimeStepCFL;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation; //also default
  divu_integrated_by_parts = true;
  gradp_integrated_by_parts = true;
  pure_dirichlet_bc = true;
  output_solver_info_every_timesteps = 1e2;
  right_hand_side = true;

  output_data.write_output = true;
  output_data.output_start_time = 0.;
  output_data.output_interval_time = 0.1;
  output_data.number_of_patches = FE_DEGREE+2;
  turb_stat_data.write_output_q_criterion = true;
  turb_stat_data.statistics_every = 10;
  turb_stat_data.statistics_end_time = end_time;
  write_restart = true;
  restart_every_timesteps = 1e9;
  restart_interval_time = 1.e9;
  restart_interval_wall_time = 1.e9;

  max_velocity = 4.;

  //solver tolerances
  rel_tol_pressure = 1.e-4;
  rel_tol_projection = 1.e-6;
  rel_tol_viscous = 1.e-4;

}

  double f_x(double x_m)
  {
    double x = x_m*1000.0;
    double y_m = 0.0;

    if (x <= 9.0)
      y_m = 0.001*(-28.0 + std::min(28.0, 2.8e1 + 6.775070969851e-3*x*x - 2.124527775800e-3*x*x*x));

    else if (x > 9.0 && x <= 14.0)
      y_m = 0.001*(-28.0 + 2.507355893131e1 + 9.754803562315e-1*x - 1.016116352781e-1*x*x + 1.889794677828e-3*x*x*x);

    else if (x > 14.0 && x <= 20.0)
      y_m = 0.001*(-28.0 + 2.579601052357e1 + 8.206693007457e-1*x - 9.055370274339e-2*x*x + 1.626510569859e-3*x*x*x);

    else if (x > 20.0 && x <= 30.0)
      y_m = 0.001*(-28.0 + 4.046435022819e1 - 1.379581654948*x + 1.945884504128e-2*x*x - 2.070318932190e-4*x*x*x);

    else if (x > 30.0 && x <= 40.0)
      y_m = 0.001*(-28.0 + 1.792461334664e1 + 8.743920332081e-1*x - 5.567361123058e-2*x*x + 6.277731764683e-4*x*x*x);

    else if (x > 40.0 && x <= 54.0)
      y_m = 0.001*(-28.0 + std::max(0.0, 5.639011190988e1 - 2.010520359035*x + 1.644919857549e-2*x*x + 2.674976141766e-5*x*x*x));

    else if (x > 54.0)
      y_m = 0.001*(-28.0);

    return y_m;
  }

  double gamma_x(double x_m)
  {
    double xmod = x_m/0.028;
    double gamma = 1.;
    if (xmod<2.8) //slightly un-stretch mesh to get a better resolution of the shear layer
    {
      gamma = (0.06*std::cos(xmod*numbers::PI/1.4)+0.94);
    }
    else if (xmod > 8.) //move mesh closer to the wall to get a lower y+ value at the peak
    {
      xmod = 9. - xmod;
      gamma = (-0.05*std::cos(xmod*numbers::PI*2.)+1.05);
    }
    return GRID_STRETCH_FAC*gamma;
  }
  template<int dim>
  class PushForward : public Function<dim>
  {
  public:
    PushForward () : Function<dim>(dim, 0.){}//,component(component) {}

    virtual ~PushForward(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const double h = 0.028;

    // data from initial block
    const double x_max = 9.0*h;//9.0*h;
    const double y_max = 2.036*h;

    const double y_FoR = h;


  };
  template<int dim>
  double PushForward<dim>::value(const Point<dim> &p,const unsigned int component) const
  {
    double result = 0;

    // x component
    if (component == 0)
      result = p[0];

    // y component
    else if (component == 1)
    {
      const double gamma = gamma_x(p[0]);
      const double tanh_gamma = std::tanh(gamma);
      double s_y = std::tanh(gamma*(2.0*((p[1] - y_FoR)/y_max)-1.0))/tanh_gamma;
      double t_y = std::tanh(gamma*(2.0*(1.0 - (p[1] - y_FoR)/y_max)-1.0))/tanh_gamma;
      if (p[0] <= x_max/2.0)
        result = y_max/2.0*s_y+4.036*h/2.0 + (0.5*t_y + 0.5)*f_x(p[0]);
                // y_max/2.0*t_y+4.036*h/2.0
      else if (p[0] > x_max/2.0)
        result = y_max/2.0*s_y+4.036*h/2.0  + (0.5*t_y + 0.5)*f_x(x_max - p[0]);
    }

    // z component
    else if (component == 2 && dim==3)
      result = p[2];

    return result;
  }
  template<int dim>
  class PullBack : public Function<dim>
  {
  public:
    PullBack () : Function<dim>(dim, 0.){}//,component(component) {}

    virtual ~PullBack(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const double h = 0.028;
    const double x_max = 9.0*h;//9.0*h;
    const double y_max = 2.036*h;

    const double y_FoR = h;
  };
  template<int dim>
  double PullBack<dim>::value(const Point<dim> &p,const unsigned int component) const
  {
    double result = 0;

    // x component
    if (component == 0)
      result = p[0];

    // y component
    else if (component == 1)
    {
      unsigned int iter = 0;
      unsigned int maxiter = 100;
      double tol = 1.0e-14;
      double eps = 1.;

      // get a good estimate for Y (pullBack without stretching the cells)
      double Y = 0.0;
      if (p[0] <= x_max/2.0)
        Y = (p[1] - f_x(p[0])*(1 + y_FoR/y_max))/(1.0 - f_x(p[0])/y_max);
      else if (p[0] > x_max/2.0)
        Y = (p[1] - f_x(x_max - p[0])*(1 + y_FoR/y_max))/(1.0 - f_x(x_max - p[0])/y_max);

      if (p[0] <= x_max/2.0)
      {
        const double gamma = gamma_x(p[0]);
        const double tanh_gamma = std::tanh(gamma);
        while(eps > tol && iter < maxiter)
        {
          const double arg = gamma*(2.0*(1.0-(Y-y_FoR)/y_max)-1.0);
          const double arg2 = gamma*(2.0*((Y-y_FoR)/y_max)-1.0);
          const double t_y = std::tanh(arg)/tanh_gamma;
          const double s_y = std::tanh(arg2)/tanh_gamma;
          const double ts_y = 1.0/(std::cosh(arg)*std::cosh(arg))*gamma*(-2.0/y_max)/tanh_gamma;
          const double ss_y = 1.0/(std::cosh(arg2)*std::cosh(arg2))*gamma*(2.0/y_max)/tanh_gamma;
          const double Yn = Y - (y_max/2.0*s_y+4.036*h/2.0 + (0.5*t_y + 0.5)*f_x(p[0]) -p[1])
              / (y_max/2.0*ss_y  + 0.5*ts_y*f_x(p[0]));

          eps = std::abs(Yn-Y);
          Y = Yn;
          iter++;
        }
        AssertThrow(iter < maxiter,
            ExcMessage("Newton within PullBack did not find the solution. "));
        result = Y;
      }
      else if (p[0] > x_max/2.0)
      {
        const double gamma = gamma_x(p[0]);
        const double tanh_gamma = std::tanh(gamma);
        while(eps > tol && iter < maxiter)
        {
          const double arg = gamma*(2.0*(1.0-(Y-y_FoR)/y_max)-1.0);
          const double arg2 = gamma*(2.0*((Y-y_FoR)/y_max)-1.0);
          const double t_y = std::tanh(arg)/tanh_gamma;
          const double s_y = std::tanh(arg2)/tanh_gamma;
          const double ts_y = 1.0/(std::cosh(arg)*std::cosh(arg))*gamma*(-2.0/y_max)/tanh_gamma;
          const double ss_y = 1.0/(std::cosh(arg2)*std::cosh(arg2))*gamma*(2.0/y_max)/tanh_gamma;
          const double Yn = Y - (y_max/2.0*s_y+4.036*h/2.0 + (0.5*t_y + 0.5)*f_x(x_max - p[0]) -p[1])
              / (y_max/2.0*ss_y  + 0.5*ts_y*f_x(x_max - p[0]));

          eps = std::abs(Yn-Y);
          Y = Yn;
          iter++;
        }
        AssertThrow(iter < maxiter,
            ExcMessage("Newton within PullBack did not find the solution. "));
        result = Y;
      }
    }

    // z component
    else if (component == 2)
      result = p[2];

    return result;
  }

template<int dim>
class Enrichment:public Function<dim>
{
public:
  Enrichment (const double max_distance) : Function<dim>(1, 0.),
                                           max_distance(max_distance)
                                           {}
  virtual ~Enrichment (){};
  virtual double value (const Point<dim> &p,const unsigned int = 0) const
  {
    PullBack<dim> pull_back;
    double test = pull_back.value(p,1);
    test -= 0.028;
    test /= (2.036*0.028);
    test *= 2.;
    test -= 1.;
//    std::cout << test << std::endl;
    if ((test > (1.0-max_distance)) || (test <(-1.0 + max_distance)))
      return 1.;
    else
      return -1.;
  }
  const double max_distance;
};

  template<int dim>
  class AnalyticalSolutionVelocity : public Function<dim>
  {
  public:
    AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                                const double        time = 0.)
      :
      Function<dim>(n_components, time)
    {}

    virtual ~AnalyticalSolutionVelocity(){};

    virtual double value (const Point<dim>    &p,
                          const unsigned int  component = 0) const;
  };

  template<int dim>
  double AnalyticalSolutionVelocity<dim>::value(const Point<dim>   &p,
                                                const unsigned int component) const
  {
    double t = this->get_time();
    double result = 0.0;
    (void)t;

    PullBack<dim> pull_back_function_PH;

    if(component == 0)
    {
      if (t < 0.001)
      {
        double y_ref = pull_back_function_PH.value(p,1); // y_ref = [0.028, 0.08508]

        // initial conditions
        const double EPSILON = 1e-3;
        if(y_ref < 0.08508-EPSILON && y_ref > 0.028+EPSILON)
          result = (-5500.0*p[1]*p[1] + 530.0*p[1] -5.27);
        else
          result = 0.0;
        if(dim == 3)
          result *= (1.0 + 0.1*((double)rand() / (RAND_MAX)));
        //start with 30% of nominal velocity and ramp it up
        result *= 0.91;//Stephan J used another density and thus the initial condition was different
        if(dim==2)
          result *= 0.5;
      }
      else
        result = 0.0;
    }
    return result;
  }

  template<int dim>
  class AnalyticalSolutionPressure : public Function<dim>
  {
  public:
    AnalyticalSolutionPressure (const double time = 0.)
      :
      Function<dim>(1 /*n_components*/, time)
    {}

    virtual ~AnalyticalSolutionPressure(){};

    virtual double value (const Point<dim>   &p,
                          const unsigned int component = 0) const;
  };

  template<int dim>
  double AnalyticalSolutionPressure<dim>::value(const Point<dim>    &,
                                                const unsigned int  /* component */) const
  {
    double result = 0.;
    return result;
  }
  template<int dim>
  class NeumannBoundaryVelocity : public Function<dim>
  {
  public:
    NeumannBoundaryVelocity (const double time = 0.)
      :
      Function<dim>(dim, time)
    {}

    virtual ~NeumannBoundaryVelocity(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double NeumannBoundaryVelocity<dim>::value(const Point<dim> &,const unsigned int ) const
  {
    double result = 0.0;
    return result;
  }

  template<int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide (const double time = 0.)
      :
      Function<dim>(dim, time),
      oldforce_(13.5)
    {
      massflows_.resize(2);
      if(dim==2)
      {
        massflows_[0] = 0.160238;
        massflows_[1] = 0.160238;
      }
      else
      {
        massflows_[0] = 40.38e-3;
        massflows_[1] = 40.38e-3;
      }
    }

    virtual ~RightHandSide(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

    void setup(const std::vector<double> &massflows, double oldforce);
  private:
    std::vector<double> massflows_;
    double oldforce_;
  };


  template<int dim>
  void RightHandSide<dim>::setup(const std::vector<double> &massflows, double oldforce)
  {
    massflows_.resize(2);
    massflows_[0] = massflows[0];
    massflows_[1] = massflows[1];
    oldforce_ = oldforce;
  }


  template<int dim>
  double RightHandSide<dim>::value(const Point<dim> &,const unsigned int component) const
  {
    // use a controller for calculating the source term fx such that the massflow is the ideal massflow
    if (USE_SOURCE_TERM_CONTROLLER)
    {
      double newforce = 0.0;
      if(component==0)
      {
        double massflow_id = 40.38e-3;
        if(dim==2)
        {
            massflow_id = 0.320476;
          //start with 10% of the ideal mass flow and increase within 0.1 time units to 1
          if(this->get_time()<0.1)
            massflow_id *= std::cos(this->get_time()*numbers::PI*10.+numbers::PI)*0.25+0.75;
        }

        //new estimated force
        //first contribution makes the system want to get back to the ideal value (spring)
        //the second contribution is a penalty on quick changes (damping)
        //the constants are empirical

        double dgoalm = massflow_id - massflows_[0];
        double dm = massflows_[0] - massflows_[1];
        if(dim==3)
          newforce = 500.0*dgoalm  - 30000.0*dm + oldforce_; //(B1)
        if(dim==2)
          newforce = 50.0*dgoalm  - 30000.0*dm + oldforce_; //(B1)
        return newforce;
      }
      else
        return 0.0;
    }

    // use an estimate for fx (see Stefan Jaeger 2015)
    else
    {
      if(component==0)
        if(this->get_time()<0.01)
          return 13.5;//*(1.0+((double)rand()/RAND_MAX)*0.0);
        else
          return 13.5;
      else
        return 0.0;
    }


  double result = 0.0;
  return result;
  }

  template<int dim>
  class PressureBC_dudt : public Function<dim>
  {
  public:
    PressureBC_dudt (const double time = 0.)
      :
      Function<dim>(dim, time)
    {}

    virtual ~PressureBC_dudt(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double PressureBC_dudt<dim>::value(const Point<dim> &,const unsigned int ) const
  {
  double result = 0.0;

  return result;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
  class TimeIntBDFDualSplittingPH : public virtual TimeIntBDFDualSplitting<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>
  {
  public:
    TimeIntBDFDualSplittingPH(
        std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > ns_operation_in,
        std_cxx11::shared_ptr<PostProcessorBase<dim> >                                                            postprocessor_in,
        InputParametersNavierStokes<dim> const                                                                   &param_in,
        unsigned int const                                                                                        n_refine_time_in,
        bool const                                                                                                use_adaptive_time_stepping)
      :
      TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
              ns_op(std::dynamic_pointer_cast<DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > (ns_operation_in)),
              old_RHS_value(13.5),
              rhs(nullptr)
              {
                massflows.resize(2);
                if(dim==2)
                {
                  massflows[0] = 0.160238;
                  massflows[1] = 0.160238;
                }
                else
                {
                  massflows[0] = 40.38e-3;
                  massflows[1] = 40.38e-3;
                }
              }

      virtual ~TimeIntBDFDualSplittingPH(){}
  protected:

    std_cxx11::shared_ptr<DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > ns_op;
    std::vector<double> massflows;
    double old_RHS_value;
    std_cxx11::shared_ptr<RightHandSide<dim> > rhs;
    void convective_step()
    {
      setup_controller();
      compute_massflow_PH(this->velocity[0]);

      TimeIntBDFDualSplitting<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>::convective_step();
    }
    void read_restart_data_ph(boost::archive::binary_iarchive & ia)
    {
      double massflows0;
      double massflows1;
      double old_force;
      ia & massflows0;
      ia & massflows1;
      ia & old_force;
      this->massflows[0] = massflows0;
      this->massflows[1] = massflows1;
      this->old_RHS_value = old_force;
    }
    void write_restart_data_ph(boost::archive::binary_oarchive & oa) const
    {
      oa & this->massflows[0];
      oa & this->massflows[1];
      oa & this->old_RHS_value;
    }
    virtual void read_restart_vectors(boost::archive::binary_iarchive & ia)
    {
      TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
      read_restart_vectors(ia);

      this->read_restart_data_ph(ia);
    }

    virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) const
    {
      TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
          write_restart_vectors(oa);

      this->write_restart_data_ph(oa);
    }

  private:
    void compute_massflow_PH (const parallel::distributed::Vector<value_type>     &src)
    {
      double dst_loc = 0.0;
      ns_op->get_data().cell_loop(&TimeIntBDFDualSplittingPH<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_compute_massflow_PH,
                                     this, dst_loc, src);

      dst_loc = Utilities::MPI::sum(dst_loc, MPI_COMM_WORLD);
      massflows[1] = massflows[0];
      massflows[0] = dst_loc/(9.0*0.028); //9.0*0.028 = 9.0*h is the length of the periodic hill domain
  #ifdef DEBUG_MASSFLOW_CONTROLLER
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
  //      std::cout << "old massflow = " << massflows[1] << std::endl;
        std::cout << "massflow = " << massflows[0] << std::endl;
      }
  #endif
    }

    void setup_controller()
    {
      if(rhs == nullptr)
        rhs = std::dynamic_pointer_cast<RightHandSide<dim> > (ns_op->get_field_functions()->right_hand_side);
      Point<dim> dummy;
      rhs->setup(massflows,old_RHS_value);
      old_RHS_value = rhs->value(dummy);
  #ifdef DEBUG_MASSFLOW_CONTROLLER
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "fx = " << old_RHS_value << "  dm/dt old = "<< massflows[1] << "  dm/dt new = "<< massflows[0] << std::endl;
      }
  #endif
      if(this->time_step_number% this->param.output_solver_info_every_timesteps == 0)
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "fx = " << old_RHS_value << "  dm/dt old = "<< massflows[1] << "  dm/dt new = "<< massflows[0] << std::endl;
        }
    }
    void local_compute_massflow_PH (const MatrixFree<dim,value_type>                       &data,
                                    double                                                 &dst,
                                    const parallel::distributed::Vector<value_type>   &src,
                                    const std::pair<unsigned int,unsigned int>             &cell_range) const
    {
      static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
      static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;

      FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall>
          fe_eval(data,&ns_op->get_fe_parameters(),0);

      double dmdt_V_loc = 0.0;
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        VectorizedArray<value_type> dmdt_V_loc_array =
                make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> u_x =
            make_vectorized_array<value_type>(0.0);
        AlignedVector<VectorizedArray<value_type> > JxW_values;
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate (true,false);

        JxW_values.resize(fe_eval.n_q_points);
        fe_eval.fill_JxW_values(JxW_values);

        // loop over quadrature points
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
  //        JxW = fe_eval.J_value[q]*fe_eval.quadrature_weights[q];
          u_x = fe_eval.get_value(q)[0];
          dmdt_V_loc_array += u_x*JxW_values[q];
        }
        // loop over vectorized array
        for (unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
        {
          dmdt_V_loc += dmdt_V_loc_array[v];
        }
      }
      dst += dmdt_V_loc;
    }

  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
  class TimeIntBDFDualSplittingXWallPH : public virtual TimeIntBDFDualSplittingXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>,
                                         public virtual TimeIntBDFDualSplittingPH<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>
  {
  public:
    TimeIntBDFDualSplittingXWallPH(
        std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > ns_operation_in,
        std_cxx11::shared_ptr<PostProcessorBase<dim> >                                                            postprocessor_in,
        InputParametersNavierStokes<dim> const                                                                   &param_in,
        unsigned int const                                                                                        n_refine_time_in,
        bool const                                                                                                use_adaptive_time_stepping)
      :
      TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
      TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
      TimeIntBDFDualSplittingPH<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping)
    {}

    virtual ~TimeIntBDFDualSplittingXWallPH(){};

  protected:
    virtual void read_restart_vectors(boost::archive::binary_iarchive & ia)
    {
      TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
      read_restart_vectors(ia);

      this->read_restart_data_ph(ia);
    }

    virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) const
    {
      TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
          write_restart_vectors(oa);

      this->write_restart_data_ph(oa);
    }
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
  class TimeIntBDFDualSplittingXWallSpalartAllmarasPH : public virtual TimeIntBDFDualSplittingXWallSpalartAllmaras<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>,
                                                        public virtual TimeIntBDFDualSplittingPH<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>
  {
  public:
    TimeIntBDFDualSplittingXWallSpalartAllmarasPH(
      std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > ns_operation_in,
      std_cxx11::shared_ptr<PostProcessorBase<dim> >                                                            postprocessor_in,
      InputParametersNavierStokes<dim> const                                                                   &param_in,
      unsigned int const                                                                                        n_refine_time_in,
      bool const                                                                                                use_adaptive_time_stepping)
      :
      TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
      TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
      TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
      TimeIntBDFDualSplittingPH<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
              (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping)
    {}

    virtual ~TimeIntBDFDualSplittingXWallSpalartAllmarasPH(){}

  protected:
    virtual void read_restart_vectors(boost::archive::binary_iarchive & ia)
    {
      TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
      read_restart_vectors(ia);

      this->read_restart_data_ph(ia);

      this->read_restart_data_sa(ia);
    }

    virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) const
    {
      TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
          write_restart_vectors(oa);

      this->write_restart_data_ph(oa);

      this->write_restart_data_sa(oa);
    }
  };

  bool exists_test0 (const std::string& name)
  {
      std::ifstream f(name.c_str());
      return f.good();
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
  class PostProcessorPH: public PostProcessor<dim,fe_degree,fe_degree_p>
  {
  public:

    PostProcessorPH(
                  std_cxx11::shared_ptr< const DGNavierStokesBase<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule> >  ns_operation,
                  PostProcessorData<dim> const &pp_data):
      PostProcessor<dim,fe_degree,fe_degree_p>(pp_data),
      output_data(pp_data.output_data),
      turb_stat_data(pp_data.turb_stat_data),
      statistics_ph(ns_operation->get_dof_handler_u(),ns_operation->get_dof_handler_p(),ns_operation->get_mapping(),true),
      matrix_free_data(nullptr),
      output_counter_q(output_data.output_counter_start),
      mapping(nullptr),
      statistics_number(0)
    {
    }

    virtual ~PostProcessorPH(){}

    virtual void setup(DoFHandler<dim> const                                        &dof_handler_velocity_in,
                       DoFHandler<dim> const                                        &dof_handler_pressure_in,
                       Mapping<dim> const                                           &mapping_in,
                       MatrixFree<dim,double> const                                 &matrix_free_data_in,
                       DofQuadIndexData const                                       &dof_quad_index_data_in,
                       std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution_in)
    {
      PostProcessor<dim,fe_degree,fe_degree_p>::setup(dof_handler_velocity_in,
                                                      dof_handler_pressure_in,
                                                      mapping_in,
                                                      matrix_free_data_in,
                                                      dof_quad_index_data_in,
                                                      analytical_solution_in);

      statistics_ph.setup(PushForward<dim>(),this->output_data.output_prefix,false);
      matrix_free_data = &matrix_free_data_in;
      dof_quad_index_data = dof_quad_index_data_in;
      mapping = &mapping_in;
      if(turb_stat_data.write_output_q_criterion == true)
      {
        matrix_free_data->initialize_dof_vector(q_criterion,dof_quad_index_data.dof_index_pressure);
        matrix_free_data->initialize_dof_vector(u_magnitude,dof_quad_index_data.dof_index_pressure);
      }

      // search for next statistics number
      std::ostringstream filename;
      filename.str("");
      filename.clear();
      filename << this->output_data.output_prefix
               << "_slice"
               << statistics_number
               << ".tauw_Yplus_flow_statistics_bottom";
      while(exists_test0(filename.str()))
      {
        statistics_number++;
        filename.str("");
        filename.clear();
        filename << this->output_data.output_prefix
                 << "_slice"
                 << statistics_number
                 << ".tauw_Yplus_flow_statistics_bottom";
      }
    }

    virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                                   parallel::distributed::Vector<double> const &intermediate_velocity,
                                   parallel::distributed::Vector<double> const &pressure,
                                   parallel::distributed::Vector<double> const &vorticity,
                                   parallel::distributed::Vector<double> const &divergence,
                                   double const                                time = 0.0,
                                   int const                                   time_step_number = -1)
    {
      PostProcessor<dim,fe_degree,fe_degree_p>::do_postprocessing(velocity,
                                                                  intermediate_velocity,
                                                                  pressure,
                                                                  vorticity,
                                                                  divergence,
                                                                  time,
                                                                  time_step_number);
      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

      if(time > this->turb_stat_data.statistics_start_time-EPSILON && time_step_number % this->turb_stat_data.statistics_every == 0)
      {
        statistics_ph.evaluate(velocity,pressure);
        if(time_step_number % 100 == 0 || time > (this->turb_stat_data.statistics_end_time-EPSILON))
          statistics_ph.write_output(this->output_data.output_prefix,this->turb_stat_data.viscosity,statistics_number);
      }

      if(turb_stat_data.write_output_q_criterion == true)
        if(time > (output_data.output_start_time + output_counter_q*output_data.output_interval_time - EPSILON))
        {
          write_output_q_criterion(velocity);
          output_counter_q++;
        }
    };

    void write_output_q_criterion(parallel::distributed::Vector<double> const &velocity)
    {
      DataOut<dim> data_out;
      compute_q_criterion(q_criterion,velocity);
      compute_u_magnitude(u_magnitude,velocity);
      data_out.add_data_vector (matrix_free_data->get_dof_handler(dof_quad_index_data.dof_index_pressure),q_criterion, "q_criterion");
      data_out.add_data_vector (matrix_free_data->get_dof_handler(dof_quad_index_data.dof_index_pressure),u_magnitude, "u_magnitude");

      std::ostringstream filename;
      filename << "output/"
               << output_data.output_prefix
               << "_q_Proc"
               << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
               << "_"
               << output_counter_q
               << ".vtu";

      data_out.build_patches (*mapping, output_data.number_of_patches, DataOut<dim>::curved_inner_cells);

      std::ofstream output (filename.str().c_str());
      data_out.write_vtu (output);

      if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
        {
          std::ostringstream filename;
          filename << output_data.output_prefix
                   << "_q_Proc"
                   << i
                   << "_"
                   << output_counter_q
                   << ".vtu";

          filenames.push_back(filename.str().c_str());
        }
        std::string master_name = "output/" + output_data.output_prefix + "_q_" + Utilities::int_to_string(output_counter_q) + ".pvtu";
        std::ofstream master_output (master_name.c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
    }

    void compute_q_criterion(parallel::distributed::Vector<double>       &dst,
                             parallel::distributed::Vector<double> const &velocity)
    {
      dst = 0;
      matrix_free_data->cell_loop (&PostProcessorPH<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>::local_compute_q_criterion,
                                   this, dst, velocity);
    }

    void compute_u_magnitude(parallel::distributed::Vector<double>       &dst,
                             parallel::distributed::Vector<double> const &velocity)
    {
      dst = 0;
      matrix_free_data->cell_loop (&PostProcessorPH<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>::local_compute_u_magnitude,
                                   this, dst, velocity);
    }

    void local_compute_q_criterion(const MatrixFree<dim,double>                &data,
                                   parallel::distributed::Vector<double>       &dst,
                                   const parallel::distributed::Vector<double> &source,
                                   const std::pair<unsigned int,unsigned int>  &cell_range)
    {
      //we use the pressure space for the q-criterion because we only need a vector with one component
      //the pressure space is also sufficient in case k,k-1 for u,p, because the q-criterion only consists of derivatives of the velocity
      FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,double> phi(data,dof_quad_index_data.dof_index_pressure,dof_quad_index_data.quad_index_velocity);
      FEEvaluation<dim,fe_degree,fe_degree_p+1,dim,double> fe_eval_u(data,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_p, 1, double> inverse(phi);
      const unsigned int dofs_per_cell = phi.dofs_per_cell;
      AlignedVector<VectorizedArray<double> > coefficients(dofs_per_cell);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit(cell);
        fe_eval_u.reinit(cell);
        fe_eval_u.read_dof_values(source);
        fe_eval_u.evaluate(false,true);
        for(unsigned int q=0;q<phi.n_q_points;q++)
        {
          Tensor<2,dim,VectorizedArray<double> > gradu = fe_eval_u.get_gradient(q);
          Tensor<2,dim,VectorizedArray<double> > Om;
          Tensor<2,dim,VectorizedArray<double> > S;
          for(unsigned int i=0;i<dim;i++)
            for(unsigned int j=0;j<dim;j++)
            {
              Om[i][j]=0.5*(gradu[i][j]-gradu[j][i]);
              S[i][j]=0.5*(gradu[i][j]+gradu[j][i]);
            }
          const VectorizedArray<double> Q = 0.5*(Om.norm_square() - S.norm_square());
          phi.submit_value(Q,q);
        }
        phi.integrate(true,false);
        inverse.fill_inverse_JxW_values(coefficients);
        inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());
        phi.distribute_local_to_global(dst);
      }
    }

    void local_compute_u_magnitude(const MatrixFree<dim,double>                &data,
                                   parallel::distributed::Vector<double>       &dst,
                                   const parallel::distributed::Vector<double> &source,
                                   const std::pair<unsigned int,unsigned int>  &cell_range)
    {
      //we use the pressure space for the q-criterion because we only need a vector with one component
      //the pressure space is also sufficient in case k,k-1 for u,p, because the q-criterion only consists of derivatives of the velocity
      FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,double> phi(data,dof_quad_index_data.dof_index_pressure,dof_quad_index_data.quad_index_velocity);
      FEEvaluation<dim,fe_degree,fe_degree_p+1,dim,double> fe_eval_u(data,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_p, 1, double> inverse(phi);
      const unsigned int dofs_per_cell = phi.dofs_per_cell;
      AlignedVector<VectorizedArray<double> > coefficients(dofs_per_cell);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit(cell);
        fe_eval_u.reinit(cell);
        fe_eval_u.read_dof_values(source);
        fe_eval_u.evaluate(true,false);
        for(unsigned int q=0;q<phi.n_q_points;q++)
        {
          Tensor<1,dim,VectorizedArray<double> > u = fe_eval_u.get_value(q);
          phi.submit_value(u.norm(),q);
        }
        phi.integrate(true,false);
        inverse.fill_inverse_JxW_values(coefficients);
        inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());
        phi.distribute_local_to_global(dst);
      }
    }

  protected:
    OutputDataNavierStokes output_data;
    TurbulenceStatisticsData turb_stat_data;
    StatisticsManagerPH<dim> statistics_ph;
    MatrixFree<dim,double> const * matrix_free_data;
    unsigned int output_counter_q;
    DofQuadIndexData dof_quad_index_data;
    parallel::distributed::Vector<double> q_criterion;
    parallel::distributed::Vector<double> u_magnitude;
    Mapping<dim> const * mapping;
    unsigned int statistics_number;

  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
  class PostProcessorPHXWall: public PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>
  {
  public:

    PostProcessorPHXWall(std_cxx11::shared_ptr< DGNavierStokesBase<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule> >  ns_operation,
                         PostProcessorData<dim> const                                                                           &pp_data)
                           :
                         PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>(ns_operation,pp_data),
                         statistics_ph(ns_operation->get_dof_handler_u(),ns_operation->get_dof_handler_p(),ns_operation->get_mapping()),
                         statistics_number(0)
    {
    }

    virtual ~PostProcessorPHXWall(){}

    void setup(DoFHandler<dim> const                                        &dof_handler_velocity_in,
               DoFHandler<dim> const                                        &dof_handler_pressure_in,
               Mapping<dim> const                                           &mapping_in,
               MatrixFree<dim,double> const                                 &matrix_free_data,
               DofQuadIndexData const                                       &dof_quad_index_data,
               std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution)
    {
      PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>::
            setup(dof_handler_velocity_in,
                  dof_handler_pressure_in,
                  mapping_in,
                  matrix_free_data,
                  dof_quad_index_data,
                  analytical_solution);
      statistics_ph.setup(PushForward<dim>(),this->pp_data.output_data.output_prefix,true);

      // search for next statistics number
      std::ostringstream filename;
      filename.str("");
      filename.clear();
      filename << this->pp_data.output_data.output_prefix
               << "_slice"
               << statistics_number
               << ".tauw_Yplus_flow_statistics_bottom";
      while(exists_test0(filename.str()))
      {
        statistics_number++;
        filename.str("");
        filename.clear();
        filename << this->pp_data.output_data.output_prefix
                 << "_slice"
                 << statistics_number
                 << ".tauw_Yplus_flow_statistics_bottom";
      }
    }

    virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                                   parallel::distributed::Vector<double> const &intermediate_velocity,
                                   parallel::distributed::Vector<double> const &pressure,
                                   parallel::distributed::Vector<double> const &vorticity,
                                   parallel::distributed::Vector<double> const &divergence,
                                   double const                                time = 0.0,
                                   int const                                   time_step_number = -1)
    {
      PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>::do_postprocessing(velocity,
                                                                                                       intermediate_velocity,
                                                                                                       pressure,
                                                                                                       vorticity,
                                                                                                       divergence,
                                                                                                       time,
                                                                                                       time_step_number);

      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

      if(time > this->pp_data.turb_stat_data.statistics_start_time-EPSILON && time_step_number % this->pp_data.turb_stat_data.statistics_every == 0)
      {
        statistics_ph.evaluate_xwall(velocity,
                                     pressure,
                                     this->ns_operation_xw_->get_dof_handler_wdist(),
                                     this->ns_operation_xw_->get_fe_parameters());
        if(time_step_number % 100 == 0 || time > (this->pp_data.turb_stat_data.statistics_end_time-EPSILON))
          statistics_ph.write_output(this->pp_data.output_data.output_prefix,this->pp_data.turb_stat_data.viscosity,statistics_number);
      }

    };
  protected:
    StatisticsManagerPH<dim> statistics_ph;
    unsigned int statistics_number;

  };

  template<int dim>
  class NavierStokesProblem
  {
  public:
    typedef typename DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>::value_type value_type;
    NavierStokesProblem(const unsigned int refine_steps_space, const bool do_restart = false);
    void solve_problem(bool do_restart);

  private:
    void create_grid();
    void print_parameters() const;

    ConditionalOStream pcout;
    // TMP
    PushForward<dim> push_forward;
    PullBack<dim> pull_back;
    FunctionManifold<dim,dim,dim> manifold;
    parallel::distributed::Triangulation<dim> triangulation;
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

    const unsigned int n_refine_space;

    std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity;
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure;

    InputParametersNavierStokes<dim> param;

    std_cxx11::shared_ptr<DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> > navier_stokes_operation;

    std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor;

    std_cxx11::shared_ptr<TimeIntBDFNavierStokes<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type> > time_integrator;

  };

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps_space, const bool do_restart):
  pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  push_forward(),
  pull_back(),
  manifold(push_forward, pull_back),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space)
  {
    PrintInputParams::Header(pcout);

    param.set_input_parameters();
    param.check_input_parameters();

    if(do_restart == false)
      AssertThrow(param.start_with_low_order == true,ExcMessage("start with low order unless you do a restart"));

    // initialize functions (analytical solution, rhs, boundary conditions)
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
    analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(dim,param.start_time));
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
    analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(param.start_time));

    std_cxx11::shared_ptr<Function<dim> > right_hand_side;
    right_hand_side.reset(new RightHandSide<dim>(param.start_time));

    field_functions.reset(new FieldFunctionsNavierStokes<dim>());
    field_functions->initial_solution_velocity = analytical_solution_velocity;
    field_functions->initial_solution_pressure = analytical_solution_pressure;
    field_functions->analytical_solution_pressure = analytical_solution_pressure;
    field_functions->right_hand_side = right_hand_side;

    boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokes<dim>());
    boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokes<dim>());

    PostProcessorData<dim> pp_data;
    pp_data.output_data = param.output_data;
    pp_data.turb_stat_data = param.turb_stat_data;
    pp_data.turb_stat_data.viscosity = param.viscosity;

    // set next empty output number if we do restart
    if(do_restart)
    {
      unsigned int output_number = 0;
      std::ostringstream filename;
      filename.str("");
      filename.clear();
      filename << "output/"
               << pp_data.output_data.output_prefix
               << "_"
               << output_number
               << ".pvtu";
      while(exists_test0(filename.str()))
      {
        output_number++;
        filename.str("");
        filename.clear();
        filename << "output/"
                 << pp_data.output_data.output_prefix
                 << "_"
                 << output_number
                 << ".pvtu";
      }
      pp_data.output_data.output_counter_start = output_number;
    }

    bool use_adaptive_time_stepping = false;
    if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
      use_adaptive_time_stepping = true;

    if(param.spatial_discretization == SpatialDiscretization::DGXWall)
    {
      analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(2*dim,param.start_time));
      field_functions->initial_solution_velocity = analytical_solution_velocity;
      if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        // initialize navier_stokes_operation
        if(param.xwall_turb == XWallTurbulenceApproach::RANSSpalartAllmaras ||
           param.xwall_turb == XWallTurbulenceApproach::ClassicalDESSpalartAllmaras)
        {
          navier_stokes_operation.reset(new DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
              (triangulation,param));
          // initialize postprocessor after initializing navier_stokes_operation
          postprocessor.reset(new PostProcessorPHXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,pp_data));
          // initialize time integrator that depends on both navier_stokes_operation and postprocessor
          time_integrator.reset(new TimeIntBDFDualSplittingXWallSpalartAllmarasPH<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
              navier_stokes_operation,postprocessor,param,0,use_adaptive_time_stepping));
        }
        else if(param.xwall_turb == XWallTurbulenceApproach::None)
        {
          navier_stokes_operation.reset(new DGNavierStokesDualSplittingXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
              (triangulation,param));
          // initialize postprocessor after initializing navier_stokes_operation
          postprocessor.reset(new PostProcessorPHXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,pp_data));
          // initialize time integrator that depends on both navier_stokes_operation and postprocessor
          time_integrator.reset(new TimeIntBDFDualSplittingXWallPH<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
              navier_stokes_operation,postprocessor,param,0,use_adaptive_time_stepping));
        }
        else if(param.xwall_turb == XWallTurbulenceApproach::Undefined)
        {
          AssertThrow(false,ExcMessage("Turbulence approach for xwall undefined"));
        }
      }
      else
      {
        AssertThrow(false,ExcMessage("XWall only implemented for the unsteady DualSplitting case up to now"));
      }
    }
    else if(param.spatial_discretization == SpatialDiscretization::DG)
    {
      if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesDualSplitting<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessorPH<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,pp_data));
        // initialize time integrator that depends on both navier_stokes_operation and postprocessor
        time_integrator.reset(new TimeIntBDFDualSplittingPH<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
            navier_stokes_operation,postprocessor,param,0,use_adaptive_time_stepping));
      }
      else if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      {
        AssertThrow(false,ExcMessage("currently not implemented. One would have to include the adaptive bodyforce in the RHS"));
      }
    }
  }

  template<int dim>
  void NavierStokesProblem<dim>::create_grid ()
  {
    /* --------------- Generate grid ------------------- */
    const double h = 0.028;
    Point<dim> coordinates;
    coordinates[0] = 9.0*h;//9.0*h;
    coordinates[1] = 3.036*h;//2.036*h;
    if (dim == 3)
      coordinates[2] = 2.25*h;//4.5*h;
    std::vector<unsigned int> refinements(dim, 1);
    refinements[0] = 2;

    // start with a cube
    Point<dim> p;
    p[0] = 0.;
    p[1] = h;
    if(dim==3)
      p[2] = -2.25*h;

    GridGenerator::subdivided_hyper_rectangle (triangulation,refinements,p,coordinates);

    triangulation.last()->vertex(0)[1] = 0.;
    if(dim==3)
      triangulation.last()->vertex(4)[1] = 0.;
    // boundary ids for refinements[0] = 2:
    //periodicity in x-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    triangulation.begin()->face(0)->set_all_boundary_ids(0+10);
    triangulation.last()->face(1)->set_all_boundary_ids(1+10);

    //periodicity in z-direction, if dim==3
    if(dim == 3)
    {
      triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
      triangulation.begin()->face(5)->set_all_boundary_ids(3+10);
      triangulation.last()->face(4)->set_all_boundary_ids(4+10);
      triangulation.last()->face(5)->set_all_boundary_ids(5+10);
    }

    GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
    if(dim == 3)
    {
      GridTools::collect_periodic_faces(triangulation, 2+10, 3+10, 2, periodic_faces);
      GridTools::collect_periodic_faces(triangulation, 4+10, 5+10, 2, periodic_faces);
    }

    triangulation.add_periodicity(periodic_faces);

    triangulation.begin()->set_all_manifold_ids(111);
    triangulation.last()->set_all_manifold_ids(111);

    triangulation.set_manifold(111, manifold);

    triangulation.refine_global(n_refine_space);

    // fill boundary descriptor velocity
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
    analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(dim,param.start_time));
    // Dirichlet boundaries: ID = 0
    boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                      (0,analytical_solution_velocity));

    std_cxx11::shared_ptr<Function<dim> > neumann_bc_velocity;
    neumann_bc_velocity.reset(new NeumannBoundaryVelocity<dim>(param.start_time));
    // Neumann boundaris: ID = 1
    boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (1,neumann_bc_velocity));

    // fill boundary descriptor pressure
    std_cxx11::shared_ptr<Function<dim> > pressure_bc_dudt;
    pressure_bc_dudt.reset(new PressureBC_dudt<dim>(param.start_time));
    // Dirichlet boundaries: ID = 0
    boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                      (0,pressure_bc_dudt));

    std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
    analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(param.start_time));
    // Neumann boundaries: ID = 1
    boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (1,analytical_solution_pressure));

    PrintInputParams::print_spatial_discretization(pcout,dim,n_refine_space,
        triangulation.n_global_active_cells(),triangulation.n_active_faces(),triangulation.n_vertices() );
  }

template<int dim>
void NavierStokesProblem<dim>::solve_problem(bool do_restart)
{
  create_grid();

  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  time_integrator->setup(do_restart);

  navier_stokes_operation->setup_solvers();

  PrintInputParams::print_solver_parameters(pcout,param);
  PrintInputParams::print_turbulence_parameters(pcout,param,GRID_STRETCH_FAC);
  PrintInputParams::print_linear_solver_tolerances_dual_splitting(pcout,param);
  if(param.spatial_discretization == SpatialDiscretization::DGXWall)
    PrintInputParams::print_xwall_parameters(pcout,param,N_Q_POINTS_1D_XWALL);

  std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolutionNavierStokes<dim>());
  analytical_solution->velocity = field_functions->initial_solution_velocity;
  analytical_solution->pressure = field_functions->initial_solution_pressure;

  DofQuadIndexData dof_quad_index_data;

  postprocessor->setup(navier_stokes_operation->get_dof_handler_u(),
                       navier_stokes_operation->get_dof_handler_p(),
                       navier_stokes_operation->get_mapping(),
                       navier_stokes_operation->get_data(),
                       dof_quad_index_data,
                       analytical_solution);

  time_integrator->timeloop();

}

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    bool do_restart = false;
    if (argc > 1)
    {
      do_restart = std::atoi(argv[1]);
    }

    NavierStokesProblem<DIMENSION> navier_stokes_problem(REFINE_STEPS_SPACE,do_restart);
    navier_stokes_problem.solve_problem(do_restart);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}