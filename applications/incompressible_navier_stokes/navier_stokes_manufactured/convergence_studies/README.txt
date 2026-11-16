Convergence studies of navier_stokes_manufactured (generalized Newtonian flow,
Carreau-Yasuda): domain (0,1)^2, pure Dirichlet, divergence formulation, L2
(DG) discretization, serial runs.

Layout: one folder per study, grouped by the scheme the study is about,
  dual_splitting/{spatial,temporal_high_viscosity,temporal_low_viscosity}/
  consistent_splitting/{spatial,temporal,temporal_moving_mesh}/
each with
  study.json                   input shared by all runs ("base") and the
                               varied values (schemes, convective term,
                               viscosity, degrees or BDF orders)
  summary_*_convergence.txt    the results
and shared by all studies
  generate_inputs.py   writes the inputs input_<name>.json of a study folder
  run_study.sh         generates the inputs and runs all of them
  summarize.py         prints the tables of a summary from the logs
The scheme names are those of TemporalDiscretization (coupled =
BDFCoupledSolution, ...); consistent_splitting_no_leray is
BDFConsistentSplitting with ApplyLerayProjection = false.

Rerun a study (8 runs at a time, 1 MPI process each, the defaults):
  ./run_study.sh dual_splitting/spatial <path to solver> 8 1
Each run writes `log_<name>.txt` (with the convergence table at the end) and
`output_<name>/` (error files `run_<id>_*`, one per refinement level) into the
study folder. The script deletes the output folder of a run before running it;
when running an input by hand, its output folder must not exist, since the
convergence table is computed from all error files in it. Then print the tables
of the summary by
  python3 summarize.py dual_splitting/spatial
Generated inputs, logs and output folders are not stored (.gitignore).


dual_splitting/
  The paper's settings (arXiv 2506.14424, ExaDG 8d3db6fbe): nu_oo = 5e-2,
  margin 5e-2, kappa 1, lambda 1, a 2, n 0.5. All combinations are covered:
  scheme (coupled, dual_splitting, pressure_correction) x convective term
  (explicit, linearly_implicit, implicit) x viscosity (explicit, implicit).

dual_splitting/spatial/
  Navier-Stokes, T = 0.1, dt = 1e-5, BDF2, degree 2, 3, 4, refinements 0..4.
  - coupled and pressure_correction: full rates (velocity k+1, pressure k) for
    all treatments, identical errors to 2-3 digits.
  - dual_splitting: levels off at its temporal (splitting) error, velocity about
    6e-8, as in the paper.

dual_splitting/temporal_high_viscosity/
  Navier-Stokes, degree 8, refinement 3, T = 1, dt = 1/512 ... 1/4096, BDF1/2/3,
  nu_oo = 5e-2, margin 5e-2 (paper's temporal_convergence_new).
  - coupled: rates 1, 2, 3; BDF3 reaches the floor (velocity ~1e-13, pressure
    ~3e-11, earlier for a (linearly) implicit convective term).
  - pressure_correction: velocity rates 1, 2, 3; pressure about 0.8 (BDF1),
    1.75-1.8 (BDF2), 2.8-2.9 (BDF3). BDF3 with EXPLICIT viscosity: the error
    increases again at dt = 1/4096 (velocity 1.5e-9 instead of ~3e-12), i.e., a
    growing perturbation; with implicit viscosity rate 3.
  - dual_splitting: rate 1 for BDF1, BDF2 and BDF3 (velocity and pressure), the
    order reduction for generalized Newtonian flow reported in the paper.

dual_splitting/temporal_low_viscosity/
  Stokes (Navier-Stokes with explicit convection is CFL-limited to dt <= 1/512,
  where dual splitting is already first order), nu_oo = 1e-6, margin 4.9e-5,
  degree 6, refinement 3, T = 10, dt = 1 ... 1/1024, BDF1/2/3 (paper's older
  temporal_convergence study). 3 schemes x 2 viscosity treatments.
  - BDF1, BDF2: rates 1 and 2 for all schemes and treatments.
  - BDF3: coupled rate 3 down to ~3e-11; dual splitting ~3 down to dt ~1/32,
    then first order from dt = 1/128 (as in the paper); pressure correction
    rate 3 down to dt = 1/128, then unstable (errors O(1)) with pressure
    extrapolation order 2 (order_time_integrator - 1 in this app).
  - With the momentum/velocity-block preconditioners updated every time step,
    the Chebyshev eigenvalue estimation (deal.II's fused CG, absolute
    tolerance, float) broke down (NaN) on the continuous cphMG levels of the
    mass-dominated operator. Fixed in ExaDG without changing deal.II:
    ChebyshevSmoother uses the non-fused deal.II algorithms on continuous
    levels (AdditionalData::use_fused_operations = is_dg), DG levels keep the
    fused vector operations.


consistent_splitting/{spatial,temporal}/
  Consistent splitting with a variable viscosity compared to the other schemes,
  Carreau-Yasuda with the blood parameters kappa 1, lambda 1.556335, a 2,
  n 0.462495 and nu_0 / nu_oo = 8.45 at nu_oo = 1e-2 (the viscosity varies by
  a factor of about 1.7, and neglecting it increases the error by 100x), all 4
  schemes, explicit convective term, explicit and implicit viscosity.
  temporal: degree 8, refinement 3, T = 1, dt = 1/512 ... 1/4096, BDF1/2/3.
  spatial: T = 0.1, dt = 1e-5, BDF2, degree 2, 3, 4, refinements 0..4.
  - consistent_splitting: rates 1, 2, 3 for velocity and pressure, BDF1/2
    errors equal to coupled to 3 digits; full spatial rates as coupled and
    pressure_correction.
  - dual_splitting: rate 1 for BDF1/2/3, spatially limited by it.
  - pressure_correction: velocity rates 1, 2, 3, pressure 0.74 and 1.75; BDF3
    with explicit viscosity unstable at dt = 1/4096.
  - Without the variable viscosity terms in its pressure Poisson equation,
    consistent_splitting is first order (BDF2, velocity 6.0e-5 ... 7.5e-6,
    pressure rate 0.77) and spatially limited (k = 3: velocity 3.5e-7).

consistent_splitting/temporal_moving_mesh/
  Consistent splitting on a moving mesh with and without the Leray projection,
  Newtonian (nu = 1e-2), mesh motion of the app (sine in space and time,
  amplitude 1/15, period 0.4; the boundary moves as well), explicit convective
  term, degree 8, refinement 3, T = 0.4, dt = 1/160 ... 1/5120, BDF1/2/3,
  2 MPI processes per run.
  - consistent_splitting (with the Leray projection, the default): rates 1, 2,
    3 for velocity and pressure, errors equal to coupled to 3 digits.
  - consistent_splitting_no_leray (Eulerian form of the convective term in the
    pressure Poisson equation): converges, but with a reduced temporal order
    (velocity / pressure: BDF1 about 0.9 / 0.8; BDF2 2 decreasing to 1.45 /
    0.91; BDF3 about 2 decreasing to 1.61 / 0.91). At the smallest dt, the
    errors are 1.4 / 2.6 (BDF1), 1.7 / 3.9 (BDF2) and 100 / 200 (BDF3) times
    those with the Leray projection.
  - The ALE form without the Leray projection (convective term with u - w and
    no volume term of the time derivative) does not converge (velocity 7e-3,
    pressure 1e-1 for all dt); it is not implemented.
