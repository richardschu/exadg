Convergence studies of navier_stokes_manufactured (generalized Newtonian flow,
Carreau-Yasuda), based on the paper's settings (arXiv 2506.14424, ExaDG 8d3db6fbe):
domain (0,1)^2, pure Dirichlet, divergence formulation, nu_oo = 5e-2, margin 5e-2,
kappa 1, lambda 1, a 2, n 0.5, L2 (DG) discretization, serial runs.

Each study is one input file; the solver writes `log_<name>.txt` (with the
convergence table at the end) and `output_<name>/` (error files `run_<id>_*`, one
per refinement level). Logs and output folders are not stored, rerun all
studies of a folder with its run script, e.g.
  cd spatial && ./run_spatial_studies.sh <path to solver> 8 1
(8 studies at a time, 1 MPI process each, the defaults). The script deletes the
output folder of a study before running it; when running a study by hand, its
output folder must not exist, since the convergence table is computed from all
error files in it. The tables of the summaries are printed from the logs by
  python3 summarize.py <folder>

All combinations are covered: scheme (coupled, dual_splitting,
pressure_correction) x convective term (explicit, linearly_implicit, implicit)
x viscosity (explicit, implicit).

spatial/  (summary_spatial_convergence.txt)
  Navier-Stokes, T = 0.1, dt = 1e-5, BDF2, degree 2, 3, 4, refinements 0..4.
  - coupled and pressure_correction: full rates (velocity k+1, pressure k) for
    all treatments, identical errors to 2-3 digits.
  - dual_splitting: levels off at its temporal (splitting) error, velocity about
    6e-8, as in the paper.

temporal/high_viscosity/ (summary_temporal_convergence.txt)
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

temporal/low_viscosity/ (summary_temporal_convergence.txt)
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

