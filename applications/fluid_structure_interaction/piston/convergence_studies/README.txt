Convergence studies of the piston (rectangular piston of Serino, Banks,
Henshaw, Schwendeman, JCP 2019, see application.h): linear elastic standing
wave in (0,1) x (-0.5,0), fluid in (0,1) x (d(t),0.5) with the uniform velocity
(0, d'(t)) and a pressure linear in y, interface displacement
d(t) = a sin(omega t). The errors of all three fields are known exactly.

Layout: one folder per study,
  temporal/   time step refinement of all schemes, update methods and traction
              transfers
  spatial/    mesh refinement for the degrees 1, 2, 3 of the structure
each with
  study.json                   input shared by all runs ("base") and the
                               varied values
  summary_*_convergence.txt    the results
and shared by both studies
  generate_inputs.py   writes the inputs input_<name>.json of a study folder
  run_study.sh         generates the inputs and runs all of them
  summarize.py         prints the tables of a summary from the logs
The piston runs one time step size and one mesh per run, so each level of a
study is a run of its own. The scheme names are those of TemporalDiscretization
(coupled = BDFCoupledSolution, ...), the update methods and traction transfers
those of UpdateMethod and TractionTransfer of the FSI parameters.

Rerun a study (8 runs at a time, 1 MPI process each, the defaults):
  ./run_study.sh temporal <path to solver> 8 1
Each run writes `log_<name>.txt` into the study folder (the piston does not
write files, the errors are printed every ErrorInterval). Then print the tables
of the summary by
  python3 summarize.py temporal
The error of a run is the maximum over its error outputs of the absolute L2
error (the fluid velocity vanishes at T = 0.5, so the error at T alone is not
representative).

Both studies use E = 200 instead of the E = 5e3 of input.json: with E = 5e3
the wave number k H = 0.19 makes the standing wave almost linear in Y, such
that the spatial errors of the structure are not representative (the temporal
rates are the same for both). FSI AbsTol is 1e-16 instead of 1e-20: the
relative criterion is relative to |v| dt, which vanishes at the zero crossing
of the velocity at T = 0.5, where one run reached PartitionedMaxIter.


temporal/
  Fluid degree 2 and structure degree 4 on refinement 1 (4x2 cells each),
  T = 0.5, dt = 0.0125 ... 7.8125e-4, BDF2 and generalized-alpha (spectral
  radius 0), Dirichlet-Neumann coupling with IQN-ILS: all schemes (coupled,
  dual_splitting, pressure_correction, consistent_splitting) x update methods
  (implicit, implicit_pressure_structure; the coupled solver updates velocity
  and pressure together) x traction transfers (point_evaluation, residual).
  - All 14 combinations: rate 2 in velocity, pressure and displacement, errors
    equal to 2-3 digits (velocity 1.4e-5 ... 5.6e-8, pressure 2.6e-5 ...
    1.0e-7, displacement 1.4e-6 ... 5.7e-9), 5-6 FSI iterations per step.
  - Smaller time steps reach the floor of the solver tolerances: the pressure
    stagnates at about 5e-8 from dt = 3.9e-4 on, the displacement at about
    1.5e-9 from dt = 2e-4 on.
  - The fixes needed for these rates: the Residual traction transfer needs the
    structure velocity as Dirichlet data of the fluid before the initial
    acceleration; implicit_pressure_structure needs the extrapolated structure
    state as initial guess, the intermediate velocity of the momentum or
    convective step in the pressure step of the later coupling iterations
    (dual_splitting, pressure_correction) and a final update of the fluid
    velocity after the coupling loop.

spatial/
  Refinements 1..4 of fluid and structure (4x2 ... 32x16 cells each;
  refinement 0 fails in the linear solver), structure degree 1, 2, 3, fluid
  degree 2 (the fluid solution is in the discrete space), coupled solver,
  dt = 7.8125e-4, where the temporal errors are u 5.6e-8, p 1.0e-7, d 5.7e-9.
  - Degree 1: rate 2 in all fields (displacement 5.9e-5 ... 9.5e-7).
  - Degree 2: displacement rate 3 (2.4e-6 ... 6.5e-9) until the temporal
    error on refinement 4; velocity and pressure reach their temporal errors
    on refinement 2.
  - Degree 3: all fields reach their temporal errors on refinement 2.
