#!/usr/bin/env python3
"""
Writes the input files input_<name>.json of a study folder from its study.json.

Usage: python3 generate_inputs.py <study folder>

study.json contains the input file shared by all runs ("base") and the varied
values: "schemes", "update_methods" (optional), "traction_transfers" (optional)
and either "time_step_refinements" (temporal study: the TimeStepSize of the
base divided by 2^i) or "degrees_structure" and "refinements" (spatial study:
the degree of the structure and RefineSpace of fluid and structure). The piston
runs one time step size and one mesh per run, so one input file is written for
each combination and each level, named as expected by summarize.py:
  <prefix>_<scheme>[_<update method>][_<traction transfer>]_<level>
with the level dt<i> (temporal) or k<degree>_r<i> (spatial). The scheme names are those of TemporalDiscretization (coupled =
BDFCoupledSolution, ...), the other names those of UpdateMethod and
TractionTransfer. The coupled solver updates velocity and pressure together, so
it is skipped with implicit_pressure_structure.
"""

import copy
import itertools
import json
import os
import sys

SCHEMES = {
    'coupled': 'BDFCoupledSolution',
    'dual_splitting': 'BDFDualSplitting',
    'pressure_correction': 'BDFPressureCorrection',
    'consistent_splitting': 'BDFConsistentSplitting'
}
UPDATE_METHODS = {
    'implicit': 'Implicit',
    'implicit_pressure_structure': 'ImplicitPressureStructure',
    'geometric_explicit': 'GeometricExplicit'
}
TRACTION_TRANSFERS = {
    'point_evaluation': 'PointEvaluation',
    'residual': 'Residual'
}


def main(folder):
    with open(os.path.join(folder, 'study.json')) as f:
        study = json.load(f)

    spatial = 'refinements' in study
    update_methods = study.get('update_methods', [None])
    traction_transfers = study.get('traction_transfers', [None])
    if spatial:
        levels = list(itertools.product(study['degrees_structure'],
                                        study['refinements']))
    else:
        levels = [(None, i) for i in study['time_step_refinements']]

    for scheme, update, traction, (degree, level) in itertools.product(
            study['schemes'], update_methods, traction_transfers, levels):
        if scheme == 'coupled' and update == 'implicit_pressure_structure':
            continue

        name = study['prefix'] + '_' + scheme
        if update is not None:
            name += '_' + update
        if traction is not None:
            name += '_' + traction
        name += '_k%d_r%d' % (degree, level) if spatial else '_dt%d' % level

        prm = copy.deepcopy(study['base'])
        prm['Fluid']['TemporalDiscretization'] = SCHEMES[scheme]
        if update is not None:
            prm['FSI']['UpdateMethod'] = UPDATE_METHODS[update]
        if traction is not None:
            prm['FSI']['TractionTransfer'] = TRACTION_TRANSFERS[traction]
        if spatial:
            prm['SpatialResolutionStructure']['Degree'] = str(degree)
            prm['SpatialResolutionFluid']['RefineSpace'] = str(level)
            prm['SpatialResolutionStructure']['RefineSpace'] = str(level)
        else:
            dt = float(prm['Application']['TimeStepSize']) / 2**level
            prm['Application']['TimeStepSize'] = repr(dt)
        prm['Output']['OutputDirectory'] = 'output_%s/' % name

        with open(os.path.join(folder, 'input_%s.json' % name), 'w') as f:
            json.dump(prm, f, indent=4)
            f.write('\n')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
