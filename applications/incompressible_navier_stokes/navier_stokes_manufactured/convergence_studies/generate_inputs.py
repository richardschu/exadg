#!/usr/bin/env python3
"""
Writes the input files input_<name>.json of a study folder from its study.json.

Usage: python3 generate_inputs.py <study folder>

study.json contains the input file shared by all runs ("base") and the varied
values: "schemes", "convective_term" (optional), "viscosity", and "degrees"
(spatial study) or "bdf_orders" (temporal study). One input file is written for
each combination, named as expected by summarize.py:
  <prefix>_<scheme>[_conv_<convective term>]_visc_<viscosity>_<k<degree>|bdf<order>>
The scheme consistent_splitting_no_leray is BDFConsistentSplitting with
ApplyLerayProjection = false, consistent_splitting_equal_order is
BDFConsistentSplitting with DegreePressure = EqualOrder.
"""

import copy
import itertools
import json
import os
import sys

# parameters of the section "Application" set by each scheme
SCHEMES = {
    'coupled': {'TemporalDiscretization': 'BDFCoupledSolution'},
    'dual_splitting': {'TemporalDiscretization': 'BDFDualSplitting'},
    'consistent_splitting': {'TemporalDiscretization': 'BDFConsistentSplitting'},
    'consistent_splitting_no_leray': {
        'TemporalDiscretization': 'BDFConsistentSplitting',
        'ApplyLerayProjection': 'false'
    },
    'consistent_splitting_equal_order': {
        'TemporalDiscretization': 'BDFConsistentSplitting',
        'DegreePressure': 'EqualOrder'
    },
    'pressure_correction': {'TemporalDiscretization': 'BDFPressureCorrection'}
}
TREATMENTS = {'explicit': 'Explicit', 'linearly_implicit': 'LinearlyImplicit', 'implicit': 'Implicit'}


def main(folder):
    with open(os.path.join(folder, 'study.json')) as f:
        study = json.load(f)

    spatial = 'degrees' in study
    numbers = study['degrees'] if spatial else study['bdf_orders']
    convective_terms = study.get('convective_term', [None])

    for scheme, conv, visc, number in itertools.product(study['schemes'], convective_terms,
                                                        study['viscosity'], numbers):
        name = study['prefix'] + '_' + scheme
        if conv is not None:
            name += '_conv_' + conv
        name += '_visc_' + visc + ('_k%d' % number if spatial else '_bdf%d' % number)

        prm = copy.deepcopy(study['base'])
        application = prm['Application']
        application.update(SCHEMES[scheme])
        if conv is not None:
            application['TreatmentOfConvectiveTerm'] = TREATMENTS[conv]
        application['TreatmentOfVariableViscosity'] = TREATMENTS[visc]
        if spatial:
            prm['SpatialResolution']['DegreeMin'] = str(number)
            prm['SpatialResolution']['DegreeMax'] = str(number)
        else:
            application['OrderTimeIntegrator'] = str(number)
        prm['Output']['OutputDirectory'] = 'output_%s/' % name

        with open(os.path.join(folder, 'input_%s.json' % name), 'w') as f:
            json.dump(prm, f, indent=4)
            f.write('\n')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
