#!/usr/bin/env python3
"""
Prints the tables of a summary_*_convergence.txt from the logs of a study
folder.

Usage: python3 summarize.py <folder with input_*.json and log_*.txt>

The runs of a study differ only in their level: the time step size (`_dt<i>`,
temporal) or the mesh refinement (`_r<i>`, spatial, grouped by the degree of
the structure `_k<k>`). For the fluid velocity, the fluid pressure and the
structure displacement, the error of a run is the maximum over its error
outputs (every ErrorInterval up to EndTime) of the absolute L2 error. The
errors on the first and last level and the rates (log2) of all levels are
given; FSI it is the number of partitioned iterations per time step averaged
over all runs, time is the sum of the wall times of all runs of a study.
"""

import glob
import json
import math
import os
import re
import sys

SCHEMES = ['coupled', 'dual_splitting', 'pressure_correction', 'consistent_splitting']
UPDATE_METHODS = ['implicit', 'implicit_pressure_structure', 'geometric_explicit']
TRACTION_TRANSFERS = ['point_evaluation', 'residual']
FIELDS = [('fluid velocity', 'u'), ('fluid pressure', 'p'),
          ('structure displacement', 'd')]

NAME = re.compile(r'(temporal|spatial)_(%s)(?:_(%s))?(?:_(%s))?_(?:dt(\d+)|k(\d+)_r(\d+))$' %
                  ('|'.join(SCHEMES), '|'.join(UPDATE_METHODS),
                   '|'.join(TRACTION_TRANSFERS)))

ERROR = re.compile(r'Calculate error for (.+?) at time t = (\S+):\n'
                   r'\s+Absolute error \(L2-norm\): (\S+)')


def read_log(path):
    log = open(path).read()

    errors = {}
    for field, time, error in ERROR.findall(log):
        errors[field] = max(errors.get(field, 0.0), float(error))
    if not errors:
        return None, None, None

    iterations = re.search(r'Partitioned iterations\s+(\S+)', log)
    wall_time = re.search(r'Computational costs:\n(?:.*\n){0,3}?\s+Wall time\s+=\s+(\S+)',
                          log)
    return (errors, float(iterations.group(1)) if iterations else float('nan'),
            float(wall_time.group(1)) if wall_time else 0.0)


def level_label(path, study):
    with open(path) as f:
        prm = json.load(f)
    if study == 'spatial':
        return 'r=' + prm['SpatialResolutionFluid']['RefineSpace']
    return 'dt=%g' % float(prm['Application']['TimeStepSize'])


def rates(e):
    return ' '.join(
        '%5.2f' % math.log2(e[i - 1] / e[i]) if e[i] > 0 and e[i - 1] > 0 else '    -'
        for i in range(1, len(e)))


def main(folder):
    # runs[(group, level)] = (label, errors, iterations, wall time)
    runs = {}
    for log in sorted(glob.glob(os.path.join(folder, 'log_*.txt'))):
        name = os.path.basename(log)[len('log_'):-len('.txt')]
        m = NAME.match(name)
        if m is None:
            continue
        study, scheme, update, traction, dt_level, degree, r_level = m.groups()
        level = int(dt_level if dt_level is not None else r_level)
        group = (study, int(degree) if degree else 0, scheme, update, traction)
        label = level_label(os.path.join(folder, 'input_%s.json' % name), study)
        runs[(group, level)] = (label, *read_log(log))

    def order(group):
        study, degree, scheme, update, traction = group
        return (degree, SCHEMES.index(scheme),
                UPDATE_METHODS.index(update) if update is not None else 0,
                TRACTION_TRANSFERS.index(traction) if traction is not None else 0)

    groups = sorted({group for group, level in runs}, key=order)
    header = 'scheme' + (' / update method' if any(g[3] for g in groups) else '') + \
             (' / traction transfer' if any(g[4] for g in groups) else '')
    width = max(len(header), max(len(' / '.join(x for x in g[2:] if x)) for g in groups))

    current = None
    for group in groups:
        study, degree, scheme, update, traction = group
        levels = sorted(level for g, level in runs if g == group)
        first, last = runs[(group, levels[0])][0], runs[(group, levels[-1])][0]
        if (study, degree) != current:
            current = (study, degree)
            if study == 'spatial':
                print('\n==== structure degree k = %d (expected rate of the displacement %d)' %
                      (degree, degree + 1))
            else:
                print('\n==== BDF2 and generalized-alpha (expected rate 2)')
            print('%-*s %10s %10s   rates' % (width + 3, header, first, last))

        label = ' / '.join(x for x in (scheme, update, traction) if x is not None)
        results = [runs[(group, level)][1:] for level in levels]
        if any(r[0] is None for r in results):
            print('%-*s  no errors (run failed?)' % (width, label))
            continue
        iterations = sum(r[1] for r in results) / len(results)
        wall_time = sum(r[2] for r in results)
        for i, (field, short) in enumerate(FIELDS):
            e = [r[0][field] for r in results]
            tail = '  FSI it %.1f  %5.0fs' % (iterations, wall_time) if i == 0 else ''
            print('%-*s %s  %.2e  %.2e  %s%s' % (width, label if i == 0 else '', short,
                                                 e[0], e[-1], rates(e), tail))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
