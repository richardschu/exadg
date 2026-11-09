#!/usr/bin/env python3
"""
Prints the tables of a summary_*_convergence.txt from the logs of a study folder.

Usage: python3 summarize.py <folder with input_*.json and log_*.txt>

Each study is grouped by its degree (`_k<k>`, spatial) or its BDF order
(`_bdf<order>`, temporal). For velocity and pressure, the relative L2 errors on
the first and last refinement level and the rates (log2) of all refinements are
given; time is the sum of the wall times of all runs of a study.
"""

import glob
import json
import math
import os
import re
import sys

SCHEMES = ['coupled', 'dual_splitting', 'pressure_correction']
TREATMENTS = ['explicit', 'linearly_implicit', 'implicit']

NAME = re.compile(r'(spatial|temporal)_(?:stokes_)?(\w+?)(?:_conv_(\w+?))?_visc_(\w+?)_(k|bdf)(\d+)$')


def read_log(path):
    lines = open(path).read().splitlines()

    # the convergence table is printed after each run, the last one is complete
    starts = [i for i, l in enumerate(lines) if l.startswith('velocity_L2_relative')]
    if not starts:
        return None, None
    fields = lines[starts[-1]].split()
    errors = {f: [] for f in fields}
    for l in lines[starts[-1] + 1:]:
        if not l.strip():
            break
        values = l.split()
        for i, f in enumerate(fields):
            errors[f].append(float(values[2 * i]))

    # each run prints its wall time once in the block `Computational costs:`
    wall_time = 0.0
    for i, l in enumerate(lines):
        if l.startswith('Computational costs:'):
            for m in lines[i + 1:i + 4]:
                if re.match(r'\s+Wall time\s+=', m):
                    wall_time += float(m.split()[3])
    return errors, wall_time


def level_labels(path, study):
    with open(path) as f:
        prm = json.load(f)
    if study == 'spatial':
        r = prm['SpatialResolution']
        return 'r=' + r['RefineSpaceMin'], 'r=' + r['RefineSpaceMax']

    r = prm['TemporalResolution']
    dt = float(prm['Application']['TimeStepSize'])

    def label(refine):
        value = dt / 2**int(refine)
        inverse = 1.0 / value
        if abs(inverse - round(inverse)) < 1e-6 * inverse:
            return 'dt=1' if round(inverse) == 1 else 'dt=1/%d' % round(inverse)
        return 'dt=%g' % value

    return label(r['RefineTimeMin']), label(r['RefineTimeMax'])


def rates(e):
    return ' '.join(
        '%5.2f' % math.log2(e[i - 1] / e[i]) if e[i] > 0 and e[i - 1] > 0 else '    -'
        for i in range(1, len(e)))


def main(folder):
    studies = {}
    for log in sorted(glob.glob(os.path.join(folder, 'log_*.txt'))):
        name = os.path.basename(log)[len('log_'):-len('.txt')]
        m = NAME.match(name)
        if m is None:
            continue
        study, scheme, conv, visc, group, number = m.groups()
        labels = level_labels(os.path.join(folder, 'input_%s.json' % name), study)
        studies[(group, int(number), scheme, conv, visc)] = (study, labels, *read_log(log))

    with_convection = any(key[3] is not None for key in studies)
    header = 'scheme / convective term / viscosity' if with_convection else 'scheme / viscosity'
    width = 52 if with_convection else 32

    def order(key):
        group, number, scheme, conv, visc = key
        return (group, number, SCHEMES.index(scheme),
                TREATMENTS.index(conv) if conv is not None else 0, TREATMENTS.index(visc))

    current = None
    for key in sorted(studies, key=order):
        group, number, scheme, conv, visc = key
        study, (first, last), errors, wall_time = studies[key]
        if (group, number) != current:
            current = (group, number)
            if group == 'k':
                print('\n==== degree k = %d (expected rates: velocity %d, pressure %d)' %
                      (number, number + 1, number))
            else:
                print('\n==== BDF%d (expected rate %d)' % (number, number))
            print('%-*s %9s %9s   rates' % (width + 2, header, first, last))

        label = ' / '.join(x for x in (scheme, conv, visc) if x is not None)
        if errors is None:
            print('%-*s  no convergence table (run failed?)' % (width, label))
            continue
        u = errors['velocity_L2_relative']
        p = errors['pressure_L2_relative']
        print('%-*s u  %.2e  %.2e  %s  %5.0fs' % (width, label, u[0], u[-1], rates(u), wall_time))
        print('%-*s p  %.2e  %.2e  %s' % (width, '', p[0], p[-1], rates(p)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
