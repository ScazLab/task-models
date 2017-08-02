#!/usr/bin/env python


"""Compares behavior over probability of preference for holding.

Compares POMCP with always hold and never hold policies for various values
of the probability for preference.
"""

import io
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from expjobs.job import Job
from expjobs.pool import Pool
from expjobs.torque import TorquePool, has_qsub
from expjobs.process import MultiprocessPool

from task_models.utils.plot import plot_var


parser = argparse.ArgumentParser(
    description="Script to run and plot evaluation of the supportive policy")
parser.add_argument('path', help='working path')
parser.add_argument('--name', default='eval-horizons', help='experiment name')

subparsers = parser.add_subparsers(dest='action')
prepare = subparsers.add_parser('prepare', help='generate configuration files')
run = subparsers.add_parser('run', help='run the experiments')
status = subparsers.add_parser('status', help='print overall status')
plot = subparsers.add_parser('plot', help='generate figures')

for p in (run, prepare, status):
    p.add_argument('-l', '--launcher', default='torque' if has_qsub() else 'process',
                   choices=['process', 'torque'])
for p in (run, status):
    p.add_argument('-w', '--watch', action='store_true')
plot.add_argument('--plot-destination', default=None)


args = parser.parse_args(sys.argv[1:])

SCRIPT = os.path.join(os.path.dirname(__file__),
                      'supportive_evaluation_preference_job.py')

p_preference = list(np.arange(0, 1.001, .05))
policies = ['never-hold', 'always-hold', 'pomcp']
exps = [('preferences-{}-{}'.format(p, pol),
         {'p_preference': p, 'policy': pol, 'n_evaluations': 100})
        for p in p_preference for pol in policies]

jobs = {name: Job(args.path, name, SCRIPT) for name, exp in exps}
exps = dict(exps)


if args.action in ('prepare', 'run', 'status'):
    if args.launcher == 'process':
        pool = MultiprocessPool()
    elif args.launcher == 'torque':
        pool = TorquePool(default_walltime=720)
else:
    pool = Pool()
pool.extend(jobs.values())


# Helpers for the printing progress

class RefreshedPrint:

    def __init__(self):
        self._to_clean = 0

    def _print(self, s):
        print(s, end='\r')

    def _clean(self):
        self._print(' ' * self._to_clean)

    def __call__(self, s):
        self._clean()
        self._to_clean = len(s)
        self._print(s)


# Plot helpers

def get_results_from_one(job):
    with io.open(os.path.join(job.path, job.name + '.json'), 'r') as f:
        results = json.load(f)['evaluations']
    return ([r['return'] for r in results],
            [r['elapsed-time'] for r in results],
            [r['simulator-calls'] for r in results])


def plot_results(var=True):
    # Load results
    results = {j: get_results_from_one(jobs[j]) for j in jobs}
    # Plot returns for preferences
    figure = plt.figure()
    cplot = plt.gca()
    cplot.set_ylabel('Average Return', labelpad=0)  # '-' sign already pushes label left
    cplot.set_xlabel('$p_H$')
    colors = ["#44ac66", "#1f82f9", "#bb0c36"]
    lines = []
    for pol, col in zip(policies, colors):
        returns_iterations = [results['preferences-{}-{}'.format(p, pol)][0]
                              for p in p_preference]
        if var:
            params = {'var_style': 'both', 'capsize': 2, 'elinewidth': .5}
        else:
            params = {'var_style': 'none'}
        lines.append(plot_var(returns_iterations, x=p_preference, label=pol,
                              linewidth=2, color=col, **params))
        # plt.scatter([p_preference] * 100, returns_iterations)
    plt.xlim([0., 1.])
    legend_lines = []
    for line in lines:
        legend_lines.append(mpatches.Patch(color=line[0].get_color()))
    plt.legend(legend_lines, policies, loc='lower right', frameon=False)
    # plt.title('Average returns for preference probability')
    return figure


if args.action == 'prepare':
    # Generate config files
    for name in exps:
        with io.open(jobs[name].config, 'w') as fp:
            json.dump(exps[name], fp, indent=2)

elif args.action == 'run':
    pool.run()
    if args.launcher == 'process' or args.watch:
        pool.log_refreshed_stats(RefreshedPrint())
    print(pool.get_stats())

elif args.action == 'status':
    if args.watch:
        pool.log_refreshed_stats(RefreshedPrint())
    else:
        print(pool.get_stats())

elif args.action == 'plot':
    if args.plot_destination is not None:
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'font.serif': 'Computer Modern Roman',
            'text.usetex': 'True',
            'text.latex.unicode': 'True',
            'axes.titlesize': 'large',
            'xtick.labelsize': 'x-small',
            'ytick.labelsize': 'x-small',
            'path.simplify': 'True',
            'savefig.pad_inches': 0.0,
            'savefig.bbox': 'tight',
            'figure.figsize': (3.5, 2.5),
        })
    figure = plot_results(var=(args.plot_destination is None))
    if args.plot_destination is None:
        plt.show()
    else:
        figure.savefig(os.path.join(args.plot_destination, 'preferences.pdf'),
                       transparent=True)
