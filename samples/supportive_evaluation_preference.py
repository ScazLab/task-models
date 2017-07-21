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
from matplotlib import pyplot as plt

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


args = parser.parse_args(sys.argv[1:])

SCRIPT = os.path.join(os.path.dirname(__file__),
                      'supportive_evaluation_preference_job.py')

p_preference = list(np.arange(0, 1.001, .05))
policies = ['pomcp', 'always-hold', 'never-hold']
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


def plot_results():
    # Load results
    results = {j: get_results_from_one(jobs[j]) for j in jobs}
    # Plot returns for preferences
    plt.figure()
    cplot = plt.gca()
    cplot.set_ylabel('Average return')
    cplot.set_xlabel('p_preference')
    for pol in policies:
        returns_iterations = [results['preferences-{}-{}'.format(p, pol)][0]
                              for p in p_preference]
        plot_var(returns_iterations, x=p_preference, label=pol, var_style='bar',
                 capsize=3)
        # plt.scatter([p_preference] * 100, returns_iterations)
    plt.legend()
    plt.title('Average returns for preference probability')


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
    plot_results()
    plt.show()
