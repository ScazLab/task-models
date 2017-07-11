#!/usr/bin/env python

import io
import os
import sys
import json
import argparse

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
                      'plot_supportive_horizons_job.py')

horizon_types = ['transitions'] * 10 + ['htm'] * 9
horizon_length_transitions = list(range(10, 101, 10))
horizon_length_htm = list(range(1, 10))
horizon_lengths = horizon_length_transitions + horizon_length_htm
exps = [('{}-{}'.format(t, l), {'horizon-type': t, 'horizon-length': l})
        for t, l in zip(horizon_types, horizon_lengths)]

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
    # Plot returns
    plots = plt.subplots(1, 2, sharey=True)[1]
    plots[0].set_ylabel('Average return')
    returns_transititions = [results['transitions-{}'.format(h)][0]
                             for h in horizon_length_transitions]
    returns_htm = [results['htm-{}'.format(h)][0]
                   for h in horizon_length_htm]
    plot_var(returns_transititions, x=horizon_length_transitions, ax=plots[0])
    plots[0].set_title('N Transitions Horizon')
    plots[0].set_xlabel('Number of Transitions')
    plot_var(returns_htm, x=horizon_length_htm, ax=plots[1])
    plots[1].set_title('N HTM Horizon')
    plots[1].set_xlabel('Number of HTM Transitions')
    # Plot simulator calls
    plots = plt.subplots(1, 2, sharey=True)[1]
    plots[0].set_ylabel('Average number of calls to simulator')
    calls_transititions = [results['transitions-{}'.format(h)][2]
                           for h in horizon_length_transitions]
    calls_htm = [results['htm-{}'.format(h)][2]
                 for h in horizon_length_htm]
    plot_var(calls_transititions, x=horizon_length_transitions, ax=plots[0])
    plots[0].set_title('N Transitions Horizon')
    plots[0].set_xlabel('Number of Transitions')
    plot_var(calls_htm, x=horizon_length_htm, ax=plots[1])
    plots[1].set_title('N HTM Horizon')
    plots[1].set_xlabel('Number of HTM Transitions')


if args.action == 'prepare':
    # Generate config files
    for name in exps:
        with io.open(jobs[name].config, 'w') as fp:
            json.dump(exps[name], fp, indent=2)

elif args.action == 'run':
    pool.run()
    if args.launcher == 'process' or args.watch:
        pool.log_refreshed_stats(RefreshedPrint())

elif args.action == 'status':
    if args.watch:
        pool.log_refreshed_stats(RefreshedPrint())
    else:
        print(pool.get_stats())

elif args.action == 'plot':
    plot_results()
    plt.show()
