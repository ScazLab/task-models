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
                      'plot_supportive_evaluation_job.py')

horizon_length_transitions = [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100]
horizon_length_htm = list(range(1, 10))
horizon_types = (['transitions'] * len(horizon_length_transitions) +
                 ['htm'] * len(horizon_length_htm))
horizon_lengths = horizon_length_transitions + horizon_length_htm
exps = [('{}-{}-{}'.format(t, l, 's' if s else 'ns'),
         {'horizon-type': t, 'horizon-length': l, 'intermediate-rewards': s})
        for t, l in zip(horizon_types, horizon_lengths)
        for s in (True, False)]  # intermediate rewards (for subtasks)
exps = []
n_iterations = [15, 20, 30] + list(range(50, 1001, 50))
n_rollout_it = [1, 5, 10, 30, 50, 75, 100]
exps.extend([('iterations-{}-{}'.format(i, r),
              {'iterations': i, 'rollout-iterations': r})
             for i in n_iterations for r in n_rollout_it])

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
    # Horizon evaluation
    # Plot returns
    plots = plt.subplots(1, 2, sharey=True)[1]
    plots[0].set_ylabel('Average return')
    returns_transititions = [[results['transitions-{}-{}'.format(h, s)][0]
                              for h in horizon_length_transitions]
                             for s in ('ns', 's')]
    returns_htm = [[results['htm-{}-{}'.format(h, s)][0]
                    for h in horizon_length_htm]
                   for s in ('ns', 's')]
    plot_var(returns_transititions[0], x=horizon_length_transitions,
             ax=plots[0], label='final rewards only')
    plot_var(returns_transititions[1], x=horizon_length_transitions,
             ax=plots[0], label='subtask rewards')
    plots[0].set_title('N Transitions Horizon')
    plots[0].set_xlabel('Number of Transitions')
    plot_var(returns_htm[0], x=horizon_length_htm, ax=plots[1],
             label='final rewards only')
    plot_var(returns_htm[1], x=horizon_length_htm, ax=plots[1],
             label='subtask rewards')
    plots[1].set_title('N HTM Horizon')
    plots[1].set_xlabel('Number of HTM Transitions')
    plots[1].legend()
    plt.title('Average returns for various horizons')
    # Plot simulator calls
    plots = plt.subplots(1, 2, sharey=True)[1]
    plots[0].set_ylabel('Average number of calls to simulator')
    calls_transititions = [[results['transitions-{}-{}'.format(h, s)][2]
                            for h in horizon_length_transitions]
                           for s in ('ns', 's')]
    calls_htm = [[results['htm-{}-{}'.format(h, s)][2]
                  for h in horizon_length_htm]
                 for s in ('ns', 's')]
    plot_var(calls_transititions[0], x=horizon_length_transitions, ax=plots[0],
             label='final rewards only')
    plot_var(calls_transititions[1], x=horizon_length_transitions, ax=plots[0],
             label='subtask rewards')
    plots[0].set_title('N Transitions Horizon')
    plots[0].set_xlabel('Number of Transitions')
    plot_var(calls_htm[0], x=horizon_length_htm, ax=plots[1],
             label='final rewards only')
    plot_var(calls_htm[1], x=horizon_length_htm, ax=plots[1],
             label='subtask rewards')
    plots[1].set_title('N HTM Horizon')
    plots[1].set_xlabel('Number of HTM Transitions')
    plots[1].legend()
    plt.title('Simulator calls for various horizons')
    # N iterations evaluation
    # Plot returns for
    plt.figure()
    cplot = plt.gca()
    cplot.set_ylabel('Average return')
    cplot.set_xlabel('Number of iterations')
    for r in n_rollout_it:
        returns_iterations = [results['iterations-{}-{}'.format(i, r)][0]
                              for i in n_iterations]
        plot_var(returns_iterations, x=n_iterations,
                 label='{} rollout iterations'.format(r))
    plt.legend()
    plt.title('Average returns for various numbers of iterations')


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
