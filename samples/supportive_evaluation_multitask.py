#!/usr/bin/env python


"""Compares behavior on various tasks.
"""

import io
import os
import sys
import json
import argparse

import numpy as np

from expjobs.job import Job
from expjobs.pool import Pool
from expjobs.torque import TorquePool, has_qsub
from expjobs.process import MultiprocessPool


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
                      'supportive_evaluation_multitask_job.py')

p_preference = list(np.arange(0, 1.001, .05))
policies = ['pomcp', 'sequence', 'random']
tasks = ['sequence', 'alternative']
exps = [('{}-{}'.format(task, pol), {'task': task, 'policy': pol})
        for task in tasks for pol in policies]

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
    raise NotImplementedError


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
    raise NotImplementedError
