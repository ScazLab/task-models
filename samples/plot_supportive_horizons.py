#!/usr/bin/env python

import io
import os
import sys
import json
import argparse

from expjobs.job import Job
from expjobs.pool import Pool
from expjobs.torque import TorquePool
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
    p.add_argument('-l', '--launcher', default='process',
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

if args.action in ('run', 'status'):
    if args.launcher == 'process':
        pool = MultiprocessPool()
    elif args.launcher == 'torque':
        pool = TorquePool(default_walltime=240)
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


if args.action == 'prepare':
    # Generate config files
    for name in exps:
        with io.open(jobs[name].config, 'w') as fp:
            json.dump(exps[name], fp, indent=2)

elif args.action == 'run':
    pool.run()
    if args.launcher == 'process' or args.watch:
        pool.log_refreshed_stats(RefreshedPrint())
        print()

elif args.action == 'status':
    if args.watch:
        pool.print_refreshed_stats(RefreshedPrint())
        print()
    else:
        print(pool.get_stats())

elif args.action == 'plot':
    raise NotImplementedError
