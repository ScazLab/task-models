#!/usr/bin/env python

import io
import os
import sys
import time
import json
import logging
import argparse

import numpy as np

from task_models.lib.pomcp import POMCPPolicyRunner, export_pomcp
from task_models.task import SequentialCombination, LeafCombination
from task_models.supportive import (SupportivePOMDP, AssembleLeg, AssembleLegToTop,
                                    NHTMHorizon)
from task_models.evaluation import FinishedOrNTransitionsHorizon, evaluate


# Arguments
parser = argparse.ArgumentParser(
    description="Script to run evaluation job of the supportive policy")
parser.add_argument('config', nargs='?', default=None,
                    help='json file containing experiment parameters')
parser.add_argument('path', nargs='?', default=None,
                    help='working path')
parser.add_argument('name', nargs='?', default=None,
                    help='experiment name')
parser.add_argument('--debug', action='store_true',
                    help='quick interactive run with dummy parameters')
args = parser.parse_args(sys.argv[1:])


PARAM = {
    # Algorithm parameters
    'n_warmup': 5000,         # initial warmup exploration
    'n_evaluations': 100,     # number of evaluations
    'iterations': 100,        # iterations for the policy (in get_action)
    'exploration': 50,
    'relative_explo': False,  # In this case use smaller exploration
    'belief_values': False,
    'n_particles': 150,
    'horizon-type': 'transitions',  # or htm
    'horizon-length': 20,
    'intermediate-rewards': False,
}

if args.config is not None:
    # Load configuration parameters on top of default values
    with io.open(args.config) as f:
        new_params = json.load(f)
    PARAM.update(new_params)

if args.debug:
    PARAM['n_warmup'] = 2
    PARAM['n_evaluations'] = 2
    PARAM['iterations'] = 3
    PARAM['n_particles'] = 20
    PARAM['horizon-length'] = 2


class NPEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NPEncoder, self).default(obj)


# Problem definition
leg_i = 'leg-{}'.format
htm = SequentialCombination([
    SequentialCombination([
        LeafCombination(AssembleLeg(leg_i(i))),
        LeafCombination(AssembleLegToTop(leg_i(i), bring_top=(i == 0)))])
    for i in range(4)])

p = SupportivePOMDP(htm)
if PARAM['intermediate-rewards']:
    p.r_subtask = 10
    p.r_final = 110
else:
    p.r_subtask = 0
    p.r_final = 200
pol = POMCPPolicyRunner(p, iterations=PARAM['iterations'],
                        horizon=(NHTMHorizon if PARAM['horizon-type'] == 'htm'
                                 else FinishedOrNTransitionsHorizon
                                 ).generator(p, n=PARAM['horizon-length']),
                        exploration=PARAM['exploration'],
                        relative_exploration=PARAM['relative_explo'],
                        belief_values=PARAM['belief_values'],
                        belief='particle',
                        belief_params={'n_particles': PARAM['n_particles']})

# Explore and evaluate
logging.basicConfig(level=logging.INFO)
info = logging.info
tostore = {'parameters': PARAM}
info('Starting warmup')
# Some initial exploration
t_0 = time.time()
pol.get_action(iterations=PARAM['n_warmup'])
tostore['t_warmup'] = time.time() - t_0
info('Warmup done in {}s.'.format(tostore['t_warmup']))
if args.path is not None:
    assert(args.name is not None)
    export_pomcp(pol, os.path.join(args.path, args.name + '.tree.json'),
                 belief_as_quotient=True)
    info('Tree stored in {}s.'.format(time.time() - t_0 - tostore['t_warmup']))
# Evaluation
tostore['evaluations'] = evaluate(p, pol, PARAM['n_evaluations'], logger=info)

# Storing current status
if args.path is not None:
    with io.open(os.path.join(args.path, args.name + '.json'), 'w') as f:
        json.dump(tostore, f, cls=NPEncoder)
