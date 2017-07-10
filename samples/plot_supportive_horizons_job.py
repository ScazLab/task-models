#!/usr/bin/env python

import io
import os
import sys
import time
import json
import logging
import argparse

import numpy as np

from task_models.utils.multiprocess import repeat, get_process_elapsed_time
from task_models.lib.pomcp import POMCPPolicyRunner, NTransitionsHorizon
from task_models.task import SequentialCombination, LeafCombination
from task_models.supportive import (SupportivePOMDP, AssembleLeg, AssembleLegToTop,
                                    NHTMHorizon)


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
    'iterations': 10,         # iterations for the policy (in get_action)
    'exploration': 50,
    'relative_explo': False,  # In this case use smaller exploration
    'belief_values': False,
    'n_particles': 150,
    'horizon-type': 'transitions',  # or htm
    'horizon-length': 3,
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
    PARAM['n_particles'] = 50


class FinishedOrNTransitionsHorizon(NTransitionsHorizon):

    def __init__(self, model, n):
        super(FinishedOrNTransitionsHorizon, self).__init__(n)
        self.model = model
        self._is_final = False

    def is_reached(self):
        return self.n <= 0 or self._is_final

    def decrement(self, a, s, new_s, o):
        super(FinishedOrNTransitionsHorizon, self).decrement(a, s, new_s, o)
        self._is_final = self.model._int_to_state(new_s).is_final()

    def copy(self):
        return FinishedOrNTransitionsHorizon(self.model, self.n)

    @classmethod
    def generator(cls, model, n=100):
        return cls._Generator(cls, model, n)


class CountingSupportivePOMDP(SupportivePOMDP):
    """Adds a counter of calls to sample_transition"""

    def __init__(self, *args, **kwargs):
        super(CountingSupportivePOMDP, self).__init__(*args, **kwargs)
        self._calls = 0

    def sample_transition(self, a, s, random=True):
        self._calls += 1
        return super(CountingSupportivePOMDP, self).sample_transition(
            a, s, random=random)


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


def episode_summary(model, full_return, h_s, h_a, h_o, h_r, n_calls,
                    elapsed=None):
    indent = 4 * " "
    return ("Evaluation: {} transitions, return: {:4.0f} [{} calls in {}]\n"
            "".format(len(h_a), full_return, n_calls, elapsed) +
            "".join(["{ind}{}: {} → {} [{}]\n".format(model._int_to_state(s),
                                                      model.actions[a],
                                                      model.observations[o],
                                                      r, ind=indent)
                     for s, a, o, r in zip(h_s, h_a, h_o, h_r)]) +
            "{ind}{}".format(model._int_to_state(h_s[-1]), ind=indent))


def simulate_one_evaluation(model, pol, max_horizon=50, logger=None):
    init_calls = model._calls
    pol.reset()
    # History init
    h_s = [model.sample_start()]
    h_a = []
    h_o = []
    h_r = []
    horizon = FinishedOrNTransitionsHorizon(model, max_horizon)
    full_return = 0
    while not horizon.is_reached():
        a = model.actions.index(pol.get_action())
        h_a.append(a)
        s, o, r = model.sample_transition(a, h_s[-1])  # real transition
        h_o.append(o)
        h_r.append(r)
        horizon.decrement(a, h_s[-1], s, o)
        pol.step(model.observations[o])
        h_s.append(s)
        full_return = r + model.discount * full_return
    elapsed = get_process_elapsed_time()
    n_calls = model._calls - init_calls
    if logger is not None:
        logger(episode_summary(model, full_return, h_s, h_a, h_o, h_r, n_calls,
                               elapsed=elapsed))
    return {'return': full_return,
            'states': h_s,
            'actions': h_a,
            'observations': h_o,
            'rewards': h_r,
            'elapsed-time': elapsed.total_seconds(),
            'simulator-calls': n_calls,
            }


def evaluate(model, pol, n_evaluation, logger=None):
    def func():
        return simulate_one_evaluation(model, pol, logger=logger)

    return repeat(func, n_evaluation)


# Problem definition
leg_i = 'leg-{}'.format
htm = SequentialCombination([
    SequentialCombination([
        LeafCombination(AssembleLeg(leg_i(i))),
        LeafCombination(AssembleLegToTop(leg_i(i), bring_top=(i == 0)))])
    for i in range(4)])

p = CountingSupportivePOMDP(htm)
# TODO put as default
p.r_subtask = 0.
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
info('Warmup done in: {}s'.format(tostore['t_warmup']))
# Evaluation
tostore['evaluations'] = evaluate(p, pol, PARAM['n_evaluations'], logger=info)

# Storing current status
if args.path is not None:
    assert(args.name is not None)
    with io.open(os.path.join(args.path, args.name + '.json'), 'w') as f:
        json.dump(tostore, f, cls=NPEncoder)
