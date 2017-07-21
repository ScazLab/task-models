import sys
import time
import logging
import argparse

import numpy as np

from task_models.lib.pomcp import POMCPPolicyRunner, export_pomcp
from task_models.task import (LeafCombination, AlternativeCombination,
                              SequentialCombination)
from task_models.supportive import (SupportivePOMDP, AssembleLeg, SupportedAction,
                                    CONSUMES)
from task_models.policy import PolicyRunner, PolicyLongSupportiveSequence
from task_models.evaluation import evaluate


parser = argparse.ArgumentParser(
    description="Script to run evaluation job on the multitask problem")
parser.add_argument('--debug', action='store_true',
                    help='quick interactive run with dummy parameters')
args = parser.parse_args(sys.argv[1:])


def task_long_sequence(n):
    leg_i = 'leg-{}'.format
    htm = SequentialCombination([LeafCombination(AssembleLeg(leg_i(i)))
                                 for i in range(n)])
    return htm


class CustomObjectAction(SupportedAction):

    hold = None

    def __init__(self, obj):
        super(CustomObjectAction, self).__init__('get-{}'.format(obj))
        self.conditions = [(CONSUMES, obj)]


def task_alternative(n):
    return AlternativeCombination(
        [LeafCombination(CustomObjectAction('{:0>{}}'.format(i, len(str(n)))))
         for i in range(n)])


class PolicySupportiveAlternatives(PolicyRunner):

    def __init__(self, model):
        super(PolicySupportiveAlternatives, self).__init__(model)
        self._a_next = self.actions.index('hold H')
        self._a_wait = self.actions.index('wait')
        self._o_none = self.observations.index('none')
        self.n_tasks = self.model.n_htm_states - 2
        self.tools = ['screws', 'screwdriver', 'joints']
        self.reset()


def average_return(evaluations):
    return np.average([x['return'] for x in evaluations])


PARAM = {
    # Algorithm parameters
    'n_warmup': 1000,         # initial warmup exploration
    'n_evaluations': 100,     # number of evaluations
    'iterations': 100,        # iterations for the policy (in get_action)
    'exploration': 50,
    'relative_explo': False,  # In this case use smaller exploration
    'belief_values': False,
    'n_particles': 150,
    'horizon-length': 50,
    'intermediate-rewards': False,
    'n_subtasks': 20,
}

if args.debug:
    PARAM['n_warmup'] = 2
    PARAM['n_evaluations'] = 2
    PARAM['iterations'] = 3
    PARAM['n_particles'] = 20
    PARAM['horizon-length'] = 2


p = SupportivePOMDP(task_long_sequence(PARAM['n_subtasks']), discount=.95)
p.p_preferences = [1.]
p.p_change_preference = 0.
# p.r_final = 10  # TODO move to parameter

logging.basicConfig(level=logging.INFO)
info = logging.info
warn = logging.warning
tostore = {'parameters': PARAM}
info('Starting warmup')
# POMCP policy
pol = POMCPPolicyRunner(p, iterations=PARAM['iterations'],
                        horizon=PARAM['horizon-length'],
                        exploration=PARAM['exploration'],
                        relative_exploration=PARAM['relative_explo'],
                        belief_values=PARAM['belief_values'],
                        belief='particle',
                        belief_params={'n_particles': PARAM['n_particles']})
# Some initial exploration
t_0 = time.time()
pol.get_action(iterations=PARAM['n_warmup'])
tostore['t_warmup'] = time.time() - t_0
info('Warmup done in {}s.'.format(tostore['t_warmup']))
tostore['evaluations-pomcp'] = evaluate(p, pol, PARAM['n_evaluations'],
                                        logger=info)
warn('POMCP average return: {}'.format(
    average_return(tostore['evaluations-pomcp'])))
# Ad-hoc policy
pol = PolicyLongSupportiveSequence(p)
tostore['evaluations-adhoc'] = evaluate(p, pol, PARAM['n_evaluations'],
                                        logger=info)
warn('Ad-hoc average return: {}'.format(
    average_return(tostore['evaluations-adhoc'])))
