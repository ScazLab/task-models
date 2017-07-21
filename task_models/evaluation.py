import io
import os
import sys
import time
import json
import logging
import argparse

from task_models.lib.utils import NPEncoder

from task_models.utils.multiprocess import repeat, get_process_elapsed_time
from task_models.lib.pomcp import NTransitionsHorizon, POMCPPolicyRunner
from task_models.supportive import NHTMHorizon


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


def transition_summary(model, s, a, o, r, indent=""):
    return "{ind}{}: {} → {} [{}]\n".format(model._int_to_state(s),
                                            model.actions[a],
                                            model.observations[o],
                                            r, ind=indent)


def episode_summary(model, full_return, h_s, h_a, h_o, h_r, n_calls,
                    elapsed=None):
    indent = 4 * " "
    return ("Evaluation: {} transitions, return: {:4.0f} [{:,} calls in {}]\n"
            "".format(len(h_a), full_return, n_calls, elapsed) +
            "".join([transition_summary(model, s, a, o, r, indent=indent)
                     for s, a, o, r in zip(h_s, h_a, h_o, h_r)]) +
            "{ind}{}".format(model._int_to_state(h_s[-1]), ind=indent))


def simulate_one_evaluation(model, pol, max_horizon=200, logger=None,
                            discount=True):
    if discount:
        gamma = model.discount
    else:
        gamma = 1.
    init_calls = model.n_simulator_calls
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
        full_return = r + gamma * full_return
    elapsed = get_process_elapsed_time()
    n_calls = model.n_simulator_calls - init_calls
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


class SupportiveExperiment(object):

    """Evaluation of one policy.
    """

    DEFAULT_PARAMETERS = {
        # Algorithm parameters
        'n_warmup': 2000,           # initial warmup exploration
        'n_evaluations': 20,        # number of evaluations
        'iterations': 100,          # iterations for the policy (in get_action)
        'rollout-iterations': 100,  # iterations for rollouts
        'exploration': 50,
        'relative-explo': False,    # In this case use smaller exploration
        'belief-values': False,
        'n_particles': 150,
        'horizon-type': 'transitions',  # or htm
        'horizon-length': 20,
        'intermediate-rewards': False,
        'p_preference': 0.3,
        'policy': 'pomcp',
    }

    def __init__(self, parameters=None):
        self.parameters = self.DEFAULT_PARAMETERS.copy()
        if parameters:
            self.parameters.update(parameters)
        logging.basicConfig(level=logging.INFO)
        self.log = logging.info
        self.results = {}
        self.path = None

    def run(self, debug=False):
        if debug:
            self.log('DEBUG mode is on')
            self.parameters['n_warmup'] = 2
            self.parameters['n_evaluations'] = 2
            self.parameters['iterations'] = 10
            self.parameters['n_particles'] = 20
            self.parameters['horizon-length'] = 2
        self.log_parameters()
        self.results['parameters'] = self.parameters
        self.init_run()
        self.log('Starting warmup')
        # Some initial exploration
        t_0 = time.time()
        self.policy.get_action(iterations=self.parameters['n_warmup'])
        self.results['t_warmup'] = time.time() - t_0
        self.log('Warmup done in {}s.'.format(self.results['t_warmup']))
        # Evaluation
        self.results['evaluations'] = evaluate(
            self.model, self.policy, self.parameters['n_evaluations'],
            logger=self.log)
        # Finishing
        self.finish_run()
        if self.path is not None:
            self.write_result(self.path)

    def log_parameters(self):
        self.log('Parameters:\n' + '\n'.join([
            '  - {}: {}'.format(k, self.parameters[k])
            for k in self.parameters]))

    def init_run(self):
        """Needs to set the following attributes:
        - model
        - policy
        """
        raise NotImplementedError

    def finish_run(self):
        pass

    def init_pomcp_policy(self):
        self.policy = POMCPPolicyRunner(
            self.model, iterations=self.parameters['iterations'],
            horizon=(NHTMHorizon if self.parameters['horizon-type'] == 'htm'
                     else FinishedOrNTransitionsHorizon
                     ).generator(self.model, n=self.parameters['horizon-length']),
            exploration=self.parameters['exploration'],
            relative_exploration=self.parameters['relative-explo'],
            belief_values=self.parameters['belief-values'],
            belief='particle',
            belief_params={'n_particles': self.parameters['n_particles']})

    def write_result(self, path):
        with io.open(path, 'w') as f:
            json.dump(self.results, f, cls=NPEncoder)

    @classmethod
    def load_from_serialized(cls, config_path):
        with open(config_path, 'r') as config_file:
            param = json.load(config_file)
        return cls(parameters=param)

    @classmethod
    def run_from_arguments(cls):
        # Parser definition
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
        # Parse arguments
        args = parser.parse_args(sys.argv[1:])
        if args.config is not None:
            exp = cls.load_from_serialized(args.config)
        else:
            exp = cls()
        if args.path is not None:
            exp.path = os.path.join(args.path, args.name + '.json')
        exp.run()
