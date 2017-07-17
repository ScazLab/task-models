import time
import logging

from task_models.lib.pomcp import POMCPPolicyRunner, export_pomcp
from task_models.task import (LeafCombination, AlternativeCombination,
                              SequentialCombination)
from task_models.supportive import (SupportivePOMDP, AssembleLeg, SupportedAction,
                                    CONSUMES)
from task_models.evaluation import simulate_one_evaluation


def task_long_sequence(n):
    leg_i = 'leg-{}'.format
    htm = SequentialCombination([LeafCombination(AssembleLeg(leg_i(i)))
                                 for i in range(n)])
    return htm


# TODO: eventually merge with POMCPPolicyRunner

class PolicyRunner(object):

    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, o):
        raise Exception(o)

    @property
    def actions(self):
        return self.model.actions

    @property
    def observations(self):
        return self.model.observations

    def reset(self):
        self.history = []

    def get_action(self):
        """Needs to store action index as self._last_action."""
        raise NotImplementedError

    def step(self, observation):
        if self._last_action is None:
            raise ValueError('Unknown last action')
        o = self.observations.index(observation)
        self.history.extend([self._last_action, o])


class PolicyLongSupportiveSequence(PolicyRunner):

    class UnexpectedObservation(RuntimeError):
        pass

    def __init__(self, model):
        super(PolicyLongSupportiveSequence, self).__init__(model)
        self._a_next = self.actions.index('hold H')
        self._a_wait = self.actions.index('wait')
        self._o_none = self.observations.index('none')
        self.n_tasks = self.model.n_htm_states - 2
        self.tools = ['screws', 'screwdriver', 'joints']
        self.reset()

    def _a_bring(self, obj):
        return self.actions.index('bring {}'.format(obj))

    def _a_clean(self, obj):
        return self.actions.index('clear {}'.format(obj))

    def reset(self):
        super(PolicyLongSupportiveSequence, self).reset()
        self._to_bring = [self._a_bring(o) for o in self.tools]
        self._to_clean = [self._a_clean(o) for o in self.tools]
        self._needs_leg = True
        self._last_action = None
        self._n_done = 0

    def get_action(self, iterations=None):
        if len(self._to_bring) > 0:
            a = self._to_bring[0]
        elif self._n_done < self.n_tasks:
            if self._needs_leg:
                a = self._a_bring('leg')
            else:
                a = self._a_next
        elif len(self._to_clean) > 0:
            a = self._to_clean[0]
        else:  # Final wait
            a = self._a_wait
        self._last_action = a
        return self.actions[a]

    def step(self, observation):
        super(PolicyLongSupportiveSequence, self).step(observation)
        unexpected = self.UnexpectedObservation('{} for action {}'.format(
            observation, self.actions[self._last_action]))
        if observation == 'fail':
            pass
        elif observation in ('none', 'not-found'):
            if self._last_action in [self._a_bring(o) for o in self.tools]:
                self._to_bring.remove(self._last_action)
            elif self._last_action in [self._a_clean(o) for o in self.tools]:
                self._to_clean.remove(self._last_action)
            elif self._last_action == self._a_bring('leg'):
                self._needs_leg = False
            elif self._last_action == self._a_next and observation == 'none':
                # Next subtask
                self._n_done += 1
                self._needs_leg = True
            elif self._last_action == self._a_wait:
                pass
            else:
                raise unexpected
        else:
            raise unexpected


class CustomObjectAction(SupportedAction):

    hold = None

    def __init__(self, obj):
        super(CustomObjectAction, self).__init__('get-{}'.format(obj))
        self.conditions = [(CONSUMES, obj)]


def task_alternative(n):
    return AlternativeCombination(
        [LeafCombination(CustomObjectAction('{:0>{}}'.format(i, len(str(n)))))
         for i in range(n)])


PARAM = {
    # Algorithm parameters
    'n_warmup': 1000,         # initial warmup exploration
    'n_evaluations': 100,     # number of evaluations
    'iterations': 100,        # iterations for the policy (in get_action)
    'exploration': 50,
    'relative_explo': False,  # In this case use smaller exploration
    'belief_values': False,
    'n_particles': 150,
    'horizon-length': 20,
    'intermediate-rewards': False,
}


p = SupportivePOMDP(task_long_sequence(20))
p.p_preferences = [1.]
p.p_change_preference = 0.
pol = POMCPPolicyRunner(p, iterations=PARAM['iterations'],
                        horizon=PARAM['horizon-length'],
                        exploration=PARAM['exploration'],
                        relative_exploration=PARAM['relative_explo'],
                        belief_values=PARAM['belief_values'],
                        belief='particle',
                        belief_params={'n_particles': PARAM['n_particles']})
pol = PolicyLongSupportiveSequence(p)

logging.basicConfig(level=logging.INFO)
info = logging.info
info('Starting warmup')
# Some initial exploration
t_0 = time.time()
pol.get_action(iterations=PARAM['n_warmup'])
info('Warmup done in {}s.'.format(time.time() - t_0))
simulate_one_evaluation(p, pol, max_horizon=1000, logger=info)
