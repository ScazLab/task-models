import itertools

import numpy as np

from task_models.task import (LeafCombination, AlternativeCombination,
                              SequentialCombination)
from task_models.supportive import (SupportivePOMDP, AssembleLeg, SupportedAction,
                                    CONSUMES)
from task_models.policy import PolicyRunner, PolicyLongSupportiveSequence
from task_models.evaluation import SupportiveExperiment


def task_long_sequence(n):
    leg_i = 'leg-{}'.format
    htm = SequentialCombination([LeafCombination(AssembleLeg(leg_i(i)))
                                 for i in range(n)])
    return htm


class CustomHoldLegAssembly(AssembleLeg):

    def __init__(self, obj, hold=None):
        super(CustomHoldLegAssembly, self).__init__(str(obj))
        self.hold = hold


def task_uniform(n):
    def task_seq(support_seq):
        return SequentialCombination([
            LeafCombination(CustomHoldLegAssembly('leg', hold=s))
            for i, s in enumerate(support_seq)])
    return AlternativeCombination([task_seq(''.join(s))
                                   for s in itertools.product('hv', repeat=n)])


def task_alternative():
    def task_seq(support_seq):
        return SequentialCombination([
            LeafCombination(CustomHoldLegAssembly('leg', hold=s))
            for i, s in enumerate(support_seq)])
    return AlternativeCombination([task_seq(s)
                                   for s in ['hvvv', 'hhhh', 'vhhv', 'vvhv']])


class PolicySupportiveAlternatives(PolicyLongSupportiveSequence):

    def __init__(self, model, sequence_length):
        super(PolicySupportiveAlternatives, self).__init__(model)
        self.n_tasks = sequence_length
        self._a_next_list = [self.actions.index('hold ' + h)
                             for h in ('H', 'V')]


class Experiment(SupportiveExperiment):

    # TODO: Is there a nicer way to do that?
    DEFAULT_PARAMETERS = SupportiveExperiment.DEFAULT_PARAMETERS.copy()
    DEFAULT_PARAMETERS.update({
        'n_subtasks': 20,  # only for sequence task
        'task': 'alternative',
        'policy': 'random',
        'n_evaluations': 100,
    })

    def init_run(self):
        task_length = 4
        if self.parameters['task'] == 'sequence':
            task_length = self.parameters['n_subtasks']
            htm = task_long_sequence(task_length)
        elif self.parameters['task'] == 'uniform':
            htm = task_uniform(task_length)
        elif self.parameters['task'] == 'alternative':
            htm = task_alternative()
        else:
            raise ValueError('Unknown task: ' + self.parameters['task'])
        self.model = SupportivePOMDP(htm)
        # discount=.95  # TODO keep?
        self.model.p_preferences = [1.]
        self.model.p_change_preference = 0.
        self.model.r_final = 10  # TODO move to parameter?
        if self.parameters['task'] != 'sequence':
            # Makes wrong action more costly to differentiate algorithms
            # (The task is used in an abstract way here so this is not really
            # about holding and preferences.)
            self.model.cost_hold = 10
            self.model.r_preference = 18
        if self.parameters['policy'] == 'pomcp':
            self.init_pomcp_policy()
        elif self.parameters['policy'] == 'repeat':
            self.policy = PolicyLongSupportiveSequence(self.model)
        elif self.parameters['policy'] == 'random':
            self.policy = PolicySupportiveAlternatives(self.model, task_length)
        else:
            raise NotImplementedError

    def finish_run(self):
        self.log('Average return: {}'.format(
            np.average([x['return'] for x in self.results['evaluations']])))


Experiment.run_from_arguments()
