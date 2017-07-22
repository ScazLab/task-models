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


class Experiment(SupportiveExperiment):

    # TODO: Is there a nicer way to do that?
    DEFAULT_PARAMETERS = SupportiveExperiment.DEFAULT_PARAMETERS.copy()
    DEFAULT_PARAMETERS.update({
        'n_subtasks': 20,
    })

    def init_run(self):
        self.model = SupportivePOMDP(
            task_long_sequence(self.parameters['n_subtasks']),
            # discount=.95,  # TODO keep?
        )
        self.model.p_preferences = [1.]
        self.model.p_change_preference = 0.
        self.model.r_final = 10  # TODO move to parameter?
        if self.parameters['policy'] == 'pomcp':
            self.init_pomcp_policy()
        elif self.parameters['policy'] == 'sequence':
            self.policy = PolicyLongSupportiveSequence(self.model)
        else:
            raise NotImplementedError

    def finish_run(self):
        self.log('Average return: {}'.format(
            np.average([x['return'] for x in self.results['evaluations']])))


Experiment.run_from_arguments()
