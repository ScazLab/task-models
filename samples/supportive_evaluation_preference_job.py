#!/usr/bin/env python


from task_models.task import LeafCombination
from task_models.policy import PolicyLongSupportiveSequence
from task_models.supportive import SupportivePOMDP, AssembleLeg
from task_models.evaluation import SupportiveExperiment


class Experiment(SupportiveExperiment):

    def init_run(self):
        # HTM definition: one leg assembly
        self.htm = LeafCombination(AssembleLeg('leg'))
        # POMDP definition
        self.model = SupportivePOMDP(self.htm)
        self.model.p_preferences = [self.parameters['p_preference']]
        self.model.reward_independent_preference = True
        self.model.r_subtask = 10
        self.model.r_preference = 10
        self.model.r_final = 10
        if 'p_change_preference' in self.parameters:
            self.model.p_change_preference = self.parameters['p_change_preference']
        # Policy
        if self.parameters['policy'] == 'pomcp':
            self.init_pomcp_policy()
        elif self.parameters['policy'] == 'always-hold':
            self.policy = PolicyLongSupportiveSequence(self.model)
        elif self.parameters['policy'] == 'never-hold':
            self.policy = PolicyLongSupportiveSequence(self.model)
            self.policy._a_next = self.policy._a_wait
        else:
            raise NotImplementedError


Experiment.run_from_arguments()
