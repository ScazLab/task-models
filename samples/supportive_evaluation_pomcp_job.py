#!/usr/bin/env python


from task_models.task import SequentialCombination, LeafCombination
from task_models.supportive import SupportivePOMDP, AssembleLeg, AssembleLegToTop
from task_models.evaluation import SupportiveExperiment


class Experiment(SupportiveExperiment):

    def init_run(self):
        # HTM definition
        leg_i = 'leg-{}'.format
        self.htm = SequentialCombination([
            SequentialCombination([
                LeafCombination(AssembleLeg(leg_i(i))),
                LeafCombination(AssembleLegToTop(leg_i(i), bring_top=(i == 0)))])
            for i in range(4)])
        # POMDP definition
        self.model = SupportivePOMDP(self.htm)
        self.model.p_preferences = [self.parameters['p_preference']]
        if self.parameters['intermediate-rewards']:
            self.model.r_subtask = 10
            self.model.r_final = 30
        else:
            self.model.r_subtask = 0
            self.model.r_final = 120
        if 'p_change_preference' in self.parameters:
            self.model.p_change_preference = self.parameters['p_change_preference']
        # Policy
        self.init_pomcp_policy()


Experiment.run_from_arguments()
