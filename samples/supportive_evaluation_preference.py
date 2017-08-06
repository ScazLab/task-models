#!/usr/bin/env python


"""Compares behavior over probability of preference for holding.

Compares POMCP with always hold and never hold policies for various values
of the probability for preference.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from expjobs.job import Job

from task_models.utils.plot import plot_var
from task_models.evaluation import ExpLauncher


SCRIPT = os.path.join(os.path.dirname(__file__),
                      'supportive_evaluation_preference_job.py')
POLICIES = ['never-hold', 'always-hold', 'pomcp']
P_PREFERENCE = list(np.arange(0, 1.001, .05))


class PreferenceLauncher(ExpLauncher):

    name = 'preferences experiment'

    def init_jobs(self):
        exps = [('preferences-{}-{}'.format(p, pol),
                {'p_preference': p, 'policy': pol, 'n_evaluations': 100})
                for p in P_PREFERENCE for pol in POLICIES]
        self.jobs = {name: Job(self.args.path, name, SCRIPT)
                     for name, exp in exps}
        self.exps = dict(exps)

    def plot_results(self, var=True):
        # Load results
        results = self.get_results()
        # Plot returns for preferences
        figure = plt.figure()
        cplot = plt.gca()
        cplot.set_ylabel('Average Return', labelpad=0)
        # (uses labelpad=0 since '-' sign already pushes label left)
        cplot.set_xlabel('$p_H$')
        colors = ["#44ac66", "#1f82f9", "#bb0c36"]
        lines = []
        for pol, col in zip(POLICIES, colors):
            returns_iterations = [results['preferences-{}-{}'.format(p, pol)][0]
                                  for p in P_PREFERENCE]
            if var:
                params = {'var_style': 'both', 'capsize': 2, 'elinewidth': .5}
            else:
                params = {'var_style': 'none'}
            lines.append(plot_var(returns_iterations, x=P_PREFERENCE,
                                  label=pol, linewidth=2, color=col,
                                  **params))
            # plt.scatter([P_PREFERENCE] * 100, returns_iterations)
        plt.xlim([0., 1.])
        legend_lines = []
        for line in lines:
            legend_lines.append(mpatches.Patch(color=line[0].get_color()))
        plt.legend(legend_lines, POLICIES, loc='lower right', frameon=False)
        # plt.title('Average returns for preference probability')
        return figure

    def action_plot(self):
        destination = self.args.plot_destination
        if destination is not None:
            self.set_matplotlib_params_for_print(
                params={'axes.titlesize': 'large'})
        figure = self.plot_results(var=(destination is None))
        if self.args.plot_destination is None:
            plt.show()
        else:
            figure.savefig(os.path.join(destination, 'preferences.pdf'),
                           transparent=True)


PreferenceLauncher().run()
