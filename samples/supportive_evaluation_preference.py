#!/usr/bin/env python


"""Compares behavior over probability of preference for holding.

Compares POMCP with always hold and never hold policies for various values
of the probability for preference.
"""

import io
import os
import json

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from expjobs.job import Job
from expjobs.helpers import Launcher

from task_models.utils.plot import plot_var


SCRIPT = os.path.join(os.path.dirname(__file__),
                      'supportive_evaluation_preference_job.py')
POLICIES = ['never-hold', 'always-hold', 'pomcp']
P_PREFERENCE = list(np.arange(0, 1.001, .05))


class ExpLauncher(Launcher):

    name = 'preferences experiment'
    torque_args = {'default_walltime': 720}

    def init_jobs(self):
        exps = [('preferences-{}-{}'.format(p, pol),
                {'p_preference': p, 'policy': pol, 'n_evaluations': 100})
                for p in P_PREFERENCE for pol in POLICIES]
        self.jobs = {name: Job(self.args.path, name, SCRIPT)
                     for name, exp in exps}
        self.exps = dict(exps)

    def get_results_from_one(self, job):
        with io.open(os.path.join(job.path, job.name + '.json'), 'r') as f:
            results = json.load(f)['evaluations']
        return ([r['return'] for r in results],
                [r['elapsed-time'] for r in results],
                [r['simulator-calls'] for r in results])

    def plot_results(self, var=True):
        # Load results
        results = {j: self.get_results_from_one(self.jobs[j])
                   for j in self.jobs}
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

    def action_prepare(self):
        for name in self.exps:
            with io.open(self.jobs[name].config, 'w') as fp:
                json.dump(self.exps[name], fp, indent=2)

    def action_plot(self):
        destination = self.args.plot_destination
        if destination is not None:
            matplotlib.rcParams.update({
                'font.family': 'serif',
                'font.size': 10,
                'font.serif': 'Computer Modern Roman',
                'text.usetex': 'True',
                'text.latex.unicode': 'True',
                'axes.titlesize': 'large',
                'xtick.labelsize': 'x-small',
                'ytick.labelsize': 'x-small',
                'path.simplify': 'True',
                'savefig.pad_inches': 0.0,
                'savefig.bbox': 'tight',
                'figure.figsize': (3.5, 2.5),
            })
        figure = self.plot_results(var=(destination is None))
        if self.args.plot_destination is None:
            plt.show()
        else:
            figure.savefig(os.path.join(destination, 'preferences.pdf'),
                           transparent=True)


ExpLauncher().run()
