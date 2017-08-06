#!/usr/bin/env python


"""Compares behavior on various tasks.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from expjobs.job import Job

from task_models.utils.plot import boxplot
from task_models.evaluation import ExpLauncher


SCRIPT = os.path.join(os.path.dirname(__file__),
                      'supportive_evaluation_multitask_job.py')

POLICIES = ['pomcp', 'repeat', 'random']
TASKS = ['sequence', 'uniform',  'alternative']


class MultitaskLauncher(ExpLauncher):

    name = 'multitask experiment'
    torque_args = {'default_walltime': 360}

    def init_jobs(self):
        self.exps = [('{}-{}'.format(task, pol),
                      {'task': task, 'policy': pol})
                     for task in TASKS for pol in POLICIES]
        self.jobs = {name: Job(self.args.path, name, SCRIPT)
                     for name, exp in self.exps}
        self.exps = dict(self.exps)

    def print_results(self):
        # Load results
        results = self.get_results()
        print("Policies: " + ", ".join(POLICIES))
        for task in TASKS:
            print("{} task: {} {} {}".format(task, *[
                np.average(results['{}-{}'.format(task, p)][0])
                for p in POLICIES]))
        return results

    def plot_results(self, results, exclude_repeat=True):
        # Plot returns for preferences
        assert(len(TASKS) == 3)
        # Share axis for uniform and alternative
        figure = plt.figure()
        ax1 = figure.add_subplot(1, 3, 1)
        ax2 = figure.add_subplot(1, 3, 2)
        ax3 = figure.add_subplot(1, 3, 3, sharey=ax2)
        plots = [ax1, ax2, ax3]
        for plot, task in zip(plots, TASKS):
            if exclude_repeat:
                cpolicies = [p for p in POLICIES
                             if task == 'sequence' or p != 'repeat']
            else:
                cpolicies = POLICIES
            returns = [results['{}-{}'.format(task, pol)][0]
                       for pol in cpolicies]
            boxplot(returns, xticklabels=cpolicies, ax=plot, color="#1f82f9",
                    median_color="#bb0c36", showfliers=False)
            plot.set_title(task.capitalize())
            plot.set_xticklabels(plot.xaxis.get_ticklabels(), rotation=30)
        plots[0].set_ylabel('Average Return')
        return figure

    def action_plot(self):
        results = self.print_results()
        if self.args.plot_destination is not None:
            self.set_matplotlib_params_for_print()
        figure = self.plot_results(results)
        if self.args.plot_destination is None:
            plt.show()
        else:
            figure.savefig(
                os.path.join(self.args.plot_destination, 'multitask.pdf'),
                transparent=True)


MultitaskLauncher().run()
