#!/usr/bin/env python


"""Compares behavior on various tasks.
"""

import io
import os
import json

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from expjobs.job import Job
from expjobs.helpers import Launcher

from task_models.utils.plot import boxplot


SCRIPT = os.path.join(os.path.dirname(__file__),
                      'supportive_evaluation_multitask_job.py')

POLICIES = ['pomcp', 'repeat', 'random']
TASKS = ['sequence', 'uniform',  'alternative']


class ExpLauncher(Launcher):

    name = 'multitask experiment'
    torque_args = {'default_walltime': 360}

    def init_jobs(self):
        self.exps = [('{}-{}'.format(task, pol),
                      {'task': task, 'policy': pol})
                     for task in TASKS for pol in POLICIES]
        self.jobs = {name: Job(self.args.path, name, SCRIPT)
                     for name, exp in self.exps}
        self.exps = dict(self.exps)

    def get_results_from_one(self, job):
        with io.open(os.path.join(job.path, job.name + '.json'), 'r') as f:
            results = json.load(f)['evaluations']
        return ([r['return'] for r in results],
                [r['elapsed-time'] for r in results],
                [r['simulator-calls'] for r in results])

    def action_prepare(self):
        for name in self.exps:
            with io.open(self.jobs[name].config, 'w') as fp:
                json.dump(self.exps[name], fp, indent=2)

    def get_results(self):
        return {j: self.get_results_from_one(self.jobs[j]) for j in self.jobs}

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
            matplotlib.rcParams.update({
                'font.family': 'serif',
                'font.size': 10,
                'font.serif': 'Computer Modern Roman',
                'text.usetex': 'True',
                'text.latex.unicode': 'True',
                'axes.titlesize': 'medium',
                'xtick.labelsize': 'xx-small',
                'ytick.labelsize': 'xx-small',
                'path.simplify': 'True',
                'savefig.pad_inches': 0.0,
                'savefig.bbox': 'tight',
                'figure.figsize': (3.5, 2.5),
            })
        figure = self.plot_results(results)
        if self.args.plot_destination is None:
            plt.show()
        else:
            figure.savefig(
                os.path.join(self.args.plot_destination, 'multitask.pdf'),
                transparent=True)


ExpLauncher().run()
