#!/usr/bin/env python

import io
import os
import json

from matplotlib import pyplot as plt

from expjobs.job import Job
from expjobs.helpers import Launcher

from task_models.utils.plot import plot_var


SCRIPT = os.path.join(os.path.dirname(__file__),
                      'supportive_evaluation_pomcp_job.py')
HORIZON_LENGTH_TRANSITIONS = [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100]
HORIZON_LENGTH_HTM = list(range(1, 10))

N_ITERATIONS = [15, 20, 30] + list(range(50, 1001, 50))
N_ROLLOUT_IT = [1, 5, 10, 30, 50, 75, 100]


class ExpLauncher(Launcher):

    name = 'multitask experiment'
    torque_args = {'default_walltime': 720}

    def init_jobs(self):
        horizon_types = (['transitions'] * len(HORIZON_LENGTH_TRANSITIONS) +
                         ['htm'] * len(HORIZON_LENGTH_HTM))
        horizon_lengths = HORIZON_LENGTH_TRANSITIONS + HORIZON_LENGTH_HTM
        self.exps = [('{}-{}-{}'.format(t, l, 's' if s else 'ns'),
                      {'horizon-type': t,
                       'horizon-length': l,
                       'intermediate-rewards': s})
                     for t, l in zip(horizon_types, horizon_lengths)
                     for s in (True, False)]  # subtask rewards
        self.exps.extend([('iterations-{}-{}'.format(i, r),
                           {'iterations': i, 'rollout-iterations': r})
                          for i in N_ITERATIONS for r in N_ROLLOUT_IT])
        self.jobs = {name: Job(self.args.path, name, SCRIPT)
                     for name, exp in self.exps}
        self.exps = dict(self.exps)

    def action_prepare(self):
        for name in self.exps:
            with io.open(self.jobs[name].config, 'w') as fp:
                json.dump(self.exps[name], fp, indent=2)

    @staticmethod
    def get_results_from_one(job):
        with io.open(os.path.join(job.path, job.name + '.json'), 'r') as f:
            results = json.load(f)['evaluations']
        return ([r['return'] for r in results],
                [r['elapsed-time'] for r in results],
                [r['simulator-calls'] for r in results])

    def plot_results(self):
        # Load results
        results = {j: self.get_results_from_one(self.jobs[j]) for j in self.jobs}
        # Horizon evaluation
        # Plot returns
        plots = plt.subplots(1, 2, sharey=True)[1]
        plots[0].set_ylabel('Average return')
        returns_transititions = [[results['transitions-{}-{}'.format(h, s)][0]
                                 for h in HORIZON_LENGTH_TRANSITIONS]
                                 for s in ('ns', 's')]
        returns_htm = [[results['htm-{}-{}'.format(h, s)][0]
                       for h in HORIZON_LENGTH_HTM]
                       for s in ('ns', 's')]
        plot_var(returns_transititions[0], x=HORIZON_LENGTH_TRANSITIONS,
                 ax=plots[0], label='final rewards only')
        plot_var(returns_transititions[1], x=HORIZON_LENGTH_TRANSITIONS,
                 ax=plots[0], label='subtask rewards')
        plots[0].set_title('N Transitions Horizon')
        plots[0].set_xlabel('Number of Transitions')
        plot_var(returns_htm[0], x=HORIZON_LENGTH_HTM, ax=plots[1],
                 label='final rewards only')
        plot_var(returns_htm[1], x=HORIZON_LENGTH_HTM, ax=plots[1],
                 label='subtask rewards')
        plots[1].set_title('N HTM Horizon')
        plots[1].set_xlabel('Number of HTM Transitions')
        plots[1].legend()
        plt.title('Average returns for various horizons')
        # Plot simulator calls
        plots = plt.subplots(1, 2, sharey=True)[1]
        plots[0].set_ylabel('Average number of calls to simulator')
        calls_transititions = [[results['transitions-{}-{}'.format(h, s)][2]
                               for h in HORIZON_LENGTH_TRANSITIONS]
                               for s in ('ns', 's')]
        calls_htm = [[results['htm-{}-{}'.format(h, s)][2]
                     for h in HORIZON_LENGTH_HTM]
                     for s in ('ns', 's')]
        plot_var(calls_transititions[0], x=HORIZON_LENGTH_TRANSITIONS, ax=plots[0],
                 label='final rewards only')
        plot_var(calls_transititions[1], x=HORIZON_LENGTH_TRANSITIONS, ax=plots[0],
                 label='subtask rewards')
        plots[0].set_title('N Transitions Horizon')
        plots[0].set_xlabel('Number of Transitions')
        plot_var(calls_htm[0], x=HORIZON_LENGTH_HTM, ax=plots[1],
                 label='final rewards only')
        plot_var(calls_htm[1], x=HORIZON_LENGTH_HTM, ax=plots[1],
                 label='subtask rewards')
        plots[1].set_title('N HTM Horizon')
        plots[1].set_xlabel('Number of HTM Transitions')
        plots[1].legend()
        plt.title('Simulator calls for various horizons')
        # N iterations evaluation
        # Plot returns for
        plt.figure()
        cplot = plt.gca()
        cplot.set_ylabel('Average return')
        cplot.set_xlabel('Number of iterations')
        for r in N_ROLLOUT_IT:
            returns_iterations = [results['iterations-{}-{}'.format(i, r)][0]
                                  for i in N_ITERATIONS]
            plot_var(returns_iterations, x=N_ITERATIONS,
                     label='{} rollout iterations'.format(r))
        plt.legend()
        plt.title('Average returns for various numbers of iterations')

    def action_plot(self):
        self.plot_results()
        plt.show()


ExpLauncher().run()
