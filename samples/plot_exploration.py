#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import time
import json
import argparse

import numpy as np
from matplotlib import pyplot as plt

from task_models.utils.plot import plot_var
from task_models.task import (SequentialCombination, LeafCombination)
from task_models.supportive import (SupportivePOMDP, AssembleLeg, AssembleLegToTop,
                                    NHTMHorizon)
from task_models.lib.multiprocess import repeat
from task_models.lib.pomcp import POMCPPolicyRunner, NTransitionsHorizon


# Arguments
parser = argparse.ArgumentParser(
    description="Script to generate plot on exploration in the supportive "
                "POMDP")
parser.add_argument('path', help='path where to write the figure', nargs='?',
                    default=None)

args = parser.parse_args(sys.argv[1:])
INTERACTIVE = args.path is None


class FinishedOrNTransitionsHorizon(NTransitionsHorizon):

    def __init__(self, model, n):
        super(FinishedOrNTransitionsHorizon, self).__init__(n)
        self.model = model
        self._is_final = False

    def is_reached(self):
        return self.n <= 0 or self._is_final

    def decrement(self, a, s, new_s, o):
        super(FinishedOrNTransitionsHorizon, self).decrement(a, s, new_s, o)
        self._is_final = self.model._int_to_state(new_s).is_final()

    def copy(self):
        return FinishedOrNTransitionsHorizon(self.model, self.n)

    @classmethod
    def generator(cls, model, n=100):
        return cls._Generator(cls, model, n)


def simulate_one_evaluation(model, pol, max_horizon=50, norandom=False):
    pol.reset()
    state = model.sample_start()
    horizon = FinishedOrNTransitionsHorizon(model, max_horizon)
    full_return = 0
    while not horizon.is_reached():
        a = model.actions.index(pol.get_action(norandom=norandom))
        new_s, o, r = model.sample_transition(a, state)  # real transition
        horizon.decrement(a, state, new_s, o)
        pol.step(model.observations[o])
        state = new_s
        full_return = r + model.discount * full_return
    return full_return


def evaluate(model, pol, n_evaluation):

    def func():
        return simulate_one_evaluation(model, pol)

    return repeat(func, n_evaluation)


# Algorithm parameters
N_EP_EXPLO = 100     # exploration episodes (number of points in plot)
N_ITER_EXPLO = 250   # iterations for each exploration episode
N_EVALUATIONS = 100  # number of evaluation episodes after each exploration
ITERATIONS = 10      # iterations for the policy (used in get_action)
EXPLORATION = 50
RELATIVE_EXPLO = False  # In this case use smaller exploration
BELIEF_VALUES = False
N_PARTICLES = 150
HORIZON = 3
FULL_NORANDOM = True  # Also use norandom during get_action in evaluation


# Problem definition
leg_i = 'leg-{}'.format
htm = SequentialCombination([
    SequentialCombination([
        LeafCombination(AssembleLeg(leg_i(i))),
        LeafCombination(AssembleLegToTop(leg_i(i), bring_top=(i == 0)))])
    for i in range(4)])

p = SupportivePOMDP(htm)
# TODO put as default
p.r_subtask = 0.
p.r_preference = 20.
p.cost_hold = 3.
p.cost_get = 20.
pol = POMCPPolicyRunner(p, iterations=ITERATIONS,
                        horizon=NHTMHorizon.generator(p, n=HORIZON),
                        exploration=EXPLORATION,
                        relative_exploration=RELATIVE_EXPLO,
                        belief_values=BELIEF_VALUES,
                        belief='particle',
                        belief_params={'n_particles': N_PARTICLES})

pol_norandom = POMCPPolicyRunner(
    p, iterations=ITERATIONS,
    horizon=NHTMHorizon.generator(p, n=HORIZON),
    exploration=EXPLORATION,
    relative_exploration=RELATIVE_EXPLO,
    belief_values=BELIEF_VALUES,
    belief='particle',
    belief_params={'n_particles': N_PARTICLES})


# Explore and evaluate
maxl = 0
timer = []
t_0 = time.time()
evals_random = []
evals_norandom = []
for i in range(N_EP_EXPLO):
    s = "Exploring.... [{:2.0f}%] ({:.1f} [random], {:.1f} [norandom])".format(
        i * 100. / N_EP_EXPLO,
        np.average(evals_random[-1]) if i > 0 else 0.,
        np.average(evals_norandom[-1]) if i > 0 else 0.)
    maxl = max(maxl, len(s))
    print(' ' * maxl, end='\r')
    print(s, end='\r')
    sys.stdout.flush()
    # Some exploration
    pol.get_action(iterations=N_ITER_EXPLO)
    pol_norandom.get_action(iterations=N_ITER_EXPLO, norandom=True)
    t_exploration = time.time() - t_0
    # Some evaluation
    print("Evaluating... [{:2.0f}%]".format(i * 100 / N_EP_EXPLO), end='\r')
    evals_random.append(evaluate(p, pol, N_EVALUATIONS))
    evals_norandom.append(evaluate(p, pol_norandom, N_EVALUATIONS),
                          norandom=FULL_NORANDOM)
    t_evaluation = time.time() - t_0
    timer.append((t_evaluation, t_evaluation))
    # Storing current status
    tostore = {'evaluations': {'random': evals_random, 'no-random': evals_norandom},
               'timer': timer,
               }
    if not INTERACTIVE:
        with open(os.path.join(args.path, 'results.json'), 'w') as f:
            json.dump(tostore, f)

print('Running... [done]')


xs = np.arange(N_ITER_EXPLO, (1 + N_EP_EXPLO) * N_ITER_EXPLO, N_ITER_EXPLO)
if INTERACTIVE:
    plt.interactive(True)
plt.figure()
plot_var(evals_random, x=xs, label="random")
plot_var(evals_norandom, x=xs, label="norandom")
plt.xlabel('Exploration episodes')
plt.ylabel('Average return for policy')
plt.legend()
if not INTERACTIVE:
    plt.savefig(os.path.join(args.path, 'policy-values.svg'))
