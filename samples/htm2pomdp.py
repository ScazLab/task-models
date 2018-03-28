"""Code to test and visualize the HTM to POMDP transformation of a basic
task.
"""


import os

from task_models.task import (HierarchicalTask, SequentialCombination,
                      AlternativeCombination, LeafCombination,
                      ParallelCombination)
from task_models.task_to_pomdp import CollaborativeAction, HTMToPOMDP


# Costs
T_WAIT = 1.
T_COMM = 2.
C_INTR = 1.
C_ERR = 5.
INF = 100.

## Tested scenarios:
# 1. with full sequence of sequential actions
R_END = 0.1
LOOP = False
# 2. with full sequence of sequential actions
# R_END = 100
# LOOP = True
R_SUBTASK = None

## Define the task
mount_central = SequentialCombination([
    LeafCombination(CollaborativeAction(
        'Get central frame', (INF, 20., 30.))),
    LeafCombination(CollaborativeAction(
        'Start Hold central frame', (3., 10., INF)))],
    name='Mount central frame')
#mount_legs = ParallelCombination([
mount_legs = SequentialCombination([
    SequentialCombination([
        LeafCombination(CollaborativeAction(
            'Get left leg', (INF, 20., 30.))),
        LeafCombination(CollaborativeAction(
            'Snap left leg', (5., INF, INF), fail_probability=.1)),
        ], name='Mount left leg'),
    SequentialCombination([
        LeafCombination(CollaborativeAction(
            'Get right leg', (INF, 20., 30.))),
        LeafCombination(CollaborativeAction(
            'Snap right leg', (5., INF, INF), fail_probability=.1)),
        ], name='Mount right leg'),
    ],
    name='Mount legs')
release_central = LeafCombination(CollaborativeAction('Release central frame', (INF, 1., 1.), no_probability=.1))
mount_top = SequentialCombination([
    LeafCombination(CollaborativeAction('Get top', (INF, 20., 30.))),
    LeafCombination(CollaborativeAction('Snap top', (5., INF, INF), fail_probability=.1))],
    name='Mount top')

chair_task = HierarchicalTask(root=SequentialCombination(
    [mount_central, mount_legs, release_central, mount_top], name='Mount chair'))

## Convert the task into a POMDP

h2p = HTMToPOMDP(T_WAIT, T_COMM, C_INTR, end_reward=R_END, loop=LOOP,
                 reward_state=False, subtask_reward=R_SUBTASK)
p = h2p.task_to_pomdp(chair_task)
#p.discount = .99

gp = p.solve(method='grid', n_iterations=500, verbose=True)
gp.save_as_json(os.path.join(os.path.dirname(__file__),
                             '../visualization/policy/json/test.json'))

from task_models.lib.pomdp import GraphPolicyBeliefRunner

pol = GraphPolicyBeliefRunner(gp, p)
pol.save_trajectories_from_starts(
    os.path.join(os.path.dirname(__file__),
                 '../visualization/trajectories/json/trajectories.json'),
    horizon=10, indent=2)
gp2 = pol.visit()
gp2.save_as_json(
    os.path.join(os.path.dirname(__file__),
                 '../visualization/policy/json/from_beliefs.json'))


from task_models.plot import plot_beliefs
import matplotlib.pyplot as plt

plt.interactive(True)
plt.close('all')


def plot_values(values, actions, p):
    b = values - values.min()
    b /= b.max(-1)[:, None]
    plot = plot_beliefs(b, states=p.states, xlabels_rotation=45,
                        ylabels=["{}: {}".format(i, a)
                                for i, a in enumerate(actions)])
    plt.colorbar(plot)


plt.figure()
plot_values(gp.values, gp.actions, p)
plt.title('Policy values')

plt.figure()
plot_values(gp2.values, gp2.actions, p)
plt.title('Belief trajectory')
