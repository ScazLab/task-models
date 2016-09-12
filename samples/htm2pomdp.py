import os

from htm.task import (HierarchicalTask, SequentialCombination,
                      AlternativeCombination, LeafCombination,
                      ParallelCombination)
from htm.task_to_pomdp import CollaborativeAction, HTMToPOMDP
# Costs
T_WAIT = 1.
T_COMM = 2.
C_INTR = 1.
C_ERR = 5.
R_END = 100

## Define the task
mount_central = SequentialCombination([
    LeafCombination(CollaborativeAction(
        'Get central frame', (10., 3., C_ERR))),
    LeafCombination(CollaborativeAction(
        'Snap central frame', (3., 10., C_ERR)))],
    name='Mount central frame')
#mount_legs = ParallelCombination([
mount_legs = SequentialCombination([
    SequentialCombination([
        LeafCombination(CollaborativeAction(
            'Get left leg', (10., 3., C_ERR))),
        LeafCombination(CollaborativeAction(
            'Snap left leg', (3., 10., C_ERR))),
        ], name='Mount left leg'),
    SequentialCombination([
        LeafCombination(CollaborativeAction(
            'Get right leg', (10., 3., C_ERR))),
        LeafCombination(CollaborativeAction(
            'Snap right leg', (3., 10., C_ERR))),
        ], name='Mount right leg'),
    ],
    name='Mount legs')
mount_top = SequentialCombination([
    LeafCombination(CollaborativeAction('Get top', (10., 3., C_ERR))),
    LeafCombination(CollaborativeAction('Snap top', (3., 10., C_ERR)))],
    name='Mount top')

chair_task = HierarchicalTask(root=SequentialCombination(
    [mount_central, mount_legs, mount_top], name='Mount chair'))

## Convert the task into a POMDP

h2p = HTMToPOMDP(T_WAIT, T_COMM, C_INTR, end_reward=R_END, loop=True)
p = h2p.task_to_pomdp(chair_task)
#p.discount = .99

gp = p.solve(method='grid', n_iterations=1000, verbose=True)
gp.dump_to(os.path.join(os.path.dirname(__file__),
                        '../visualization/policy/json/test.json'))

from htm.lib.pomdp import GraphPolicyBeliefRunner

pol = GraphPolicyBeliefRunner(gp, p)
pol.save_trajectories_from_starts(
    os.path.join(os.path.dirname(__file__),
                 '../visualization/trajectories/json/trajectories.json'),
    horizon=10, indent=2)
gp2 = pol.visit()
gp2.dump_to(os.path.join(os.path.dirname(__file__),
                         '../visualization/policy/json/from_beliefs.json'))


from htm.plot import plot_beliefs
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
