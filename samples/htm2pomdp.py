from htm.task import (HierarchicalTask, SequentialCombination,
                      AlternativeCombination, LeafCombination)
from htm.task_to_pomdp import CollaborativeAction, HTMToPOMDP

## Define the task
mount_central = SequentialCombination([
    LeafCombination(CollaborativeAction('Get central frame', (10., 3., 5.))),
    LeafCombination(CollaborativeAction('Snap central frame', (3., 10., 5.)))],
    name='Mount central frame')
mount_legs = AlternativeCombination([
    SequentialCombination(
        [LeafCombination(CollaborativeAction(
            'Get leg {} ({} first)'.format(sides[0], sides[0]),
            (10., 3., 5.))),
         LeafCombination(CollaborativeAction(
             'Snap leg {}'.format(sides[1], sides[0]), (10., 3., 5.)))
         ],
        name='Mount legs ({} first)'.format(sides[0]))
    for sides in [('left', 'right'), ('right', 'left')]
    ], name='Mount legs')
# Use a simpler one until Alternative to POMDP is implemented
mount_legs = SequentialCombination([
    LeafCombination(CollaborativeAction('Get left leg', (10., 3., 5.))),
    LeafCombination(CollaborativeAction('Snap left leg', (3., 10., 5.))),
    LeafCombination(CollaborativeAction('Get right leg', (10., 3., 5.))),
    LeafCombination(CollaborativeAction('Snap right leg', (3., 10., 5.))),
    ],
    name='Mount legs')
mount_top = SequentialCombination([
    LeafCombination(CollaborativeAction('Get top', (10., 3., 5.))),
    LeafCombination(CollaborativeAction('Snap top', (3., 10., 5.)))],
    name='Mount top')

chair_task = HierarchicalTask(root=SequentialCombination(
    [mount_central, mount_legs, mount_top], name='Mount chair'))

## Convert the task into a POMDP
T_WAIT = 1.
T_COMM = 2.
C_INTR = 1.

h2p = HTMToPOMDP(T_WAIT, T_COMM, C_INTR, False, False)

p = h2p.task_to_pomdp(HierarchicalTask(root=mount_legs))
# p.dump_to('/tmp/', 'legs')

gp = p.solve(method='grid', n_iterations=1000)
print(gp.to_json())
# gp.dump_to('../visualization/policy/json/test.json');

from htm.plot import plot_beliefs
import matplotlib.pyplot as plt

plt.interactive(True)
plt.figure()
b = gp.values - gp.values.min()
b /= b.max(-1)[:, None]
p = plot_beliefs(b, states=p.states, xlabels_rotation=45,
                 ylabels=["{}: {}".format(i, a)
                          for i, a in enumerate(gp.actions)])
plt.colorbar(p)
