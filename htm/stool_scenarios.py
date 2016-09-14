from htm.task import (HierarchicalTask, SequentialCombination,
                      LeafCombination)
from htm.task_to_pomdp import CollaborativeAction


# Costs
T_WAIT = 1.
T_COMM = 2.
C_INTR = 1.
C_ERR = 5.
INF = 100.

mount_central = SequentialCombination([
    LeafCombination(CollaborativeAction(
        'Get central frame', (INF, 20., 30.))),
    LeafCombination(CollaborativeAction(
        'Hold central frame', (.01, 15., INF)))],
    name='Mount central frame')

mount_legs = SequentialCombination([
    SequentialCombination([
        LeafCombination(CollaborativeAction(
            'Bring left leg', (INF, 20., 30.))),
        LeafCombination(CollaborativeAction(
            'Snap left leg', (5., INF, INF), fail_probability=.1)),
        ], name='Mount left leg'),
    SequentialCombination([
        LeafCombination(CollaborativeAction(
            'Bring right leg', (INF, 20., 30.))),
        LeafCombination(CollaborativeAction(
            'Snap right leg', (5., INF, INF), fail_probability=.1)),
        ], name='Mount right leg'),
    ],
    name='Mount legs')

release_central = LeafCombination(CollaborativeAction(
    'Release central frame', (INF, 1., 1.), no_probability=.1))

mount_top = SequentialCombination([
    LeafCombination(CollaborativeAction('Bring top', (INF, 20., 30.))),
    LeafCombination(CollaborativeAction('Snap top', (5., INF, INF),
                                        fail_probability=.1))],
    name='Mount top')

stool_task_sequential = HierarchicalTask(root=SequentialCombination(
    [mount_central, mount_legs, release_central, mount_top],
    name='Mount stool'))
