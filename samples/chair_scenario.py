import json

from htm.task import (HierarchicalTask, AbstractAction, SequentialCombination,
                      ParallelCombination, LeafCombination)


take_base = LeafCombination(AbstractAction('Take base'))
mount_leg_combinations = [
    SequentialCombination(
        [LeafCombination(AbstractAction('Take leg {}'.format(i))),
         LeafCombination(AbstractAction('Attach leg {}'.format(i)))
         ],
        name='Mount leg {}'.format(i))
    for i in range(4)
    ]
mount_frame = SequentialCombination(
    [LeafCombination(AbstractAction('Take frame')),
     LeafCombination(AbstractAction('Attach frame'))
     ],
    name='Mount frame')

chair_task = HierarchicalTask(
    root=SequentialCombination(
        [take_base,
         ParallelCombination(mount_leg_combinations, name='Mount legs'),
         mount_frame,
         ],
        name='Mount chair')
    )

print(json.dumps(chair_task.as_dictionary(), indent=2))
