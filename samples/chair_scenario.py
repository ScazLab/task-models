import json

from htm.task import (HierarchicalTask, AbstractAction, SequentialCombination,
                      ParallelCombination)


take_base = AbstractAction('Take base')
mount_leg_combinations = [
    SequentialCombination([AbstractAction('Take leg {}'.format(i)),
                           AbstractAction('Attach leg {}'.format(i))],
                          name='Mount leg {}'.format(i))
    for i in range(4)
    ]
mount_frame = SequentialCombination(
    [AbstractAction('Take frame'), AbstractAction('Attach frame')],
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
