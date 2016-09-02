import json

from htm.task import (HierarchicalTask, AbstractAction, SequentialCombination,
                      ParallelCombination, AlternativeCombination, LeafCombination)

bench_task = LeafCombination(AbstractAction('Mount Bench'))
chair_task = LeafCombination(AbstractAction('Mount Chair'))

central_frame = LeafCombination(AbstractAction('Take central frame'))

mount_leg_combinations = [
    SequentialCombination(
        [LeafCombination(AbstractAction('Take {} leg'.format(i))),
         LeafCombination(AbstractAction('Snap {} leg'.format(i)))
         ],
        name='Mount {} leg'.format(i))
    for i in ['left','right']
    ]

mount_top_combination = SequentialCombination(
    [LeafCombination(AbstractAction('Take  top'.format(i)), highlighted=True),
     LeafCombination(AbstractAction('Place top'.format(i)))
     ],
    name='Mount top'.format(i))

stool_task = SequentialCombination(
    [central_frame,
     ParallelCombination(mount_leg_combinations, name='Mount legs'),
     mount_top_combination,
     ],
    name='Mount Stool')

workshop_task = HierarchicalTask(
    root=AlternativeCombination(
        [bench_task,
         stool_task,
         chair_task,
         ],
        name='Available Tasks')
    )

print(json.dumps(workshop_task.as_dictionary(), indent=2))
