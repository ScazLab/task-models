import json

from htm.task import (HierarchicalTask, AbstractAction, SequentialCombination,
                      ParallelCombination, AlternativeCombination, LeafCombination)

bench_task = LeafCombination(AbstractAction('Mount Bench'))
chair_task = LeafCombination(AbstractAction('Mount Chair'))

take_central_frame = LeafCombination(AbstractAction('Take central frame'))
hold_central_frame = LeafCombination(AbstractAction('Hold central frame'))
release_central_frame = LeafCombination(AbstractAction('Drop central frame'))

mount_leg_combinations = [
    SequentialCombination(
        [LeafCombination(AbstractAction('Take {} leg'.format(i))),
         LeafCombination(AbstractAction('Snap {} leg'.format(i)))
         ],
        name='Mount {} leg'.format(i))
    for i in ['left','right']
    ]

mount_top_combination = SequentialCombination(
    [LeafCombination(AbstractAction('Take  top')),
     LeafCombination(AbstractAction('Place top'))
     ],
    name='Mount top')

stool_task = SequentialCombination(
    [take_central_frame,
     hold_central_frame,
     ParallelCombination(mount_leg_combinations, name='Mount legs'),
     release_central_frame,
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

# print(json.dumps(workshop_task.as_dictionary(), indent=2))
