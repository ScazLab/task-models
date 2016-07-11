import json

import numpy as np

from htm.snap_circuit import (SnapCircuitState, SnapCircuitPart, PlaceAction,
                              NORTH, EAST)
from htm.task import TaskGraph


BOARD = (7, 10)


def path_make_L(position):
    assert(position[0] < BOARD[0] - 2)  # L impossible on last two rows
    assert(position[1] < BOARD[1] - 1)  # L impossible on last column
    part0 = SnapCircuitPart(0, '2')
    part1 = SnapCircuitPart(1, '2')
    location0 = (position[0], position[1], NORTH)
    location1 = (position[0] + 2, position[1], EAST)
    return [
        SnapCircuitState(BOARD, []),
        PlaceAction(BOARD, part0, location0),
        SnapCircuitState(BOARD, [(location0, part0)]),
        PlaceAction(BOARD, part1, location1),
        SnapCircuitState(BOARD, [(location0, part0), (location1, part1)]),
        ]


def random_paths_make_L():
    return path_make_L((np.random.randint(BOARD[0] - 2),
                        np.random.randint(BOARD[1] - 1)))


tg = TaskGraph()
for __ in range(3):
    tg.add_path(random_paths_make_L())
d = tg.as_dictionary()
print(json.dumps(d, indent=2))
