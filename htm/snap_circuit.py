import numpy as np


from .state import State
from .action import PrePostConditionAction, Condition


# Orientations of parts in SnapCircuit
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
ORIENTATION_NAMES = ["north", "east", "south", "west"]
# Locations are always given as (row, column)


class SnapCircuitPart:

    def __init__(self, _id, label):
        self.id = _id
        self.label = label

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return (isinstance(other, SnapCircuitPart) and
                self.id == other.id and
                self.label == other.label)

    def __repr__(self):
        return "{}<{}>".format(self.label, self.id)


class SnapCircuitState(State):
    """Represent a SnapCircuit board state.

    A board is given as number of rows and columns.
    Its state corresponds to a list of parts with locations.
    """

    N_PARTS = 20  # TODO improve this
    N_ORIENTATIONS = len(ORIENTATION_NAMES)

    def __init__(self, board, parts):
        self.board_rows = board[0]
        self.board_cols = board[1]
        self._parts = set(parts)

    def __hash__(self):
        return hash((self._shape, tuple(self.parts)))

    def __eq__(self, other):
        return (isinstance(other, SnapCircuitState) and
                self._shape == other._shape and
                self.parts == other.parts)

    def __str__(self):
        parts = ', '.join(
            '{} at ({}, {}) oriented {}'.format(part, loc[0], loc[1],
                                                ORIENTATION_NAMES[loc[2]])
            for loc, part in self._parts)
        return '{' + parts + '}'

    @property
    def _shape(self):
        return self.feature_shape((self.board_rows, self.board_cols))

    @property
    def parts(self):  # Immutable parts
        return self._parts.copy()

    @property
    def n_dim(self):
        return np.prod(self._shape)

    @property
    def features(self):
        features = np.zeros(self._shape)  # TODO: use sparse vectors
        for (loc, part) in self.parts:
            (x, y, o) = loc
            features[part.id, x, y, o] = 1
        return features

    def get_features(self):
        return self.features.flatten()

    @staticmethod
    def feature_shape(board):
        return (SnapCircuitState.N_PARTS, board[0], board[1],
                SnapCircuitState.N_ORIENTATIONS)


class PartPresenceCondition(Condition):

    def __init__(self, board, part, location, is_there=True):
        dim = SnapCircuitState.feature_shape(board)
        mask = np.zeros(dim)
        mask[part.id, location[0], location[1], location[2]] = 1
        value = np.zeros(dim)
        if is_there:
            value[part.id, location[0], location[1], location[2]] = 1
        super().__init__(mask.flatten(), value.flatten())


class PlaceAction(PrePostConditionAction):

    def __init__(self, board, part, location):
        x, y, o = location
        self.board = board
        pre = PartPresenceCondition(board, part, location, is_there=False)
        post = PartPresenceCondition(board, part, location, is_there=True)
        super().__init__(pre, post, "Place {} at {} oriented {}".format(
            part.label, (x, y), ORIENTATION_NAMES[o]))
        self.part = part
        self.location = location

    @property
    def position(self):
        return self.location[0], self.location[1]

    @property
    def orientation(self):
        return self.location[2]
