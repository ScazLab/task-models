"""Environment state models."""


import numpy as np


class State:
    """Any kind of state. Must be hashable and are immutable.

    Any state implementation must implement an identifier for  states.
    The equality of identifiers must imply the equality of states.
    """

    def get_features(self):
        raise NotImplemented

    def __hash__(self):
        raise NotImplemented

    def __eq__(self, other):
        raise NotImplemented


class NDimensionalState(State):

    def __init__(self, vector):
        self.features = vector

    def __hash__(self):
        return hash(self.features.tostring())

    @property
    def n_dim(self):
        return len(self.features)

    def get_features(self):
        return self.features


# Orientations of parts in SnapCircuit
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


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

    def __str__(self):
        return "{}<{}>".format(self.label, self.id)


class SnapCircuitState(State):
    """Represent a SnapCircuit board state.

    A board is given as number of rows and columns.
    It state corresponds to a list of parts with locations.
    """

    N_PARTS = 20  # TODO improve this
    N_ORIENTATIONS = 4

    def __init__(self, board, parts):
        self.board_cols = board[0]
        self.board_rows = board[1]
        self._parts = set(parts)

    def __hash__(self):
        return hash((self._shape, tuple(self.parts)))

    def __eq__(self, other):
        return (isinstance(other, SnapCircuitState) and
                self._shape == other._shape and
                self.parts == other.parts)

    @property
    def _shape(self):
        return (self.N_PARTS, self.board_cols, self.board_rows,
                self.N_ORIENTATIONS)

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
