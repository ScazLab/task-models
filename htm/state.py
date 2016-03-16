"""Environment state models."""


import numpy as np


class State:

    def get_features(self):
        raise NotImplemented


class NDimensionalState(State):

    def __init__(self, vector):
        self.features = vector

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
        self.parts = parts

    @property
    def _shape(self):
        return (self.N_PARTS, self.board_cols, self.board_rows,
                self.N_ORIENTATIONS)

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
