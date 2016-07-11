"""Environment state models."""


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
