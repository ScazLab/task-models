"""
Tools for task representation.

These make no sense if states are not discrete (although they may be
represented as continuous vectors).
"""


from htm.state import State
from htm.action import Action


def check_path(path):
    """Validates a path. Raises error on invalid path.

    A valid path is a list of alternate states and actions. It must start
    and finish by a state. In all successive (s, a, s'), s must validate
    pre-condition of a and s' its post-conditions.
    Empty paths are allowed.
    """
    try:
        return (len(path) == 0 or  # Empty path
                isinstance(path[0], State) and (
                    len(path) == 1 or (  # [s] or [s, a, s, ...]
                        isinstance(path[1], Action) and
                        isinstance(path[2], State) and
                        path[1].check(path[0], path[2]) and  # pre/post
                        check_path(path[2:])
                        )
                    )
                )
    except IndexError:
        return False


def split_path(path):
    return [(path[i], path[i + 1], path[i + 2])
            for i in range(0, len(path) - 2, 2)]


class TaskGraph:
    """Represents valid transitions in a task model.
    """

    def __init__(self):
        self.transitions = {}
        self.initials = set()
        self.terminals = set()

    def add_transition(self, s1, a, s2):
        if s1 not in self.transitions:
            self.transitions[s1] = []
        if (a, s2) not in self.transitions[s1]:
            self.transitions[s1].append((a, s2))

    def has_transition(self, s1, a, s2):
        return (s1 in self.transitions and (a, s2) in self.transitions[s1])

    def add_path(self, path):
        if not check_path(path):
            raise ValueError('Invalid path.')
        if len(path) > 0:
            self.initials.add(path[0])
            self.terminals.add(path[-1])
        for (s1, a, s2) in split_path(path):
            self.add_transition(s1, a, s2)
