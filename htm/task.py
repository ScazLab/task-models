"""
Tools for task representation.

These make no sense if states are not discrete (although they may be
represented as continuous vectors).
"""


from htm.state import State
from htm.action import Action, MatchAllCondition


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


class BaseGraph:
    """Transitions (s, l, d) are stored as {s: {(l, d), ...}, ...}, that is
    a dictionary of sets of pairs.
    """

    def __init__(self):
        self.transitions = {}

    def __eq__(self, other):
        return (isinstance(other, BaseGraph) and
                self.transitions == other.transitions)

    def add_transition(self, source, label, destination):
        if source not in self.transitions:
            self.transitions[source] = set()
        self.transitions[source].add((label, destination))

    def has_transition(self, source, label, destination):
        return (source in self.transitions and
                (label, destination) in self.transitions[source])

    def all_transitions(self):
        for s in self.transitions:
            for (a, s_next) in self.transitions[s]:
                yield (s, a, s_next)


class TaskGraph(BaseGraph):
    """Represents valid transitions in a task model.
    """

    def __init__(self):
        super().__init__()
        self.initial = set()
        self.terminal = set()

    def __eq__(self, other):
        return (super().__eq__(other) and
                isinstance(other, TaskGraph) and
                self.initial == other.initial and
                self.terminal == other.terminal)

    def add_path(self, path):
        if not check_path(path):
            raise ValueError('Invalid path.')
        if len(path) > 0:
            self.initial.add(path[0])
            self.terminal.add(path[-1])
        for (s1, a, s2) in split_path(path):
            self.add_transition(s1, a, s2)

    def has_path(self, path):
        """Checks if the path is a valid path for this task model.
        """
        if not check_path(path):
            raise ValueError('Invalid path.')
        return ((len(path) == 0) or
                (path[0] in self.initial and
                 path[-1] in self.terminal and
                 all([self.has_transition(s1, a, s2)
                      for (s1, a, s2) in split_path(path)
                      ])
                 ))

    def check_only_deterministic_transitions_from_state(self, s):
        """See get_deterministic_transitions."""
        outgoing_actions = set()
        for (a, sn) in self.transitions[s]:
            if a in outgoing_actions:
                raise ValueError(
                    "Non-deterministic transition from state {} "
                    "for action {}.".format(s, a))
            outgoing_actions.add(a)

    def check_only_deterministic_transitions(self):
        """Deterministic means: for each (s, a) there is at most one s' such
        as (s, a, s') is a transition.
        Raises ValueError if there are non-deterministic transitions.
        """
        for s in self.transitions:
            self.check_only_deterministic_transitions_from_state(s)

    def conjugate(self):
        return ConjugateTaskGraph.from_task_graph(self)


class AbstractAction(Action):

    def __init__(self, name):
        super().__init__(MatchAllCondition(), MatchAllCondition(), name=name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, AbstractAction) and self.name == other.name


class ConjugateTaskGraph(BaseGraph):

    initial = AbstractAction('initial')
    terminal = AbstractAction('terminal')

    @classmethod
    def from_task_graph(cls, tg):
        ctg = cls()
        tg.check_only_deterministic_transitions()
        # Add transitions of the form: initial --(s_init)--> a
        for s in tg.initial:
            for (a, _) in tg.transitions[s]:
                ctg.add_transition(ctg.initial, s, a)
        for outgoings in tg.transitions.values():  # Whatever origin state
            for (a, next_state) in outgoings:
                if next_state in tg.terminal:
                    # Add transitions of the form: a --(s_term)--> terminal
                    ctg.add_transition(a, next_state, ctg.terminal)
                else:
                    # Look ahead...
                    for (next_action, _) in tg.transitions[next_state]:
                        # ...and add transitions of the form: a --(s')--> a'
                        ctg.add_transition(a, next_state, next_action)
        return ctg
