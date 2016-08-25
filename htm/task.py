"""
Tools for task representation.

These make no sense if states are not discrete (although they may be
represented as continuous vectors).
"""


from .state import State
from .action import Action


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


class BaseGraph(object):
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

    def all_nodes(self):
        nodes = set()
        for s, l, d in self.all_transitions():
            nodes.add(s)
            nodes.add(d)
        return nodes

    def as_dictionary(self, name=''):
        d = {'name': name}
        all_nodes = list(enumerate(self.all_nodes()))
        d['nodes'] = [{'id': i, 'value': {'label': str(node)}}
                      for i, node in all_nodes]
        nodes_ids = dict([(n, i) for i, n in all_nodes])
        d['links'] = [{'u': nodes_ids[u],
                       'v': nodes_ids[v],
                       'value': {'label': str(l)}}
                      for u, l, v in self.all_transitions()]
        return d


class TaskGraph(BaseGraph):
    """Represents valid transitions in a task model.
    """

    def __init__(self):
        super(TaskGraph, self).__init__()
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

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, AbstractAction) and self.name == other.name

    def check(self, pre, post):
        return True


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


# Hierarchical task definition

def int_generator():
    i = -1
    while True:
        i += 1
        yield i


class Combination(object):

    kind = 'Undefined'

    def __init__(self, children, name='unnamed', highlighted=False):
        self.children = children  # Actions or combinations
        self.name = name
        self.highlighted = highlighted

    def _meta_dictionary(self, parent_id, id_generator):
        attr = []
        if self.highlighted:
            attr.append('highlighted')
        return {'name': self.name,
                'id': next(id_generator),
                'parent': parent_id,
                'combination': self.kind,
                'attributes': attr,
                }

    def as_dictionary(self, parent_id, id_generator):
        d = self._meta_dictionary(parent_id, id_generator)
        d['children'] = [
            c.as_dictionary(d['id'], id_generator)
            for c in self.children
            ]
        return d


class LeafCombination(Combination):

    kind = None

    def __init__(self, action, highlighted=False):
        self.highlighted = highlighted
        self.action = action

    @property
    def name(self):
        return self.action.name

    def as_dictionary(self, parent, id_generator):
        return self._meta_dictionary(parent, id_generator)


class SequentialCombination(Combination):

    kind = 'Sequence'


class AlternativeCombination(Combination):

    kind = 'Alternative'

    def __init__(self, children, human_probabilities=None, **xargs):
        super(AlternativeCombination, self).__init__(children, **xargs)
        self.h_proba = human_probabilities


class ParallelCombination(Combination):

    kind = 'Parallel'


class HierarchicalTask:
    """Tree representing a hierarchy of tasks which leaves are actions."""

    def __init__(self, root=None):
        self.root = root

    def is_empty(self):
        return self.root is None

    def as_dictionary(self, name=None):
        return {
            'name': 'Hierarchical task tree' if name is None else name,
            'nodes': None if self.is_empty() else self.root.as_dictionary(
                None, int_generator()),
            }
