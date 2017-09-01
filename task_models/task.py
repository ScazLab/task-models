# coding: utf-8

from __future__ import unicode_literals

"""
Tools for task representation.

These make no sense if states are not discrete (although they may be
represented as continuous vectors).
"""


from itertools import permutations

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
                    ))
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

    def remove_transition(self, source, label, destination):
        self.transitions[source].remove((label, destination))

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

    def compact(self, nodes, new_node):
        """Compact several nodes into on meta node.
        There is no difference about nodes and meta nodes, the code is agnostic
        about the nature of the node. However it replaces all label with
        an empty string, hence breaking any convention such as labels being
        states in a TaskGraph.
        """
        nodes = set(nodes)
        for (s, l, d) in list(self.all_transitions()):
            sin = s in nodes
            din = d in nodes
            if sin or din:
                self.remove_transition(s, l, d)
                if not (sin and din):  # not an inner node
                    self.add_transition(
                        new_node if sin else s, '', new_node if din else d)


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

    def copy(self, rename_format='{}'):
        return AbstractAction(rename_format.format(self.name))


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

    def get_max_chains(self, exclude=[]):
        in_degree = {}
        unique_transitions = {}
        for s in self.all_nodes():
            unique_transitions[s] = set()
        for (s, l, d) in self.all_transitions():
            if d not in unique_transitions[s]:
                in_degree[d] = in_degree.get(d, 0) + 1
            unique_transitions[s].add(d)
        to_explore = set([self.initial])
        explored = set()
        chain = None

        def explore_successors(node):
            to_explore.update(unique_transitions[node].difference(explored))

        while len(to_explore) > 0 or chain is not None:
            if chain is None:
                node = to_explore.pop()
                explored.add(node)
                if len(unique_transitions[node]) == 1:  # out degree == 1
                    chain = [node]
                else:
                    explore_successors(node)
            else:
                node = chain[-1]
                assert(len(unique_transitions[node]) == 1)
                next_node = list(unique_transitions[node])[0]
                done = False
                if in_degree[next_node] == 1:
                    chain.append(next_node)
                    explored.add(next_node)
                    if len(unique_transitions[next_node]) != 1:
                        done = True
                        explore_successors(next_node)
                else:
                    to_explore.add(next_node)
                    done = True
                if done:
                    if len(chain) > 1:
                        yield chain
                    chain = None

    def get_max_cliques(self):
        raise NotImplementedError


# Hierarchical task definition

class MetaAction(AbstractAction):

    SEP = {'sequence': '→',
           'parallel': '||',
           'alternative': '∨',
           }

    def __init__(self, kind, actions):
        self.kind = kind
        self.actions = actions

    def __eq__(self, other):
        return isinstance(other, MetaAction) and self.actions == other.actions

    @property
    def name(self):
        return self.SEP[self.kind].join([a.name for a in self.actions])

    def to_combination(self):
        children = [
            a.to_combination() if isinstance(a, MetaAction) else LeafCombination(a)
            for a in self.actions]
        return COMBINATION_CLASSES[self.kind](children)


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

    def _deep_copy_children(self, rename_format='{}'):
        return [c.deep_copy(rename_format=rename_format)
                for c in self.children]


class LeafCombination(Combination):

    kind = None

    def __init__(self, action, highlighted=False):
        self.highlighted = highlighted
        self.action = action

    @property
    def name(self):
        return self.action.name

    @property
    def children(self):
        raise ValueError('A leaf does not have children.')

    def as_dictionary(self, parent, id_generator):
        return self._meta_dictionary(parent, id_generator)

    def deep_copy(self, rename_format='{}'):
        return LeafCombination(self.action.copy(rename_format=rename_format),
                               highlighted=self.highlighted)


class SequentialCombination(Combination):

    kind = 'Sequence'

    def deep_copy(self, rename_format='{}'):
        return SequentialCombination(
            self._deep_copy_children(rename_format=rename_format),
            name=rename_format.format(self.name),
            highlighted=self.highlighted)


class AlternativeCombination(Combination):

    kind = 'Alternative'

    def __init__(self, children, probabilities=None, **xargs):
        super(AlternativeCombination, self).__init__(children, **xargs)
        self.proba = probabilities

    def deep_copy(self, rename_format='{}'):
        return AlternativeCombination(
            self._deep_copy_children(rename_format=rename_format),
            probabilities=self.proba,
            name=rename_format.format(self.name),
            highlighted=self.highlighted)


class ParallelCombination(Combination):

    kind = 'Parallel'

    def __init__(self, children, **xargs):
        super(ParallelCombination, self).__init__(children, **xargs)

    def deep_copy(self, rename_format='{}'):
        return ParallelCombination(
            self._deep_copy_children(rename_format=rename_format),
            probabilities=self.proba,
            name=rename_format.format(self.name),
            highlighted=self.highlighted)

    def to_alternative(self):
        sequences = [
            SequentialCombination(
                [c.deep_copy('{{}} order-{}'.format(i)) for c in p],
                name='{} order-{}'.format(self.name, i))
            for i, p in enumerate(permutations(self.children))
        ]
        return AlternativeCombination(sequences, name=self.name,
                                      highlighted=self.highlighted)


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


COMBINATION_CLASSES = {'sequence': SequentialCombination,
                       'parallel': ParallelCombination,
                       'alternative': AlternativeCombination,
                       }
