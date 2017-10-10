# coding: utf-8

from __future__ import unicode_literals
import numpy as np
import itertools as iter

"""
Tools for task representation.

These make no sense if states are not discrete (although they may be
represented as continuous vectors).
"""

from itertools import permutations

from task_models.state import State
from task_models.action import Action


# from task_models import State
# from .action import Action


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


def max_cliques(graph):
    """Searches non-trivial maximum cliques in graph.

    Uses Bron-Kerbosch algorithm with pivot.

    :graph: dictionary of node: set(neighbours) representing symmetric graph
        (must verify: v in graph[u] => u in graph[v])
    :returns: iterator over cliques (as sets)
    """
    # yield from
    for clique in _bron_kerbosch(graph, set(), set(graph), set()):
        if len(clique) > 1:  # non-trivial
            yield clique


def _bron_kerbosch(graph, r, p, x):
    if len(p) == 0 and len(x) == 0:
        yield r
    elif len(p) > 0:
        pivot = max(p, key=lambda u: len(graph[u].intersection(p)))
        # (node with the most neighbours in p)
        for u in p.difference(graph[pivot]):
            # yield from...
            for clique in _bron_kerbosch(graph, r.union([u]),
                                         p.intersection(graph[u]),
                                         x.intersection(graph[u])):
                yield clique
            p.remove(u)
            x.add(u)


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
        return (super(TaskGraph, self).__eq__(other) and
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


class PredAction(AbstractAction):
    """Action abstraction to be used for the prediction project.

    :param name: str (must be unique)
    :param durations: (float: human, float: robot, float: error)
        Duration for human and robot to perform the action as well as error
        time when robot starts action at wrong moment (the error time includes
        the time of acting before interruption as well as the recovery time).
    :param human_probability: float
        Probability that the human would take care of this action. If not
        defined, will have to be estimated.
    :param fail_probability: float
        Probability that the robot action fails.
    :param no_probability: float
        Probability that the human answers no to the robot asking if he can
        take the action.
    """

    def __init__(self, name, bin_feats):
        super(PredAction, self).__init__(name=name)
        self.bin_feats = bin_feats

    def copy(self, rename_format='{}'):
        return PredAction(rename_format.format(self.name), self.bin_feats)


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

    def compact(self, nodes, new_node):
        # TODO: is this really the desired behavior?
        super(ConjugateTaskGraph, self).compact(nodes, new_node)
        if self.initial in nodes:
            self.initial = new_node
        if self.terminal in nodes:
            self.terminal = new_node

    def get_max_chains(self, exclude=[]):
        """Search non-trivial maximum chains in the graph.

        :returns: iterator over chains (as lists)
        """
        in_degree = {}
        unique_transitions = {n: set() for n in self.all_nodes()}
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
                assert (len(unique_transitions[node]) == 1)
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
        """Searches non-trivial maximum cliques in the symmetrized graph.

        :returns: iterator over cliques (as sets)
        """
        # compute symmetric graph
        unique = set((s, d) for (s, _, d) in self.all_transitions())
        symmetric = {n: set() for n in self.all_nodes()}
        for s, d in unique:
            if (d, s) in unique:
                symmetric[s].add(d)
                symmetric[d].add(s)
        return max_cliques(symmetric)


# Hierarchical task definition

class MetaAction(AbstractAction):
    SEP = {'sequence': '→',
           'parallel': '||',
           'alternative': '∨',
           }

    def __init__(self, kind, actions):
        super(MetaAction, self).__init__(self._name(kind, actions))
        self.kind = kind
        self.actions = actions

    def __eq__(self, other):
        return (isinstance(other, MetaAction) and
                self.kind == other.kind and
                self.actions == other.actions)

    def __hash__(self):
        return hash(self.name)
        # Needs to re-implement __hash__ despite inheritence (see following
        # from datamodels doc).
        # A class that overrides __eq__() and does not define __hash__() will
        # have its __hash__() implicitly set to None. When the __hash__()
        # method of a class is None, instances of the class will raise an
        # appropriate TypeError when a program attempts to retrieve their hash
        # value, and will also be correctly identified as unhashable when
        # checking isinstance(obj, collections.Hashable).

    def to_combination(self):
        children = [
            a.to_combination() if isinstance(a, MetaAction) else LeafCombination(a)
            for a in self.actions]
        return COMBINATION_CLASSES[self.kind](children)

    @classmethod
    def _name(cls, kind, actions):
        return '({})'.format(cls.SEP[kind].join([a.name for a in actions]))


def int_generator():
    i = -1
    while True:
        i += 1
        yield i


class Combination(object):
    kind = 'Undefined'

    def __init__(self, children, name='unnamed', highlighted=False,
                 probabilities=None):
        self.children = children  # Actions or combinations
        self.name = name
        self.highlighted = highlighted
        self.proba = probabilities

    @property
    def proba(self):
        return self.__proba

    @proba.setter
    def proba(self, probabilities):
        num_children = len(self.children)
        if len(probabilities) != num_children:
            raise ValueError("Length of probabilities array should be equal \
                             to number of children of combination.")
        else:
            self.__proba = probabilities

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

    def __init__(self, children, **xargs):
        super(SequentialCombination, self).__init__(children, **xargs)

    @property
    def proba(self):
        return super(SequentialCombination, self).proba
        # return self.__proba

    @proba.setter
    def proba(self, probabilities):
        correct = False
        num_children = len(self.children)
        if probabilities is None:
            correct = True
        elif all(prob == 1 for prob in probabilities):
            correct = True
        if correct is True:
            return super(SequentialCombination, self.__class__). \
                proba.fset(self, [1] * num_children)
        else:
            raise ValueError("Sequential combinations have default probs \
                             set to [1] * num children.")

    def deep_copy(self, rename_format='{}'):
        return SequentialCombination(
            self._deep_copy_children(rename_format=rename_format),
            probabilities=self.proba,
            name=rename_format.format(self.name),
            highlighted=self.highlighted)


class AlternativeCombination(Combination):
    kind = 'Alternative'

    def __init__(self, children, **xargs):
        super(AlternativeCombination, self).__init__(children, **xargs)

    @property
    def proba(self):
        return super(AlternativeCombination, self).proba

    @proba.setter
    def proba(self, probabilities):
        if probabilities is None:
            num_children = len(self.children)
            prob = float(1) / num_children
            return super(AlternativeCombination, self.__class__). \
                proba.fset(self, [prob] * num_children)
        elif np.min(probabilities) < 0:
            raise ValueError("At least one prob value is < 0.")
        elif np.max(probabilities) > 1:
            raise ValueError("At least one prob value is > 1.")
        elif not(np.isclose(np.sum(probabilities), 1)):
            raise ValueError("Probs should sum to 1.")
        else:
            return super(AlternativeCombination, self.__class__). \
                proba.fset(self, probabilities)

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

    @property
    def proba(self):
        return super(ParallelCombination, self).proba

    @proba.setter
    def proba(self, probabilities):
        correct = False
        if probabilities is None:
            correct = True
        elif all(prob is None for prob in probabilities):
            correct = True
        if correct is True:
            num_children = len(self.children)
            return super(ParallelCombination, self.__class__). \
                proba.fset(self, [None] * num_children)
        else:
            raise ValueError("Parallel combinations shouldn't have \
            probs specified.")

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


class TrajectoryElement(object):
    """Objects out of which we build trajectories.
    Trajectories are sequences of leaves generated from an HTM.
    """

    def __init__(self, node, prob):
        self.node = node
        self.prob = prob

    @property
    def prob(self):
        return self.__prob

    @prob.setter
    def prob(self, prob):
        if prob < 0:
            raise ValueError("Prob < 0.")
        elif prob > 1:
            raise ValueError("Prob > 1.")
        else:
            self.__prob = prob


class HierarchicalTask(object):
    """Tree representing a hierarchy of tasks which leaves are actions."""

    def __init__(self, root=None):
        self.root = root
        self.all_trajectories = []
        self.traj_probs = []

    def is_empty(self):
        return self.root is None

    def as_dictionary(self, name=None):
        return {
            'name': 'Hierarchical task tree' if name is None else name,
            'nodes': None if self.is_empty() else self.root.as_dictionary(
                None, int_generator()),
        }

    def gen_all_trajectories(self):
        self.all_trajectories = \
            self.gen_trajectories_rec(self.root, proba=1)

    # def gen_trajectories_rec(self, node, proba=None):
    #     """Generates all possible trajectories from an HTM.
    #     """
    #     if isinstance(node, LeafCombination):
    #         return [(1, [node])]
    #     elif isinstance(node, ParallelCombination):
    #         return self.gen_trajectories_rec(node.to_alternative())
    #     elif isinstance(node, AlternativeCombination):
    #         return list(iter.chain.from_iterable(
    #             [(p * node.proba[c_idx], seq)
    #              for p, seq in self.gen_trajectories_rec(c, node.proba[c_idx])
    #              for c_idx, c in enumerate(node.children)]))
    #     elif isinstance(node, SequentialCombination):
    #         children_trajectories = [
    #             self.gen_trajectories_rec(c, node.proba[c_idx])
    #             for c_idx, c in enumerate(node.children)]
    #         new_trajectories = [[]]
    #         for child in children_trajectories:
    #             product_trajectories = list(iter.product(
    #                         new_trajectories, *child))
    #             new_trajectories = []
    #             for product in product_trajectories:
    #                 new_trajectories.append((list(iter.chain.from_iterable(product)))
    #         return new_trajectories
    #     else:
    #         raise ValueError("Reached invalid type during recursion.")

    def gen_trajectories_rec(self, node, proba=None):
        """Generates all possible trajectories from an HTM.
        """
        if isinstance(node, LeafCombination):
            return [[node]]
        elif isinstance(node, ParallelCombination):
            return self.gen_trajectories_rec(node.to_alternative())
        elif isinstance(node, AlternativeCombination):
            return list(iter.chain.from_iterable(
                [self.gen_trajectories_rec(c, node.proba[c_idx])
                 for c_idx, c in enumerate(node.children)]))
        elif isinstance(node, SequentialCombination):
            children_trajectories = [
                self.gen_trajectories_rec(c, node.proba[c_idx])
                for c_idx, c in enumerate(node.children)]
            new_trajectories = []
            product_trajectories = list(iter.product(
                *children_trajectories))
            for product in product_trajectories:
                new_trajectories.append(list(iter.chain.from_iterable(product)))
            return new_trajectories
        else:
            raise ValueError("Reached invalid type during recursion.")

    def sample_trajectory(self, node, prob):
        """Samples a trajectory from an HTM.
        """
        return 1


def debugging():
    ## Define the task
    mount_central = SequentialCombination([
        LeafCombination(PredAction(
            'Get central frame', (1, 0, 0, 0, 0, 0, 0, 0, 0))),
        LeafCombination(PredAction(
            'Start Hold central frame', (1, 1, 0, 0, 0, 0, 0, 0, 0)))],
        name='Mount central frame')
    mount_legs = ParallelCombination([
        SequentialCombination([
            LeafCombination(PredAction(
                'Get left leg', (1, 1, 1, 0, 0, 0, 0, 0, 0))),
            LeafCombination(PredAction(
                'Snap left leg', (1, 1, 1, 1, 0, 0, 0, 0, 0))),
        ], name='Mount left leg'),
        SequentialCombination([
            LeafCombination(PredAction(
                'Get right leg', (1, 1, 1, 1, 1, 0, 0, 0, 0))),
            LeafCombination(PredAction(
                'Snap right leg', (1, 1, 1, 1, 1, 1, 0, 0, 0))),
        ], name='Mount right leg'),
    ],
        name='Mount legs')
    release_central = LeafCombination(
        PredAction('Release central frame', (1, 1, 1, 1, 1, 1, 1, 0, 0)))
    mount_top = SequentialCombination([
        LeafCombination(PredAction('Get top', (1, 1, 1, 1, 1, 1, 1, 1, 0))),
        LeafCombination(PredAction('Snap top', (1, 1, 1, 1, 1, 1, 1, 1, 1)))],
        name='Mount top')

    # mount_central_alt = mount_central.to_alternative()
    # mount_legs_alt = mount_legs.to_alternative()
    # print(mount_central_alt.children)
    # print(mount_legs_alt.children)

    mount_legs_alt = mount_legs.to_alternative()

    chair_task_root = SequentialCombination(
        [mount_central, mount_legs, release_central, mount_top], name='Mount chair')

    test_parallel = ParallelCombination([mount_top, release_central])
    test_seq = SequentialCombination([mount_top, release_central])

    chair_task = HierarchicalTask(root=chair_task_root)
    chair_task_parallel = HierarchicalTask(root=test_parallel)
    chair_task_seq = HierarchicalTask(root=test_seq)

    chair_task_dict = chair_task.as_dictionary()
    # print("yay")
    # print(chair_task_dict)

    test = 1

    leaf = LeafCombination(PredAction(
        'l', (1, 0, 0, 0, 0, 0, 0, 0, 0)))
    simple_task_leaf = HierarchicalTask(root=leaf)

    a = LeafCombination(PredAction(
        'a', (1, 0, 0, 0, 0, 0, 0, 0, 0)))
    a1 = LeafCombination(PredAction(
        'a1', (1, 0, 0, 0, 0, 0, 0, 0, 0)))
    a2 = LeafCombination(PredAction(
        'a2', (1, 0, 0, 0, 0, 0, 0, 0, 0)))
    a11 = LeafCombination(PredAction(
        'a1', (1, 0, 0, 0, 0, 0, 0, 0, 0)))
    b = LeafCombination(PredAction(
        'b', (1, 0, 0, 0, 0, 0, 0, 0, 0)))
    c = LeafCombination(PredAction(
        'c', (1, 0, 0, 0, 0, 0, 0, 0, 0)))
    ab = SequentialCombination([a, b])
    a1a2 = SequentialCombination([a1, a2])
    a1a2a11 = ParallelCombination([a1a2, a11])

    alt_a = LeafCombination(AbstractAction('a'))
    alt_b = LeafCombination(AbstractAction('b'))
    alt_c = LeafCombination(AbstractAction('c'))
    alt_p1 = ParallelCombination([alt_a, alt_b])
    alt_p2 = ParallelCombination([alt_p1, alt_c])
    alt_alt = alt_p2.to_alternative()
    alt_probs = AlternativeCombination([a, b, c])

    simple_task_parallel = HierarchicalTask(root=ParallelCombination([a, b, c]))
    simple_task_parallel2 = HierarchicalTask(root=ParallelCombination([a, b]))
    simple_task_seq = HierarchicalTask(root=SequentialCombination([a, b, c]))
    simple_task_alt = HierarchicalTask(root=AlternativeCombination([a, b, c]))
    two_level_task1 = HierarchicalTask(root=ParallelCombination([ab, c]))
    two_level_task2 = HierarchicalTask(root=AlternativeCombination([SequentialCombination([a, b]), c]))
    two_level_task3 = HierarchicalTask(root=SequentialCombination([SequentialCombination([a, b]), c]))
    two_level_task4 = HierarchicalTask(root=AlternativeCombination([c, SequentialCombination([a, b])]))
    two_level_task5 = HierarchicalTask(root=ParallelCombination([ParallelCombination([a, b]), c]))
    three_level_task1 = HierarchicalTask(root=ParallelCombination([SequentialCombination([a1a2, b]), c]))
    three_level_task2 = HierarchicalTask(root=SequentialCombination([SequentialCombination([a1a2, b]), c]))
    four_level_task1 = HierarchicalTask(root=SequentialCombination([SequentialCombination([a1a2a11, b]), c]))

    #chair_task.gen_all_trajectories()
    # chair_task_parallel.gen_all_trajectories()
    # chair_task_seq.gen_all_trajectories()
    simple_task_leaf.gen_all_trajectories()
    #simple_task_parallel2.gen_all_trajectories()
    #simple_task_parallel.gen_all_trajectories()
    simple_task_seq.gen_all_trajectories()
    simple_task_alt.gen_all_trajectories()
    two_level_task1.gen_all_trajectories()
    two_level_task2.gen_all_trajectories()
    two_level_task3.gen_all_trajectories()
    two_level_task4.gen_all_trajectories()
    two_level_task5.gen_all_trajectories()
    three_level_task1.gen_all_trajectories()
    three_level_task2.gen_all_trajectories()
    four_level_task1.gen_all_trajectories()
    print("---")
    # print("Length of generated trajectories ", len(chair_task.all_trajectories))
    # for traj in chair_task.all_trajectories:
    #    print(traj.leaf.name)


def main():
    debugging()


if __name__ == '__main__':
    main()

COMBINATION_CLASSES = {'sequence': SequentialCombination,
                       'parallel': ParallelCombination,
                       'alternative': AlternativeCombination,
                       }
