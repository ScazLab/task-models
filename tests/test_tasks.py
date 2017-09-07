from unittest import TestCase
from types import GeneratorType

import numpy as np

from task_models.action import Condition, PrePostConditionAction
from task_models.state import NDimensionalState
from task_models.task import (check_path, split_path, TaskGraph,
                              ConjugateTaskGraph, AbstractAction,
                              ParallelCombination, LeafCombination,
                              AlternativeCombination, SequentialCombination,
                              max_cliques)


class DummyState(NDimensionalState):
    """State representing an int as an array of size dim (default to 6)
    of its binary expansion.
    """

    dim = 6

    def __init__(self, id_):
        self.id_ = id_
        super(DummyState, self).__init__(DummyState.to_array(id_))

    def __hash__(self):
        return self.id_

    def __eq__(self, other):
        return isinstance(other, DummyState) and self.id_ == other.id_

    def __repr__(self):
        return "DummyState<{}>".format(str(self))

    def __str__(self):
        return self.to_str(self.id_)

    @classmethod
    def to_array(cls, i):
        return np.array(list(cls.to_str(i)), dtype='b')

    @classmethod
    def to_str(cls, i):
        if i >= 2 ** cls.dim:
            raise ValueError('State id too big')
        return '{:0{dim}b}'.format(i, dim=cls.dim)


class TestDummyState(TestCase):

    def test_raises_value_error(self):
        with self.assertRaises(ValueError):
            DummyState.to_array(64)

    def test_to_array(self):
        np.testing.assert_array_equal(DummyState.to_array(3),
                                      np.array([0, 0, 0, 0, 1, 1]))

    def test_state(self):
        np.testing.assert_array_equal(DummyState(3).get_features(),
                                      np.array([0, 0, 0, 0, 1, 1]))


def get_action(pre, post, name=""):
    """Shortcut to get action from conditions represented as pairs of int.
    """
    return PrePostConditionAction(
        Condition(DummyState.to_array(pre[0]),
                  DummyState.to_array(pre[1])),
        Condition(DummyState.to_array(post[0]),
                  DummyState.to_array(post[1])),
        name=name)


class TestPathCheck(TestCase):

    def test_empty_path_is_valid(self):
        self.assertTrue(check_path([]))

    def test_singleton_state_is_valid(self):
        self.assertTrue(check_path([DummyState(2)]))

    def test_singleton_action_is_not_valid(self):
        self.assertFalse(check_path([get_action((1, 0), (1, 1))]))

    def test_must_finish_with_state(self):
        path = [DummyState(0),
                get_action((0, 0), (0, 0)),  # Catchall action
                ]
        self.assertFalse(check_path(path))

    def test_must_start_with_state(self):
        path = [get_action((0, 0), (0, 0)),  # Catchall action
                DummyState(0),
                ]
        self.assertFalse(check_path(path))

    def test_must_alternate(self):
        path = [DummyState(0),
                DummyState(0),
                get_action((0, 0), (0, 0)),  # Catchall action
                DummyState(0),
                ]
        self.assertFalse(check_path(path))

    def test_must_alternate2(self):
        path = [DummyState(0),
                get_action((0, 0), (0, 0)),  # Catchall action
                get_action((0, 0), (0, 0)),
                DummyState(0),
                ]
        self.assertFalse(check_path(path))

    def test_must_validate_pre_condition(self):
        path = [DummyState(1),
                get_action((1, 0), (0, 0)),  # Catchall pre
                DummyState(0),
                ]
        self.assertFalse(check_path(path))

    def test_must_validate_post_condition(self):
        path = [DummyState(1),
                get_action((0, 0), (1, 1)),  # Catchall post
                DummyState(0),
                ]
        self.assertFalse(check_path(path))

    def test_must_validate_pre_condition2(self):
        path = [DummyState(1),
                get_action((0, 0), (0, 0)),  # Catchall action
                DummyState(1),
                get_action((1, 0), (0, 0)),  # Catchall pre
                DummyState(0),
                ]
        self.assertFalse(check_path(path))

    def test_must_validate_post_condition2(self):
        path = [DummyState(1),
                get_action((0, 0), (0, 0)),  # Catchall action
                DummyState(1),
                get_action((0, 0), (1, 1)),  # Catchall post
                DummyState(0),
                ]
        self.assertFalse(check_path(path))

    def test_path_is_valid(self):
        path = [DummyState(0),
                get_action((1, 0), (1, 1)),
                DummyState(1 + 4),
                get_action((1 + 2, 1), (1 + 2, 1 + 2)),
                DummyState(1 + 2 + 4),
                ]
        self.assertTrue(check_path(path))


class TestSplitPath(TestCase):

    def test_split_empty_path(self):
        self.assertEqual(split_path([]), [])

    def test_split1(self):
        self.assertEqual(split_path([1]), [])

    def test_split2(self):
        self.assertEqual(split_path([1, 2]), [])

    def test_split3(self):
        self.assertEqual(split_path([1, 2, 3]), [(1, 2, 3)])

    def test_split5(self):
        self.assertEqual(split_path(range(1, 6)), [(1, 2, 3), (3, 4, 5)])

    def test_split7(self):
        self.assertEqual(split_path(range(1, 8)),
                         [(1, 2, 3), (3, 4, 5), (5, 6, 7)])


class TestMaxClique(TestCase):

    def setUp(self):
        self.g = {1: set([2, 5]),
                  2: set([1, 3, 5]),
                  3: set([2, 4]),
                  4: set([3, 5, 6]),
                  5: set([1, 2, 4]),
                  6: set([4]),
                  }

    def test_max_cliques(self):
        cliques = max_cliques(self.g)
        self.assertIsInstance(cliques, GeneratorType)
        cliques = list(cliques)
        self.assertEqual(len(cliques), 5)
        self.assertIn(set([1, 2, 5]), cliques)
        self.assertIn(set([2, 3]), cliques)
        self.assertIn(set([3, 4]), cliques)
        self.assertIn(set([4, 5]), cliques)
        self.assertIn(set([4, 6]), cliques)


class TestTaskGraph(TestCase):

    def setUp(self):
        self.path = [DummyState(0),
                     get_action((1, 0), (1, 1)),
                     DummyState(1 + 4),
                     get_action((1 + 2, 1), (1 + 2, 1 + 2)),
                     DummyState(1 + 2 + 4),
                     ]
        self.graph = TaskGraph()

    def test_add_transition(self):
        self.assertFalse(self.graph.has_transition(
            self.path[0], self.path[1], self.path[2]))
        self.graph.add_transition(*self.path[:3])
        self.assertTrue(self.graph.has_transition(
            self.path[0], self.path[1], self.path[2]))

    def test_add_path(self):
        self.assertFalse(self.graph.has_transition(*self.path[:3]))
        self.assertFalse(self.graph.has_transition(*self.path[2:5]))
        self.graph.add_path(self.path)
        self.assertTrue(self.graph.has_transition(*self.path[:3]))
        self.assertTrue(self.graph.has_transition(*self.path[2:5]))

    def test_always_has_empty_path(self):
        self.assertTrue(self.graph.has_path([]))
        self.graph.add_path(self.path)
        self.assertTrue(self.graph.has_path([]))

    def test_has_singleton_path(self):
        s = DummyState(8)
        self.assertFalse(self.graph.has_path([DummyState(8)]))
        self.graph.initial.add(s)  # Needs to be initial
        self.assertFalse(self.graph.has_path([DummyState(8)]))
        self.graph.terminal.add(s)  # Needs to be terminal
        self.assertTrue(self.graph.has_path([DummyState(8)]))
        self.graph.add_path(self.path)  # Should not matter
        self.assertTrue(self.graph.has_path([DummyState(8)]))
        # Final but not initial:
        self.assertFalse(self.graph.has_path([DummyState(1 + 2 + 4)]))
        # Should be false too
        self.assertFalse(self.graph.has_path([DummyState(1 + 4)]))

    def test_invalid_paths_raises_error(self):
        self.graph.add_path(self.path)
        with self.assertRaises(ValueError):
            self.graph.has_path([
                DummyState(0),
                get_action((1 + 2, 0), (1 + 2, 1)),
                DummyState(1 + 2 + 4),
            ])

    def test_must_have_same_transitions(self):
        path = [DummyState(0),
                get_action((1, 0), (1, 1)),
                DummyState(1),  # Different intermediate state
                get_action((1 + 2, 1), (1 + 2, 1 + 2)),
                DummyState(1 + 2 + 4),
                ]
        self.graph.add_path(self.path)
        self.assertFalse(self.graph.has_path(path))
        self.graph.add_path(path)
        self.assertTrue(self.graph.has_path(path))

    def test_must_have_same_transitions2(self):
        path = [DummyState(0),
                get_action((1, 0), (1, 1)),
                DummyState(1 + 4),
                get_action((1 + 2, 1), (1 + 4, 1 + 4)),  # Different action
                DummyState(1 + 2 + 4),
                ]
        self.graph.add_path(self.path)
        self.assertFalse(self.graph.has_path(path))
        self.graph.add_path(path)
        self.assertTrue(self.graph.has_path(path))

    def test_is_not_deterministic(self):
        path = [DummyState(0),
                get_action((1, 0), (1, 1)),
                DummyState(1),  # Different intermediate state
                get_action((1 + 2, 1), (1 + 2, 1 + 2)),
                DummyState(1 + 2 + 4),
                ]
        self.graph.add_path(self.path)
        self.graph.add_path(path)
        with self.assertRaises(ValueError):
            self.graph.check_only_deterministic_transitions()

    def test_empty_is_deterministic(self):
        self.graph.check_only_deterministic_transitions()

    def test_deterministic(self):
        self.graph.add_path(self.path)
        self.graph.check_only_deterministic_transitions()

    def test_deterministic2(self):
        self.graph.add_path(self.path)
        self.graph.add_path([
            DummyState(0),
            get_action((1, 0), (1, 1)),
            DummyState(1 + 4),
            get_action((1 + 2, 1), (1 + 4, 1 + 4)),  # Different action
            DummyState(1 + 2 + 4),
        ])
        self.graph.check_only_deterministic_transitions()

    def test_remove_transition(self):
        self.graph.add_path(self.path)
        t1 = self.path[:3]
        self.assertTrue(self.graph.has_transition(*t1))
        self.graph.remove_transition(*t1)
        self.assertFalse(self.graph.has_transition(*t1))
        with self.assertRaises(KeyError):
            self.graph.remove_transition(*t1)

    def test_compact_node_to_itself_is_same(self):
        self.graph.add_path(self.path)
        trans = [self.path[:3], self.path[2:]]
        trans[0][1] = ''  # replaces label
        trans = [tuple(t) for t in trans]
        self.graph.compact([self.path[0]], self.path[0])
        self.assertEqual(list(self.graph.all_transitions()), trans)

    def test_compact_all_nodes_is_empty(self):
        self.graph.add_path(self.path)
        self.graph.compact([self.path[i] for i in [0, 2, 4]], None)
        self.assertEqual(list(self.graph.all_transitions()), [])

    def test_compacts_two(self):
        self.graph.add_path(self.path)
        self.graph.compact([self.path[i] for i in [0, 2]], None)
        trans = [(None, '', self.path[4])]
        self.assertEqual(list(self.graph.all_transitions()), trans)


class TestConjugateTaskGraph(TestCase):

    def setUp(self):
        self.s0 = DummyState(0)
        self.s1 = DummyState(1 + 4)
        self.s2 = DummyState(1 + 2 + 4)
        self.a0 = get_action((1, 0), (1, 1))
        self.a1 = get_action((1 + 2, 1), (1 + 2, 1 + 2))

    def test_check_deterministic(self):
        graph = TaskGraph()
        graph.add_path([self.s0, self.a0, self.s1, self.a1, self.s2])
        graph.add_path([self.s0, self.a0,
                        DummyState(1),  # Different intermediate state
                        self.a1, self.s2,
                        ])
        with self.assertRaises(ValueError):
            graph.conjugate()

    def test_conjugate_empty_is_empty(self):
        self.assertEqual(TaskGraph().conjugate(), ConjugateTaskGraph())

    def test_conjugate_chain(self):
        graph = TaskGraph()
        graph.add_path([self.s0, self.a0, self.s1, self.a1, self.s2])
        c = ConjugateTaskGraph()
        c.add_transition(c.initial, self.s0, self.a0)
        c.add_transition(self.a0, self.s1, self.a1)
        c.add_transition(self.a1, self.s2, c.terminal)
        self.assertEqual(graph.conjugate(), c)

    def test_conjugate_chair(self):
        HAVE_LPEG = 1
        HAVE_RPEG = 2
        PLACED_LPEG = 4
        PLACED_RPEG = 8
        HAVE_FRAME = 16
        PLACED_FRAME = 32
        # states
        init = DummyState(0)
        final = DummyState(PLACED_LPEG + PLACED_RPEG + PLACED_FRAME)
        # actions
        get_lpeg = get_action((HAVE_LPEG, 0), (HAVE_LPEG, HAVE_LPEG),
                              name='get left peg')
        get_rpeg = get_action((HAVE_RPEG, 0), (HAVE_RPEG, HAVE_RPEG),
                              name='get right peg')
        place_lpeg = get_action((HAVE_LPEG + PLACED_LPEG, HAVE_LPEG),
                                (HAVE_LPEG + PLACED_LPEG, PLACED_LPEG),
                                name='place left peg')
        place_rpeg = get_action((HAVE_RPEG + PLACED_RPEG, HAVE_RPEG),
                                (HAVE_RPEG + PLACED_RPEG, PLACED_RPEG),
                                name='place right peg')
        get_frame = get_action((HAVE_FRAME, 0), (HAVE_FRAME, HAVE_FRAME),
                               name='get frame')
        place_frame = get_action(
            (HAVE_FRAME + PLACED_FRAME + PLACED_LPEG + PLACED_RPEG,
             HAVE_FRAME + PLACED_LPEG + PLACED_RPEG),
            (HAVE_FRAME + PLACED_FRAME + PLACED_LPEG + PLACED_RPEG,
             PLACED_FRAME + PLACED_LPEG + PLACED_RPEG),
            name='place frame')
        graph = TaskGraph()
        # add paths
        graph.add_path([
            init,
            get_lpeg, DummyState(HAVE_LPEG),
            place_lpeg, DummyState(PLACED_LPEG),
            get_rpeg, DummyState(PLACED_LPEG + HAVE_RPEG),
            place_rpeg, DummyState(PLACED_LPEG + PLACED_RPEG),
            get_frame, DummyState(PLACED_LPEG + PLACED_RPEG + HAVE_FRAME),
            place_frame, final])
        graph.add_path([
            init,
            get_rpeg, DummyState(HAVE_RPEG),
            place_rpeg, DummyState(PLACED_RPEG),
            get_lpeg, DummyState(PLACED_RPEG + HAVE_LPEG),
            place_lpeg, DummyState(PLACED_RPEG + PLACED_LPEG),
            get_frame, DummyState(PLACED_LPEG + PLACED_RPEG + HAVE_FRAME),
            place_frame, final])
        # expected conjugate
        cgraph = ConjugateTaskGraph()
        cgraph.add_transition(cgraph.initial, init, get_lpeg)
        cgraph.add_transition(get_lpeg, DummyState(HAVE_LPEG), place_lpeg)
        cgraph.add_transition(get_lpeg,
                              DummyState(PLACED_RPEG + HAVE_LPEG),
                              place_lpeg),
        cgraph.add_transition(place_lpeg, DummyState(PLACED_LPEG), get_rpeg),
        cgraph.add_transition(place_lpeg,
                              DummyState(PLACED_RPEG + PLACED_LPEG),
                              get_frame)
        cgraph.add_transition(cgraph.initial, init, get_rpeg)
        cgraph.add_transition(get_rpeg, DummyState(HAVE_RPEG), place_rpeg)
        cgraph.add_transition(get_rpeg,
                              DummyState(PLACED_LPEG + HAVE_RPEG),
                              place_rpeg),
        cgraph.add_transition(place_rpeg, DummyState(PLACED_RPEG), get_lpeg),
        cgraph.add_transition(place_rpeg,
                              DummyState(PLACED_LPEG + PLACED_RPEG),
                              get_frame)
        cgraph.add_transition(get_frame,
                              DummyState(PLACED_LPEG + PLACED_RPEG +
                                         HAVE_FRAME),
                              place_frame)
        cgraph.add_transition(place_frame, final, cgraph.terminal)
        self.assertEqual(graph.conjugate(), cgraph)

    def test_get_max_chains_on_chain(self):
        graph = TaskGraph()
        graph.add_path([self.s0, self.a0, self.s1, self.a1, self.s2])
        c = graph.conjugate()
        chains = c.get_max_chains()
        self.assertIsInstance(chains, GeneratorType)
        self.assertEqual(list(chains),
                         [[c.initial, self.a0, self.a1, c.terminal]])

    def test_get_max_chains_on_clique(self):
        graph = TaskGraph()
        a0 = get_action((1, 0), (1, 1))
        a1 = get_action((2, 0), (2, 2))
        a2 = get_action((4, 0), (4, 4))
        si = DummyState(0)
        sf = DummyState(1 + 2 + 4)
        graph.add_path([si, a0, DummyState(1), a1, DummyState(1 + 2), a2, sf])
        graph.add_path([si, a0, DummyState(1), a2, DummyState(1 + 4), a1, sf])
        graph.add_path([si, a1, DummyState(2), a0, DummyState(1 + 2), a2, sf])
        graph.add_path([si, a1, DummyState(2), a2, DummyState(2 + 4), a0, sf])
        graph.add_path([si, a2, DummyState(4), a0, DummyState(1 + 4), a1, sf])
        graph.add_path([si, a2, DummyState(4), a1, DummyState(2 + 4), a0, sf])
        c = graph.conjugate()
        chains = c.get_max_chains()
        self.assertIsInstance(chains, GeneratorType)
        self.assertEqual(list(chains), [])

    def test_get_two_max_chains(self):
        graph = TaskGraph()
        a0 = get_action((1, 0), (1, 1), name='a0')
        a1 = get_action((2, 0), (2, 2), name='a1')
        b0 = get_action((4, 0), (4, 4), name='b0')
        b1 = get_action((8, 0), (8, 8), name='b0')
        si = DummyState(0)
        sf = DummyState(1 + 2 + 4 + 8)
        graph.add_path([si, a0, DummyState(1), a1, DummyState(1 + 2),
                        b0, DummyState(1 + 2 + 4), b1, sf])
        graph.add_path([si, b0, DummyState(4), b1, DummyState(4 + 8),
                        a0, DummyState(1 + 4 + 8), a1, sf])
        c = graph.conjugate()
        chains = c.get_max_chains()
        self.assertIsInstance(chains, GeneratorType)
        chains = list(chains)
        self.assertEqual(len(chains), 2)
        self.assertIn([a0, a1], chains)
        self.assertIn([b0, b1], chains)
        # (accounts for all orders)

    def test_get_max_clique_on_chain(self):
        graph = TaskGraph()
        graph.add_path([self.s0, self.a0, self.s1, self.a1, self.s2])
        c = graph.conjugate()
        chains = c.get_max_cliques()
        self.assertIsInstance(chains, GeneratorType)
        self.assertEqual(list(chains), [])

    def test_get_max_clique_on_clique(self):
        graph = TaskGraph()
        a0 = get_action((1, 0), (1, 1), name='0')
        a1 = get_action((2, 0), (2, 2), name='1')
        a2 = get_action((4, 0), (4, 4), name='2')
        si = DummyState(0)
        sf = DummyState(1 + 2 + 4)
        graph.add_path([si, a0, DummyState(1), a1, DummyState(1 + 2), a2, sf])
        graph.add_path([si, a0, DummyState(1), a2, DummyState(1 + 4), a1, sf])
        graph.add_path([si, a1, DummyState(2), a0, DummyState(1 + 2), a2, sf])
        graph.add_path([si, a1, DummyState(2), a2, DummyState(2 + 4), a0, sf])
        graph.add_path([si, a2, DummyState(4), a0, DummyState(1 + 4), a1, sf])
        graph.add_path([si, a2, DummyState(4), a1, DummyState(2 + 4), a0, sf])
        c = graph.conjugate()
        chains = c.get_max_cliques()
        self.assertIsInstance(chains, GeneratorType)
        self.assertEqual(list(chains), [set([a0, a1, a2])])


class TestParallelToAlternatives(TestCase):

    def test_is_correct(self):
        a = LeafCombination(AbstractAction('a'))
        b = LeafCombination(AbstractAction('b'))
        c = LeafCombination(AbstractAction('c'))
        p = ParallelCombination([a, b, c])
        alt = p.to_alternative()
        self.assertIsInstance(alt, AlternativeCombination)
        self.assertEqual(len(alt.children), 6)
        self.assertIsInstance(alt.children[0], SequentialCombination)
        self.assertTrue(all([len(c.children) == 3 for c in alt.children]))
