from unittest import TestCase

import numpy as np

from htm.action import Condition, Action
from htm.state import NDimensionalState
from htm.task import check_path, split_path, TaskGraph


class DummyState(NDimensionalState):

    dim = 5

    def __init__(self, id_):
        self.id_ = id_
        super().__init__(DummyState.to_array(id_))

    def __hash__(self):
        return self.id_

    def __eq__(self, other):
        return isinstance(other, DummyState) and self.id_ == other.id_

    @classmethod
    def to_array(cls, i):
        if i >= 2 ** cls.dim:
            raise ValueError('State id too big')
        return np.array(list('{:0{dim}b}'.format(i, dim=cls.dim)), dtype='b')


class TestDummyState(TestCase):

    def test_raises_value_error(self):
        with self.assertRaises(ValueError):
            DummyState.to_array(32)

    def test_to_array(self):
        np.testing.assert_array_equal(DummyState.to_array(3),
                                      np.array([0, 0, 0, 1, 1]))

    def test_state(self):
        np.testing.assert_array_equal(DummyState(3).get_features(),
                                      np.array([0, 0, 0, 1, 1]))


def get_action(pre, post):
    """Shortcut to get action from conditions represented as pairs of int.
    """
    return Action(Condition(DummyState.to_array(pre[0]),
                            DummyState.to_array(pre[1])),
                  Condition(DummyState.to_array(post[0]),
                            DummyState.to_array(post[1])))


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
