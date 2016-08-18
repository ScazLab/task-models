import os
from unittest import TestCase

import numpy as np

from htm.lib.pomdp import parse_value_function, parse_policy_graph


TEST_VF = os.path.join(os.path.dirname(__file__), 'samples/example.alpha')
TEST_PG = os.path.join(os.path.dirname(__file__), 'samples/example.pg')


class TestParseValueFunction(TestCase):

    def test_parses_ok(self):
        correct_actions = [1, 0, 2, 0, 0]
        correct_vectors = np.array([[-10.5,  0.,  0.0],
                                    [- 9.4, -1., -1.0],
                                    [- 8.5, -2., -2.0],
                                    [- 8.2, -3., -3.1],
                                    [- 8.1, -4., -4.3]])
        with open(TEST_VF, 'r') as f:
            actions, vectors = parse_value_function(f)
            self.assertEqual(actions, correct_actions)
            np.testing.assert_array_equal(np.vstack(vectors), correct_vectors)


class TestParsePolicyGraph(TestCase):

    def test_parses_ok(self):
        correct_actions = [1, 0, 2, 0, 0]
        correct_transitions = [[0,    4,    0],
                               [0, None, None],
                               [0,    4,    0],
                               [2, None, None],
                               [3, None, None]]
        with open(TEST_PG, 'r') as f:
            actions, transitions = parse_policy_graph(f)
            self.assertEqual(actions, correct_actions)
            self.assertEqual(transitions, correct_transitions)
