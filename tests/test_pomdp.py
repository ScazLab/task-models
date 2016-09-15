import os
import io
from unittest import TestCase

import numpy as np

from htm.lib.pomdp import (parse_value_function, parse_policy_graph, POMDP,
                           _dump_list, _dump_1d_array, _dump_2d_array,
                           _dump_3d_array, _dump_4d_array, GraphPolicy)


TEST_VF = os.path.join(os.path.dirname(__file__), 'samples/example.alpha')
TEST_PG = os.path.join(os.path.dirname(__file__), 'samples/example.pg')
TEST_POMDP = os.path.join(os.path.dirname(__file__), 'samples/dump.pomdp')


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


class TestDumpArrays(TestCase):

    def test_dump_list_string(self):
        s = _dump_list(['a', 'bla', 'blah'])
        self.assertEqual(s, 'a bla blah')

    def test_dump_list_ints(self):
        s = _dump_list([1, 2, 3])
        self.assertEqual(s, '1 2 3')

    def test_dump_1d_array(self):
        s = _dump_1d_array(np.array([1, 2.3, 0.1234567]))
        self.assertEqual(s, '1.00000 2.30000 0.12346')

    def test_dump_2d_array(self):
        s = _dump_2d_array(np.array([[1, 2.2, 3],
                                     [4,   5, 6]]))
        self.assertEqual(s, '1.00000 2.20000 3.00000\n'
                            '4.00000 5.00000 6.00000')

    def test_dump_3d_array(self):
        s = _dump_3d_array(np.array([[[1, 2.2,  3],
                                      [4,   5,  6]],
                                     [[7,   8,  9],
                                      [10, 11, 12]]]),
                           'NAME', ['first', 'second'])
        self.assertEqual(s, 'NAME : first\n'
                            '1.00000 2.20000 3.00000\n'
                            '4.00000 5.00000 6.00000\n'
                            'NAME : second\n'
                            '7.00000 8.00000 9.00000\n'
                            '10.00000 11.00000 12.00000')

    def test_dump_4d_array(self):
        s = _dump_4d_array(
            1 + np.array(range(2 * 3 * 2 * 3)).reshape((2, 3, 2, 3)),
            'NAME', ['first', 'second'], ['a', 'b', 'c'])
        self.assertEqual(s, 'NAME : first : a\n'
                            '1.00000 2.00000 3.00000\n'
                            '4.00000 5.00000 6.00000\n'
                            'NAME : first : b\n'
                            '7.00000 8.00000 9.00000\n'
                            '10.00000 11.00000 12.00000\n'
                            'NAME : first : c\n'
                            '13.00000 14.00000 15.00000\n'
                            '16.00000 17.00000 18.00000\n'
                            'NAME : second : a\n'
                            '19.00000 20.00000 21.00000\n'
                            '22.00000 23.00000 24.00000\n'
                            'NAME : second : b\n'
                            '25.00000 26.00000 27.00000\n'
                            '28.00000 29.00000 30.00000\n'
                            'NAME : second : c\n'
                            '31.00000 32.00000 33.00000\n'
                            '34.00000 35.00000 36.00000')


class TestPOMDP(TestCase):

    def setUp(self):
        s = 3
        a = 4
        o = 2
        self.T = np.random.dirichlet(np.ones((s,)), (a, s))
        self.O = np.random.dirichlet(np.ones((o,)), (a, s))
        self.R = np.random.random((a, s, s, o))
        self.start = np.random.dirichlet(np.ones((s)))

    def test_init_stores_list(self):
        p = POMDP(self.T, self.O, self.R, self.start, 1,
                  states=range(3),
                  actions=set(['a', 'b', 'c', 'd']),
                  observations=[True, False])
        self.assertIsInstance(p.states, list)
        self.assertIsInstance(p.actions, list)
        self.assertIsInstance(p.observations, list)

    def test_raises_ValueErorr_on_discount_gt_1(self):
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, 1.1,
                  states=range(3), actions=range(4), observations=range(2))

    def test_raises_ValueErorr_on_discount_lt_0(self):
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, -.5,
                  states=range(3), actions=range(4), observations=range(2))

    def test_raises_ValueErorr_on_wrong_shape(self):
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, 1.,
                  states=range(4), actions=range(4), observations=range(2))
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, 1.,
                  states=range(3), actions=range(2), observations=range(2))
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, 1.,
                  states=range(3), actions=range(4), observations=range(3))
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, np.random.dirichlet(np.ones((2))),
                  1., states=range(3), actions=range(4), observations=range(2))

    def test_ValueError_on_nonnormal(self):
        with self.assertRaises(ValueError):
            T = np.random.random((4, 3, 3))
            POMDP(T, self.O, self.R, self.start, 1.,
                  states=range(3), actions=range(4), observations=range(2))
        with self.assertRaises(ValueError):
            O = np.random.random((4, 3, 2))
            POMDP(self.T, O, self.R, self.start, 1.,
                  states=range(3), actions=range(4), observations=range(2))
        with self.assertRaises(ValueError):
            start = np.random.random((3,))
            POMDP(self.T, self.O, self.R, start, 1.,
                  states=range(3), actions=range(4), observations=range(2))

    def test_ValueError_on_duplicates(self):
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, 1.,
                  states=['a', 'b', 'a'])
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, 1.,
                  actions=['a', 'b', 'a', 'c'])
        with self.assertRaises(ValueError):
            POMDP(self.T, self.O, self.R, self.start, 1.,
                  observations=['a', 'a'])

    def test_default_states(self):
        p = POMDP(self.T, self.O, self.R, self.start, 1.,
                  actions=range(4), observations=range(2))
        self.assertEqual(p.states, list(range(3)))

    def test_default_actions(self):
        p = POMDP(self.T, self.O, self.R, self.start, 1.,
                  states=range(3), observations=range(2))
        self.assertEqual(p.actions, list(range(4)))

    def test_default_observations(self):
        p = POMDP(self.T, self.O, self.R, self.start, 1.,
                  states=range(3), actions=range(4))
        self.assertEqual(p.observations, list(range(2)))

    def test_non_default_states(self):
        states = list(range(3, 0, -1))
        p = POMDP(self.T, self.O, self.R, self.start, 1., states=states)
        self.assertEqual(p.states, states)

    def test_non_default_actions(self):
        actions = list(range(4, 0, -1))
        p = POMDP(self.T, self.O, self.R, self.start, 1., actions=actions)
        self.assertEqual(p.actions, actions)

    def test_non_default_observations(self):
        observations = ['a', 'b']
        p = POMDP(self.T, self.O, self.R, self.start, 1.,
                  observations=observations)
        self.assertEqual(p.observations, observations)

    def test_cost_is_negated_in_R(self):
        p = POMDP(self.T, self.O, self.R, self.start, 1., values='cost')
        np.testing.assert_array_equal(p.R, -self.R)

    def test_reward_is_not_negated_in_R(self):
        p = POMDP(self.T, self.O, self.R, self.start, 1., values='reward')
        np.testing.assert_array_equal(p.R, self.R)

    def test_dump_pomdp(self):
        self.maxDiff = None
        T = np.ones((3, 2, 2)) / 2.
        O = np.zeros((3, 2, 2))
        O[:, :, 1] = 1.
        R = np.array(range(24)).reshape((3, 2, 2, 2))
        start = np.array([.2, .8])
        p = POMDP(T, O, R, start, 1.,
                  values='reward',
                  states=['x', 'y'],
                  actions=['a', 'b', 'c'],
                  observations=['o', 'p'])
        s = p.dump()
        with open(TEST_POMDP, 'r') as f:
            correct = f.read().rstrip('\n')  # Remove ending newline
        self.assertEqual(s, correct)

    def test_solver_runs(self):
        p = POMDP(self.T, self.O, self.R, self.start, .8)
        p.solve(n_iterations=5, timeout=1)

    def test_solver_runs_grid(self):
        p = POMDP(self.T, self.O, self.R, self.start, .8)
        p.solve(n_iterations=5, timeout=1, method='grid')

    def test_belief_update_is_proba(self):
        p = POMDP(self.T, self.O, self.R, self.start, .8)
        b = np.random.dirichlet([1, 1, 1])
        c = p.belief_update(np.random.randint(4), np.random.randint(2), b)
        self.assertTrue((c >= 0).all())
        self.assertAlmostEqual(c.sum(), 1.)

    def test_belief_update_from_full_obs(self):
        O = np.zeros((4, 3, 3))
        O[...] = np.eye(3)
        R = np.zeros((4, 3, 3, 3))
        p = POMDP(self.T, O, R, self.start, .8)
        a = np.random.randint(4)
        o = np.random.randint(3)
        b = np.array([.3, .3, .4])
        c = p.belief_update(a, o, b)
        cc = np.zeros((3,))
        cc[o] = 1.
        np.testing.assert_array_equal(c, cc)

    def test_belief_update_from_uniform_obs_is_T(self):
        O = np.ones((4, 3, 2)) * .5
        p = POMDP(self.T, O, self.R, self.start, .8)
        a = np.random.randint(4)
        o = np.random.randint(2)
        s = np.random.randint(3)
        b = np.zeros((3,))
        b[s] = 1.
        c = p.belief_update(a, o, b)
        np.testing.assert_allclose(c, self.T[a, s, :])


class TestPolicy(TestCase):

    def setUp(self):
        self.a = ['a', 'b', 'a', 'c', 'c']
        self.o = ['d', 'e']
        self.t = [[0, 1], [2, 3], [4, 4], [3, 2], [1, 0]]
        self.v = np.random.random((5, 12))
        self.i = np.random.randint(5)

    def test_init_with_start(self):
        i = np.argmax(np.square(self.v).sum(-1))  # get the one with max norm
        p = GraphPolicy(self.a, self.o, self.t, self.v, start=self.v[i])
        # so that it also has maximum scalar product with itself
        self.assertEqual(p.init, i)

    def test_init_with_init(self):
        p = GraphPolicy(self.a, self.o, self.t, self.v, init=self.i)
        self.assertEqual(p.init, self.i)

    def test_init_fails_without_init_or_start(self):
        with self.assertRaises(ValueError):
            GraphPolicy(self.a, self.o, self.t, self.v)

    def test_save_load(self):
        pol = GraphPolicy(self.a, self.o, self.t, self.v, init=self.i)
        dump = pol.to_json()
        try:
            p = GraphPolicy.from_json(io.StringIO(dump))
        except TypeError:  # Quick hack for python2
            p = GraphPolicy.from_json(io.StringIO(dump.decode()))
        self.assertEqual(self.a, p.actions)
        self.assertEqual(self.o, p.observations)
        np.testing.assert_allclose(pol.transitions, p.transitions)
        np.testing.assert_allclose(pol.values, p.values)
        self.assertEqual(self.i, p.init)
