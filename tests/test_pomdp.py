import os
from numbers import Integral
from unittest import TestCase, skip

import numpy as np

from htm.lib.pomdp import (parse_value_function, parse_policy_graph, POMDP,
                           _dump_list, _dump_1d_array, _dump_2d_array,
                           _dump_3d_array, _dump_4d_array, GraphPolicy,
                           _SearchNode, _SearchObservationNode,
                           _SearchActionNode, _SearchTree, ArrayBelief,
                           ParticleBelief, POMCPPolicyRunner)


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

    def test_save_load(self):
        p = POMDP(self.T, self.O, self.R, self.start, .8)
        dump = p.as_json()
        pp = POMDP.from_json(dump)
        np.testing.assert_allclose(p.T, pp.T)
        np.testing.assert_allclose(p.O, pp.O)
        np.testing.assert_allclose(p.R, pp.R)
        np.testing.assert_allclose(p.start, pp.start)
        self.assertEqual(p.discount, pp.discount)
        self.assertEqual(p.states, pp.states)
        self.assertEqual(p.actions, pp.actions)
        self.assertEqual(p.observations, pp.observations)


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
        p = GraphPolicy.from_json(dump)
        self.assertEqual(self.a, p.actions)
        self.assertEqual(self.o, p.observations)
        np.testing.assert_allclose(pol.transitions, p.transitions)
        np.testing.assert_allclose(pol.values, p.values)
        self.assertEqual(self.i, p.init)


class TestSearchNode(TestCase):

    def setUp(self):
        self.node = _SearchNode(alpha=0.)

    def test_n_simulations_is_0(self):
        self.assertEqual(self.node.n_simulations, 0)

    def test_value_is_0(self):
        self.assertIsInstance(self.node._avg.total_value, float)
        self.assertEqual(self.node._avg.total_value, 0)
        self.assertIsInstance(self.node.value, float)
        self.assertEqual(self.node.value, 0)

    def test_n_simulation_increases_after_update(self):
        self.node.update(3)
        self.assertEqual(self.node.n_simulations, 1)
        self.node.update(2)
        self.assertEqual(self.node.n_simulations, 2)

    def test_value_is_float_after_update(self):
        self.node.update(3)
        self.assertIsInstance(self.node._avg.total_value, float)
        self.assertIsInstance(self.node.value, float)

    def test_value_is_average(self):
        self.node.update(1)
        self.node.update(2)
        self.assertEqual(self.node.value, 1.5)

    def test_str(self):
        child = _SearchNode()
        self.node.children[1] = child
        child.children[2] = _SearchNode()
        self.node.children[3] = _SearchNode()
        self.assertEqual(str(self.node), "[1: [2: []], 3: []]")


class TestSearchObservationNode(TestCase):

    class DummyBelief:

        def to_list(self):
            return []

    class DummyModel:

        actions = ['x', 'y', 'z', 'a', 'b', 'c', 'm', 'n', 'o', 'p']
        observations = ['a', 'b', 'c']

    def setUp(self):
        self.dummy_belief = object()
        self.n_actions = 10
        self.node = _SearchObservationNode(self.dummy_belief, self.n_actions)

    def test_safe_get_child_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.node.safe_get_child(11)

    def test_safe_get_child_init_child(self):
        child = self.node.safe_get_child(3)
        self.assertIsNot(child, None)
        self.assertIs(child, self.node.children[3])

    def test_get_best_action_is_unset(self):
        for i in range(9):
            c = self.node.safe_get_child(self.node.get_best_action())
            c.update(0)
        self.node._avg.n_simulations = 10
        unset = self.node.children.index(None)
        a = self.node.get_best_action()
        self.assertEqual(a, unset)

    def test_get_best_action_is_unexplored(self):
        for i in range(10):
            c = self.node.safe_get_child(i)
            if i != 3:
                c.update(0)
        self.node._avg.n_simulations = 9
        self.assertEqual(self.node.get_best_action(), 3)

    def test_get_best_action_is_best(self):
        for i in range(10):
            c = self.node.safe_get_child(i)
            c.update(0)
        self.node._avg.n_simulations = 10
        self.node.children[7].update(10)
        a = self.node.get_best_action()
        self.assertEqual(a, 7)

    def test_get_best_action_is_not_best(self):
        for i in range(10):
            c = self.node.safe_get_child(i)
            c.update(0)
            c.update(0)
        self.node.children[4]._avg.n_simulations = 1
        self.node.children[4]._avg.total_value = 9.3  # n_simu = 1
        self.node.children[7].update(30)  # n_simu = 3
        self.node._avg.n_simulations = 20  # log(20) is slightly less than 3
        a = self.node.get_best_action(exploration=1.)  # sqrt(3) > 1.72
        self.assertEqual(a, 4)

    def test_get_best_action_is_still_best_for_small_exploration(self):
        for i in range(10):
            c = self.node.safe_get_child(i)
            c.update(0)
            c.update(0)
        self.node.children[4]._avg.n_simulations = 1
        self.node.children[4]._avg.total_value = 9.3  # n_simu = 1
        self.node.children[7].update(30)  # n_simu = 3
        self.node._avg.n_simulations = 20  # log(20) is slightly less than 3
        a = self.node.get_best_action(exploration=.5)  # sqrt(3) > 1.72
        self.assertEqual(a, 7)

    def test_get_best_action_is_always_best_without_exploration(self):
        for i in range(10):
            c = self.node.safe_get_child(i)
            for i in range(1 + np.random.randint(9)):  # at least one sample
                c.update(10. * np.random.random())
        self.node._avg.n_simulations = sum([self.node.children[i].n_simulations
                                            for i in range(10)])
        best = np.argmax([self.node.children[i].value for i in range(10)])
        a = self.node.get_best_action()
        self.assertEqual(a, best)

    def test_str(self):
        self.node.safe_get_child(2).children[1] = _SearchNode()
        self.assertEqual(str(self.node), "[2: [1: []]]")

    def _node_with_best_action(self, ai):
        node = _SearchObservationNode(self.DummyBelief(), self.n_actions)
        for i in range(10):
            c = node.safe_get_child(i)
            c.update(0)
        node._avg.n_simulations = 10
        node.children[ai]._avg.n_simulations = 1
        node.children[ai]._avg.total_value = 10.
        return node

    @skip("outdated")
    def test_to_dict_no_child(self):
        node = self._node_with_best_action(1)
        self.assertEqual(node.to_dict(self.DummyModel()),
                         {'belief': [],
                          'action': 'y',
                          'node': None,
                          'observations': [],
                          'children': [],
                          })

    @skip("outdated")
    def test_to_dict_children(self):
        node = self._node_with_best_action(1)
        child = node.children[1]
        child.children[0] = self._node_with_best_action(2)
        child.children[2] = self._node_with_best_action(0)
        model = self.DummyModel()
        self.assertEqual(node.to_dict(model),
                         {'belief': [],
                          'action': 'y',
                          'actions': model.actions,
                          'node': None,
                          'observations': ['a', 'c'],
                          'children': [
                                {'belief': [],
                                 'action': 'z',
                                 'actions': model.actions,
                                 'node': None,
                                 'observations': [],
                                 'children': [],
                                 },
                                {'belief': [],
                                 'action': 'x',
                                 'actions': model.actions,
                                 'node': None,
                                 'observations': [],
                                 'children': [],
                                 },
                              ],
                          })


class _FakeModel:

    discount = .9

    def __init__(self, start, n_actions, n_observations):
        self.start = start
        self.n_states = len(self.start)
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.action = [a for a in range(self.n_actions)]
        self.reset()

    def sample_start(self):
        return np.random.choice(self.n_states, p=self.start)

    def sample_transition(self, action, state):
        self.transitions_history.append((action, state))
        return self.transitions.pop(0)

    def belief_update(self, action, observation, belief):
        self.successors_history.append((action, observation, belief))
        return self.successors.pop(0)

    def reset(self):
        self.transitions = []
        self.transitions_history = []
        self.successors = []
        self.successors_history = []


class BeliefBaseTest(object):

    class FakeModel:

        def __init__(self, successor_0, successor_1):
            self.sampled_on = []
            self.successors = [successor_0, successor_1]

        def belief_update(self, a, o, array):
            return self.successors[o]  # only accepts o = (0|1)

        def sample_transition(self, a, s):
            self.sampled_on.append((a, s))
            if np.random.random() < .5:
                o = 0
            else:
                o = 1
            return np.random.choice(3, p=self.successors[o]), o, -1.5

    def setUp(self):
        self.p = np.array([.7, 0., .3])

    def test_sample_is_int(self):
        self.assertIsInstance(self.belief.sample(), Integral)

    def test_sample_only_from_non_zeros(self):
        for i in range(10):
            self.assertNotEqual(self.p[self.belief.sample()], 0.)


class TestArrayBelief(BeliefBaseTest, TestCase):

    def setUp(self):
        super(TestArrayBelief, self).setUp()
        self.belief = ArrayBelief(self.p)

    def test_successor(self):
        p_succ = np.array([.6, .4, 0.])
        model = self.FakeModel(np.array([0., 0.1, 0.9]), p_succ)
        succ = self.belief.successor(model, 2, 1)
        self.assertIsInstance(succ, ArrayBelief)
        np.testing.assert_array_equal(succ.array, p_succ)


class TestParticleBelief(BeliefBaseTest, TestCase):

    def setUp(self):
        super(TestParticleBelief, self).setUp()

        def sampler():
            return int(np.random.choice(3, p=self.p))

        self.belief = ParticleBelief(sampler, 3, 10)

    def test_successor(self):
        p_succ = np.array([.6, .4, 0.])
        model = self.FakeModel(np.array([0., 0.1, 0.9]), p_succ)
        succ = self.belief.successor(model, 2, 1)
        self.assertIsInstance(succ, ParticleBelief)
        self.assertNotIn(2, succ.part_states)
        self.assertTrue(all([a == 2 for (a, _) in model.sampled_on]))
        self.assertTrue(all([s in self.belief.part_states
                             for (_, s) in model.sampled_on]))


class TestSearchTree(TestCase):

    def setUp(self):
        self.start = np.zeros((10,))
        self.start[-1] = 1.
        self.model = _FakeModel(self.start, 3, 2)
        self.tree = _SearchTree(self.model, 3, 1., node_params={'alpha': 0.})

    def test_belief_type(self):
        tree = _SearchTree(self.model, 3, 1., belief='array')
        self.assertIsInstance(tree.root.belief, ArrayBelief)
        tree = _SearchTree(self.model, 3, 1., belief='particle',
                           belief_params={'n_particles': 37})
        self.assertIsInstance(tree.root.belief, ParticleBelief)
        self.assertEqual(tree.root.belief.n_particles, 37)
        self.assertEqual(tree.root.belief.n_states, 10)

    def test_get_node(self):
        ca = self.tree.root.safe_get_child(0)
        co = _SearchObservationNode(ArrayBelief(self.start),
                                    self.model.n_actions)
        ca.children[5] = co
        cca = co.safe_get_child(1)
        cco = _SearchObservationNode(ArrayBelief(self.start),
                                     self.model.n_actions)
        cca.children[6] = cco
        self.assertIs(self.tree.get_node([0, 5, 1, 6]), cco)
        with self.assertRaises(ValueError):
            self.tree.get_node([0, 5, 1, 7])

    def test_rollout_from_node_with_horizon_0_is_0(self):
        self.assertEqual(self.tree.rollout_from_node(self.tree.root, 1, 0), 0)
        self.assertEqual(len(self.model.transitions_history), 0)

    def test_rollout_from_node_with_horizon_1_is_reward(self):
        r = 3.2
        self.model.transitions = [(1, 0, r)]
        self.assertEqual(self.tree.rollout_from_node(self.tree.root, 2, 1), r)
        self.assertEqual(len(self.model.transitions_history), 1)
        self.assertEqual(self.model.transitions_history[0][1], 2)

    def test_rollout_from_node_with_horizon_2(self):
        self.model.transitions = [(1, None, 11.), (2, None, 10.)]
        self.assertEqual(self.tree.rollout_from_node(self.tree.root, 3, 2), 20)
        self.assertEqual(len(self.model.transitions_history), 2)
        self.assertEqual(self.model.transitions_history[0][1], 3)
        self.assertEqual(self.model.transitions_history[1][1], 1)

    def test_simulate_from_node_with_horizon_0(self):
        self.tree.horizon = 0
        self.tree.simulate_from_node(self.tree.root)
        self.assertEqual(len(self.model.transitions_history), 0)
        self.assertEqual(str(self.tree.root), "[]")
        self.assertEqual(self.tree.root.n_simulations, 0)

    def test_simulate_from_node_with_horizon_1(self):
        self.model.transitions = [(1, 1, 11.)]
        belief2 = np.zeros((10))
        belief2[1] = 1.
        self.model.successors = [belief2]
        self.tree.horizon = 1
        self.tree.simulate_from_node(self.tree.root)
        self.assertEqual(len(self.model.transitions_history), 1)
        a = self.model.transitions_history[0][0]
        self.assertEqual(str(self.tree.root), "[{}: [1: []]]".format(a))
        self.assertEqual(self.tree.root.n_simulations, 1)

    def test_simulate_from_node_with_horizon_3(self):
        self.model.discount = 1.
        belief = np.zeros((10))
        belief[1] = 1.

        def ret_1(exploration=None, relative_exploration=None):
            return 1

        self.tree.horizon = 3
        # Always use action "1" at first
        self.tree.root.get_best_action = ret_1
        # First run
        self.model.transitions = [(1, 1, 11.), (2, 0, 13.), (4, 0, 0.)]
        self.model.successors = [belief]
        self.tree.simulate_from_node(self.tree.root)
        self.assertEqual(len(self.model.transitions_history), 3)
        self.assertEqual(str(self.tree.root), "[1: [1: []]]")
        self.assertEqual(self.tree.root.n_simulations, 1)
        self.assertEqual(self.tree.root._avg.total_value, 24.)
        self.assertEqual(self.tree.get_node([1])._avg.total_value, 24.)
        self.assertEqual(self.tree.get_node([1, 1])._avg.total_value, 13.)
        # Second run
        self.model.reset()
        self.model.transitions = [(1, 1, 3.), (2, 0, 1.), (4, 0, 5.)]
        self.model.successors = [belief]
        self.tree.simulate_from_node(self.tree.root)
        self.assertEqual(len(self.model.transitions_history), 3)
        a1 = self.model.transitions_history[1][0]
        self.assertEqual(str(self.tree.root),
                         "[1: [1: [{}: [0: []]]]]".format(a1))
        self.assertEqual(self.tree.root.n_simulations, 2)
        self.assertEqual(self.tree.root._avg.total_value, 33.)
        self.assertEqual(self.tree.get_node([1])._avg.total_value, 33.)
        self.assertEqual(self.tree.get_node([1, 1])._avg.total_value, 19.)
        self.assertEqual(self.tree.get_node([1, 1, a1])._avg.total_value, 6.)
        self.assertEqual(self.tree.get_node([1, 1, a1, 0])._avg.total_value, 5.)
        # Third run
        self.model.reset()
        self.model.transitions = [(1, 0, 2.), (2, 1, 3.), (4, 0, 4.)]
        self.model.successors = [belief]
        self.tree.simulate_from_node(self.tree.root)
        self.assertEqual(len(self.model.transitions_history), 3)
        self.assertEqual(str(self.tree.root),
                         "[1: [0: [], 1: [{}: [0: []]]]]".format(a1))
        self.assertEqual(self.tree.root.n_simulations, 3)
        self.assertEqual(self.tree.root._avg.total_value, 42.)
        self.assertEqual(self.tree.get_node([1])._avg.total_value, 42.)
        self.assertEqual(self.tree.get_node([1, 1])._avg.total_value, 19.)
        self.assertEqual(self.tree.get_node([1, 1, a1])._avg.total_value, 6.)
        self.assertEqual(self.tree.get_node([1, 1, a1, 0])._avg.total_value, 5.)
        self.assertEqual(self.tree.get_node([1, 0])._avg.total_value, 7.)


class TestPOMCPPolicyRunner(TestCase):

    def setUp(self):
        s = 4
        a = 3
        o = 2
        T = np.random.dirichlet(np.ones((s,)), (a, s))
        O = np.ones((a, s, o)) * 1. / o  # ensures frequent observations
        R = np.random.random((a, s, s, o))
        start = np.random.dirichlet(np.ones((s)))
        self.pomdp = POMDP(T, O, R, start, 1, states=range(4),
                           actions=set(['a', 'b', 'c']),
                           observations=[True, False])
        self.policy = POMCPPolicyRunner(self.pomdp, iterations=20, horizon=5)

    def test_get_action_is_action(self):
        a = self.policy.get_action()
        self.assertIn(a, self.pomdp.actions)

    def test_step_updates_history(self):
        self.policy.get_action()
        self.policy.history = [0, 1]  # Note: Might fail if '1' unobserved
        a = self.pomdp.actions.index(self.policy.get_action())
        self.policy.step(True)
        self.assertEqual(self.policy.history, [0, 1, a, 0])
