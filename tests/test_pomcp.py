from unittest import TestCase, skip

import numpy as np

from task_models.lib.pomdp import POMDP
from task_models.lib.pomcp import (
    _SearchNode, _SearchObservationNode, _SearchActionNode, _SearchTree,
    ArrayBelief, ParticleBelief, POMCPPolicyRunner, NTransitionsHorizon,
    Horizon, _ValueAverage)


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


class TestSearchTree(TestCase):

    def setUp(self):
        self.start = np.zeros((10,))
        self.start[-1] = 1.
        self.model = _FakeModel(self.start, 3, 2)
        self.tree = _SearchTree(self.model, 3, 1., node_params={'alpha': 0.})
        self.tree.rollout_it = 1

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

    def test_get_node_creates_child(self):
        ca = self.tree.root.safe_get_child(0)
        co = _SearchObservationNode(ArrayBelief(self.start),
                                    self.model.n_actions)
        ca.children[5] = co
        n = self.tree.get_node([0, 5, 1])
        self.assertIsInstance(n, _SearchActionNode)
        b = np.zeros((10,))
        b[1] = .5
        b[2] = .5
        self.model.successors.append(b)
        n = self.tree.get_node([0, 5, 1, 6])
        self.assertIsInstance(n, _SearchObservationNode)
        np.testing.assert_array_equal(n.belief.array, b)

    def test_rollout_from_state_with_horizon_0_is_0(self):
        h = NTransitionsHorizon(0)
        self.assertEqual(self.tree._one_rollout_from_state(1, h), 0)
        self.assertEqual(len(self.model.transitions_history), 0)

    def test_rollout_from_state_with_horizon_1_is_reward(self):
        r = 3.2
        h = NTransitionsHorizon(1)
        self.model.transitions = [(1, 0, r)]
        self.assertEqual(self.tree._one_rollout_from_state(2, h), r)
        self.assertEqual(len(self.model.transitions_history), 1)
        self.assertEqual(self.model.transitions_history[0][1], 2)

    def test_rollout_from_state_with_horizon_2(self):
        h = NTransitionsHorizon(2)
        self.model.transitions = [(1, None, 11.), (2, None, 10.)]
        self.assertEqual(self.tree._one_rollout_from_state(3, h), 20)
        self.assertEqual(len(self.model.transitions_history), 2)
        self.assertEqual(self.model.transitions_history[0][1], 3)
        self.assertEqual(self.model.transitions_history[1][1], 1)

    def test_simulate_from_node_with_horizon_0(self):
        self.tree.horizon_gen = NTransitionsHorizon.generator(self.model, n=0)
        self.tree.simulate_from_node(self.tree.root)
        self.assertEqual(len(self.model.transitions_history), 0)
        self.assertEqual(str(self.tree.root), "[]")
        self.assertEqual(self.tree.root.n_simulations, 0)

    def test_simulate_from_node_with_horizon_1(self):
        self.model.transitions = [(1, 1, 11.)]
        belief2 = np.zeros((10))
        belief2[1] = 1.
        self.model.successors = [belief2]
        self.tree.horizon_gen = NTransitionsHorizon.generator(self.model, n=1)
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

        self.tree.horizon_gen = NTransitionsHorizon.generator(self.model, n=3)
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

    def test_rollout_from_node_multiple_rollouts(self):
        self.tree.rollout_it = 10
        self.model.transitions = [(1, 1, 11.)] * 10
        belief2 = np.zeros((10))
        belief2[1] = 1.
        self.tree.rollout_from_node(self.tree.root,
                                    NTransitionsHorizon(n=1))
        self.assertEqual(len(self.model.transitions_history), 10)
        self.assertEqual(self.tree.root.n_simulations, 1)

    def test_rollout_from_node_multiple_rollouts_multiprocess(self):
        self.tree.multiprocess = True
        self.tree.rollout_it = 10
        self.model.transitions = [(1, 1, 11.)] * 10
        belief2 = np.zeros((10))
        belief2[1] = 1.
        self.tree.rollout_from_node(self.tree.root,
                                    NTransitionsHorizon(n=1))
        self.assertEqual(self.tree.root.n_simulations, 1)


class TestPOMCPPolicyRunner(TestCase):

    multiprocess = False

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
        self.policy = POMCPPolicyRunner(self.pomdp, iterations=20, horizon=5,
                                        multiprocess_rollouts=self.multiprocess)
        self.policy.tree.rollout_it = 10

    def test_get_action_is_action(self):
        a = self.policy.get_action()
        self.assertIn(a, self.pomdp.actions)

    def test_step_updates_history(self):
        self.policy.get_action()
        self.policy.history = [0, 1]  # Note: Might fail if '1' unobserved
        a = self.pomdp.actions.index(self.policy.get_action())
        self.policy.step(True)
        self.assertEqual(self.policy.history, [0, 1, a, 0])

    def test_horizon_generator_is_one(self):
        # Default, from int
        h = self.policy.tree.horizon_gen()
        self.assertIsInstance(h, Horizon)
        self.assertEqual(h.n, 5)
        # Explicit generator
        policy = POMCPPolicyRunner(
            self.pomdp, iterations=20,
            horizon=NTransitionsHorizon.generator(self.pomdp, 13),
            multiprocess_rollouts=self.multiprocess)
        h = policy.tree.horizon_gen()
        self.assertIsInstance(h, NTransitionsHorizon)
        self.assertEqual(h.n, 13)


class TestPOMCPPolicyRunnerMultiprocess(TestPOMCPPolicyRunner):

    multiprocess = True


class Test_ValueAverage(TestCase):

    def setUp(self):
        self.avg = _ValueAverage(alpha=.1)

    def test_n_simulations(self):
        self.assertEqual(self.avg.n_simulations, 0)
        self.avg.update(3)
        self.assertEqual(self.avg.n_simulations, 1)
        self.avg.update(2)
        self.assertEqual(self.avg.n_simulations, 2)
        self.avg.update(1)
        self.assertEqual(self.avg.n_simulations, 3)

    def test_is_average_for_alpha_equal_zero(self):
        avg = _ValueAverage(0.)
        a = np.random.random(10)
        for i, x in enumerate(a):
            avg.update(x)
            self.assertAlmostEqual(avg.value, np.average(a[:i + 1]))

    def test_is_last_for_alpha_equal_one(self):
        avg = _ValueAverage(1.)
        a = np.random.random(10)
        for x in a:
            avg.update(x)
            self.assertAlmostEqual(avg.value, x)

    def test_increment_moves_towards_value(self):
        a = np.random.random(10)
        for x in a:
            self.avg.update(x)
        val = self.avg.value
        self.avg.update(5)
        self.assertLessEqual(np.abs(self.avg.value - 5), np.abs(val - 5))
