from unittest import TestCase

import numpy as np

from htm.task import HierarchicalTask, LeafCombination, SequentialCombination
from htm.task_to_pomdp import (HTMToPOMDP, CollaborativeAction, _name_radix,
                               _NodeToPOMDP)


class TestNameRadix(TestCase):

    def test_name_radix(self):
        a = CollaborativeAction('My mixed Case action', (3., 2., 5.))
        self.assertEqual(_name_radix(a), 'my-mixed-case-action')


class TestLeafToPOMDP(TestCase):

    def setUp(self):
        leaf = LeafCombination(
            CollaborativeAction('Do it', (3., 2., 5.), human_probability=.3))
        self.l2p = _NodeToPOMDP.from_node(leaf, 2.)

    def test_times(self):
        self.assertEqual(self.l2p.t_hum, 3.)
        self.assertEqual(self.l2p.t_rob, 2.)
        self.assertEqual(self.l2p.t_err, 5.)

    def test_durations(self):
        self.assertEqual(self.l2p.durations, [5., 2.])

    def test_start(self):
        self.assertEqual(self.l2p.start, [.3, .7])

    def test_start_default_to_half(self):
        l = _NodeToPOMDP.from_node(LeafCombination(
            CollaborativeAction('Do it', (3., 2., 5.))), 2.)
        self.assertEqual(l.start, [.5, .5])

    def test_update_T_stays_in_range(self):
        T0 = np.random.random((7, 8, 8))
        T = T0.copy()
        self.l2p.update_T(T, 0, 2, 4, [6, 7], [.2, .8], list(range(7)))
        T0[:, 4:6, :] = T[:, 4:6, :]
        np.testing.assert_allclose(T, T0)

    def test_updated_T_is_proba(self):
        T = np.zeros((7, 8, 8))
        self.l2p.update_T(T, 0, 2, 4, [6, 7], [.2, .8], list(range(7)))
        np.testing.assert_allclose(T[:, 4:6, :].sum(-1), np.ones((7, 2)))

    def test_update_O_stays_in_range(self):
        O0 = np.random.random((7, 8, 3))
        O = O0.copy()
        self.l2p.update_O(O, 0, 2, 4, [1, 5], [6], [4])
        O0[:, 4:6, :] = O[:, 4:6, :]
        np.testing.assert_allclose(O, O0)

    def test_updated_O_is_proba(self):
        O = np.zeros((7, 8, 3))
        self.l2p.update_O(O, 0, 2, 4, [1, 5], [6], [4])
        np.testing.assert_allclose(O[:, 4:6, :].sum(-1), np.ones((7, 2)))

    def test_updated_O_on_com(self):
        O = np.zeros((7, 8, 3))
        self.l2p.update_O(O, 0, 2, 4, [1, 5], [6], [4])
        yes = [0., 1., 0.]
        no = [0., 0., 1.]
        np.testing.assert_allclose(
            O[[1, 5, 6, 3], ...][:, 4:6, :],
            [[yes, yes], [yes, yes], [no, no], [no, no]])

    def test_updated_O_on_act(self):
        O = np.zeros((7, 8, 3))
        self.l2p.update_O(O, 0, 2, 4, [1, 5], [6], [4])
        no = [0., 0., 1.]
        none = [1., 0., 0.]
        np.testing.assert_allclose(O[[4, 2], ...][:, 4:6, :],
                                   [[no, no], [no, none]])

    def test_update_R_stays_in_range(self):
        R0 = np.random.random((7, 8, 8, 3))
        R = R0.copy()
        self.l2p.update_R(R, 0, 2, 4, list(range(7)))
        R0[:, 4:6, :, :] = R[:, 4:6, :, :]
        np.testing.assert_allclose(R, R0)

    def test_updated_R_values(self):
        R = np.zeros((7, 8, 8, 3))
        self.l2p.update_R(R, 0, 2, 4, list(range(2, 9)))
        np.testing.assert_allclose(R[:, 4:6, :, :], np.broadcast_to(
            np.array([[2, 2], [3, 3], [4, 2], [5, 5], [6, 6], [7, 7], [8, 8]]
                     )[:, :, None, None],
            (7, 2, 8, 3)))


class TestSequenceToPOMDP(TestCase):

    def setUp(self):
        l1 = LeafCombination(CollaborativeAction('Do a', (3., 2., 5.),
                                                 human_probability=.3))
        l2 = LeafCombination(CollaborativeAction('Do b', (2., 3., 4.),
                                                 human_probability=.7))
        seq = SequentialCombination([l1, l2], name='Do all')
        self.s2p = _NodeToPOMDP.from_node(seq, 2.)

    def test_durations(self):
        self.assertEqual(self.s2p.durations, [5., 2., 4., 2.])

    def test_init(self):
        self.assertEqual(self.s2p.init, [0, 1])

    def test_start(self):
        self.assertEqual(self.s2p.start, [.3, .7])

    def test_update_T_stays_in_range(self):
        T0 = np.random.random((7, 8, 8))
        T = T0.copy()
        self.s2p.update_T(T, 0, 2, 3, [7], [1.], list(range(7)))
        T0[:, 3:7, :] = T[:, 3:7, :]
        np.testing.assert_allclose(T, T0)

    def test_updated_T_is_proba(self):
        T = np.zeros((7, 8, 8))
        self.s2p.update_T(T, 0, 2, 3, [7], [1.], list(range(7)))
        np.testing.assert_allclose(T[:, 3:7, :].sum(-1), np.ones((7, 4)))

    def test_update_O_stays_in_range(self):
        O0 = np.random.random((7, 8, 3))
        O = O0.copy()
        self.s2p.update_O(O, 0, 2, 3, [], [6], [1, 2, 4])
        O0[:, 3:7, :] = O[:, 3:7, :]
        np.testing.assert_allclose(O, O0)

    def test_updated_O_is_proba(self):
        O = np.zeros((7, 8, 3))
        self.s2p.update_O(O, 0, 2, 3, [1], [6], [2, 4])
        np.testing.assert_allclose(O[:, 3:7, :].sum(-1), np.ones((7, 4)))

    def test_updated_O_on_com(self):
        O = np.zeros((7, 8, 3))
        self.s2p.update_O(O, 0, 2, 3, [6], [1], [2, 4])
        yes = [0., 1., 0.]
        no = [0., 0., 1.]
        np.testing.assert_allclose(
            O[[1, 3, 5, 6], ...][:, 3:7, :],
            [[no] * 4, [no, no, yes, yes], [no] * 4, [yes] * 4])

    def test_updated_O_on_act(self):
        O = np.zeros((7, 8, 3))
        self.s2p.update_O(O, 0, 2, 3, [6], [], [1, 2, 4])
        no = [0., 0., 1.]
        none = [1., 0., 0.]
        np.testing.assert_allclose(
            O[[1, 2, 4], ...][:, 3:7, :],
            [[no] * 4, [no, none, no, no], [no, no, no, none]])

    def test_update_R_stays_in_range(self):
        R0 = np.random.random((7, 8, 8, 3))
        R = R0.copy()
        self.s2p.update_R(R, 0, 2, 3, list(range(7)))
        R0[:, 3:7, :, :] = R[:, 3:7, :, :]
        np.testing.assert_allclose(R, R0)

    def test_updated_R_values(self):
        R = np.zeros((7, 8, 8, 3))
        self.s2p.update_R(R, 0, 2, 3, [1, 2, 5, 2, 4, 2, 7])
        np.testing.assert_allclose(R[:, 3:7, :, :], np.broadcast_to(
            np.array([[1] * 4, [2] * 4, [5, 2, 5, 5], [2] * 4, [4, 4, 4, 3],
                      [2] * 4, [7] * 4])[:, :, None, None],
            (7, 4, 8, 3)))


class TestHTM2POMDP(TestCase):
    def setUp(self):
        self.h2p = HTMToPOMDP(1., 2.)

    def test_leaf_to_pomdp(self):
        task = HierarchicalTask(root=LeafCombination(
            CollaborativeAction('Do it', (3., 2., 5.), human_probability=.3)))
        p = self.h2p.task_to_pomdp(task)
        self.assertEqual(p.states,
                         ['before-H-do-it', 'before-R-do-it', 'end'])
        self.assertEqual(p.actions, ['wait', 'phy-do-it', 'com-do-it'])
        self.assertEqual(p.observations, ['none', 'yes', 'no'])
        np.testing.assert_array_equal(p.start, np.array([.3, .7, 0.]))
        # checked manually:
        T = np.array([[[0.71653131,  0.,          0.28346869],
                       [0.,          1.,          0.],
                       [0.,          0.,          1.]],
                      [[0.1888756,   0.,          0.8111244],
                       [0.,          0.,          1.],
                       [0.,          0.,          1.]],
                      [[0.51341712,  0.,          0.48658288],
                       [0.,          1.,          0.],
                       [0.,          0.,          1.]]])
        np.testing.assert_allclose(T, p.T)
        O = np.array([[[1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.]],
                      [[0., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.]],
                      [[0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 0.]]])  # convention: nothing observed in end
        np.testing.assert_array_equal(O, p.O)
        R = -np.broadcast_to(np.array([[1, 1, 0],
                                       [5, 2, 1],
                                       [2, 2, 1]]
                                      )[:, :, None, None], (3, 3, 3, 3))
        np.testing.assert_array_equal(R, p.R)

    def test_seq_to_pomdp(self):
        task = HierarchicalTask(root=SequentialCombination([
            LeafCombination(CollaborativeAction('Do a', (3., 2., 5.),
                                                human_probability=.3)),
            LeafCombination(CollaborativeAction('Do b', (2., 3., 4.),
                                                human_probability=.7))
            ], name='Do all'))
        p = self.h2p.task_to_pomdp(task)
        self.assertEqual(p.states, ['before-H-do-a', 'before-R-do-a',
                                    'before-H-do-b', 'before-R-do-b', 'end'])
        self.assertEqual(p.actions, ['wait', 'phy-do-a', 'com-do-a',
                                     'phy-do-b', 'com-do-b'])
        self.assertEqual(p.observations, ['none', 'yes', 'no'])
        np.testing.assert_array_equal(p.start, np.array([.3, .7, 0., 0., 0.]))
        # checked manually:
        T = np.array([
            [[0.71653,  0.,       0.19843,  0.08504,  0.],
             [0.,       1.,       0.,       0.,       0.],
             [0.,       0.,       0.60653,  0.,       0.39347],
             [0.,       0.,       0.,       1.,       0.],
             [0.,       0.,       0.,       0.,       1.]],
            [[0.18888,  0.,       0.56779,  0.24334,  0.],
             [0.,       0.,       0.7,      0.3,      0.],
             [0.,       0.,       0.08208,  0.,       0.91792],
             [0.,       0.,       0.,       1.,       0.],
             [0.,       0.,       0.,       0.,       1.]],
            [[0.51342,  0.,       0.34061,  0.14597,  0.],
             [0.,       1.,       0.,       0.,       0.],
             [0.,       0.,       0.36788,  0.,       0.63212],
             [0.,       0.,       0.,       1.,       0.],
             [0.,       0.,       0.,       0.,       1.]],
            [[0.2636,   0.,       0.51548,  0.22092,  0.],
             [0.,       1.,       0.,       0.,       0.],
             [0.,       0.,       0.13534,  0.,       0.86466],
             [0.,       0.,       0.,       0.,       1.],
             [0.,       0.,       0.,       0.,       1.]],
            [[0.51342,  0.,       0.34061,  0.14597,  0.],
             [0.,       1.,       0.,       0.,       0.],
             [0.,       0.,       0.36788,  0.,       0.63212],
             [0.,       0.,       0.,       1.,       0.],
             [0.,       0.,       0.,       0.,       1.]]
            ])
        np.testing.assert_allclose(T, p.T, atol=1.e-4)
        O = np.array([[[1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.]],
                      [[0., 0., 1.],
                       [1., 0., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 0.]],
                      [[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0., 0.]],  # convention: nothing observed in end
                      [[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.]],
                      [[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 0.]]])  # convention: nothing observed in end
        np.testing.assert_array_equal(O, p.O)
        R = -np.broadcast_to(np.array([[1] * 4 + [0],
                                       [5, 2, 5, 5, 1],
                                       [2] * 4 + [1],
                                       [4, 4, 4, 3, 1],
                                       [2] * 4 + [1]]
                                      )[:, :, None, None],
                             (5, 5, 5, 3))
        np.testing.assert_array_equal(R, p.R)
