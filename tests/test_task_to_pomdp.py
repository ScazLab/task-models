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
        self.assertEqual(self.l2p.durations, [5., 2, 2, 2])

    def test_start(self):
        self.assertEqual(self.l2p.start, [1])

    def test_update_T_stays_in_range(self):
        T0 = np.random.random((7, 8, 8))
        T = T0.copy()
        self.l2p.update_T(T, 0, 2, 4, [6, 7], [.2, .8], list(range(7)))
        T0[:, 4:7, :] = T[:, 4:7, :]
        np.testing.assert_allclose(T, T0)

    def test_updated_T_is_proba(self):
        T = np.zeros((7, 8, 8))
        self.l2p.update_T(T, 0, 2, 4, [6, 7], [.2, .8], list(range(7)))
        np.testing.assert_allclose(T[:, 4:6, :].sum(-1), np.ones((7, 2)))

    def test_update_O_stays_in_range(self):
        O0 = np.random.random((7, 8, 4))
        O = O0.copy()
        self.l2p.update_O(O, 2, 4, [7], [6, 7], [1, 5])
        O0[2:6, :, :] = O[2:6, :, :]
        np.testing.assert_allclose(O, O0)

    def test_updated_O_is_proba(self):
        O = np.zeros((7, 8, 4))
        O[:, :, -1] = 1  # initialized with o_NONE
        self.l2p.update_O(O, 2, 4, [7], [6, 7], [1, 5])
        np.testing.assert_allclose(O[2:6, :, :].sum(-1), np.ones((4, 8)))

    def test_update_R_stays_in_range(self):
        R0 = np.random.random((7, 8, 8, 4))
        R = R0.copy()
        self.l2p.update_R(R, 0, 2, 4, list(range(7)))
        R0[:, 4:7, :, :] = R[:, 4:7, :, :]
        np.testing.assert_allclose(R, R0)

    def test_updated_R_values(self):
        R = np.zeros((7, 8, 8, 4))
        self.l2p.update_R(R, 0, 2, 4, list(range(2, 9)))
        np.testing.assert_allclose(R[:, 4:7, :, :], np.broadcast_to(
            np.array([[2, 2, 2], [3, 3, 3], [4, 4, 2], [5, 5, 5], [6, 6, 6],
                      [7, 7, 7], [8, 8, 8]]
                     )[:, :, None, None],
            (7, 3, 8, 4)))


class TestSequenceToPOMDP(TestCase):

    def setUp(self):
        l1 = LeafCombination(CollaborativeAction('Do a', (3., 2., 5.),
                                                 human_probability=.3))
        l2 = LeafCombination(CollaborativeAction('Do b', (2., 3., 4.),
                                                 human_probability=.7))
        seq = SequentialCombination([l1, l2], name='Do all')
        self.s2p = _NodeToPOMDP.from_node(seq, 2.)

    def test_durations(self):
        self.assertEqual(self.s2p.durations, [5., 2, 2, 2, 4, 2, 2, 2])

    def test_init(self):
        self.assertEqual(self.s2p.init, [0])

    def test_start(self):
        self.assertEqual(self.s2p.start, [1])

    def test_update_T_stays_in_range(self):
        T0 = np.random.random((11, 10, 10))
        T = T0.copy()
        self.s2p.update_T(T, 0, 2, 3, [7], [1.], list(range(1, 12)))
        T0[:, 3:9, :] = T[:, 3:9, :]
        np.testing.assert_allclose(T, T0)

    def test_updated_T_is_proba(self):
        T = np.zeros((11, 10, 10))
        self.s2p.update_T(T, 0, 2, 3, [9], [1.], list(range(11)))
        np.testing.assert_allclose(T[:, 3:9, :].sum(-1), np.ones((11, 6)))

    def test_update_O_stays_in_range(self):
        O0 = np.random.random((11, 10, 4))
        O = O0.copy()
        self.s2p.update_O(O, 2, 3, [9], [0, 1, 2, 10], [9])
        O0[2:10, :, :] = O[2:10, :, :]
        np.testing.assert_allclose(O, O0)

    def test_updated_O_is_proba(self):
        O = np.zeros((11, 10, 4))
        O[:, :, 2] = 1.  # O is initialized with NO
        self.s2p.update_O(O, 2, 3, [9], [1, 0], [2, 9])
        np.testing.assert_allclose(O[2:10, :, :].sum(-1), np.ones((8, 10)))

    def test_update_R_stays_in_range(self):
        R0 = np.random.random((11, 10, 10, 4))
        R = R0.copy()
        self.s2p.update_R(R, 0, 2, 3, list(range(1, 12)))
        R0[:, 3:9, :, :] = R[:, 3:9, :, :]
        np.testing.assert_allclose(R, R0)

    def test_updated_R_values(self):
        R = np.zeros((11, 10, 10, 4))
        self.s2p.update_R(R, 0, 2, 3, [1, 2, 5, 2, 2, 2, 4, 2, 2, 2, 7])
        correct = np.broadcast_to(
            np.array([[1] * 6, [2] * 6, [5, 5, 2, 5, 5, 5], [2] * 6,
                      [2] * 6, [2] * 6, [4, 4, 4, 4, 4, 3], [2] * 6,
                      [2] * 6, [2] * 6, [7] * 6])[:, :, None, None],
            (11, 6, 10, 4))
        np.testing.assert_allclose(R[:, 3:9, :, :], correct)


class TestHTM2POMDP(TestCase):
    def setUp(self):
        self.h2p = HTMToPOMDP(1., 2.)

    def test_leaf_to_pomdp(self):
        task = HierarchicalTask(root=LeafCombination(
            CollaborativeAction('Do it', (3., 2., 5.), human_probability=.3)))
        p = self.h2p.task_to_pomdp(task)
        self.assertEqual(p.states,
                         ['init-do-it', 'H-do-it', 'R-do-it', 'end'])
        self.assertEqual(p.actions, [
            'wait', 'phy-do-it', 'com-ask-intention-do-it',
            'com-tell-intention-do-it', 'com-ask-finished-do-it'])
        self.assertEqual(p.observations, ['none', 'yes', 'no', 'error'])
        np.testing.assert_array_equal(p.start, np.array([1, 0., 0., 0.]))
        # checked manually:
        T = np.array([[[1., 0.,          0.,          0.],
                       [0., 0.71653131,  0.,          0.28346869],
                       [0., 0.,          1.,          0.],
                       [0., 0.,          0.,          1.]],
                      [[1., 0.,          0.,          0.],
                       [0., 0.1888756,   0.,          0.8111244],
                       [0., 0.,          0.,          1.],
                       [0., 0.,          0.,          1.]],
                      [[0., 1.,          0.,          0.],
                       [0., 0.51341712,  0.,          0.48658288],
                       [0., 0.,          1.,          0.],
                       [0., 0.,          0.,          1.]],
                      [[0., 0.,          1.,          0.],
                       [0., 0.51341712,  0.,          0.48658288],
                       [0., 0.,          1.,          0.],
                       [0., 0.,          0.,          1.]],
                      [[1., 0.,          0.,          0.],
                       [0., 0.51341712,  0.,          0.48658288],
                       [0., 0.,          1.,          0.],
                       [0., 0.,          0.,          1.]],
                      ])
        np.testing.assert_allclose(T, p.T)
        O = np.array([
                      # Wait
                      [[1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.]],
                      # Act
                      [[0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [1., 0., 0., 0.]],
                      # Ask intention
                      [[1., 0., 0., 0.],   # not possible in T
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],   # robot has told its int. to act
                       [1., 0., 0., 0.]],  # human acts while robot ask again
                      #                    # TODO: change to Yes?
                      # Tell intention
                      [[1., 0., 0., 0.],   # not possible in T
                       [1., 0., 0., 0.],   # TODO: No?
                       [1., 0., 0., 0.],   # TODO: maybe H answers to R tell
                       [1., 0., 0., 0.]],  # TODO: answer?
                      # Ask finished
                      [[0., 0., 1., 0.],   # not started
                       [0., 0., 1., 0.],
                       [0., 0., 1., 0.],
                       [0., 1., 0., 0.]],
                      ])
        np.testing.assert_array_equal(O, p.O)
        R = -np.broadcast_to(np.array([[1, 1, 1, 0],
                                       [5, 5, 2, 1],
                                       [2, 2, 2, 1],
                                       [2, 2, 2, 1],
                                       [2, 2, 2, 1],
                                       ]
                                      )[:, :, None, None],
                             (5, 4, 4, 4))
        np.testing.assert_array_equal(R, p.R)

    def test_seq_to_pomdp(self):
        task = HierarchicalTask(root=SequentialCombination([
            LeafCombination(CollaborativeAction('Do a', (3., 2., 5.))),
            LeafCombination(CollaborativeAction('Do b', (2., 3., 4.))),
            ], name='Do all'))
        p = self.h2p.task_to_pomdp(task)
        self.assertEqual(p.states, ['init-do-a', 'H-do-a', 'R-do-a',
                                    'init-do-b', 'H-do-b', 'R-do-b', 'end'])
        self.assertEqual(p.actions, [
            'wait',
            'phy-do-a', 'com-ask-intention-do-a', 'com-tell-intention-do-a',
            'com-ask-finished-do-a',
            'phy-do-b', 'com-ask-intention-do-b', 'com-tell-intention-do-b',
            'com-ask-finished-do-b'])
        self.assertEqual(p.observations, ['none', 'yes', 'no', 'error'])
        np.testing.assert_array_equal(p.start, np.array([1, 0, 0, 0, 0, 0, 0]))
        # checked manually:
        T = np.array([
            # Wait
            [[1., 0.,      0., 0.,      0.,      0., 0.],
             [0., 0.71653, 0., 0.28347, 0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.60653, 0., 0.39347],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Physical a
            [[1., 0.,      0., 0.,      0.,      0., 0.],
             [0., 0.18888, 0., 0.81112, 0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.08208, 0., 0.91792],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Ask intention a
            [[0., 1.,      0., 0.,      0.,      0., 0.],
             [0., 0.51342, 0., 0.48658, 0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.36788, 0., 0.63212],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Tell intention a
            [[0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.51342, 0., 0.48658, 0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.36788, 0., 0.63212],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Ask finished a
            [[1., 0.,      0., 0.,      0.,      0., 0.],
             [0., 0.51342, 0., 0.48658, 0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.36788, 0., 0.63212],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Physical b
            [[1., 0.,      0., 0.,      0.,      0., 0.],
             [0., 0.2636,  0., 0.7364,  0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.13534, 0., 0.86466],
             [0., 0.,      0., 0.,      0.,      0., 1.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Ask intention b
            [[1., 0.,      0., 0.,      0.,      0., 0.],
             [0., 0.51342, 0., 0.48658, 0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      1.,      0., 0.],
             [0., 0.,      0., 0.,      0.36788, 0., 0.63212],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Tell intention b
            [[1., 0.,      0., 0.,      0.,      0., 0.],
             [0., 0.51342, 0., 0.48658, 0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.36788, 0., 0.63212],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            # Ask finished b
            [[1., 0.,      0., 0.,      0.,      0., 0.],
             [0., 0.51342, 0., 0.48658, 0.,      0., 0.],
             [0., 0.,      1., 0.,      0.,      0., 0.],
             [0., 0.,      0., 1.,      0.,      0., 0.],
             [0., 0.,      0., 0.,      0.36788, 0., 0.63212],
             [0., 0.,      0., 0.,      0.,      1., 0.],
             [0., 0.,      0., 0.,      0.,      0., 1.]],
            ])
        np.testing.assert_allclose(T, p.T, atol=1.e-4)
        O = np.array([
            # Wait
            [[1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],
            # Physical a
            [[0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [1., 0., 0., 0.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.]],
            # Ask intention a
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],
            # Tell intention a
            [[1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],
            # Ask finished
            [[0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 1., 0., 0.],
             [0., 1., 0., 0.],
             [0., 1., 0., 0.],
             [0., 1., 0., 0.]],
            # Physical b
            [[0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [1., 0., 0., 0.]],
            # Ask intention
            [[1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [1., 0., 0., 0.]],
            # Tell intention
            [[1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],
            # Ask finished b
            [[0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 1., 0., 0.]],
            ])
        np.testing.assert_array_equal(O, p.O)
        R = -np.broadcast_to(np.array([[1] * 6 + [0],
                                       [5, 5, 2, 5, 5, 5, 1],
                                       [2] * 6 + [1],
                                       [2] * 6 + [1],
                                       [2] * 6 + [1],
                                       [4, 4, 4, 4, 4, 3, 1],
                                       [2] * 6 + [1],
                                       [2] * 6 + [1],
                                       [2] * 6 + [1]]
                                      )[:, :, None, None],
                             (9, 7, 7, 4))
        np.testing.assert_array_equal(R, p.R)
