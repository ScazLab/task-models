from unittest import TestCase

import numpy as np

from htm.task import (HierarchicalTask, LeafCombination, SequentialCombination,
                      AlternativeCombination)
from htm.bring_next_to_pomdp import HTMToPOMDP, CollaborativeAction


class TestHTM2POMDP(TestCase):

    def test_leaf_to_pomdp(self):
        h2p = HTMToPOMDP(2., 8., 5., ['A1'], end_reward=50.)
        task = HierarchicalTask(root=LeafCombination(CollaborativeAction(
            'bottom left', 'A1')))
        p = h2p.task_to_pomdp(task)
        self.assertEqual(p.states, ['before-bottom-left', 'end'])
        self.assertEqual(p.actions, ['get-A1', 'ask-A1'])
        self.assertEqual(p.observations, ['none', 'yes', 'no', 'error'])
        np.testing.assert_array_equal(p.start, np.array([1, 0.]))
        # checked manually:
        T = np.array([
                      # get
                      [[0., 1.],
                       [1., 0.]],
                      # ask
                      [[1., 0.],
                       [1., 0.]],
                      ])
        np.testing.assert_allclose(T, p.T)
        O = np.array([
                      # get
                      [[0., 0., 0., 1.],
                       [.5, 0., 0., .5]],
                      # ask
                      [[0., 1., 0., 0.],
                       [0., 0., 1., 0.]],
                      ])
        np.testing.assert_array_equal(O, p.O)
        R = np.broadcast_to(np.array([[[-5., -8.],
                                       [50., 50.]],
                                      [[-2., -2.],
                                       [50., 50.]],
                                      ])[..., None],
                            (2, 2, 2, 4))
        np.testing.assert_array_equal(R, p.R)

    def test_seq_to_pomdp(self):
        # No probability of failure or human saying no here
        task = HierarchicalTask(root=SequentialCombination([
            LeafCombination(CollaborativeAction('Bottom left', 'A1')),
            LeafCombination(CollaborativeAction('Top left', 'A2')),
            ], name='Do all'))
        h2p = HTMToPOMDP(2., 8., 5., ['A2', 'A1'], end_reward=50.)
        p = h2p.task_to_pomdp(task)
        self.assertEqual(p.states,
                         ['before-bottom-left', 'before-top-left', 'end'])
        self.assertEqual(p.actions, ['get-A2', 'get-A1', 'ask-A2', 'ask-A1'])
        self.assertEqual(p.observations, ['none', 'yes', 'no', 'error'])
        np.testing.assert_array_equal(p.start, np.array([1, 0, 0]))
        # checked manually:
        T = np.array([
            # get A2
            [[1., 0., 0.],
             [0., 0., 1.],
             [1., 0., 0.]],
            # get A1
            [[0., 1., 0.],
             [0., 1., 0.],
             [1., 0., 0.]],
            # ask A2
            [[1., 0., 0.],
             [0., 1., 0.],
             [1., 0., 0.]],
            # ask A1
            [[1., 0., 0.],
             [0., 1., 0.],
             [1., 0., 0.]],
            ])
        np.testing.assert_allclose(T, p.T, atol=1.e-4)
        O = np.array([
            # get A2
            [[0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [.5, 0., 0., .5]],
            # get A1
            [[0., 0., 0., 1.],
             [.5, 0., 0., .5],
             [0., 0., 0., 1.]],
            # ask A2
            [[0., 0., 1., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.]],
            # ask A1
            [[0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.]],
            ])
        np.testing.assert_array_equal(O, p.O)
        R = np.broadcast_to(np.array([
                                      # get A2
                                      [[-5.] * 3,
                                       [-5, -5, -8],
                                       [50, 50, 50]],
                                      # get A1
                                      [[-5, -8, -5],
                                       [-5] * 3,
                                       [50, 50, 50]],
                                      # ask A2
                                      [[-2.] * 3,
                                       [-2.] * 3,
                                       [50, 50, 50]],
                                      # ask A1
                                      [[-2.] * 3,
                                       [-2.] * 3,
                                       [50, 50, 50]],
                                      ])[..., None],
                            (4, 3, 3, 4))
        np.testing.assert_array_equal(R, p.R)

    def test_alt_to_pomdp(self):
        # No probability of failure or human saying no here
        task = HierarchicalTask(root=AlternativeCombination([
            LeafCombination(CollaborativeAction('Bottom left', 'A1')),
            LeafCombination(CollaborativeAction('Top left', 'A2')),
            ], name='Do all'))
        h2p = HTMToPOMDP(2., 8., 5., ['A1', 'A2'], end_reward=50.)
        p = h2p.task_to_pomdp(task)
        self.assertEqual(p.states,
                         ['before-bottom-left', 'before-top-left', 'end'])
        self.assertEqual(p.actions, ['get-A1', 'get-A2', 'ask-A1', 'ask-A2'])
        self.assertEqual(p.observations, ['none', 'yes', 'no', 'error'])
        np.testing.assert_array_equal(p.start, np.array([.5, .5, 0]))
        # checked manually:
        T = np.array([
            # get A1
            [[0., 0., 1.],
             [0., 1., 0.],
             [.5, .5, 0.]],
            # get A2
            [[1., 0., 0.],
             [0., 0., 1.],
             [.5, .5, 0.]],
            # ask A1
            [[1., 0., 0.],
             [0., 1., 0.],
             [.5, .5, 0.]],
            # ask A2
            [[1., 0., 0.],
             [0., 1., 0.],
             [.5, .5, 0.]],
            ])
        np.testing.assert_allclose(T, p.T, atol=1.e-4)
        O = np.array([
            # get A1
            [[0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [.5, 0., 0., .5]],
            # get A2
            [[0., 0., 0., 1.],
             [0., 0., 0., 1.],
             [.5, 0., 0., .5]],
            # ask A1
            [[0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.]],
            # ask A2
            [[0., 0., 1., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.]],
            ])
        np.testing.assert_array_equal(O, p.O)
        R = np.broadcast_to(np.array([
            # get A1
            [[-5, -5, -8],
             [-5] * 3,
             [50, 50, 50]],
            # get A2
            [[-5.] * 3,
             [-5, -5, -8],
             [50, 50, 50]],
            # ask A1
            [[-2.] * 3,
             [-2.] * 3,
             [50, 50, 50]],
            # ask A2
            [[-2.] * 3,
             [-2.] * 3,
             [50, 50, 50]],
            ])[..., None],
                            (4, 3, 3, 4))
        np.testing.assert_array_equal(R, p.R)
