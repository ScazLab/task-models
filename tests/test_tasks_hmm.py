from unittest import TestCase
import numpy as np
import math
from task_models.task import (AbstractAction,
                              LeafCombination,
                              AlternativeCombination,
                              SequentialCombination,
                              ParallelCombination)
from task_models.task_hmm import HierarchicalTaskHMM


class TestGenAllTrajectoriesWithProbs(TestCase):
    def test_leaf(self):
        leaf = LeafCombination(AbstractAction('l'))
        task = HierarchicalTaskHMM(root=leaf)
        task.gen_all_trajectories()
        trajectories = task.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 1)
        self.assertIsInstance(trajectories[0], tuple)
        self.assertEqual(trajectories[0][0], 1)
        self.assertIsInstance(trajectories[0][1], list)
        self.assertEqual(len(trajectories[0][1]), 1)
        self.assertEqual(trajectories[0][1][0].name, leaf.name)

    def test_parallel(self):
        a = LeafCombination(AbstractAction('a'))
        b = LeafCombination(AbstractAction('b'))
        c = LeafCombination(AbstractAction('c'))
        nodes = [a, b, c]
        task = HierarchicalTaskHMM(root=ParallelCombination(nodes))
        task.gen_all_trajectories()
        trajectories = task.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), math.factorial(len(nodes)))
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 6])
                            and isinstance(traj[1], list)
                            and len(traj[1]) == len(nodes)
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))

    def test_sequential(self):
        a = LeafCombination(AbstractAction('a'))
        b = LeafCombination(AbstractAction('b'))
        c = LeafCombination(AbstractAction('c'))
        task = HierarchicalTaskHMM(root=SequentialCombination([a, b, c]))
        task.gen_all_trajectories()
        trajectories = task.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 1)
        self.assertIsInstance(trajectories[0][1], list)
        self.assertEqual(len(trajectories[0][1]), 3)
        self.assertTrue(all(isinstance(node, LeafCombination)
                            for node in trajectories[0][1]))
        self.assertEqual(trajectories[0][0], 1)

    def test_alternative(self):
        a = LeafCombination(AbstractAction('a'))
        b = LeafCombination(AbstractAction('b'))
        c = LeafCombination(AbstractAction('c'))
        nodes = [a, b, c]
        task = HierarchicalTaskHMM(root=AlternativeCombination([a, b, c]))
        task.gen_all_trajectories()
        trajectories = task.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), len(nodes))
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 3])
                            and isinstance(traj[1], list)
                            and len(traj[1]) == 1
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))

    def test_two_level(self):
        a = LeafCombination(AbstractAction('a'))
        b = LeafCombination(AbstractAction('b'))
        c = LeafCombination(AbstractAction('c'))
        ab = SequentialCombination([a, b])
        two_level_task1 = HierarchicalTaskHMM(root=ParallelCombination(
            [ab, c]))
        two_level_task2 = HierarchicalTaskHMM(root=AlternativeCombination(
            [SequentialCombination([a, b]), c]))
        two_level_task3 = HierarchicalTaskHMM(root=SequentialCombination(
            [SequentialCombination([a, b]), c]))
        two_level_task4 = HierarchicalTaskHMM(root=AlternativeCombination(
            [c, SequentialCombination([a, b])]))
        two_level_task5 = HierarchicalTaskHMM(root=ParallelCombination(
            [ParallelCombination([a, b]), c]))
        two_level_task6 = HierarchicalTaskHMM(root=ParallelCombination(
            [c, ParallelCombination([a, b])]))
        two_level_task1.gen_all_trajectories()
        trajectories = two_level_task1.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 2)
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 2])
                            and isinstance(traj[1], list)
                            and len(traj[1]) == 3
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))
        two_level_task2.gen_all_trajectories()
        trajectories = two_level_task2.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 2)
        self.assertTrue(len(trajectories[0][1]) == 2 and len(trajectories[1][1]) == 1)
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 2])
                            and isinstance(traj[1], list)
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))
        two_level_task3.gen_all_trajectories()
        trajectories = two_level_task3.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertIsInstance(trajectories[0], tuple)
        self.assertEqual(trajectories[0][0], 1)
        self.assertIsInstance(trajectories[0][1], list)
        self.assertEqual(len(trajectories[0][1]), 3)
        self.assertTrue(all(isinstance(node, LeafCombination)
                            for node in trajectories[0][1]))
        two_level_task4.gen_all_trajectories()
        trajectories = two_level_task4.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 2)
        self.assertTrue(len(trajectories[0][1]) == 1 and len(trajectories[1][1]) == 2)
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 2])
                            and isinstance(traj[1], list)
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))
        two_level_task5.gen_all_trajectories()
        trajectories = two_level_task5.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 4)
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 4])
                            and isinstance(traj[1], list)
                            and len(traj[1]) == 3
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))
        two_level_task6.gen_all_trajectories()
        trajectories = two_level_task6.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 4)
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 4])
                            and isinstance(traj[1], list)
                            and len(traj[1]) == 3
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))

    def test_three_level(self):
        b = LeafCombination(AbstractAction('b'))
        c = LeafCombination(AbstractAction('c'))
        a1 = LeafCombination(AbstractAction('a1'))
        a2 = LeafCombination(AbstractAction('a2'))
        a1a2 = SequentialCombination([a1, a2])
        three_level_task1 = HierarchicalTaskHMM(root=ParallelCombination(
            [SequentialCombination([a1a2, b]), c]))
        three_level_task2 = HierarchicalTaskHMM(root=SequentialCombination(
            [SequentialCombination([a1a2, b]), c]))
        three_level_task1.gen_all_trajectories()
        trajectories = three_level_task1.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 2)
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 2])
                            and isinstance(traj[1], list)
                            and len(traj[1]) == 4
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))
        three_level_task2.gen_all_trajectories()
        trajectories = three_level_task2.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertIsInstance(trajectories[0], tuple)
        self.assertEqual(trajectories[0][0], 1)
        self.assertIsInstance(trajectories[0][1], list)
        self.assertEqual(len(trajectories[0][1]), 4)
        self.assertTrue(all(isinstance(node, LeafCombination)
                            for node in trajectories[0][1]))

    def test_custom_probs_alt(self):
        a = LeafCombination(AbstractAction('a'))
        b = LeafCombination(AbstractAction('b'))
        c = LeafCombination(AbstractAction('c'))
        ab = SequentialCombination([a, b])
        ba = SequentialCombination([b, a])
        alt_aux1 = AlternativeCombination([ab, ba], probabilities=[0.8, 0.2])
        alt_aux2 = AlternativeCombination([ab, ba], probabilities=[0.1, 0.9])
        custom_task1 = HierarchicalTaskHMM(root=AlternativeCombination(
            [SequentialCombination([alt_aux1, c]), SequentialCombination([c, alt_aux2])],
            probabilities=[0.7, 0.3]))
        custom_task1.gen_all_trajectories()
        trajectories = custom_task1.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 4)
        self.assertTrue(all(isinstance(traj, tuple)
                            and isinstance(traj[1], list)
                            and len(traj[1]) == 3
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))
        # TODO: self.assertRaises doesn't work
        # with self.assertRaises(ValueError):
        #     AlternativeCombination([a, b], probabilities=[-2, 0.2])

    def test_complex_task(self):
        mount_central = SequentialCombination([
            LeafCombination(AbstractAction('Get central frame')),
            LeafCombination(AbstractAction('Start Hold central frame'))],
            name='Mount central frame')
        mount_legs = ParallelCombination([
            SequentialCombination([
                LeafCombination(AbstractAction('Get left leg')),
                LeafCombination(AbstractAction('Snap left leg')),
            ], name='Mount left leg'),
            SequentialCombination([
                LeafCombination(AbstractAction('Get right leg')),
                LeafCombination(AbstractAction('Snap right leg')),
            ], name='Mount right leg'),
        ],
            name='Mount legs')
        release_central = LeafCombination(
            AbstractAction('Release central frame'))
        mount_top = SequentialCombination([
            LeafCombination(AbstractAction('Get top')),
            LeafCombination(AbstractAction('Snap top'))],
            name='Mount top')

        chair_task_root = SequentialCombination(
            [mount_central, mount_legs, release_central, mount_top], name='Mount chair')
        chair_task = HierarchicalTaskHMM(root=chair_task_root)
        chair_task.gen_all_trajectories()
        trajectories = chair_task.all_trajectories
        self.assertIsInstance(trajectories, list)
        self.assertEqual(len(trajectories), 2)
        self.assertTrue(all(isinstance(traj, tuple)
                            and np.isclose([traj[0]], [float(1) / 2])
                            and isinstance(traj[1], list)
                            and len(traj[1]) == 9
                            and isinstance(el, LeafCombination)
                            for traj in trajectories for el in traj[1]))
        self.assertEqual(trajectories[0][1][2].name, 'Get left leg order-0')
        self.assertEqual(trajectories[0][1][3].name, 'Snap left leg order-0')
        self.assertEqual(trajectories[0][1][4].name, 'Get right leg order-0')
        self.assertEqual(trajectories[0][1][5].name, 'Snap right leg order-0')
        self.assertEqual(trajectories[0][1][6].name, 'Release central frame')
        self.assertEqual(trajectories[1][1][2].name, 'Get right leg order-1')
        self.assertEqual(trajectories[1][1][3].name, 'Snap right leg order-1')
        self.assertEqual(trajectories[1][1][4].name, 'Get left leg order-1')
        self.assertEqual(trajectories[1][1][5].name, 'Snap left leg order-1')
        self.assertEqual(trajectories[1][1][6].name, 'Release central frame')


class TestGenBinTrajectories(TestCase):
    def test_sim_task(self):
        b_l1 = LeafCombination(AbstractAction('bring_leg1'))
        b_l2 = LeafCombination(AbstractAction('bring_leg2'))
        b_l3 = LeafCombination(AbstractAction('bring_leg3'))
        b_l4 = LeafCombination(AbstractAction('bring_leg4'))
        b_s = LeafCombination(AbstractAction('bring_seat'))
        b_b = LeafCombination(AbstractAction('bring_back'))
        b_scr = LeafCombination(AbstractAction('bring_screwdriver'))
        a_legs_1 = ParallelCombination([b_l1, b_l2, b_l3, b_l4], name='attach_legs')
        a_rest_1 = ParallelCombination([b_s, b_b], name='attach_rest')
        a_l1_2 = ParallelCombination([b_l1, b_scr], name='attach_leg1')
        a_l2_2 = ParallelCombination([b_l2, b_scr], name='attach_leg2')
        a_l3_2 = ParallelCombination([b_l3, b_scr], name='attach_leg3')
        a_l4_2 = ParallelCombination([b_l4, b_scr], name='attach_leg4')
        a_s_2 = ParallelCombination([b_s, b_scr], name='attach_seat')
        a_b_2 = ParallelCombination([b_b, b_scr], name='attach_back')
        a_legs_2 = ParallelCombination([a_l1_2, a_l2_2, a_l3_2, a_l4_2], name='attach_legs')
        a_rest_2 = ParallelCombination([a_s_2, a_b_2], name='attach_rest')

        sim_task_action1 = HierarchicalTaskHMM(root=
                                            SequentialCombination([b_scr, a_legs_1, a_rest_1], name='complete'),
                                            name='sim_task_action1')

        sim_task_action1.gen_all_trajectories()
        sim_task_action1.gen_bin_feats_traj(False)
        self.assertTrue(all(el == 7
                            for el in np.sum(np.sum(sim_task_action1.bin_trajectories, axis=1), axis=1)))
