from unittest import TestCase
import numpy as np

from htm.state import SnapCircuitState, SnapCircuitPart, NORTH, WEST


class TestSnapCircuitState(TestCase):

    def test_dimensions(self):
        state = SnapCircuitState((5, 6), [])
        self.assertEqual(state.n_dim, 5 * 6 * 4 * 20)

    def test_no_part(self):
        state = SnapCircuitState((5, 6), [])
        expected = np.zeros(state.n_dim)
        np.testing.assert_equal(expected, state.get_features())

    def test_features_one_part(self):
        part1 = SnapCircuitPart(5, "first part")
        state = SnapCircuitState((2, 3), [((1, 2, WEST), part1)])
        features = state.features
        self.assertEqual(features[5, 1, 2, WEST], 1)
        features[5, 1, 2, WEST] = 0
        self.assertTrue((features == 0).all())

    def test_parts(self):
        part1 = SnapCircuitPart(0, "first part")
        part2 = SnapCircuitPart(2, "second part")
        state = SnapCircuitState(
            (2, 3),
            [((1, 2, WEST), part1), ((0, 1, NORTH), part2)]
            )
        state.N_PARTS = 3
        expected_features = np.zeros((state.n_dim))
        i = WEST + state.N_ORIENTATIONS * (2 + state.board_rows * (
            1 + state.board_cols * part1.id))
        expected_features[i] = 1
        i = NORTH + state.N_ORIENTATIONS * (1 + state.board_rows * (
            0 + state.board_cols * part2.id))
        expected_features[i] = 1
        np.testing.assert_equal(expected_features, state.get_features())

    def test_equals(self):
        part1 = SnapCircuitPart(0, "first part")
        part2 = SnapCircuitPart(2, "second part")
        state = SnapCircuitState(
            (2, 3),
            [((1, 2, WEST), part1), ((0, 1, NORTH), part2)]
            )
        part1 = SnapCircuitPart(0, "first part")
        part2 = SnapCircuitPart(2, "second part")
        state_bis = SnapCircuitState(
            (2, 3),
            [((1, 2, WEST), part1), ((0, 1, NORTH), part2)]
            )
        self.assertEqual(state, state_bis)

    def test_mismatch_part(self):
        part1 = SnapCircuitPart(0, "first part")
        part2 = SnapCircuitPart(2, "second part")
        state = SnapCircuitState(
            (2, 3),
            [((1, 2, WEST), part1), ((0, 1, NORTH), part2)]
            )
        part1 = SnapCircuitPart(0, "first part")
        part2 = SnapCircuitPart(3, "second part")
        state_bis = SnapCircuitState(
            (2, 3),
            [((1, 2, WEST), part1), ((0, 1, NORTH), part2)]
            )
        self.assertNotEqual(state, state_bis)

    def test_mismatch_dimensions(self):
        part1 = SnapCircuitPart(0, "first part")
        part2 = SnapCircuitPart(2, "second part")
        state = SnapCircuitState(
            (2, 3),
            [((1, 2, WEST), part1), ((0, 1, NORTH), part2)]
            )
        part1 = SnapCircuitPart(0, "first part")
        part2 = SnapCircuitPart(2, "second part")
        state_bis = SnapCircuitState(
            (2, 4),
            [((1, 2, WEST), part1), ((0, 1, NORTH), part2)]
            )
        self.assertNotEqual(state, state_bis)
