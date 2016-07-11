from unittest import TestCase
import numpy as np

from htm.snap_circuit import (SnapCircuitState, SnapCircuitPart, PlaceAction,
                              PartPresenceCondition,
                              NORTH, WEST, EAST, SOUTH, ORIENTATION_NAMES)


BOARD = (7, 10)


class TestSnapCircuitState(TestCase):

    def test_order_is_row_column(self):
        state = SnapCircuitState((5, 6), [])
        self.assertEqual(state.board_rows, 5)
        self.assertEqual(state.board_cols, 6)

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
        i = WEST + state.N_ORIENTATIONS * (2 + state.board_cols * (
            1 + state.board_rows * part1.id))
        expected_features[i] = 1
        i = NORTH + state.N_ORIENTATIONS * (1 + state.board_cols * (
            0 + state.board_rows * part2.id))
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

    def test_str(self):
        part1 = SnapCircuitPart(5, "first-part")
        part2 = SnapCircuitPart(2, "second part")
        state = SnapCircuitState((2, 3), [((1, 2, WEST), part1)])
        self.assertEqual(str(state),
                         "{first-part<5> at (1, 2) oriented west}")
        state = SnapCircuitState((2, 3), [((1, 2, WEST), part1),
                                          ((0, 1, NORTH), part2)])
        self.assertEqual(str(state),
                         "{first-part<5> at (1, 2) oriented west, "
                         "second part<2> at (0, 1) oriented north}")


class TestOrientationNames(TestCase):

    def test_four_orientations(self):
        self.assertEqual(len(ORIENTATION_NAMES), 4)

    def test_orientation_names_fits_variables(self):
        self.assertEqual(
            [ORIENTATION_NAMES[o] for o in [NORTH, EAST, SOUTH, WEST]],
            ["north", "east", "south", "west"]
            )


class TestPartPresenceCondition(TestCase):

    def setUp(self):
        self.part = SnapCircuitPart(1, '2')
        self.location = (2, 3, NORTH)

    def test_is_there_true(self):
        cond = PartPresenceCondition(BOARD, self.part, self.location)
        state = SnapCircuitState(BOARD, [(self.location, self.part)])
        self.assertTrue(cond.check(state))

    def test_is_there_false(self):
        cond = PartPresenceCondition(BOARD, self.part, self.location)
        state = SnapCircuitState(BOARD, [])
        self.assertFalse(cond.check(state))

    def test_is_there_false_wrong_orientation(self):
        cond = PartPresenceCondition(BOARD, self.part, self.location)
        state = SnapCircuitState(BOARD, [((2, 3, EAST), self.part)])
        self.assertFalse(cond.check(state))

    def test_is_there_false_wrong_part(self):
        cond = PartPresenceCondition(BOARD, self.part, self.location)
        state = SnapCircuitState(BOARD,
                                 [(self.location, SnapCircuitPart(0, '2'))])
        self.assertFalse(cond.check(state))

    def test_is_not_there_true(self):
        cond = PartPresenceCondition(BOARD, self.part, self.location,
                                     is_there=False)
        state = SnapCircuitState(BOARD, [])
        self.assertTrue(cond.check(state))

    def test_is_not_there_false(self):
        cond = PartPresenceCondition(BOARD, self.part, self.location,
                                     is_there=False)
        state = SnapCircuitState(BOARD, [(self.location, self.part)])
        self.assertFalse(cond.check(state))


class TestAction(TestCase):

    def setUp(self):
        self.part = SnapCircuitPart(1, '2')
        self.location = (2, 3, NORTH)

    def test_is_place_valid(self):
        action = PlaceAction(BOARD, self.part, self.location)
        before = SnapCircuitState(BOARD, [])
        after = SnapCircuitState(BOARD, [(self.location, self.part)])
        self.assertTrue(action.check(before, after))

    def test_place_not_valid_if_no_place(self):
        action = PlaceAction(BOARD, self.part, self.location)
        before = SnapCircuitState(BOARD, [])
        after = SnapCircuitState(BOARD, [])
        self.assertFalse(action.check(before, after))

    def test_place_not_valid_if_already_there(self):
        action = PlaceAction(BOARD, self.part, self.location)
        before = SnapCircuitState(BOARD, [(self.location, self.part)])
        after = SnapCircuitState(BOARD, [(self.location, self.part)])
        self.assertFalse(action.check(before, after))
