import unittest

import baduk

from . import symmetry


class SymmetryTest(unittest.TestCase):
    def test_rotations(self):
        orig = baduk.Point(3, 6)
        rotations = [
            baduk.Point(3, 6),
            baduk.Point(6, 3),
            baduk.Point(14, 3),
            baduk.Point(17, 6),
            baduk.Point(3, 14),
            baduk.Point(6, 17),
            baduk.Point(17, 14),
            baduk.Point(14, 17),
        ]

        rotated = [symmetry.rotate_point(orig, i, 19) for i in range(8)]
        self.assertCountEqual(rotated, rotations)

    def test_rotate_board(self):
        board = baduk.Board(5, 5)
        board.place_stone(baduk.Player.black, baduk.Point(3, 3))
        board.place_stone(baduk.Player.white, baduk.Point(3, 2))
        board.place_stone(baduk.Player.black, baduk.Point(4, 2))

        rotated = symmetry.rotate_board(board, 1)
        self.assertEqual(baduk.Player.black, rotated.get(baduk.Point(3, 3)))
        self.assertEqual(baduk.Player.white, rotated.get(baduk.Point(3, 2)))
        self.assertEqual(baduk.Player.black, rotated.get(baduk.Point(2, 2)))

    def test_rotate_game_record(self):
         # Handicap-ish setup
        board = baduk.Board(5, 5)
        board.place_stone(baduk.Player.black, baduk.Point(2, 2))
        board.place_stone(baduk.Player.black, baduk.Point(2, 4))
        game = baduk.GameState.from_board(board, baduk.Player.white)
        game = game.apply_move(baduk.Move.play(baduk.Point(4, 3)))

        # 1 == Reflect rows
        rot_game = symmetry.rotate_game_record(game, 1)
        self.assertEqual(
            baduk.Move.play(baduk.Point(2, 3)),
            rot_game.last_move)
        rot_board = rot_game.board
        self.assertEqual(baduk.Player.black, rot_board.get(baduk.Point(4, 2)))
        self.assertEqual(baduk.Player.black, rot_board.get(baduk.Point(4, 4)))
        self.assertEqual(baduk.Player.white, rot_board.get(baduk.Point(2, 3)))

        rot_game = rot_game.previous_state
        self.assertIsNone(rot_game.last_move)
        rot_board = rot_game.board
        self.assertEqual(baduk.Player.black, rot_board.get(baduk.Point(4, 2)))
        self.assertEqual(baduk.Player.black, rot_board.get(baduk.Point(4, 4)))
        self.assertIsNone(rot_board.get(baduk.Point(2, 3)))
