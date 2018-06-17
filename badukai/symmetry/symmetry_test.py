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
