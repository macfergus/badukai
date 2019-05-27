import unittest

from . import schedules


class SchedulesTest(unittest.TestCase):
    def test_parse_decaying(self):
        s = schedules.parse_schedule('0.5 until 10 then 0')
        self.assertEqual(0.5, s.get(9))
        self.assertEqual(0.0, s.get(11))

    def test_parse_constant(self):
        s = schedules.parse_schedule('0.25')
        self.assertEqual(0.25, s.get(1))
        self.assertEqual(0.25, s.get(1000))
