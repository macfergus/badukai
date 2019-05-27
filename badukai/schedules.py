import re


class Schedule:
    def get(self, move_num):
        raise NotImplementedError()


class ConstantSchedule:
    def __init__(self, value):
        self.value = value

    def get(self, move_num):
        return self.value


class DecayingSchedule:
    def __init__(self, init_val, breakpoint, later_val):
        self.init_val = init_val
        self.breakpoint = breakpoint
        self.later_val = later_val

    def get(self, move_num):
        if move_num < self.breakpoint:
            return self.init_val
        return self.later_val

def parse_schedule(schedule_str):
    mo = re.match('^(\d+\.?\d*) until (\d+) then (\d+\.?\d*)$', schedule_str)
    if mo:
        return DecayingSchedule(
            init_val=float(mo.group(1)),
            breakpoint=int(mo.group(2)),
            later_val=float(mo.group(3)))
    return ConstantSchedule(float(schedule_str))
