import shlex
import subprocess

import baduk

from .handler import encode_gtp_move, parse_gtp_move
from .handler import encode_gtp_color, parse_gtp_color

__all__ = [
    'GTPClient',
]


class GTPClient:
    def __init__(self, cmdline):
        self._proc = subprocess.Popen(
            shlex.split(cmdline),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open('/dev/null', 'w'))
        self.send_command('boardsize 19')

    def send_command(self, cmdstring):
        self._proc.stdin.write((cmdstring + '\n').encode('ascii'))
        self._proc.stdin.flush()
        response_str = ''
        while not response_str.startswith('='):
            response_str = self._proc.stdout.readline().decode('ascii')
        return response_str.strip()

    def select_move(self, game_state):
        response_str = self.send_command(
            'genmove {}'.format(encode_gtp_color(game_state.next_player)))
        _, gtp_move = response_str.split()
        return parse_gtp_move(gtp_move)

    def send_move(self, player, move):
        cmd_string = 'play {} {}'.format(
            encode_gtp_color(player),
            encode_gtp_move(move))
        self.send_command(cmd_string)

    def shutdown(self):
        self._proc.terminate()
