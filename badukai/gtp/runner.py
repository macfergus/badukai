import sys
from collections import namedtuple

__all__ = [
    'BotRunner',
]


class UnknownCommandError(Exception):
    pass


class Command(namedtuple('Command', 'id body')):
    pass


class CommandBody:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class Response:
    def __init__(self, success, content):
        self.success = success
        self.content = content


def success(content):
    return Response(True, content)


def failure(content):
    return Response(False, content)


class BotRunner:
    def __init__(self, bot, inf=None, outf=None):
        self.bot = bot

        if inf:
            self.inf = inf
        else:
            self.inf = sys.stdin

        if outf:
            self.outf = outf
        else:
            self.outf = sys.stdout

        self.should_quit = False

    def run_forever(self):
        while not self.should_quit:
            command = self.parse_command()
            response = self.handle_command(command.body)
            self.send_response(command.id, response)

    def handle_command(self, command):
        try:
            handler = self.get_handler(command.name)
            return handler(command)
        except UnknownCommandError:
            return failure('unknown command')

    def parse_command(self):
        text = self.inf.readline()
        pieces = text.strip().split()
        command_id = None
        if pieces[0].isdigit():
            command_id = pieces[0]
            pieces = pieces[1:]
        name = pieces[0]
        args = pieces[1:]
        return Command(command_id, CommandBody(name, args))

    def get_handler(self, name):
        method_name = 'handle_{}'.format(name)
        if not hasattr(self, method_name):
            raise UnknownCommandError(name)
        return getattr(self, method_name)

    def send_response(self, response_id, response):
        self.outf.write('{indicator}{response_id} {result}\n\n'.format(
            indicator='=' if response.success else '?',
            response_id='' if response_id is None else response_id,
            result=response.content))

    def handle_quit(self, _):
        self.should_quit = True
        return success('bye!')

