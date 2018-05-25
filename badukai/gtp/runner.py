import sys

from . import command

__all__ = [
    'Runner',
]


class Runner:
    def __init__(self, handler, inf=None, outf=None):
        self.handler = handler

        if inf:
            self.inf = inf
        else:
            self.inf = sys.stdin

        if outf:
            self.outf = outf
        else:
            self.outf = sys.stdout

    def run_forever(self):
        while not self.handler.is_done:
            cmd = self.parse_command()
            response = self.dispatch_command(cmd.body)
            self.send_response(cmd.id, response)

    def dispatch_command(self, cmd):
        try:
            handler = self.get_handler(cmd.name)
            return handler(cmd)
        except command.UnknownCommandError:
            return command.failure('unknown command')

    def parse_command(self):
        text = self.inf.readline()
        pieces = text.strip().split()
        command_id = None
        if pieces[0].isdigit():
            command_id = pieces[0]
            pieces = pieces[1:]
        name = pieces[0]
        args = pieces[1:]
        return command.Command(command_id, command.Body(name, args))

    def get_handler(self, name):
        method_name = 'handle_{}'.format(name)
        if not hasattr(self.handler, method_name):
            raise command.UnknownCommandError(name)
        return getattr(self.handler, method_name)

    def send_response(self, response_id, response):
        self.outf.write('{indicator}{response_id} {result}\n\n'.format(
            indicator='=' if response.success else '?',
            response_id='' if response_id is None else response_id,
            result=response.content))
