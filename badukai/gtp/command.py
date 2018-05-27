from collections import namedtuple


class UnknownCommandError(Exception):
    pass


class Command(namedtuple('Command', 'id body')):
    pass


class Body:
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
