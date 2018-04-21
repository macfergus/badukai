import contextlib
import os
import tempfile

import boto3

__all__ = [
    'get_input',
]


@contextlib.contextmanager
def open_local_input(identifier):
    yield identifier


@contextlib.contextmanager
def open_s3_input(identifier):
    assert identifier.startswith('s3://')
    identifier = identifier[5:]
    bucket, key = identifier.split('/', 1)

    client = boto3.client('s3')
    obj_resp = client.get_object(Bucket=bucket, Key=key)
    body_bytes = obj_resp['Body'].read()

    tmpfd, local_filename = tempfile.mkstemp(prefix='tmp-s3')
    os.write(tmpfd, body_bytes)
    os.close(tmpfd)

    yield local_filename

    os.unlink(local_filename)


def get_input(identifier):
    if identifier.startswith('s3://'):
        return open_s3_input(identifier)
    return open_local_input(identifier)
