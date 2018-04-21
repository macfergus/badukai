import contextlib
import os
import tempfile

import boto3

__all__ = [
    'open_output',
]


@contextlib.contextmanager
def open_local_output(identifier):
    outf = open(identifier, 'w')
    yield outf
    outf.close()


@contextlib.contextmanager
def open_s3_output(identifier):
    assert identifier.startswith('s3://')
    identifier = identifier[5:]
    bucket, key = identifier.split('/', 1)

    tempfd, local_output = tempfile.mkstemp(prefix='tmp-badukai-s3-output')
    os.close(tempfd)

    outf = open(local_output, 'w')
    yield outf
    outf.close()

    try:
        s3 = boto3.resource('s3')
        s3.meta.client.upload_file(local_output, bucket, key)
    finally:
        os.unlink(local_output)


def open_output(identifier):
    if identifier.startswith('s3://'):
        return open_s3_output(identifier)
    return open_local_output(identifier)
