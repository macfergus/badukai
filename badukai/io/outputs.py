import contextlib
import os
import tempfile

import boto3

__all__ = [
    'open_output',
    'open_output_filename',
]


@contextlib.contextmanager
def open_local_output(identifier, as_filename=False):
    if as_filename:
        yield identifier
    else:
        outf = open(identifier, 'w')
        yield outf
        outf.close()


@contextlib.contextmanager
def open_s3_output(identifier, as_filename=False):
    assert identifier.startswith('s3://')
    identifier = identifier[5:]
    bucket, key = identifier.split('/', 1)

    tempfd, local_output = tempfile.mkstemp(prefix='tmp-badukai-s3-output')
    os.close(tempfd)

    if as_filename:
        yield local_output
    else:
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
        return open_s3_output(identifier, as_filename=False)
    return open_local_output(identifier, as_filename=False)


def open_output_filename(identifier):
    if identifier.startswith('s3://'):
        return open_s3_output(identifier, as_filename=True)
    return open_local_output(identifier, as_filename=True)
