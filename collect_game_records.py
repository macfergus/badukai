import argparse
import boto3
import io
import os
import tempfile

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-input', required=True)
    parser.add_argument('--output', '-o', required=True)
    args = parser.parse_args()

    records = []
    bucket, path = args.s3_input.split('/', 1)
    client = boto3.client('s3')
    keep_going = True
    continuation_token = None
    s3_args = {'Bucket': bucket, 'Prefix': path, 'MaxKeys': 100}
    while keep_going:
        response = client.list_objects_v2(**s3_args)
        for c in response['Contents']:
            obj_resp = client.get_object(
                Bucket=bucket,
                Key=c['Key'])
            body_bytes = obj_resp['Body'].read()
            body_text = io.StringIO(body_bytes.decode('utf-8'))
            records += badukai.bots.zero.load_game_records(body_text)
        if response['IsTruncated']:
            s3_args['ContinuationToken'] = response['NextContinuationToken']
        else:
            keep_going = False

    with badukai.io.open_output(args.output) as outf:
        badukai.bots.zero.save_game_records(records, outf)


if __name__ == '__main__':
    main()
