import argparse
import boto3
import os
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-input')
    parser.add_argument('--s3-output')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    clean_up_local_output = False
    local_output = args.output
    if local_output is None:
        tempfd, local_output = tempfile.mkstemp(prefix='tmp-gameout')
        os.close(tempfd)
        clean_up_local_output = True

    try:
        combined_body = bytes()

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
                combined_body += obj_resp['Body'].read()
            if response['IsTruncated']:
                s3_args['ContinuationToken'] = response['NextContinuationToken']
            else:
                keep_going = False

        with open(local_output, 'wb') as outf:
            outf.write(combined_body)

        if args.s3_output:
            bucket, key = args.s3_output.split('/', 1)
            s3 = boto3.resource('s3')
            s3.meta.client.upload_file(local_output, bucket, key)
    finally:
        if clean_up_local_output:
            os.unlink(local_output)


if __name__ == '__main__':
    main()
