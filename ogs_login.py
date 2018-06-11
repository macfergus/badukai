import argparse
import json

import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--secret', type=str, required=True)
    parser.add_argument('--creds-file', type=str, required=True)
    parser.add_argument('username')
    parser.add_argument('password')
    args = parser.parse_args()

    secrets = json.load(open(args.secret))
    resp = requests.post(
        'https://online-go.com/oauth2/token/',
        data={
            'client_id': secrets['clientid'],
            'client_secret': secrets['secret'],
            'grant_type': 'password',
            'username': args.username,
            'password': args.password,
        })
    assert resp.status_code < 300
    with open(args.creds_file, 'w') as creds_out:
        json.dump(resp.json(), creds_out)


if __name__ == '__main__':
    main()
