import time
import json
from urllib.parse import urlparse, urlunparse

import requests

from ..io import get_input, open_output

__all__ = [
    'AuthorizationError',
    'OGSClient',
    'get_client',
    'get_game_records',
]


class AuthorizationError(Exception):
    pass


class OGSClient:
    def __init__(self, base_url, token, rate_limit=2.0):
        self.base_url = urlparse(base_url)
        self.token = token
        self.last_request = 0.0
        self.time_between_reqs = 1.0 / rate_limit

    def make_request(self, api_path):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.time_between_reqs:
            sleep_time = self.time_between_reqs - elapsed
            time.sleep(sleep_time)
        parsed = urlparse(api_path)
        url = self.base_url._replace(
            path=parsed.path,
            query=parsed.query,
            fragment=parsed.fragment)
        headers = {}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        response = requests.get(
            urlunparse(url),
            headers={'Authorization': 'Bearer ' + self.token})
        self.last_request = time.time()
        if response.status_code == 401:
            raise AuthorizationError(response.status_code)
        return response.json()


def get_game_records(client):
    game_list = client.make_request('/api/v1/megames')
    while True:
        for g in game_list['results']:
            yield g
        if game_list.get('next'):
            game_list = client.make_request(game_list['next'])
        else:
            break


def token_is_valid(base_url, token):
    client = OGSClient(base_url, token)
    try:
        client.make_request('/api/v1/me')
    except AuthorizationError:
        return False
    return True


def get_client(base_url, auth_file, secrets):
    """
    Args:
        base_url: e.g., 'https://online-go.com'
        auth_file: writable filename containing an access token. If the
            access token is expired, this function will get a new token
            (via the refresh token) and update the credentials in the
            auth file.
        secrets: dictionary containing clientid and secret
    """
    # Check if current access token is still valid.
    with get_input(auth_file) as physical_auth_file:
        old_creds = json.load(open(physical_auth_file))

    token = old_creds['access_token']

    if not token_is_valid(base_url, token):
        url = urlparse(base_url)._replace(path='/oauth2/token/')
        response = requests.post(
            urlunparse(url),
            data = {
                'client_id': secrets['clientid'],
                'client_secret': secrets['secret'],
                'refresh_token': old_creds['refresh_token'],
                'grant_type': 'refresh_token',
            }
        )
        if response.status_code >= 300:
            raise AuthorizationError(
                'Could not refresh token: {}'.format(response.status_code))
        new_creds = response.json()
        token = new_creds['access_token']
        with open_output(auth_file) as auth_outf:
            json.dump(new_creds, auth_outf)

    return OGSClient(base_url, token)
