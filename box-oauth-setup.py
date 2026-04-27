import os
import sys
import webbrowser
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import find_dotenv, load_dotenv, set_key


TOKEN_URL = "https://api.box.com/oauth2/token"
AUTH_URL = "https://account.box.com/api/oauth2/authorize"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_authorize_url(client_id: str, redirect_uri: str) -> str:
    return (
        f"{AUTH_URL}?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
    )


def extract_code(user_input: str) -> str:
    user_input = user_input.strip()

    if "code=" not in user_input:
        print("No 'code=' found in input. Assuming input is the code itself.")
        return user_input
    
    print("Extracting code from URL...")

    parsed = urlparse(user_input)

    if parsed.query:
        params = parse_qs(parsed.query)
        codes = params.get("code", [])
        if codes:
            return codes[0]

    return user_input


def exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
) -> dict:
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed ({response.status_code}): {response.text}"
        )

    return response.json()


def main() -> int:
    load_dotenv()
    env_path = find_dotenv(usecwd=True) or ".env"

    try:
        client_id = require_env("CLIENT_ID")
        client_secret = require_env("CLIENT_SECRET")
        redirect_uri = require_env("REDIRECT_URI")
    except ValueError as exc:
        print(str(exc))
        print("Add missing values to .env and run this script again.")
        return 1

    auth_url = build_authorize_url(client_id, redirect_uri)

    print("Open this URL in your browser and approve access:")
    print(auth_url)

    try:
        opened = webbrowser.open(auth_url)
        if opened:
            print("Opened browser automatically.")
    except Exception:
        pass

    print("\nPaste either the full redirected URL or just the code value.")
    user_input = input("Code/URL: ")
    code = extract_code(user_input)

    if not code:
        print("No code provided.")
        return 1

    try:
        token_data = exchange_code_for_tokens(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            code=code,
        )
    except Exception as exc:
        print(str(exc))
        return 1

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    if not access_token or not refresh_token:
        print("Token response did not include both access_token and refresh_token.")
        print(token_data)
        return 1

    set_key(env_path, "ACCESS_TOKEN", access_token)
    set_key(env_path, "REFRESH_TOKEN", refresh_token)

    print("\nSuccess. Updated .env with ACCESS_TOKEN and REFRESH_TOKEN.")
    expires_in = token_data.get("expires_in")
    if expires_in is not None:
        print(f"Access token expires in approximately {expires_in} seconds.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
