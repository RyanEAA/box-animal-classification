import os
import sys
import webbrowser
import time
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import requests
from dotenv import find_dotenv, load_dotenv, set_key

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


TOKEN_URL = "https://api.box.com/oauth2/token"
AUTH_URL = "https://account.box.com/api/oauth2/authorize"

# Global to store the authorization code from redirect
captured_code = None


class RedirectHandler(BaseHTTPRequestHandler):
    """Handle the OAuth redirect and capture the authorization code."""
    
    def do_GET(self):
        global captured_code
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        
        if code:
            captured_code = code
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>")
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Authorization failed!</h1><p>No code received.</p></body></html>")
    
    def log_message(self, format, *args):
        pass  # Suppress logging


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
    max_retries: int = 3,
) -> dict:
    for attempt in range(max_retries):
        try:
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
                error_msg = f"Token exchange failed ({response.status_code}): {response.text}"
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {error_msg}")
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError(error_msg)

            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"HTTP error on attempt {attempt + 1}: {e}")
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                raise RuntimeError(f"Token exchange failed after {max_retries} attempts: {e}")


def auto_grant_access(auth_url: str, redirect_uri: str, use_auto_grant: bool = False) -> str:
    """
    Open browser, optionally auto-click grant button, and capture the authorization code.
    
    Returns the authorization code if successful.
    """
    global captured_code
    
    if not use_auto_grant or not SELENIUM_AVAILABLE:
        # Manual mode: just open the browser
        try:
            opened = webbrowser.open(auth_url)
            if opened:
                print("Opened browser automatically.")
        except Exception:
            pass
        
        print("\nPaste either the full redirected URL or just the code value.")
        user_input = input("Code/URL: ")
        return extract_code(user_input)
    
    # Auto-grant mode using Selenium
    print("Opening browser with Selenium to auto-grant access...")
    
    # Start local redirect server
    captured_code = None
    redirect_port = int(redirect_uri.split(":")[-1])
    
    server = HTTPServer(("localhost", redirect_port), RedirectHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Started local redirect server on port {redirect_port}")
    
    driver = None
    try:
        # Use headless Chrome if available
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")  # Comment out to see the browser
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=options)
        print("Opening authorization URL...")
        driver.get(auth_url)
        
        # Wait for the grant button and click it
        print("Waiting for grant access button...")
        try:
            # Wait up to 60 seconds for the button
            button = WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@data-target-id, 'Button-grantAccessButtonLabel')]"))
            )
            print("Found grant button, clicking...")
            button.click()
            
            # Wait for redirect
            print("Waiting for authorization redirect...")
            for _ in range(30):  # Wait up to 30 seconds
                if captured_code:
                    print(f"Authorization code captured: {captured_code[:10]}...")
                    return captured_code
                time.sleep(1)
            
            if not captured_code:
                raise RuntimeError("Authorization timed out - no code captured")
        
        except Exception as e:
            print(f"Error during auto-grant: {e}")
            print("Falling back to manual code entry...")
            print("\nPaste either the full redirected URL or just the code value.")
            user_input = input("Code/URL: ")
            return extract_code(user_input)
    
    finally:
        if driver:
            driver.quit()
        server.shutdown()


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

    # Check if running in auto-grant mode (set USE_AUTO_GRANT=true to enable)
    use_auto_grant = os.getenv("USE_AUTO_GRANT", "false").lower() == "true"
    
    if use_auto_grant and not SELENIUM_AVAILABLE:
        print("\nWarning: USE_AUTO_GRANT is enabled but Selenium is not installed.")
        print("Install it with: pip install selenium")
        print("Falling back to manual mode.\n")
        use_auto_grant = False

    code = auto_grant_access(auth_url, redirect_uri, use_auto_grant)

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
        print("\nOAuth setup failed. Please check your credentials and try again.")
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
