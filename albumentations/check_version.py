import requests

from albumentations import __version__ as current_version

SUCCES_HTML_CODE = 200


def check_for_updates() -> None:
    try:
        response = requests.get("https://pypi.org/pypi/albumentations/json", timeout=0.5)
        if response.status_code == SUCCES_HTML_CODE:
            latest_version = response.json()["info"]["version"]
            if latest_version != current_version:
                print(f"A new version of Albumentations is available: {latest_version} (you have {current_version}).")
                print("Upgrade using: pip install --upgrade albumentations")
    except (requests.Timeout, requests.RequestException):
        # Silently handle any request-related errors or timeouts
        pass
