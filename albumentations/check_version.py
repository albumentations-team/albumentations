import logging

import requests

from albumentations import __version__ as current_version

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

SUCCESS_HTML_CODE = 200


def check_for_updates() -> None:
    try:
        response = requests.get("https://pypi.org/pypi/albumentations/json", timeout=2)
        if response.status_code == SUCCESS_HTML_CODE:
            latest_version = response.json()["info"]["version"]
            if latest_version != current_version:
                logger.info(
                    f"A new version of Albumentations is available: {latest_version}"
                    f" (you have {current_version}). Upgrade using: pip install -U albumentations"
                )
    except (requests.Timeout, requests.RequestException):
        # Silently handle any request-related errors or timeouts
        pass
