import json
import logging
import urllib.request
from urllib.request import OpenerDirector

from albumentations._version import __version__ as current_version

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUCCESS_HTML_CODE = 200

opener = None


def get_opener() -> OpenerDirector:
    global opener  # noqa: PLW0603
    if opener is None:
        opener = urllib.request.build_opener(urllib.request.HTTPHandler(), urllib.request.HTTPSHandler())
    return opener


def fetch_version_info() -> str:
    logger.debug("Starting to fetch version info...")
    opener = urllib.request.build_opener(urllib.request.HTTPHandler(), urllib.request.HTTPSHandler())
    url = "https://pypi.org/pypi/albumentations/json"
    try:
        with opener.open(url, timeout=2) as response:
            logger.debug(f"HTTP status: {response.status}")
            if response.status == SUCCESS_HTML_CODE:
                data = response.read()
                logger.debug(f"Raw data: {data}")
                encoding = response.info().get_content_charset("utf-8")
                decoded_data = data.decode(encoding)
                logger.debug(f"Decoded data: {decoded_data}")
                return decoded_data
    except Exception:
        logger.exception("Error fetching version info")
    return ""


def parse_version(data: str) -> str:
    """Parses the version from the given JSON data."""
    if data:
        try:
            json_data = json.loads(data)
            # Use .get() to avoid KeyError if 'version' is not present
            return json_data.get("info", {}).get("version", "")
        except json.JSONDecodeError:
            # This will handle malformed JSON data
            return ""
    return ""


def check_for_updates() -> None:
    try:
        data = fetch_version_info()
        latest_version = parse_version(data)
        if latest_version and latest_version != current_version:
            logger.info(
                f"A new version of Albumentations is available: {latest_version} (you have {current_version})."
                " Upgrade using: pip install --upgrade albumentations",
            )
    except Exception as e:  # General exception catch to ensure silent failure  # noqa: BLE001
        logger.info(f"Failed to check for updates due to an unexpected error: {e}")
