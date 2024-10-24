import json
import urllib.request
from urllib.request import OpenerDirector
from warnings import warn

from albumentations import __version__ as current_version

__version__: str = current_version  # type: ignore[has-type, unused-ignore]

SUCCESS_HTML_CODE = 200

opener = None


def get_opener() -> OpenerDirector:
    global opener  # noqa: PLW0603
    if opener is None:
        opener = urllib.request.build_opener(urllib.request.HTTPHandler(), urllib.request.HTTPSHandler())
    return opener


def fetch_version_info() -> str:
    opener = get_opener()
    url = "https://pypi.org/pypi/albumentations/json"
    try:
        with opener.open(url, timeout=2) as response:
            if response.status == SUCCESS_HTML_CODE:
                data = response.read()
                encoding = response.info().get_content_charset("utf-8")
                return data.decode(encoding)
    except Exception as e:  # noqa: BLE001
        warn(f"Error fetching version info {e}", stacklevel=2)
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
            warn(
                f"A new version of Albumentations is available: {latest_version} (you have {current_version}). "  # noqa: S608
                "Upgrade using: pip install -U albumentations. "
                "To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.",
                UserWarning,
                stacklevel=2,
            )
    except Exception as e:  # General exception catch to ensure silent failure  # noqa: BLE001
        warn(
            f"Failed to check for updates due to an unexpected error: {e}. "  # noqa: S608
            "To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.",
            UserWarning,
            stacklevel=2,
        )
