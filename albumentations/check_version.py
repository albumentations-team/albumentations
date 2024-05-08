import json
import logging
import urllib.request

from albumentations import __version__ as current_version

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUCCESS_HTML_CODE = 200


def check_for_updates() -> None:
    try:
        # Create an OpenerDirector that only allows HTTP and HTTPS
        opener = urllib.request.build_opener(urllib.request.HTTPHandler(), urllib.request.HTTPSHandler())
        url = "https://pypi.org/pypi/albumentations/json"

        # Open the URL using the restricted opener
        with opener.open(url, timeout=2) as response:
            if response.status == SUCCESS_HTML_CODE:
                data = response.read()
                encoding = response.info().get_content_charset("utf-8")
                json_data = json.loads(data.decode(encoding))
                latest_version = json_data["info"]["version"]
                if latest_version != current_version:
                    logger.info(
                        f"A new version of Albumentations is available: {latest_version} (you have {current_version})."
                        " Upgrade using: pip install --upgrade albumentations"
                    )
    except urllib.error.URLError:
        # Silently handle URL errors
        pass
    except urllib.error.HTTPError:
        # Silently handle HTTP errors
        pass
