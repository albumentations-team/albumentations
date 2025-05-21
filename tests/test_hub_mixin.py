"""Tests for the HubMixin class."""

import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from albumentations.core.hub_mixin import HubMixin, is_huggingface_hub_available

# Skip tests if huggingface_hub is not available
pytestmark = pytest.mark.skipif(
    not is_huggingface_hub_available,
    reason="huggingface_hub is not available"
)


class TestTransform(HubMixin):
    """Test class for HubMixin."""

    def __init__(self):
        """Initialize test transform."""
        pass


@pytest.mark.parametrize(
    ["path_string", "expected_posix"],
    [
        ("normal/path/format", "normal/path/format"),
        ("windows\\path\\format", "windows/path/format"),
        ("mixed/path\\format", "mixed/path/format"),
        (Path("windows\\path\\format"), "windows/path/format"),
    ],
)
def test_windows_path_handling(path_string, expected_posix):
    """Test that Windows paths are handled correctly in from_pretrained.

    This test verifies that backslashes in Windows paths are properly converted to forward slashes
    when passed to huggingface_hub.hf_hub_download in the from_pretrained method.

    Args:
        path_string: Input path with various formats
        expected_posix: Expected path after conversion to POSIX format
    """
    with patch("albumentations.core.hub_mixin.hf_hub_download") as mock_download:
        mock_download.return_value = "config.json"

        # Also mock _from_pretrained to avoid file operations
        with patch.object(TestTransform, "_from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = "mocked_transform"

            transform = TestTransform()
            transform.from_pretrained(path_string)

            # Check that the repo_id argument was properly formatted
            called_args, _ = mock_download.call_args
            assert called_args == ()
            called_kwargs = mock_download.call_args.kwargs
            assert called_kwargs["repo_id"] == expected_posix


@pytest.mark.skipif(platform.system() != "Windows", reason="Test only relevant on Windows")
def test_real_windows_paths():
    """Test with real Windows paths when running on Windows."""
    with patch("albumentations.core.hub_mixin.hf_hub_download") as mock_download:
        mock_download.return_value = "mocked_file_path"

        with patch.object(TestTransform, "_from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = "mocked_transform"

            # Use a Windows-style path string
            windows_path = "C:\\Users\\test\\models\\my_model"
            TestTransform.from_pretrained(windows_path)

            # Verify the repo_id
            repo_id = mock_download.call_args[1]["repo_id"]
            assert "\\" not in repo_id, f"Backslash found in repo_id: {repo_id}"
            assert repo_id == "C:/Users/test/models/my_model"
