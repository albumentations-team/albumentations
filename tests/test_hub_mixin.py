"""Tests for the HubMixin class."""

import os
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import albumentations as A
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
    ],
)
def test_windows_path_handling(path_string, expected_posix):
    """Test that Windows backslashes in paths are correctly handled.

    This test verifies that the from_pretrained method correctly converts
    Windows backslash path separators to forward slashes before using them
    as repo_id for Hugging Face Hub.
    """
    # Create a path object from the string
    path = path_string  # Use string directly, not Path

    # Mock the hf_hub_download function to verify the repo_id parameter
    with patch("albumentations.core.hub_mixin.hf_hub_download") as mock_download:
        # Mock file object as return value
        mock_download.return_value = "mocked_file_path"

        # Mock _from_pretrained to avoid actual file loading
        with patch.object(TestTransform, "_from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = "mocked_transform"

            # Call from_pretrained with the path
            TestTransform.from_pretrained(path)

            # Verify that the repo_id passed to hf_hub_download uses forward slashes
            repo_id = mock_download.call_args[1]["repo_id"]
            assert "\\" not in repo_id, f"Backslash found in repo_id: {repo_id}"

            # Verify that the original path is properly converted
            assert repo_id == expected_posix, f"Expected {expected_posix}, got {repo_id}"


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
