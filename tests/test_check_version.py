from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from albumentations.check_version import (
    check_for_updates,
    fetch_version_info,
    get_opener,
    parse_version,
    parse_version_parts,
    compare_versions,
)


def test_get_opener():
    opener = get_opener()
    assert opener is not None
    assert opener == get_opener()  # Should return the same opener on subsequent calls


@pytest.mark.parametrize(
    "status_code,expected_result",
    [
        (200, '{"info": {"version": "1.0.0"}}'),
        (404, ""),
    ],
)
def test_fetch_version_info(status_code, expected_result):
    mock_response = MagicMock()
    mock_response.status = status_code
    mock_response.read.return_value = expected_result.encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    mock_info = MagicMock()
    mock_info.get_content_charset.return_value = "utf-8"
    mock_response.info.return_value = mock_info

    mock_opener = MagicMock()
    mock_opener.open.return_value = mock_response

    with patch("albumentations.check_version.get_opener", return_value=mock_opener):
        result = fetch_version_info()
        assert result == expected_result

@pytest.mark.parametrize(
    "input_data,expected_version",
    [
        ('{"info": {"version": "1.0.0"}}', "1.0.0"),
        ('{"info": {}}', ""),
        ('{"other": "data"}', ""),
        ("invalid json", ""),
        ("", ""),
    ],
)
def test_parse_version(input_data, expected_version):
    assert parse_version(input_data) == expected_version


@pytest.mark.parametrize(
    "fetch_data,current_version,expected_warning",
    [
        ('{"info": {"version": "1.0.1"}}', "1.0.0", True),
        ('{"info": {"version": "1.0.0"}}', "1.0.0", False),
        ('{"info": {}}', "1.0.0", False),
        ("invalid json", "1.0.0", False),
    ],
)
def test_check_for_updates(fetch_data, current_version, expected_warning):
    with (
        patch("albumentations.check_version.fetch_version_info", return_value=fetch_data),
        patch("albumentations.check_version.current_version", current_version),
        patch("albumentations.check_version.warn") as mock_warn,
    ):
        check_for_updates()
        assert mock_warn.called == expected_warning


def test_check_for_updates_exception():
    with (
        patch("albumentations.check_version.fetch_version_info", side_effect=Exception("Test error")),
        patch("albumentations.check_version.warn") as mock_warn,
    ):
        check_for_updates()
        mock_warn.assert_called_once()
        assert "Failed to check for updates" in mock_warn.call_args[0][0]

@pytest.mark.parametrize("response_data, expected_version", [
    ('{"info": {"version": "1.0.0"}}', "1.0.0"),
    ('{"info": {}}', ""),
    ('{}', ""),
    ('', ""),
    ('invalid json', ""),
])
def test_parse_version(response_data: str, expected_version: str):
    assert parse_version(response_data) == expected_version

def test_fetch_version_info_success():
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = b'{"info": {"version": "1.0.0"}}'
    mock_response.info.return_value.get_content_charset.return_value = "utf-8"
    # Set up the context manager behavior
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    mock_opener = MagicMock()
    mock_opener.open.return_value = mock_response

    with patch("albumentations.check_version.get_opener", return_value=mock_opener):
        data = fetch_version_info()
        assert data == '{"info": {"version": "1.0.0"}}'

def test_fetch_version_info_failure():
    with patch("urllib.request.OpenerDirector.open", side_effect=Exception("Network error")):
        data = fetch_version_info()
        assert data == ""

def test_check_for_updates_no_update():
    with patch("albumentations.check_version.fetch_version_info", return_value='{"info": {"version": "1.0.0"}}'):
        with patch("albumentations.check_version.__version__", "1.0.0"):
            with patch("warnings.warn") as mock_warn:
                check_for_updates()
                mock_warn.assert_not_called()

def test_check_for_updates_with_update():
    with patch("albumentations.check_version.fetch_version_info", return_value='{"info": {"version": "2.0.0"}}'):
        with patch("albumentations.check_version.current_version", "1.0.0"):
            with patch("albumentations.check_version.warn") as mock_warn:  # Patch the imported warn
                check_for_updates()
                mock_warn.assert_called_once()



@pytest.mark.parametrize("version_str, expected", [
    # Standard versions
    ("1.4.24", (1, 4, 24)),
    ("0.0.1", (0, 0, 1)),
    ("10.20.30", (10, 20, 30)),

    # Pre-release versions
    ("1.4beta", (1, 4, "beta")),
    ("1.4beta2", (1, 4, "beta", 2)),
    ("1.4.beta2", (1, 4, "beta", 2)),
    ("1.4.alpha2", (1, 4, "alpha", 2)),
    ("1.4rc1", (1, 4, "rc", 1)),
    ("1.4.rc.1", (1, 4, "rc", 1)),

    # Mixed case handling
    ("1.4Beta2", (1, 4, "beta", 2)),
    ("1.4ALPHA2", (1, 4, "alpha", 2)),
])
def test_parse_version_parts(version_str: str, expected: tuple[int | str, ...]) -> None:
    assert parse_version_parts(version_str) == expected

# Update the test to use the new comparison function
@pytest.mark.parametrize("version1, version2, expected", [
    # Pre-release ordering
    ("1.4beta2", "1.4beta1", True),
    ("1.4", "1.4beta", True),
    ("1.4beta", "1.4alpha", True),
    ("1.4alpha2", "1.4alpha1", True),
    ("1.4rc", "1.4beta", True),
    ("2.0", "2.0rc1", True),

    # Standard version ordering
    ("1.5", "1.4", True),
    ("1.4.1", "1.4", True),
    ("1.4.24", "1.4.23", True),
])
def test_version_comparison(version1: str, version2: str, expected: bool) -> None:
    """Test that version1 > version2 matches expected result."""
    v1 = parse_version_parts(version1)
    v2 = parse_version_parts(version2)
    assert compare_versions(v1, v2) == expected
