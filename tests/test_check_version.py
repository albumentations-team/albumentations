import json
from unittest.mock import patch, MagicMock
import pytest
from albumentations.check_version import (
    get_opener, fetch_version_info, parse_version, check_for_updates
)

def test_get_opener():
    opener = get_opener()
    assert opener is not None
    assert opener == get_opener()  # Should return the same opener on subsequent calls

@pytest.mark.parametrize("status_code,expected_result", [
    (200, '{"info": {"version": "1.0.0"}}'),
    (404, ""),
])
def test_fetch_version_info(status_code, expected_result):
    mock_response = MagicMock()
    mock_response.status = status_code
    mock_response.read.return_value = expected_result.encode('utf-8')
    mock_response.__enter__.return_value = mock_response

    # Mock the info() method and its get_content_charset() method
    mock_info = MagicMock()
    mock_info.get_content_charset.return_value = 'utf-8'
    mock_response.info.return_value = mock_info

    mock_opener = MagicMock()
    mock_opener.open.return_value = mock_response

    with patch('urllib.request.build_opener', return_value=mock_opener):
        result = fetch_version_info()
        assert result == expected_result

    # Verify that open was called with the correct URL and timeout
    mock_opener.open.assert_called_once_with("https://pypi.org/pypi/albumentations/json", timeout=2)

@pytest.mark.parametrize("input_data,expected_version", [
    ('{"info": {"version": "1.0.0"}}', "1.0.0"),
    ('{"info": {}}', ""),
    ('{"other": "data"}', ""),
    ('invalid json', ""),
    ('', ""),
])
def test_parse_version(input_data, expected_version):
    assert parse_version(input_data) == expected_version

@pytest.mark.parametrize("fetch_data,current_version,expected_warning", [
    ('{"info": {"version": "1.0.1"}}', "1.0.0", True),
    ('{"info": {"version": "1.0.0"}}', "1.0.0", False),
    ('{"info": {}}', "1.0.0", False),
    ('invalid json', "1.0.0", False),
])
def test_check_for_updates(fetch_data, current_version, expected_warning):
    with patch('albumentations.check_version.fetch_version_info', return_value=fetch_data), \
         patch('albumentations.check_version.current_version', current_version), \
         patch('albumentations.check_version.warn') as mock_warn:
        check_for_updates()
        assert mock_warn.called == expected_warning

def test_check_for_updates_exception():
    with patch('albumentations.check_version.fetch_version_info', side_effect=Exception("Test error")), \
         patch('albumentations.check_version.warn') as mock_warn:
        check_for_updates()
        mock_warn.assert_called_once()
        assert "Failed to check for updates" in mock_warn.call_args[0][0]
