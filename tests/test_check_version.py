import logging
import pytest
from unittest.mock import MagicMock
import urllib.error

# Correct import statements based on your project structure
from albumentations.check_version import fetch_version_info, parse_version, check_for_updates, SUCCESS_HTML_CODE

@pytest.fixture
def mock_response_success():
    mock_response = MagicMock()
    mock_response.__enter__.return_value.read.return_value = b'{"info": {"version": "1.0.1"}}'
    mock_response.__enter__.return_value.info.return_value.get_content_charset.return_value = 'utf-8'
    mock_response.__enter__.return_value.status = SUCCESS_HTML_CODE
    return mock_response

@pytest.fixture
def mock_response_failure():
    mock_response = MagicMock()
    mock_response.__enter__.return_value.read.return_value = b'{}'
    mock_response.__enter__.return_value.info.return_value.get_content_charset.return_value = 'utf-8'
    mock_response.__enter__.return_value.status = 500  # use a different status code for failure
    return mock_response

def test_fetch_version_info_success(mocker, mock_response_success, caplog):
    mocker.patch('urllib.request.OpenerDirector.open', return_value=mock_response_success)
    with caplog.at_level(logging.DEBUG):
        result = fetch_version_info()
        assert "1.0.1" in result, "Should return version data when HTTP status is 200"

def test_fetch_version_info_failure(mocker, mock_response_failure):
    mocker.patch('urllib.request.OpenerDirector.open', return_value=mock_response_failure)
    result = fetch_version_info()
    assert result == "", "Should return empty string on HTTP failure"

def test_fetch_version_info_timeout(mocker, caplog):
    mocker.patch('urllib.request.OpenerDirector.open', side_effect=urllib.error.URLError("timeout"))
    result = fetch_version_info()
    assert result == "", "Should return empty string on timeout error"

def test_check_for_updates_new_version_available(mocker):
    mocker.patch('albumentations.check_version.fetch_version_info', return_value='{"info": {"version": "1.0.2"}}')
    mocker.patch('albumentations.check_version.current_version', '1.0.1')
    log_mock = mocker.patch('logging.Logger.info')
    check_for_updates()
    log_mock.assert_called_once_with(
        "A new version of Albumentations is available: 1.0.2 (you have 1.0.1). Upgrade using: pip install --upgrade albumentations"
    )

def test_check_for_updates_no_update_needed(mocker):
    mocker.patch('albumentations.check_version.fetch_version_info', return_value='{"info": {"version": "1.0.1"}}')
    mocker.patch('albumentations.check_version.current_version', '1.0.1')
    log_mock = mocker.patch('logging.Logger.info')
    check_for_updates()
    log_mock.assert_not_called()

def test_check_for_updates_exception_handling(mocker):
    mocker.patch('albumentations.check_version.fetch_version_info', side_effect=Exception("Error"))
    log_mock = mocker.patch('logging.Logger.info')
    check_for_updates()
    log_mock.assert_called_once_with("Failed to check for updates due to an unexpected error: Error")

def test_parse_version_correct_input():
    json_input = '{"info": {"version": "1.0.2"}}'
    expected_version = "1.0.2"
    assert parse_version(json_input) == expected_version, "Should correctly parse the version from valid JSON data"

def test_parse_version_empty_input():
    json_input = ''
    assert parse_version(json_input) == "", "Should return an empty string when provided with an empty input"

def test_parse_version_malformed_json():
    json_input = '{"info": {}}'  # Malformed JSON
    assert parse_version(json_input) == "", "Should handle malformed JSON data gracefully by returning an empty string"

def test_parse_version_no_version_field():
    json_input = '{"info": {"release": "2021"}}'  # Missing 'version' field
    assert parse_version(json_input) == "", "Should return an empty string when 'version' field is missing"
