import pytest
import numpy as np
from albumentations.core.utils import LabelEncoder

@pytest.mark.parametrize("input_labels, expected_encoded, expected_decoded", [
    (["a", "b", "c", "a", "b"], [0, 1, 2, 0, 1], ["a", "b", "c", "a", "b"]),
    ([1, 2, 3, 1, 2], [0, 1, 2, 0, 1], [1, 2, 3, 1, 2]),
    ([1.1, 2.2, 3.3, 1.1, 2.2], [0, 1, 2, 0, 1], [1.1, 2.2, 3.3, 1.1, 2.2]),
    (["a", "b", "c", "d", "e"], [0, 1, 2, 3, 4], ["a", "b", "c", "d", "e"]),
    ([], [], []),
    (np.array(["a", "b", "c", "a", "b"]), [0, 1, 2, 0, 1], ["a", "b", "c", "a", "b"]),
    (np.array([1, 2, 3, 1, 2]), [0, 1, 2, 0, 1], [1, 2, 3, 1, 2]),
    (np.array([1.1, 2.2, 3.3, 1.1, 2.2]), [0, 1, 2, 0, 1], [1.1, 2.2, 3.3, 1.1, 2.2]),
])
def test_label_encoder(input_labels, expected_encoded, expected_decoded):
    encoder = LabelEncoder()

    # Test fit and transform separately
    encoder.fit(input_labels)
    encoded = encoder.transform(input_labels)
    np.testing.assert_array_equal(encoded, expected_encoded)

    # Test fit_transform
    encoded = encoder.fit_transform(input_labels)
    np.testing.assert_array_equal(encoded, expected_encoded)

    # Test inverse_transform
    decoded = encoder.inverse_transform(encoded)
    np.testing.assert_array_equal(decoded, expected_decoded)

@pytest.mark.parametrize("input_labels", [
    ["a", "b", "c", "a", "b"],
    [1, 2, 3, 1, 2],
    [1.1, 2.2, 3.3, 1.1, 2.2],
    np.array(["a", "b", "c", "a", "b"]),
    np.array([1, 2, 3, 1, 2]),
    np.array([1.1, 2.2, 3.3, 1.1, 2.2]),
])
def test_label_encoder_with_numpy_array(input_labels):
    encoder = LabelEncoder()
    input_array = np.array(input_labels)

    encoded = encoder.fit_transform(input_array)
    assert isinstance(encoded, np.ndarray)

    decoded = encoder.inverse_transform(encoded)
    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, input_array)

def test_label_encoder_empty_input():
    encoder = LabelEncoder()
    empty_list = []
    empty_array = np.array([])

    encoded_list = encoder.fit_transform(empty_list)
    assert len(encoded_list) == 0

    encoded_array = encoder.fit_transform(empty_array)
    assert len(encoded_array) == 0

def test_label_encoder_unknown_label():
    encoder = LabelEncoder()
    encoder.fit(["a", "b", "c"])

    with pytest.raises(KeyError):
        encoder.transform(["d"])

def test_label_encoder_unknown_code():
    encoder = LabelEncoder()
    encoder.fit(["a", "b", "c"])

    with pytest.raises(KeyError):
        encoder.inverse_transform([3])

def test_label_encoder_2d_array():
    encoder = LabelEncoder()
    input_array = np.array([[1, 2], [3, 1], [2, 3]])

    encoded = encoder.fit_transform(input_array)
    assert encoded.shape == (6,)
    np.testing.assert_array_equal(encoded, [0, 1, 2, 0, 1, 2])

    decoded = encoder.inverse_transform(encoded)
    np.testing.assert_array_equal(decoded, [1, 2, 3, 1, 2, 3])
