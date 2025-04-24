import numpy as np
import pytest
from collections.abc import Sequence

from albumentations.core.label_manager import LabelEncoder, LabelManager


@pytest.mark.parametrize(
    "input_labels, expected_encoded, expected_decoded",
    [
        (["a", "b", "c", "a", "b"], [0, 1, 2, 0, 1], ["a", "b", "c", "a", "b"]),
        ([1, 2, 3, 1, 2], [1, 2, 3, 1, 2], [1, 2, 3, 1, 2]),
        ([1.1, 2.2, 3.3, 1.1, 2.2], [1.1, 2.2, 3.3, 1.1, 2.2], [1.1, 2.2, 3.3, 1.1, 2.2]),
        (["a", "b", "c", "d", "e"], [0, 1, 2, 3, 4], ["a", "b", "c", "d", "e"]),
        ([], [], []),
        (np.array(["a", "b", "c", "a", "b"]), [0, 1, 2, 0, 1], ["a", "b", "c", "a", "b"]),
        (np.array([1, 2, 3, 1, 2]), np.array([1, 2, 3, 1, 2]), [1, 2, 3, 1, 2]),
        (np.array([1.1, 2.2, 3.3, 1.1, 2.2]), np.array([1.1, 2.2, 3.3, 1.1, 2.2]), [1.1, 2.2, 3.3, 1.1, 2.2]),
        (["a", 1, "b", 2, "a", 1], [2, 0, 3, 1, 2, 0], ["a", 1, "b", 2, "a", 1]),
    ],
)
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


@pytest.mark.parametrize(
    "input_labels",
    [
        ["a", "b", "c", "a", "b"],
        [1, 2, 3, 1, 2],
        [1.1, 2.2, 3.3, 1.1, 2.2],
        np.array(["a", "b", "c", "a", "b"]),
        np.array([1, 2, 3, 1, 2]),
        np.array([1.1, 2.2, 3.3, 1.1, 2.2]),
    ],
)
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
    np.testing.assert_array_equal(encoded, [1, 2, 3, 1, 2, 3])

    decoded = encoder.inverse_transform(encoded)
    np.testing.assert_array_equal(decoded, [1, 2, 3, 1, 2, 3])


# Tests for LabelEncoder Update Functionality
@pytest.mark.parametrize(
    "initial_labels, update_labels, expected_classes, expected_inverse_classes, final_num_classes",
    [
        # Add new distinct labels
        (["a", "b"], ["c", "d"], {"a": 0, "b": 1, "c": 2, "d": 3}, {0: "a", 1: "b", 2: "c", 3: "d"}, 4),
        # Add mixed old and new labels
        (["a", "b"], ["b", "c", "d", "c"], {"a": 0, "b": 1, "c": 2, "d": 3}, {0: "a", 1: "b", 2: "c", 3: "d"}, 4),
        # Add only existing labels
        (["a", "b"], ["a", "b", "a"], {"a": 0, "b": 1}, {0: "a", 1: "b"}, 2),
        # Add numeric labels to existing non-numeric - existing labels keep their indices, new labels are sorted and appended
        (["a", "b"], [1, 2, "a", 1], {"a": 0, "b": 1, 1: 2, 2: 3}, {0: "a", 1: "b", 2: 1, 3: 2}, 4),
        # Initial empty, then update
        ([], ["x", "y"], {"x": 0, "y": 1}, {0: "x", 1: "y"}, 2),
        # Update with empty/None
        (["a", "b"], [], {"a": 0, "b": 1}, {0: "a", 1: "b"}, 2),
        (["a", "b"], None, {"a": 0, "b": 1}, {0: "a", 1: "b"}, 2),
        # Update single item
        (["a"], "b", {"a": 0, "b": 1}, {0: "a", 1: "b"}, 2),
    ]
)
def test_label_encoder_update(initial_labels, update_labels, expected_classes, expected_inverse_classes, final_num_classes):
    encoder = LabelEncoder()
    encoder.fit(initial_labels)
    encoder.update(update_labels)

    assert encoder.classes_ == expected_classes
    assert encoder.inverse_classes_ == expected_inverse_classes
    assert encoder.num_classes == final_num_classes

    # Ensure transform/inverse_transform still work with original labels
    if initial_labels:
        original_encoded = encoder.transform(initial_labels)
        original_decoded = encoder.inverse_transform(original_encoded)
        np.testing.assert_array_equal(original_decoded, np.array(initial_labels).flatten())

    # Ensure transform/inverse_transform work with updated labels (if not empty)
    if update_labels:
        if isinstance(update_labels, str):
             update_labels_list = [update_labels]
        elif isinstance(update_labels, Sequence):
            update_labels_list = list(update_labels)
        else:
            update_labels_list = [] # Should not happen based on params, but safety

        if update_labels_list:
            updated_encoded = encoder.transform(update_labels_list)
            updated_decoded = encoder.inverse_transform(updated_encoded)
            np.testing.assert_array_equal(updated_decoded, np.array(update_labels_list).flatten())

def test_label_encoder_update_numeric_noop():
    """Test that update does nothing if the encoder was fit on numeric data."""
    encoder = LabelEncoder()
    encoder.fit([1, 2, 3])
    initial_classes = encoder.classes_.copy()
    initial_inverse = encoder.inverse_classes_.copy()
    initial_num = encoder.num_classes

    encoder.update([4, 5, "a"])

    assert encoder.classes_ == initial_classes
    assert encoder.inverse_classes_ == initial_inverse
    assert encoder.num_classes == initial_num
    assert encoder.is_numerical is True

# Tests for LabelManager Implicit Update via process_field
@pytest.mark.parametrize(
    "data_name, label_field, initial_data, update_data, expected_final_labels, expected_dtype_after_decode",
    [
        # String labels
        ("bboxes", "class_labels", ["cat", "dog"], ["bird", "cat"], ["cat", "dog", "bird", "cat"], object),
        # Mixed labels
        ("keypoints", "kp_labels", ["head", 1], [2, "tail", "head"], ["head", 1, 2, "tail", "head"], object),
        # Initial empty, then update
        ("bboxes", "instance_ids", [], ["obj1", "obj2"], ["obj1", "obj2"], object),
        # Update with only existing
        ("bboxes", "class_labels", ["cat", "dog"], ["dog", "cat"], ["cat", "dog", "dog", "cat"], object),
    ]
)
def test_label_manager_process_field_updates_encoder(
    data_name, label_field, initial_data, update_data, expected_final_labels, expected_dtype_after_decode
):
    manager = LabelManager()

    # Process initial data
    encoded_initial = manager.process_field(data_name, label_field, initial_data)
    metadata_initial = manager.metadata[data_name][label_field]
    encoder_initial = metadata_initial.encoder
    num_classes_initial = encoder_initial.num_classes if encoder_initial else 0

    # Process update data (should trigger implicit update)
    encoded_update = manager.process_field(data_name, label_field, update_data)
    metadata_updated = manager.metadata[data_name][label_field]
    encoder_updated = metadata_updated.encoder
    num_classes_updated = encoder_updated.num_classes if encoder_updated else 0

    # Check if encoder was updated (if new labels were present)
    if set(update_data) - set(initial_data):
        assert num_classes_updated > num_classes_initial
        assert encoder_updated is encoder_initial # Should be the same instance
    else:
        assert num_classes_updated == num_classes_initial

    # Check restoration of both batches using the final updated encoder
    restored_initial = manager.restore_field(data_name, label_field, encoded_initial)
    restored_update = manager.restore_field(data_name, label_field, encoded_update)

    # Combine original + update data for checking restored combined labels
    combined_original_data = np.concatenate([np.array(initial_data).flatten(), np.array(update_data).flatten()])
    combined_encoded = np.concatenate([encoded_initial, encoded_update])

    # Decode the combined encoded data
    decoded_combined = manager.restore_field(data_name, label_field, combined_encoded)

    # Check final decoded labels (order might differ from input due to sorting in encoder)
    np.testing.assert_array_equal(sorted(decoded_combined, key=str), sorted(expected_final_labels, key=str))

    # Check type preservation
    assert isinstance(restored_initial, list if isinstance(initial_data, np.ndarray) else type(initial_data))
    assert isinstance(restored_update, list if isinstance(update_data, np.ndarray) else type(update_data))


def test_label_manager_process_field_numeric_no_update():
    manager = LabelManager()
    initial_data = [1, 2, 3]
    update_data = [4, 5, 1]

    encoded_initial = manager.process_field("bboxes", "scores", initial_data)
    metadata_initial = manager.metadata["bboxes"]["scores"]
    assert metadata_initial.is_numerical is True
    assert metadata_initial.encoder is None

    encoded_update = manager.process_field("bboxes", "scores", update_data)
    metadata_updated = manager.metadata["bboxes"]["scores"]
    assert metadata_updated.is_numerical is True
    assert metadata_updated.encoder is None

    # Restore and check
    restored_initial = manager.restore_field("bboxes", "scores", encoded_initial)
    restored_update = manager.restore_field("bboxes", "scores", encoded_update)

    assert restored_initial == initial_data
    assert restored_update == update_data


def test_label_manager_process_field_multiple_fields():
    manager = LabelManager()
    initial_data1 = {"bboxes": [1, 2], "class_labels": ["cat", "dog"]}
    update_data1 = {"bboxes": [3, 4], "class_labels": ["bird", "cat"]}

    # Process field 1 (numeric - scores)
    encoded_scores_initial = manager.process_field("bboxes", "scores", initial_data1["bboxes"])
    encoded_scores_update = manager.process_field("bboxes", "scores", update_data1["bboxes"])

    # Process field 2 (categorical - labels)
    encoded_labels_initial = manager.process_field("bboxes", "class_labels", initial_data1["class_labels"])
    encoded_labels_update = manager.process_field("bboxes", "class_labels", update_data1["class_labels"])

    # Check encoders
    assert manager.metadata["bboxes"]["scores"].is_numerical
    assert not manager.metadata["bboxes"]["class_labels"].is_numerical
    assert manager.metadata["bboxes"]["class_labels"].encoder.num_classes == 3 # cat, dog, bird

    # Restore scores
    restored_scores_initial = manager.restore_field("bboxes", "scores", encoded_scores_initial)
    restored_scores_update = manager.restore_field("bboxes", "scores", encoded_scores_update)
    assert restored_scores_initial == initial_data1["bboxes"]
    assert restored_scores_update == update_data1["bboxes"]

    # Restore labels
    restored_labels_initial = manager.restore_field("bboxes", "class_labels", encoded_labels_initial)
    restored_labels_update = manager.restore_field("bboxes", "class_labels", encoded_labels_update)
    assert restored_labels_initial == initial_data1["class_labels"]
    assert restored_labels_update == update_data1["class_labels"]
