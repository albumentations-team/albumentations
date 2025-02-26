from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np


def custom_sort(item: Any) -> tuple[int, Real | str]:
    return (0, item) if isinstance(item, Real) else (1, str(item))


class LabelEncoder:
    def __init__(self) -> None:
        self.classes_: dict[str | Real, int] = {}
        self.inverse_classes_: dict[int, str | Real] = {}
        self.num_classes: int = 0
        self.is_numerical: bool = True

    def fit(self, y: Sequence[Any] | np.ndarray) -> LabelEncoder:
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        self.is_numerical = all(isinstance(label, Real) for label in y)

        if self.is_numerical:
            return self

        unique_labels = sorted(set(y), key=custom_sort)
        for label in unique_labels:
            if label not in self.classes_:
                self.classes_[label] = self.num_classes
                self.inverse_classes_[self.num_classes] = label
                self.num_classes += 1
        return self

    def transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

        return np.array([self.classes_[label] for label in y])

    def fit_transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

        return np.array([self.inverse_classes_[label] for label in y])


@dataclass
class LabelMetadata:
    """Stores metadata about a label field."""

    input_type: type
    is_numerical: bool
    dtype: np.dtype | None = None
    encoder: LabelEncoder | None = None


class LabelManager:
    def __init__(self) -> None:
        self.metadata: dict[str, dict[str, LabelMetadata]] = defaultdict(dict)

    def process_field(self, data_name: str, label_field: str, field_data: Any) -> np.ndarray:
        """Process a label field and store its metadata."""
        metadata = self._analyze_input(field_data)
        self.metadata[data_name][label_field] = metadata
        return self._encode_data(field_data, metadata)

    def restore_field(self, data_name: str, label_field: str, encoded_data: np.ndarray) -> Any:
        """Restore a label field to its original format."""
        metadata = self.metadata[data_name][label_field]
        decoded_data = self._decode_data(encoded_data, metadata)
        return self._restore_type(decoded_data, metadata)

    def _analyze_input(self, field_data: Any) -> LabelMetadata:
        """Analyze input data and create metadata."""
        input_type = type(field_data)
        dtype = field_data.dtype if isinstance(field_data, np.ndarray) else None

        # Check if input is numpy array or if all elements are numerical
        is_numerical = (isinstance(field_data, np.ndarray) and np.issubdtype(field_data.dtype, np.number)) or all(
            isinstance(label, (int, float)) for label in field_data
        )

        metadata = LabelMetadata(
            input_type=input_type,
            is_numerical=is_numerical,
            dtype=dtype,
        )

        if not is_numerical:
            metadata.encoder = LabelEncoder()

        return metadata

    def _encode_data(self, field_data: Any, metadata: LabelMetadata) -> np.ndarray:
        """Encode field data for processing."""
        if metadata.is_numerical:
            # For numerical values, convert to float32 for processing
            if isinstance(field_data, np.ndarray):
                return field_data.reshape(-1, 1).astype(np.float32)
            return np.array(field_data, dtype=np.float32).reshape(-1, 1)

        # For non-numerical values, use LabelEncoder
        if metadata.encoder is None:
            raise ValueError("Encoder not initialized for non-numerical data")
        return metadata.encoder.fit_transform(field_data).reshape(-1, 1)

    def _decode_data(self, encoded_data: np.ndarray, metadata: LabelMetadata) -> np.ndarray:
        """Decode processed data."""
        if metadata.is_numerical:
            if metadata.dtype is not None:
                return encoded_data.astype(metadata.dtype)
            return encoded_data.flatten()  # Flatten for list conversion

        if metadata.encoder is None:
            raise ValueError("Encoder not found for non-numerical data")

        decoded = metadata.encoder.inverse_transform(encoded_data.astype(int))
        return decoded.reshape(-1)  # Ensure 1D array

    def _restore_type(self, decoded_data: np.ndarray, metadata: LabelMetadata) -> Any:
        """Restore data to its original type."""
        # If original input was a list or sequence, convert back to list
        if isinstance(metadata.input_type, type) and issubclass(metadata.input_type, (list, Sequence)):
            return decoded_data.tolist()

        # If original input was a numpy array, restore original dtype
        if isinstance(metadata.input_type, type) and issubclass(metadata.input_type, np.ndarray):
            if metadata.dtype is not None:
                return decoded_data.astype(metadata.dtype)
            return decoded_data

        # For any other type, convert to list by default
        return decoded_data.tolist()

    def handle_empty_data(self) -> list[Any]:
        """Handle empty data case."""
        return []
