"""Transforms for mixing and overlaying images.

This module provides transform classes for image mixing operations, including
overlay elements like stickers, logos, or other images onto the input image.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Annotated, Any, Literal
from warnings import warn

import cv2
import numpy as np
from albucore import get_num_channels
from pydantic import AfterValidator

from albumentations.augmentations.mixing import functional as fmixing
from albumentations.core.bbox_utils import check_bboxes, denormalize_bboxes
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import LENGTH_RAW_BBOX, Targets

__all__ = ["Mosaic", "OverlayElements"]


class OverlayElements(DualTransform):
    """Apply overlay elements such as images and masks onto an input image. This transformation can be used to add
    various objects (e.g., stickers, logos) to images with optional masks and bounding boxes for better placement
    control.

    Args:
        metadata_key (str): Additional target key for metadata. Default `overlay_metadata`.
        p (float): Probability of applying the transformation. Default: 0.5.

    Possible Metadata Fields:
        - image (np.ndarray): The overlay image to be applied. This is a required field.
        - bbox (list[int]): The bounding box specifying the region where the overlay should be applied. It should
                            contain four floats: [y_min, x_min, y_max, x_max]. If `label_id` is provided, it should
                            be appended as the fifth element in the bbox. BBox should be in Albumentations format,
                            that is the same as normalized Pascal VOC format
                            [x_min / width, y_min / height, x_max / width, y_max / height]
        - mask (np.ndarray): An optional mask that defines the non-rectangular region of the overlay image. If not
                             provided, the entire overlay image is used.
        - mask_id (int): An optional identifier for the mask. If provided, the regions specified by the mask will
                         be labeled with this identifier in the output mask.

    Targets:
        image, mask

    Image types:
        uint8, float32

    References:
        doc-augmentation: https://github.com/danaaubakirova/doc-augmentation

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        metadata_key: str

    def __init__(
        self,
        metadata_key: str = "overlay_metadata",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        """Get list of targets that should be passed as parameters to transforms.

        Returns:
            list[str]: List containing the metadata key name

        """
        return [self.metadata_key]

    @staticmethod
    def preprocess_metadata(
        metadata: dict[str, Any],
        img_shape: tuple[int, int],
        random_state: random.Random,
    ) -> dict[str, Any]:
        """Process overlay metadata to prepare for application.

        Args:
            metadata (dict[str, Any]): Dictionary containing overlay data such as image, mask, bbox
            img_shape (tuple[int, int]): Shape of the target image as (height, width)
            random_state (random.Random): Random state object for reproducible randomness

        Returns:
            dict[str, Any]: Processed overlay data including resized overlay image, mask,
                            offset coordinates, and bounding box information

        """
        overlay_image = metadata["image"]
        overlay_height, overlay_width = overlay_image.shape[:2]
        image_height, image_width = img_shape[:2]

        if "bbox" in metadata:
            bbox = metadata["bbox"]
            bbox_np = np.array([bbox])
            check_bboxes(bbox_np)
            denormalized_bbox = denormalize_bboxes(bbox_np, img_shape[:2])[0]

            x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])

            if "mask" in metadata:
                mask = metadata["mask"]
                mask = cv2.resize(mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8)

            overlay_image = cv2.resize(overlay_image, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_AREA)
            offset = (y_min, x_min)

            if len(bbox) == LENGTH_RAW_BBOX and "bbox_id" in metadata:
                bbox = [x_min, y_min, x_max, y_max, metadata["bbox_id"]]
            else:
                bbox = (x_min, y_min, x_max, y_max, *bbox[4:])
        else:
            if image_height < overlay_height or image_width < overlay_width:
                overlay_image = cv2.resize(overlay_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
                overlay_height, overlay_width = overlay_image.shape[:2]

            mask = metadata["mask"] if "mask" in metadata else np.ones_like(overlay_image, dtype=np.uint8)

            max_x_offset = image_width - overlay_width
            max_y_offset = image_height - overlay_height

            offset_x = random_state.randint(0, max_x_offset)
            offset_y = random_state.randint(0, max_y_offset)

            offset = (offset_y, offset_x)

            bbox = [
                offset_x,
                offset_y,
                offset_x + overlay_width,
                offset_y + overlay_height,
            ]

            if "bbox_id" in metadata:
                bbox = [*bbox, metadata["bbox_id"]]

        result = {
            "overlay_image": overlay_image,
            "overlay_mask": mask,
            "offset": offset,
            "bbox": bbox,
        }

        if "mask_id" in metadata:
            result["mask_id"] = metadata["mask_id"]

        return result

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Generate parameters for overlay transform based on input data.

        Args:
            params (dict[str, Any]): Dictionary of existing parameters
            data (dict[str, Any]): Dictionary containing input data with image and metadata

        Returns:
            dict[str, Any]: Dictionary containing processed overlay data ready for application

        """
        metadata = data[self.metadata_key]
        img_shape = params["shape"]

        if isinstance(metadata, list):
            overlay_data = [self.preprocess_metadata(md, img_shape, self.py_random) for md in metadata]
        else:
            overlay_data = [self.preprocess_metadata(metadata, img_shape, self.py_random)]

        return {
            "overlay_data": overlay_data,
        }

    def apply(
        self,
        img: np.ndarray,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        """Apply overlay elements to the input image.

        Args:
            img (np.ndarray): Input image
            overlay_data (list[dict[str, Any]]): List of dictionaries containing overlay information
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Image with overlays applied

        """
        for data in overlay_data:
            overlay_image = data["overlay_image"]
            overlay_mask = data["overlay_mask"]
            offset = data["offset"]
            img = fmixing.copy_and_paste_blend(img, overlay_image, overlay_mask, offset=offset)
        return img

    def apply_to_mask(
        self,
        mask: np.ndarray,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        """Apply overlay masks to the input mask.

        Args:
            mask (np.ndarray): Input mask
            overlay_data (list[dict[str, Any]]): List of dictionaries containing overlay information
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Mask with overlay masks applied using the specified mask_id values

        """
        for data in overlay_data:
            if "mask_id" in data and data["mask_id"] is not None:
                overlay_mask = data["overlay_mask"]
                offset = data["offset"]
                mask_id = data["mask_id"]

                y_min, x_min = offset
                y_max = y_min + overlay_mask.shape[0]
                x_max = x_min + overlay_mask.shape[1]

                mask_section = mask[y_min:y_max, x_min:x_max]
                mask_section[overlay_mask > 0] = mask_id

        return mask


class Mosaic(DualTransform):
    """Combine multiple images into one in an NxM grid mosaic style. This transformation requires the primary input
    image and a sequence of additional images provided via metadata.

    Args:
        grid_yx (tuple[int, int]): The number of rows (y) and columns (x) in the mosaic grid.
            Determines the total number of images required (grid_yx[0] * grid_yx[1]). Default: (2, 2).
        target_size (tuple[int, int]):
            The desired output (height, width) for the final mosaic image.
        metadata_key (str): Key in the input dictionary specifying the list of additional data dictionaries
            for the mosaic. Each dictionary in the list should represent one image and its properties.
            Expected keys within each dictionary are 'image' (required), and optionally 'mask', 'bboxes', 'keypoints'.
            The transform requires grid_yx[0] * grid_yx[1] - 1 additional data dictionaries.
            - If more are provided, a random subset is chosen.
            - If fewer are provided, the primary input image data (image, mask, bboxes, keypoints) is replicated
              to fill the remaining slots.
            Default: "mosaic_metadata".
        center_range (tuple[float, float]):
            The range relative to the target size to sample the center point of the mosaic view.
            Affects which parts of the assembled grid are visible in the final crop. Default: (0.3, 0.7).
        interpolation (OpenCV flag):
            OpenCV interpolation flag used for resizing images. Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag):
            OpenCV interpolation flag used for resizing masks. Default: cv2.INTER_NEAREST.
        fill (tuple[float, ...] | float):
            The constant value(s) to use for padding images if keep_aspect_ratio is True.
            Default: 0.
        fill_mask (tuple[float, ...] | float):
            The constant value(s) to use for padding masks if keep_aspect_ratio is True.
            Default: 0.
        p (float): Probability of applying the transform. Default: 0.5.


    Note: Bounding boxes and keypoints of the additional data must be formatted in the same way as specified in
        BboxParams and KeypointParams of the Compose() method. Especially, bounding boxes must include at least
        a fifth element encoding the bbox label.

    Algorithm Description (Conceptual for 2x2):
        1. Takes the input image and 3 additional images provided via `metadata_key`.
        2. Randomly samples a center point based on `center_range` relative to `target_size`.
        3. Resizes the input image and the 3 additional images to `target_size` (potentially with padding).
        4. Builds an empty oversized 2x2 mosaic (twice the `target_size`).
        5. Places the 4 resized images in the quadrants of the oversized mosaic.
        6. Crops the oversized mosaic around the calculated center point to the final `target_size`.
        (Note: The functional backend might use a different but equivalent assembly method).

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Example (2x2):
        >>> import numpy as np
        >>> import albumentations as A

        >>> # Prepare additional mosaic data (3 images for a 2x2 grid)
        >>> mosaic_metadata_input = [
        ...     {'image': np.full((200, 100, 3), fill_value=(255, 255, 0), dtype=np.uint8)},
        ...     {'image': np.full((50, 100, 3), fill_value=(255, 0, 255), dtype=np.uint8)},
        ...     {'image': np.full((50, 50, 3), fill_value=(0, 255, 255), dtype=np.uint8)}
        ... ]

        >>> transform = A.Compose([
        ...     A.Mosaic(
        ...         p=1,
        ...         grid_yx=(2, 2),
        ...         target_size=(100, 100),
        ...         keep_aspect_ratio=True,
        ...         metadata_key="mosaic_metadata"
        ...     )
        ... ])

        >>> # Apply augmentations
        >>> image = np.full((100, 100, 3), fill_value=(255, 255, 255), dtype=np.uint8)
        >>> transformed = transform(image=image, mosaic_metadata=mosaic_metadata_input)
        >>> transformed_image = transformed["image"]

    Reference:
        YOLOv4: Optimal Speed and Accuracy of Object Detection: https://arxiv.org/pdf/2004.10934

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        grid_yx: tuple[int, int]
        target_size: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
        ]
        metadata_key: str
        center_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        grid_yx: tuple[int, int] = (2, 2),
        target_size: tuple[int, int] = (512, 512),
        metadata_key: str = "mosaic_metadata",
        center_range: tuple[float, float] = (0.3, 0.7),
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.grid_yx = grid_yx
        self.target_size = target_size

        self.metadata_key = metadata_key
        self.center_range = center_range
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.fill = fill
        self.fill_mask = fill_mask

    @property
    def targets_as_params(self) -> list[str]:
        """Get list of targets that should be passed as parameters to transforms.

        Returns:
            list[str]: List containing the metadata key name

        """
        return [self.metadata_key]

    def _validate_metadata_item(self, item: Any, index: int) -> bool:
        """Validates a single metadata item.

        Checks if the item is a dictionary and contains the required 'image' key.
        Issues a warning if invalid.

        Args:
            item (Any): The metadata item to validate.
            index (int): The original index of the item for warning messages.

        Returns:
            bool: True if the item is valid, False otherwise.

        """
        if not isinstance(item, dict) or "image" not in item:
            msg = (
                f"Item at index {index} in '{self.metadata_key}' is "
                "invalid (not a dict or lacks 'image' key) and will be skipped."
            )
            warn(msg, UserWarning, stacklevel=4)
            return False
        return True

    def _validate_metadata(self, metadata_input: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Validates the provided metadata sequence.

        Args:
            metadata_input (Sequence | None): The sequence retrieved using metadata_key.

        Returns:
            list[dict[str, Any]]: A new list containing only valid metadata dictionaries.

        """
        if not isinstance(metadata_input, Sequence):
            msg = (
                f"Metadata under key '{self.metadata_key}' is not a "
                "Sequence (e.g., list or tuple). Returning empty list."
            )
            warn(msg, UserWarning, stacklevel=3)
            return []

        # Validate provided items using the helper method
        return [item for i, item in enumerate(metadata_input) if self._validate_metadata_item(item, i)]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Calculate all necessary parameters and process cell data for the mosaic transform."""
        # --- Step 1: Validate raw metadata ---
        raw_metadata_list = data.get(self.metadata_key)
        valid_metadata = self._validate_metadata(raw_metadata_list)

        # --- Step 2: Prepare grid assignment (handles sampling/replication) ---
        primary_data = {k: v for k, v in data.items() if k != self.metadata_key}
        prepared_grid_data = fmixing.prepare_mosaic_inputs(
            primary_data=primary_data,
            additional_data_list=valid_metadata,
            grid_yx=self.grid_yx,
            py_random=self.py_random,
        )

        if not prepared_grid_data:
            warn("Failed to prepare mosaic data. Applying as no-op.", UserWarning, stacklevel=2)
            return {"final_processed_data": {}, "final_placements": {}}  # Signal no-op

        # --- Step 3: Calculate center_point ---
        center_x, center_y = fmixing.calculate_mosaic_center_point(
            grid_yx=self.grid_yx,
            target_size=self.target_size,
            center_range=self.center_range,
            py_random=self.py_random,
        )

        # --- Step 4: Calculate transform coordinates ---
        transforms_coords = fmixing.get_mosaic_transforms_coords(
            grid_yx=self.grid_yx,
            target_size=self.target_size,
            center_point=(center_x, center_y),
        )

        if not transforms_coords:
            warn(
                "Failed to calculate mosaic transform coordinates (likely invalid center point). Applying as no-op.",
                UserWarning,
                stacklevel=2,
            )
            return {"final_processed_data": {}, "final_placements": {}}  # Signal no-op

        # --- Step 5: Process each cell using the new functional helper ---
        final_processed_data, final_placements = fmixing.process_mosaic_grid(
            prepared_grid_data=prepared_grid_data,
            transforms_coords=transforms_coords,
            grid_yx=self.grid_yx,
            interpolation=self.interpolation,
            mask_interpolation=self.mask_interpolation,
            fill=self.fill,
            fill_mask=self.fill_mask if self.fill_mask is not None else self.fill,
        )

        # Final check if we processed any cells
        if not final_processed_data:
            warn("No cells were successfully processed for mosaic. Applying as no-op.", UserWarning, stacklevel=2)
            return {"final_processed_data": {}, "final_placements": {}}

        return {"final_processed_data": final_processed_data, "final_placements": final_placements}

    def apply(
        self,
        img: np.ndarray,
        final_processed_data: dict[tuple[int, int], dict[str, Any]] | None = None,
        final_placements: dict[tuple[int, int], tuple[int, int, int, int]] | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Apply the mosaic transform to the input image.

        Args:
            img (np.ndarray): The input image.
            final_processed_data (dict): The final processed data.
            final_placements (dict): The final placements.
            params (dict): The parameters for the transform.

        Returns:
            np.ndarray: The applied image.

        """
        if not final_processed_data or not final_placements:
            return img  # No-op if params step failed

        assembled_image = fmixing.assemble_mosaic_image_mask(
            processed_data=final_processed_data,
            placements=final_placements,
            target_size=self.target_size,
            num_channels=get_num_channels(img),
            fill=self.fill,
            dtype=img.dtype,
            data_key="image",
        )
        # assemble_mosaic_image_mask should always return an ndarray if called for 'image'
        # but add a fallback just in case
        return assembled_image if assembled_image is not None else img

    def apply_to_mask(
        self,
        mask: np.ndarray,
        final_processed_data: dict[tuple[int, int], dict[str, Any]] | None = None,
        final_placements: dict[tuple[int, int], tuple[int, int, int, int]] | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Apply mask to the final processed data.

        Args:
            mask (np.ndarray): The mask to apply.
            final_processed_data (dict): The final processed data.
            final_placements (dict): The final placements.
            params (dict): The parameters for the transform.

        Returns:
            np.ndarray: The applied mask.

        """
        if not final_processed_data or not final_placements:
            return mask

        # Check if all relevant cells have masks
        all_masks_present = True
        for grid_pos, cell_data in final_processed_data.items():
            if cell_data.get("mask") is None:
                all_masks_present = False
                warn(
                    f"Cell {grid_pos} is missing required mask data. Cannot apply mosaic to mask.",
                    UserWarning,
                    stacklevel=2,
                )
                break

        if not all_masks_present:
            return mask

        assembled_mask = fmixing.assemble_mosaic_image_mask(
            processed_data=final_processed_data,
            placements=final_placements,
            target_size=self.target_size,
            num_channels=get_num_channels(mask),
            fill=self.fill_mask if self.fill_mask is not None else self.fill,
            dtype=mask.dtype,
            data_key="mask",
        )

        return assembled_mask if assembled_mask is not None else mask

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        final_processed_data: dict[tuple[int, int], dict[str, Any]] | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Apply bboxes to the final processed data.

        Args:
            bboxes (np.ndarray): The bboxes to apply.
            final_processed_data (dict): The final processed data.
            params (dict): The parameters for the transform.

        Returns:
            np.ndarray: The applied bboxes.

        """
        # Determine expected shape for empty return based on input bboxes
        empty_shape_cols = bboxes.shape[1] if bboxes is not None and bboxes.ndim == 2 else 5
        empty_dtype = bboxes.dtype if bboxes is not None else np.float32
        empty_bboxes = np.empty((0, empty_shape_cols), dtype=empty_dtype)

        if not final_processed_data:
            return empty_bboxes

        target_h, target_w = self.target_size
        collected_bboxes = []

        for grid_pos, cell_data in final_processed_data.items():
            cell_bboxes = cell_data.get("bboxes")
            if cell_bboxes is not None:
                cell_bboxes_np = np.array(cell_bboxes) if not isinstance(cell_bboxes, np.ndarray) else cell_bboxes
                if cell_bboxes_np.size > 0:
                    # Ensure correct number of columns before appending
                    if cell_bboxes_np.shape[1] == empty_shape_cols:
                        collected_bboxes.append(cell_bboxes_np)
                    else:
                        msg = (
                            f"Bbox shape mismatch in cell {grid_pos}. Expected {empty_shape_cols} columns, "
                            f"got {cell_bboxes_np.shape[1]}. Skipping."
                        )
                        warn(
                            msg,
                            UserWarning,
                            stacklevel=2,
                        )

        if not collected_bboxes:
            return empty_bboxes

        # Concatenate all collected bboxes (already shifted in get_params)
        concat_bboxes = np.concatenate(collected_bboxes, axis=0).astype(np.float32)

        # Clip coordinates to the final target size boundaries
        concat_bboxes[:, 0] = np.clip(concat_bboxes[:, 0], 0, target_w)
        concat_bboxes[:, 1] = np.clip(concat_bboxes[:, 1], 0, target_h)
        concat_bboxes[:, 2] = np.clip(concat_bboxes[:, 2], 0, target_w)
        concat_bboxes[:, 3] = np.clip(concat_bboxes[:, 3], 0, target_h)

        # Remove bboxes with zero area after clipping
        widths = concat_bboxes[:, 2] - concat_bboxes[:, 0]
        heights = concat_bboxes[:, 3] - concat_bboxes[:, 1]
        # Use a small epsilon for float comparison, ensure width/height > 0
        valid_indices = (widths > 1e-3) & (heights > 1e-3)

        return concat_bboxes[valid_indices]

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        final_processed_data: dict[tuple[int, int], dict[str, Any]] | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Apply keypoints to the final processed data.

        Args:
            keypoints (np.ndarray): The keypoints to apply.
            final_processed_data (dict): The final processed data.
            params (dict): The parameters for the transform.

        Returns:
            np.ndarray: The applied keypoints.

        """
        # Determine expected shape for empty return
        empty_shape_cols = (
            keypoints.shape[1] if keypoints is not None and keypoints.ndim == 2 else 4
        )  # Default to x,y,angle,scale
        empty_dtype = keypoints.dtype if keypoints is not None else np.float32
        empty_keypoints = np.empty((0, empty_shape_cols), dtype=empty_dtype)

        if not final_processed_data:
            return empty_keypoints

        target_h, target_w = self.target_size
        collected_keypoints = []

        for grid_pos, cell_data in final_processed_data.items():
            cell_keypoints = cell_data.get("keypoints")
            if cell_keypoints is not None:
                cell_keypoints_np = (
                    np.array(cell_keypoints) if not isinstance(cell_keypoints, np.ndarray) else cell_keypoints
                )
                if cell_keypoints_np.size > 0:
                    if cell_keypoints_np.shape[1] == empty_shape_cols:
                        collected_keypoints.append(cell_keypoints_np)
                    else:
                        msg = (
                            f"Keypoint shape mismatch in cell {grid_pos}. Expected {empty_shape_cols} columns, "
                            f"got {cell_keypoints_np.shape[1]}. Skipping."
                        )
                        warn(
                            msg,
                            UserWarning,
                            stacklevel=2,
                        )

        if not collected_keypoints:
            return empty_keypoints

        # Concatenate all collected keypoints (already shifted)
        concat_keypoints = np.concatenate(collected_keypoints, axis=0).astype(np.float32)

        # Filter out keypoints outside the target canvas boundaries
        valid_indices = (
            (concat_keypoints[:, 0] >= 0)
            & (concat_keypoints[:, 0] < target_w)
            & (concat_keypoints[:, 1] >= 0)
            & (concat_keypoints[:, 1] < target_h)
        )

        return concat_keypoints[valid_indices]
