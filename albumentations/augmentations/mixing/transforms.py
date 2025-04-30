"""Transforms that combine multiple images and their associated annotations.

This module contains transformations that take multiple input sources (e.g., a primary image
and additional images provided via metadata) and combine them into a single output.
Examples include overlaying elements (`OverlayElements`) or creating complex compositions
like `Mosaic`.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Annotated, Any, Literal, cast

import cv2
import numpy as np
from pydantic import AfterValidator, model_validator
from typing_extensions import Self

from albumentations.augmentations.mixing import functional as fmixing
from albumentations.core.bbox_utils import BboxProcessor, check_bboxes, denormalize_bboxes, filter_bboxes
from albumentations.core.keypoints_utils import KeypointsProcessor
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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare primary data (base image and mask)
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> mask = np.zeros((300, 300), dtype=np.uint8)
        >>>
        >>> # 1. Create a simple overlay image (a red square)
        >>> overlay_image1 = np.zeros((50, 50, 3), dtype=np.uint8)
        >>> overlay_image1[:, :, 0] = 255  # Red color
        >>>
        >>> # 2. Create another overlay with a mask (a blue circle with transparency)
        >>> overlay_image2 = np.zeros((80, 80, 3), dtype=np.uint8)
        >>> overlay_image2[:, :, 2] = 255  # Blue color
        >>> overlay_mask2 = np.zeros((80, 80), dtype=np.uint8)
        >>> # Create a circular mask
        >>> center = (40, 40)
        >>> radius = 30
        >>> for i in range(80):
        ...     for j in range(80):
        ...         if (i - center[0])**2 + (j - center[1])**2 < radius**2:
        ...             overlay_mask2[i, j] = 255
        >>>
        >>> # 3. Create an overlay with both bbox and mask_id
        >>> overlay_image3 = np.zeros((60, 120, 3), dtype=np.uint8)
        >>> overlay_image3[:, :, 1] = 255  # Green color
        >>> # Create a rectangular mask with rounded corners
        >>> overlay_mask3 = np.zeros((60, 120), dtype=np.uint8)
        >>> cv2.rectangle(overlay_mask3, (10, 10), (110, 50), 255, -1)
        >>>
        >>> # Create the metadata list - each item is a dictionary with overlay information
        >>> overlay_metadata = [
        ...     {
        ...         'image': overlay_image1,
        ...         # No bbox provided - will be placed randomly
        ...     },
        ...     {
        ...         'image': overlay_image2,
        ...         'bbox': [0.6, 0.1, 0.9, 0.4],  # Normalized coordinates [x_min, y_min, x_max, y_max]
        ...         'mask': overlay_mask2,
        ...         'mask_id': 1  # This overlay will update the mask with id 1
        ...     },
        ...     {
        ...         'image': overlay_image3,
        ...         'bbox': [0.1, 0.7, 0.5, 0.9],  # Bottom left placement
        ...         'mask': overlay_mask3,
        ...         'mask_id': 2  # This overlay will update the mask with id 2
        ...     }
        ... ]
        >>>
        >>> # Create the transform
        >>> transform = A.Compose([
        ...     A.OverlayElements(p=1.0),
        ... ])
        >>>
        >>> # Apply the transform
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     overlay_metadata=overlay_metadata  # Pass metadata using the default key
        ... )
        >>>
        >>> # Get results with overlays applied
        >>> result_image = result['image']  # Image with the three overlays applied
        >>> result_mask = result['mask']    # Mask with regions labeled using the mask_id values
        >>>
        >>> # Let's verify the mask contains the specified mask_id values
        >>> has_mask_id_1 = np.any(result_mask == 1)  # Should be True
        >>> has_mask_id_2 = np.any(result_mask == 2)  # Should be True

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
    """Combine multiple images and their annotations into a single image using a mosaic grid layout.

    This transform takes a primary input image (and its annotations) and combines it with
    additional images/annotations provided via metadata. It calculates the geometry for
    a mosaic grid, selects additional items, preprocesses annotations consistently
    (handling label encoding updates), applies geometric transformations, and assembles
    the final output.

    Args:
        grid_yx (tuple[int, int]): The number of rows (y) and columns (x) in the mosaic grid.
            Determines the maximum number of images involved (grid_yx[0] * grid_yx[1]).
            Default: (2, 2).
        target_size (tuple[int, int]): The desired output (height, width) for the final mosaic image.
            after cropping the mosaic grid.
        cell_shape (tuple[int, int]): cell shape of each cell in the mosaic grid.
        metadata_key (str): Key in the input dictionary specifying the list of additional data dictionaries
            for the mosaic. Each dictionary in the list should represent one potential additional item.
            Expected keys: 'image' (required, np.ndarray), and optionally 'mask' (np.ndarray),
            'bboxes' (np.ndarray), 'keypoints' (np.ndarray), and any relevant label fields
            (e.g., 'class_labels') corresponding to those specified in `Compose`'s `bbox_params` or
            `keypoint_params`. Default: "mosaic_metadata".
        center_range (tuple[float, float]): Range [0.0-1.0] to sample the center point of the mosaic view
            relative to the valid central region of the conceptual large grid. This affects which parts
            of the assembled grid are visible in the final crop. Default: (0.3, 0.7).
        interpolation (int): OpenCV interpolation flag used for resizing images during geometric processing.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (int): OpenCV interpolation flag used for resizing masks during geometric processing.
            Default: cv2.INTER_NEAREST.
        fill (tuple[float, ...] | float): Value used for padding images if needed during geometric processing.
            Default: 0.
        fill_mask (tuple[float, ...] | float): Value used for padding masks if needed during geometric processing.
            Default: 0.
        p (float): Probability of applying the transform. Default: 0.5.

    Workflow (`get_params_dependent_on_data`):
        1. Calculate Geometry & Visible Cells: Determine which grid cells are visible in the final
           `target_size` crop and their placement coordinates on the output canvas.
        2. Validate Raw Additional Metadata: Filter the list provided via `metadata_key`,
           keeping only valid items (dicts with an 'image' key).
        3. Select Subset of Raw Additional Metadata: Choose a subset of the valid raw items based
           on the number of visible cells requiring additional data.
        4. Preprocess Selected Raw Additional Items: Preprocess bboxes/keypoints for the *selected*
           additional items *only*. This uses shared processors from `Compose`, updating their
           internal state (e.g., `LabelEncoder`) based on labels in these selected items.
        5. Prepare Primary Data: Extract preprocessed primary data fields from the input `data` dictionary
            into a `primary` dictionary.
        6. Determine & Perform Replication: If fewer additional items were selected than needed,
           replicate the preprocessed primary data as required.
        7. Combine Final Items: Create the list of all preprocessed items (primary, selected additional,
           replicated primary) that will be used.
        8. Assign Items to VISIBLE Grid Cells
        9. Process Geometry & Shift Coordinates: For each assigned item:
            a. Apply geometric transforms (Crop, Resize, Pad) to image/mask.
            b. Apply geometric shift to the *preprocessed* bboxes/keypoints based on cell placement.
       10. Return Parameters: Return the processed cell data (image, mask, shifted bboxes, shifted kps)
           keyed by placement coordinates.

    Label Handling:
        - The transform relies on `bbox_processor` and `keypoint_processor` provided by `Compose`.
        - `Compose.preprocess` initially fits the processors' `LabelEncoder` on the primary data.
        - This transform (`Mosaic`) preprocesses the *selected* additional raw items using the same
          processors. If new labels are found, the shared `LabelEncoder` state is updated via its
          `update` method.
        - `Compose.postprocess` uses the final updated encoder state to decode all labels present
          in the mosaic output for the current `Compose` call.
        - The encoder state is transient per `Compose` call.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Reference:
        YOLOv4: Optimal Speed and Accuracy of Object Detection: https://arxiv.org/pdf/2004.10934

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare primary data
        >>> primary_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> primary_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> primary_bboxes = np.array([[10, 10, 40, 40], [50, 50, 90, 90]], dtype=np.float32)
        >>> primary_labels = [1, 2]
        >>>
        >>> # Prepare additional images for mosaic
        >>> additional_image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> additional_mask1 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> additional_bboxes1 = np.array([[20, 20, 60, 60]], dtype=np.float32)
        >>> additional_labels1 = [3]
        >>>
        >>> additional_image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> additional_mask2 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> additional_bboxes2 = np.array([[30, 30, 70, 70]], dtype=np.float32)
        >>> additional_labels2 = [4]
        >>>
        >>> additional_image3 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> additional_mask3 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> additional_bboxes3 = np.array([[5, 5, 45, 45]], dtype=np.float32)
        >>> additional_labels3 = [5]
        >>>
        >>> # Create metadata for additional images - structured as a list of dicts
        >>> mosaic_metadata = [
        ...     {
        ...         'image': additional_image1,
        ...         'mask': additional_mask1,
        ...         'bboxes': additional_bboxes1,
        ...         'labels': additional_labels1
        ...     },
        ...     {
        ...         'image': additional_image2,
        ...         'mask': additional_mask2,
        ...         'bboxes': additional_bboxes2,
        ...         'labels': additional_labels2
        ...     },
        ...     {
        ...         'image': additional_image3,
        ...         'mask': additional_mask3,
        ...         'bboxes': additional_bboxes3,
        ...         'labels': additional_labels3
        ...     }
        ... ]
        >>>
        >>> # Create the transform with Mosaic
        >>> transform = A.Compose([
        ...     A.Mosaic(
        ...         grid_yx=(2, 2),
        ...         target_size=(200, 200),
        ...         cell_shape=(120, 120),
        ...         center_range=(0.4, 0.6),
        ...         fit_mode="cover",
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=primary_image,
        ...     mask=primary_mask,
        ...     bboxes=primary_bboxes,
        ...     labels=primary_labels,
        ...     mosaic_metadata=mosaic_metadata  # Pass the metadata using the default key
        ... )
        >>>
        >>> # Access the transformed data
        >>> mosaic_image = transformed['image']        # Combined mosaic image
        >>> mosaic_mask = transformed['mask']          # Combined mosaic mask
        >>> mosaic_bboxes = transformed['bboxes']      # Combined and repositioned bboxes
        >>> mosaic_labels = transformed['labels']      # Combined labels from all images

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        grid_yx: tuple[int, int]
        target_size: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
        ]
        cell_shape: Annotated[
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
        fit_mode: Literal["cover", "contain"]

        @model_validator(mode="after")
        def _check_cell_shape(self) -> Self:
            if (
                self.cell_shape[0] * self.grid_yx[0] < self.target_size[0]
                or self.cell_shape[1] * self.grid_yx[1] < self.target_size[1]
            ):
                raise ValueError("Target size should be smaller than cell cell_size * grid_yx")
            return self

    def __init__(
        self,
        grid_yx: tuple[int, int] = (2, 2),
        target_size: tuple[int, int] = (512, 512),
        cell_shape: tuple[int, int] = (512, 512),
        center_range: tuple[float, float] = (0.3, 0.7),
        fit_mode: Literal["cover", "contain"] = "cover",
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
        metadata_key: str = "mosaic_metadata",
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
        self.fit_mode = fit_mode
        self.cell_shape = cell_shape

    @property
    def targets_as_params(self) -> list[str]:
        """Get list of targets that should be passed as parameters to transforms.

        Returns:
            list[str]: List containing the metadata key name

        """
        return [self.metadata_key]

    def _calculate_geometry(self, data: dict[str, Any]) -> list[tuple[int, int, int, int]]:
        # Step 1: Calculate Geometry & Cell Placements
        center_xy = fmixing.calculate_mosaic_center_point(
            grid_yx=self.grid_yx,
            cell_shape=self.cell_shape,
            target_size=self.target_size,
            center_range=self.center_range,
            py_random=self.py_random,
        )

        return fmixing.calculate_cell_placements(
            grid_yx=self.grid_yx,
            cell_shape=self.cell_shape,
            target_size=self.target_size,
            center_xy=center_xy,
        )

    def _select_additional_items(self, data: dict[str, Any], num_additional_needed: int) -> list[dict[str, Any]]:
        valid_items = fmixing.filter_valid_metadata(data.get(self.metadata_key), self.metadata_key, data)
        if len(valid_items) > num_additional_needed:
            return self.py_random.sample(valid_items, num_additional_needed)
        return valid_items

    def _preprocess_additional_items(
        self,
        additional_items: list[dict[str, Any]],
        data: dict[str, Any],
    ) -> list[fmixing.ProcessedMosaicItem]:
        if "bboxes" in data or "keypoints" in data:
            bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
            keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
            return fmixing.preprocess_selected_mosaic_items(additional_items, bbox_processor, keypoint_processor)
        return cast("list[fmixing.ProcessedMosaicItem]", list(additional_items))

    def _prepare_final_items(
        self,
        primary: fmixing.ProcessedMosaicItem,
        additional_items: list[fmixing.ProcessedMosaicItem],
        num_needed: int,
    ) -> list[fmixing.ProcessedMosaicItem]:
        num_replications = max(0, num_needed - len(additional_items))
        replicated = [deepcopy(primary) for _ in range(num_replications)]
        return [primary, *additional_items, *replicated]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Orchestrates the steps to calculate mosaic parameters by calling helper methods."""
        cell_placements = self._calculate_geometry(data)

        num_cells = len(cell_placements)
        num_additional_needed = max(0, num_cells - 1)

        additional_items = self._select_additional_items(data, num_additional_needed)

        preprocessed_additional = self._preprocess_additional_items(additional_items, data)

        primary = self.get_primary_data(data)
        final_items = self._prepare_final_items(primary, preprocessed_additional, num_additional_needed)

        placement_to_item_index = fmixing.assign_items_to_grid_cells(
            num_items=len(final_items),
            cell_placements=cell_placements,
            py_random=self.py_random,
        )

        processed_cells = fmixing.process_all_mosaic_geometries(
            canvas_shape=self.target_size,
            cell_shape=self.cell_shape,
            placement_to_item_index=placement_to_item_index,
            final_items_for_grid=final_items,
            fill=self.fill,
            fill_mask=self.fill_mask if self.fill_mask is not None else self.fill,
            fit_mode=self.fit_mode,
            interpolation=self.interpolation,
            mask_interpolation=self.mask_interpolation,
        )

        if "bboxes" in data or "keypoints" in data:
            processed_cells = fmixing.shift_all_coordinates(processed_cells, canvas_shape=self.target_size)

        result = {"processed_cells": processed_cells, "target_shape": self._get_target_shape(data["image"].shape)}
        if "mask" in data:
            result["target_mask_shape"] = self._get_target_shape(data["mask"].shape)
        return result

    @staticmethod
    def get_primary_data(data: dict[str, Any]) -> fmixing.ProcessedMosaicItem:
        """Get a copy of the primary data (data passed in `data` parameter) to avoid modifying the original data.

        Args:
            data (dict[str, Any]): Dictionary containing the primary data.

        Returns:
            fmixing.ProcessedMosaicItem: A copy of the primary data.

        """
        mask = data.get("mask")
        if mask is not None:
            mask = mask.copy()
        bboxes = data.get("bboxes")
        if bboxes is not None:
            bboxes = bboxes.copy()
        keypoints = data.get("keypoints")
        if keypoints is not None:
            keypoints = keypoints.copy()
        return {
            "image": data["image"],
            "mask": mask,
            "bboxes": bboxes,
            "keypoints": keypoints,
        }

    def _get_target_shape(self, np_shape: tuple[int, ...]) -> list[int]:
        target_shape = list(np_shape)
        target_shape[0] = self.target_size[0]
        target_shape[1] = self.target_size[1]
        return target_shape

    def apply(
        self,
        img: np.ndarray,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply mosaic transformation to the input image.

        Args:
            img (np.ndarray): Input image
            processed_cells (dict[tuple[int, int, int, int], dict[str, Any]]): Dictionary of processed cell data
            target_shape (tuple[int, int]): Shape of the target image.
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Mosaic transformed image

        """
        return fmixing.assemble_mosaic_from_processed_cells(
            processed_cells=processed_cells,
            target_shape=target_shape,
            dtype=img.dtype,
            data_key="image",
            fill=self.fill,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_mask_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply mosaic transformation to the input mask.

        Args:
            mask (np.ndarray): Input mask.
            processed_cells (dict): Dictionary of processed cell data containing cropped/padded mask segments.
            target_mask_shape (tuple[int, int]): Shape of the target mask.
            **params (Any): Additional parameters (unused).

        Returns:
            np.ndarray: Mosaic transformed mask.

        """
        return fmixing.assemble_mosaic_from_processed_cells(
            processed_cells=processed_cells,
            target_shape=target_mask_shape,
            dtype=mask.dtype,
            data_key="mask",
            fill=self.fill_mask,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,  # Original bboxes - ignored
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        """Applies mosaic transformation to bounding boxes.

        Args:
            bboxes (np.ndarray): Original bounding boxes (ignored).
            processed_cells (dict): Dictionary mapping placement coords to processed cell data
                                    (containing shifted bboxes in absolute pixel coords).
            **params (Any): Additional parameters (unused).

        Returns:
            np.ndarray: Final combined, filtered, bounding boxes.

        """
        all_shifted_bboxes = []

        for cell_data in processed_cells.values():
            shifted_bboxes = cell_data["bboxes"]
            if shifted_bboxes.size > 0:
                all_shifted_bboxes.append(shifted_bboxes)

        if not all_shifted_bboxes:
            return np.empty((0, bboxes.shape[1]), dtype=bboxes.dtype)

        # Concatenate (these are absolute pixel coordinates)
        combined_bboxes = np.concatenate(all_shifted_bboxes, axis=0)

        # Apply filtering using processor parameters
        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
        # Assume processor exists if bboxes are being processed
        shape_dict: dict[Literal["depth", "height", "width"], int] = {
            "height": self.target_size[0],
            "width": self.target_size[1],
        }
        return filter_bboxes(
            combined_bboxes,
            shape_dict,
            min_area=bbox_processor.params.min_area,
            min_visibility=bbox_processor.params.min_visibility,
            min_width=bbox_processor.params.min_width,
            min_height=bbox_processor.params.min_height,
            max_accept_ratio=bbox_processor.params.max_accept_ratio,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,  # Original keypoints - ignored
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        """Applies mosaic transformation to keypoints.

        Args:
            keypoints (np.ndarray): Original keypoints (ignored).
            processed_cells (dict): Dictionary mapping placement coords to processed cell data
                                    (containing shifted keypoints).
            **params (Any): Additional parameters (unused).

        Returns:
            np.ndarray: Final combined, filtered keypoints.

        """
        all_shifted_keypoints = []

        for cell_data in processed_cells.values():
            shifted_keypoints = cell_data["keypoints"]
            if shifted_keypoints.size > 0:
                all_shifted_keypoints.append(shifted_keypoints)

        if not all_shifted_keypoints:
            return np.empty((0, keypoints.shape[1]), dtype=keypoints.dtype)

        combined_keypoints = np.concatenate(all_shifted_keypoints, axis=0)

        # Filter out keypoints outside the target canvas boundaries
        target_h, target_w = self.target_size
        valid_indices = (
            (combined_keypoints[:, 0] >= 0)
            & (combined_keypoints[:, 0] < target_w)
            & (combined_keypoints[:, 1] >= 0)
            & (combined_keypoints[:, 1] < target_h)
        )

        return combined_keypoints[valid_indices]
