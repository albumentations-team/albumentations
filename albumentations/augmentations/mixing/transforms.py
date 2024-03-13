import random
import types
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from warnings import warn

import numpy as np

from albumentations.augmentations.crops.transforms import RandomSizedBBoxSafeCrop
from albumentations.augmentations.functional import split_uniform_grid
from albumentations.augmentations.geometric.resize import LongestMaxSize, Resize, SmallestMaxSize
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from albumentations.augmentations.utils import is_grayscale_image
from albumentations.core.composition import Compose
from albumentations.core.transforms_interface import ReferenceBasedTransform
from albumentations.core.types import BoxType, KeypointType, ReferenceImage, Targets
from albumentations.random_utils import beta, choice, shuffle

from .functional import mix_arrays

__all__ = ["MixUp", "Mosaic"]


class MixUp(ReferenceBasedTransform):
    """Performs MixUp data augmentation, blending images, masks, and class labels with reference data.

    MixUp augmentation linearly combines an input (image, mask, and class label) with another set from a predefined
    reference dataset. The mixing degree is controlled by a parameter λ (lambda), sampled from a Beta distribution.
    This method is known for improving model generalization by promoting linear behavior between classes and
    smoothing decision boundaries.

    Reference:
        Zhang, H., Cisse, M., Dauphin, Y.N., and Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization.
        In International Conference on Learning Representations. https://arxiv.org/abs/1710.09412

    Args:
        reference_data (Optional[Union[Generator[Any, None, None], Sequence[Any]]]):
            A sequence or generator of dictionaries containing the reference data for mixing
            If None or an empty sequence is provided, no operation is performed and a warning is issued.
        read_fn (Callable[[Any], ReferenceImage]):
            A function to process items from reference_data. It should accept items from reference_data
            and return a dictionary containing processed data:
                - The returned dictionary must include an 'image' key with a numpy array value.
                - It may also include 'mask', 'global_label' each associated with numpy array values.
            Defaults to a function that assumes input dictionary contains numpy arrays and directly returns it.
        mix_coef_return_name (str): Name used for the applied alpha coefficient in the returned dictionary.
            Defaults to "mix_coef".
        alpha (float):
            The alpha parameter for the Beta distribution, influencing the mix's balance. Must be ≥ 0.
            Higher values lead to more uniform mixing. Defaults to 0.4.
        p (float):
            The probability of applying the transformation. Defaults to 0.5.

    Targets:
        image, mask, global_label

    Image types:
        - uint8, float32

    Raises:
        - ValueError: If the alpha parameter is negative.
        - NotImplementedError: If the transform is applied to bounding boxes or keypoints.

    Notes:
        - If no reference data is provided, a warning is issued, and the transform acts as a no-op.
        - Notes if images are in float32 format, they should be within [0, 1] range.

    Example Usage:
        import albumentations as A
        import numpy as np
        from albumentations.core.types import ReferenceImage

        # Prepare reference data
        # Note: This code generates random reference data for demonstration purposes only.
        # In real-world applications, it's crucial to use meaningful and representative data.
        # The quality and relevance of your input data significantly impact the effectiveness
        # of the augmentation process. Ensure your data closely aligns with your specific
        # use case and application requirements.
        reference_data = [ReferenceImage(image=np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8),
                                         mask=np.random.randint(0, 4, (100, 100, 1), dtype=np.uint8),
                                         global_label=np.random.choice([0, 1], size=3)) for i in range(10)]

        # In this example, the lambda function simply returns its input, which works well for
        # data already in the expected format. For more complex scenarios, where the data might not be in
        # the required format or additional processing is needed, a more sophisticated function can be implemented.
        # Below is a hypothetical example where the input data is a file path, # and the function reads the image
        # file, converts it to a specific format, and possibly performs other preprocessing steps.

        # Example of a more complex read_fn that reads an image from a file path, converts it to RGB, and resizes it.
        # def custom_read_fn(file_path):
        #     from PIL import Image
        #     image = Image.open(file_path).convert('RGB')
        #     image = image.resize((100, 100))  # Example resize, adjust as needed.
        #     return np.array(image)

        # aug = A.Compose([A.RandomRotate90(), A.MixUp(p=1, reference_data=reference_data, read_fn=lambda x: x)])

        # For simplicity, the original lambda function is used in this example.
        # Replace `lambda x: x` with `custom_read_fn`if you need to process the data more extensively.

        # Apply augmentations
        image = np.empty([100, 100, 3], dtype=np.uint8)
        mask = np.empty([100, 100], dtype=np.uint8)
        global_label = np.array([0, 1, 0])
        data = aug(image=image, global_label=global_label, mask=mask)
        transformed_image = data["image"]
        transformed_mask = data["mask"]
        transformed_global_label = data["global_label"]

        # Print applied mix coefficient
        print(data["mix_coef"])  # Output: e.g., 0.9991580344142427
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.GLOBAL_LABEL)

    def __init__(
        self,
        reference_data: Optional[Union[Generator[Any, None, None], Sequence[Any]]] = None,
        read_fn: Callable[[Any], ReferenceImage] = lambda x: {"image": x, "mask": None, "global_label": None},
        alpha: float = 0.4,
        mix_coef_return_name: str = "mix_coef",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.mix_coef_return_name = mix_coef_return_name

        if alpha < 0:
            msg = "Alpha must be >= 0."
            raise ValueError(msg)

        self.read_fn = read_fn
        self.alpha = alpha

        if reference_data is None:
            warn("No reference data provided for MixUp. This transform will act as a no-op.")
            # Create an empty generator
            self.reference_data: List[Any] = []
        elif (
            isinstance(reference_data, types.GeneratorType)
            or isinstance(reference_data, Iterable)
            and not isinstance(reference_data, str)
        ):
            self.reference_data = reference_data  # type: ignore[assignment]
        else:
            msg = "reference_data must be a list, tuple, generator, or None."
            raise TypeError(msg)

    def apply(self, img: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any) -> np.ndarray:
        mix_img = mix_data.get("image")

        if not is_grayscale_image(img) and img.shape != img.shape:
            msg = "The shape of the reference image should be the same as the input image."
            raise ValueError(msg)

        return mix_arrays(img, mix_img, mix_coef) if mix_img is not None else img

    def apply_to_mask(self, mask: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any) -> np.ndarray:
        mix_mask = mix_data.get("mask")
        return mix_arrays(mask, mix_mask, mix_coef) if mix_mask is not None else mask

    def apply_to_global_label(
        self, label: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any
    ) -> np.ndarray:
        mix_label = mix_data.get("global_label")
        if mix_label is not None and label is not None:
            return mix_coef * label + (1 - mix_coef) * mix_label
        return label

    def apply_to_bboxes(self, bboxes: Sequence[BoxType], mix_data: ReferenceImage, **params: Any) -> Sequence[BoxType]:
        msg = "MixUp does not support bounding boxes yet, feel free to submit pull request to https://github.com/albumentations-team/albumentations/."
        raise NotImplementedError(msg)

    def apply_to_keypoints(
        self, keypoints: Sequence[KeypointType], *args: Any, **params: Any
    ) -> Sequence[KeypointType]:
        msg = "MixUp does not support keypoints yet, feel free to submit pull request to https://github.com/albumentations-team/albumentations/."
        raise NotImplementedError(msg)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_data", "alpha"

    def get_params(self) -> Dict[str, Any]:
        mix_data = None
        # Check if reference_data is not empty and is a sequence (list, tuple, np.array)
        if isinstance(self.reference_data, Sequence) and not isinstance(self.reference_data, (str, bytes)):
            if len(self.reference_data) > 0:  # Additional check to ensure it's not empty
                mix_idx = random.randint(0, len(self.reference_data) - 1)
                mix_data = self.reference_data[mix_idx]
        # Check if reference_data is an iterator or generator
        elif isinstance(self.reference_data, Iterator):
            try:
                mix_data = next(self.reference_data)  # Attempt to get the next item
            except StopIteration:
                warn(
                    "Reference data iterator/generator has been exhausted. "
                    "Further mixing augmentations will not be applied.",
                    RuntimeWarning,
                )
                return {"mix_data": {}, "mix_coef": 1}

        # If mix_data is None or empty after the above checks, return default values
        if mix_data is None:
            return {"mix_data": {}, "mix_coef": 1}

        # If mix_data is not None, calculate mix_coef and apply read_fn
        mix_coef = beta(self.alpha, self.alpha)  # Assuming beta is defined elsewhere
        return {"mix_data": self.read_fn(mix_data), "mix_coef": mix_coef}

    def apply_with_params(self, params: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        res = super().apply_with_params(params, *args, **kwargs)
        if self.mix_coef_return_name:
            res[self.mix_coef_return_name] = params["mix_coef"]
        return res


class Mosaic(ReferenceBasedTransform):
    """Performs Mosaic data augmentation, combining multiple images into a single image for enhanced model training.

    This transformation creates a composite image from multiple source images arranged in a grid, which can be uniform
    or random based on the `split_mode`. The mosaic augmentation introduces variations in context, scale, and object
    combinations, beneficial for object detection models.

    Args:
        reference_data (Optional[Union[Generator[ReferenceImage, None, None], Sequence[Any]]]):
            A sequence or generator of dictionaries with the reference data for the mosaic.
            If None or an empty sequence is provided, no operation is performed.
        read_fn (Callable[[ReferenceImage], Dict[str, Any]]):
            A function to process items from `reference_data`. It must return a dictionary containing 'image',
            and optionally 'mask' and 'global_label', each associated with numpy array values.
        grid_size (Tuple[int, int]):
            The size (rows, columns) of the grid to arrange images in the mosaic. Defaults to (3, 3).
        split_mode (str):
            Determines how the images are split and arranged in the mosaic. Can be 'uniform' for equal-sized tiles,
            or 'random' for randomly sized tiles. Defaults to 'uniform'.
        preprocessing_mode (str): resize, longest_max_size_pad, smallest_max_size_crop, random_sized_bbox_safe_crop,
        p (float):
            The probability of applying the transformation. Defaults to 0.5.

    Targets:
        image, mask, global_label

    Image types:
        uint8, float32

    Raises:
        - ValueError: For invalid `grid_size` or `split_mode` values.
        - NotImplementedError: If applied to bounding boxes or keypoints.
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.GLOBAL_LABEL)

    def __init__(
        self,
        reference_data: Optional[Union[Generator[ReferenceImage, None, None], Sequence[Any]]] = None,
        read_fn: Callable[[Any], ReferenceImage] = lambda x: {
            "image": x,
            "mask": None,
            "global_label": None,
        },
        grid_size: Tuple[int, int] = (3, 3),
        split_mode: str = "uniform",
        preprocessing_mode: str = "resize",
        target_size: Tuple[int, int] = (1024, 1024),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.grid_size = grid_size
        self.split_mode = split_mode
        self.target_size = target_size

        if any(x <= 0 for x in self.grid_size):
            msg = "grid_size must contain positive integers."
            raise ValueError(msg)
        if split_mode not in ["uniform", "random"]:
            msg = "split_mode must be 'uniform' or 'random'."
            raise ValueError(msg)

        self.read_fn = read_fn

        if reference_data is None:
            warn("No reference data provided for Mosaic. This transform will act as a no-op.")
            self.reference_data: List[Any] = []
        elif isinstance(reference_data, (types.GeneratorType, Iterable)) and not isinstance(reference_data, str):
            if isinstance(reference_data, Sequence) and len(reference_data) < self.grid_size[0] * self.grid_size[1] - 1:
                msg = "Not enough reference data to fill the mosaic grid."
                raise ValueError(msg)
            self.reference_data = reference_data  # type: ignore[assignment]
        else:
            msg = "reference_data must be a list, tuple, generator, or None."
            raise TypeError(msg)

        self.preprocessing_mode = preprocessing_mode

    def apply(
        self,
        img: np.ndarray,
        mix_data: List[ReferenceImage],
        tiles: np.ndarray,
        preprocessing_pipeline: Compose,
        **params: Any,
    ) -> np.ndarray:
        transformed_img = preprocessing_pipeline(image=img)["image"]
        return self.apply_to_image_or_mask(transformed_img, "image", mix_data, tiles)

    def apply_to_mask(self, mask: np.ndarray, mix_data: List[ReferenceImage], *args: Any, **params: Any) -> np.ndarray:
        msg = "Mosaic does not support keypoints yet"
        raise NotImplementedError(msg)

    def apply_to_bbox(self, bbox: Any, *args: Any, **params: Any) -> Any:
        msg = "Mosaic does not support bbox yet"
        raise NotImplementedError(msg)

    def apply_to_global_label(
        self, label: np.ndarray, mix_data: List[ReferenceImage], *args: Any, **params: Any
    ) -> np.ndarray:
        msg = "Mosaic does not support global label yet"
        raise NotImplementedError(msg)

    def sample_reference_data(self) -> List[Dict[str, Any]]:
        total_tiles = self.grid_size[0] * self.grid_size[1]
        sampled_reference_data: List[Any] = []

        if isinstance(self.reference_data, Sequence) and len(self.reference_data):
            # Select data without replacement if there are more items than needed, else with replacement
            return choice(self.reference_data, self.grid_size[0] * self.grid_size[1] - 1, replace=False)

        if isinstance(self.reference_data, Iterator):
            # Get the necessary number of elements from the iterator
            sampled_reference_data = []

            try:
                for _ in range(total_tiles - 1):
                    next_element = next(self.reference_data, None)
                    if next_element is None:
                        # The iterator doesn't have enough elements
                        warn("Reference data iterator has insufficient data to fill the mosaic grid.", RuntimeWarning)
                        # Reset mix_data as we can't fulfill the required grid tiles
                        return []
                    sampled_reference_data.append(next_element)
            except StopIteration:
                # This block is in case the iterator was shorter than expected and ran out before total_tiles
                warn("Reference data iterator was exhausted before filling all tiles.", RuntimeWarning)
                # Reset mix_data as we can't fulfill the required grid tiles
                sampled_reference_data = []

        return sampled_reference_data

    def get_params(self) -> Dict[str, Any]:
        sampled_reference_data = self.sample_reference_data()

        tiles = split_uniform_grid(self.target_size, self.grid_size)

        shuffle(tiles)

        mix_data = []

        for idx, tile in enumerate(tiles[:-1]):  # last position in shuffled tiles is for target
            element = sampled_reference_data[idx]
            processed_element = self.read_fn(element)
            # Extract the tile dimensions
            tile_height, tile_width = tile[2] - tile[0], tile[3] - tile[1]
            # Preprocess the element based on the tile size
            processed_element = self.preprocess_element(processed_element, tile_height, tile_width)
            mix_data.append(processed_element)

        last_tile = tiles[-1]
        last_tile_width = last_tile[3] - last_tile[1]
        last_tile_height = last_tile[2] - last_tile[0]
        preprocessing_pipeline = self.get_preprocessing_pipeline(last_tile_height, last_tile_width)

        return {"mix_data": sampled_reference_data, "preprocessing_pipepline": preprocessing_pipeline}

    def get_preprocessing_pipeline(self, tile_height: int, tile_width: int) -> Compose:
        if self.preprocessing_mode == "resize":
            return Compose([Resize(height=tile_height, width=tile_width)])
        if self.preprocessing_mode == "longest_max_size_pad":
            return Compose(
                [
                    LongestMaxSize(max_size=max(tile_height, tile_width)),
                    PadIfNeeded(min_height=tile_height, min_width=tile_width),
                ]
            )
        if self.preprocessing_mode == "smallest_max_size_crop":
            return Compose(
                [
                    SmallestMaxSize(max_size=min(tile_height, tile_width)),
                    RandomSizedBBoxSafeCrop(height=tile_height, width=tile_width),
                ]
            )
        if self.preprocessing_mode == "random_sized_bbox_safe_crop":
            return Compose(
                [
                    RandomSizedBBoxSafeCrop(height=tile_height, width=tile_width),
                    LongestMaxSize(max_size=max(tile_height, tile_width)),
                    PadIfNeeded(min_height=tile_height, min_width=tile_width),
                ]
            )

        raise ValueError(f"Unknown preprocessing_mode {self.preprocessing_mode}")

    def preprocess_element(self, element: ReferenceImage, tile_height: int, tile_width: int) -> ReferenceImage:
        preprocessing_pipeline = self.get_preprocessing_pipeline(tile_height, tile_width)

        # Apply the preprocess pipeline to the image, mask, and other elements
        return cast(ReferenceImage, preprocessing_pipeline(**element))

    def apply_to_image_or_mask(
        self, data: np.ndarray, data_key: Literal["image", "mask"], mix_data: List[ReferenceImage], tiles: np.ndarray
    ) -> np.ndarray:
        """Apply transformations to an image or mask based on mixed data and tile positioning.

        Args:
            data (np.ndarray): The original image or mask data.
            data_key (str): The key in the processed elements dictionary that corresponds to the data
                ('image' or 'mask').
            mix_data (List[ReferenceImage]): List of processed elements (dictionaries containing 'image', 'mask').
            tiles (List[Tuple[int, int, int, int]]): List of tile coordinates.

        Returns:
            np.ndarray: The new image or mask after applying the mosaic transformations.
        """
        new_data = np.empty(self.target_size)
        for element, tile in zip(mix_data, tiles):
            if data_key in element:
                y_min, x_min, y_max, x_max = tile
                element_data = element[data_key]
                new_data[y_min:y_max, x_min:x_max] = element_data

        last_tile = tiles[-1]
        y_min, x_min, y_max, x_max = last_tile

        new_data[y_min:y_max, x_min:x_max] = data

        return new_data

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_data", "grid_size", "split_mode", "preprocessing_mode"
