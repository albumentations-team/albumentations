# Coding Guidelines

This document outlines the coding standards and best practices for contributing to Albumentations.

## Important Note About Guidelines

These guidelines represent our current best practices, developed through experience maintaining and expanding the Albumentations codebase. While some existing code may not strictly follow these standards (due to historical reasons), we are gradually refactoring the codebase to align with these guidelines.

**For new contributions:**

- All new code must follow these guidelines
- All modifications to existing code should move it closer to these standards
- Pull requests that introduce patterns we're trying to move away from will not be accepted

**For existing code:**

- You may encounter patterns that don't match these guidelines (e.g., transforms with "Random" prefix or Union types for parameters)
- These are considered technical debt that we're working to address
- When modifying existing code, take the opportunity to align it with current standards where possible

## Code Style and Formatting

### Pre-commit Hooks

We use pre-commit hooks to maintain consistent code quality. These hooks automatically check and format your code before each commit.

- Install pre-commit if you haven't already:

  ```bash
  pip install pre-commit
  pre-commit install
  ```

- The hooks will run automatically on `git commit`. To run manually:

  ```bash
  pre-commit run --files $(find albumentations -type f)
  ```

### Python Version and Type Hints

- Use Python 3.9+ features and syntax
- Always include type hints using Python 3.10+ typing syntax:

  ```python
  # Correct
  def transform(self, value: float, range: tuple[float, float]) -> float:

  # Incorrect - don't use capital-case types
  def transform(self, value: float, range: Tuple[float, float]) -> Float:
  ```

- Use `|` instead of `Union` and for optional types:

  ```python
  # Correct
  def process(value: int | float | None) -> str:

  # Incorrect
  def process(value: Optional[Union[int, float]) -> str:
  ```

## Naming Conventions

### Transform Names

- Avoid adding "Random" prefix to new transforms

  ```python
  # Correct
  class Brightness(ImageOnlyTransform):

  # Incorrect (historical pattern)
  class RandomBrightness(ImageOnlyTransform):
  ```

### Parameter Naming

- Use `_range` suffix for interval parameters:

  ```python
  # Correct
  brightness_range: tuple[float, float]
  shadow_intensity_range: tuple[float, float]

  # Incorrect
  brightness_limit: tuple[float, float]
  shadow_intensity: tuple[float, float]
  ```

### Standard Parameter Names

For transforms that handle gaps or boundaries, use these consistent names:

- `border_mode`: Specifies how to handle gaps, not `mode` or `pad_mode`
- `fill`: Defines how to fill holes (pixel value or method), not `fill_value`, `cval`, `fill_color`, `pad_value`, `pad_cval`, `value`, `color`
- `fill_mask`: Same as `fill` but for mask filling, not `fill_mask_value`, `fill_mask_color`, `fill_mask_cval`

## Parameter Types and Ranges

### Parameter Definitions

- Prefer range parameters over fixed values:

  ```python
  # Correct
  def __init__(self, brightness_range: tuple[float, float] = (-0.2, 0.2)):

  # Avoid
  def __init__(self, brightness: float = 0.2):
  ```

### Avoid Union Types for Parameters

- Don't use `Union[float, tuple[float, float]]` for parameters
- Instead, always use ranges where sampling is needed:

  ```python
  # Correct
  scale_range: tuple[float, float] = (0.5, 1.5)

  # Avoid
  scale: float | tuple[float, float] = 1.0
  ```

- For fixed values, use same value for both range ends:

  ```python
  brightness_range = (0.1, 0.1)  # Fixed brightness of 0.1
  ```

## Transform Design Principles

### Relative Parameters

- Prefer parameters that are relative to image dimensions rather than fixed pixel values:

  ```python
  # Correct - relative to image size
  def __init__(self, crop_size_range: tuple[float, float] = (0.1, 0.3)):
      # crop_size will be fraction of min(height, width)

  # Avoid - fixed pixel values
  def __init__(self, crop_size_range: tuple[int, int] = (32, 96)):
      # crop_size will be fixed regardless of image size
  ```

### Data Type Consistency

- Ensure transforms produce consistent results regardless of input data type
- Use provided decorators to handle type conversions:
  - `@uint8_io`: For transforms that work with uint8 images
  - `@float32_io`: For transforms that work with float32 images

The decorators will:

- Pass through images that are already in the target type without conversion
- Convert other types as needed and convert back after processing

```python
@uint8_io  # If input is uint8 => use as is; if float32 => convert to uint8, process, convert back
def apply(self, img: np.ndarray, **params) -> np.ndarray:
    # img is guaranteed to be uint8
    # if input was float32 => result will be converted back to float32
    # if input was uint8 => result will stay uint8
    return cv2.blur(img, (3, 3))

@float32_io  # If input is float32 => use as is; if uint8 => convert to float32, process, convert back
def apply(self, img: np.ndarray, **params) -> np.ndarray:
    # img is guaranteed to be float32 in range [0, 1]
    # if input was uint8 => result will be converted back to uint8
    # if input was float32 => result will stay float32
    return img * 0.5

# Avoid - manual type conversion
def apply(self, img: np.ndarray, **params) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    result = cv2.blur(img, (3, 3))
    if img.dtype != np.uint8:
        result = result.astype(np.float32) / 255
    return result
```

### Channel Flexibility

- Support arbitrary number of channels unless specifically constrained:

  ```python
  # Correct - works with any number of channels
  def apply(self, img: np.ndarray, **params) -> np.ndarray:
      # img shape is (H, W, C), works for any C
      return img * self.factor

  # Also correct - explicitly requires RGB
  def apply(self, img: np.ndarray, **params) -> np.ndarray:
      if img.shape[-1] != 3:
          raise ValueError("Transform requires RGB image")
      return rgb_to_hsv(img)  # RGB-specific processing
  ```

### Handling Auxiliary Data via Metadata

When a transform requires complex or variable auxiliary data beyond simple configuration parameters (e.g., additional images and labels for `Mosaic`, extra images for domain adaptation transforms like `FDA` or `HistogramMatching`), **do not pass this data directly through the `__init__` constructor**.

Instead, follow this preferred pattern:

1. **Pass the auxiliary data** within the main `data` dictionary provided to the transform's `__call__` method, using a descriptive key (e.g., `mosaic_metadata`, `target_image`).
2. **Declare this key** in the transform's `targets_as_params` class attribute. This attribute signals to the `Compose` pipeline that the associated auxiliary data should be extracted from the main data dictionary and forwarded to the transform. In other words, targets_as_params enables the dynamic inclusion of per-sample metadata into the `get_params_dependent_on_data` method, ensuring the transform has access to this additional context during processing.
3. **Access the data** inside `get_params_dependent_on_data` using `data.get("your_metadata_key")`.

This approach keeps the transform's initialization focused on static configuration and allows the dynamic, per-sample auxiliary data to flow through the standard pipeline mechanism.

**Example (`Mosaic` transform):**

```python
# Correct - Data passed via `data` dict, key declared in `targets_as_params`
class Mosaic(DualTransform):
    def __init__(self, target_size: tuple[int, int] = (512, 512), p=0.5, metadata_key="mosaic_metadata"):
        super().__init__(p=p)
        self.target_size = target_size
        # NO auxiliary data like list of images/bboxes in __init__
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        """Get list of targets that should be passed as parameters to transforms."""
        return [self.metadata_key]

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        # Access auxiliary data here
        additional_items = data.get(self.metadata_key, [])
        # ... process additional_items ...
        return { ... } # Return calculated parameters

# Usage
transform = A.Mosaic(target_size=(640, 640))
result = transform(image=img1, mask=mask1, bboxes=bboxes1,
                   mosaic_metadata=[
                       {'image': img2, 'bboxes': bboxes2},
                       {'image': img3, 'bboxes': bboxes3},
                       # ... more items
                   ])


# Incorrect - Avoid passing variable data structures via __init__
class BadMosaic(DualTransform):
    def __init__(self, additional_items: list[dict], target_size: tuple[int, int] = (512, 512), p=0.5):
         super().__init__(p=p)
         self.additional_items = additional_items # Discouraged!
         self.target_size = target_size

    # ... might not even need get_params_dependent_on_data if data is in self ...
```

Passing data via `__init__` couples the transform instance to specific data, making it less reusable and potentially breaking serialization or pipeline composition.

## Random Number Generation

### Using Random Generators

- Use class-level random generators instead of direct numpy or random calls:

  ```python
  # Correct
  value = self.random_generator.uniform(0, 1, size=image.shape)
  choice = self.py_random.choice(options)

  # Incorrect
  value = np.random.uniform(0, 1, size=image.shape)
  choice = random.choice(options)
  ```

- Prefer Python's standard library `random` over `numpy.random`:

  ```python
  # Correct - using standard library random (faster)
  value = self.py_random.uniform(0, 1)
  choice = self.py_random.choice(options)

  # Use numpy.random only when needed
  value = self.random_generator.randint(0, 255, size=image.shape)
  ```

### Parameter Sampling

- Handle all probability calculations in `get_params` or `get_params_dependent_on_data`
- Don't perform random operations in `apply_xxx` or `__init__` methods:

  ```python
  def get_params(self):
      return {
          "brightness": self.random_generator.uniform(
              self.brightness_range[0],
              self.brightness_range[1]
          )
      }
  ```

## Transform Development

### Method Definitions

- Don't use default arguments in `apply_xxx` methods:

  ```python
  # Correct
  def apply_to_mask(self, mask: np.ndarray, fill_mask: int) -> np.ndarray:

  # Incorrect
  def apply_to_mask(self, mask: np.ndarray, fill_mask: int = 0) -> np.ndarray:
  ```

### Parameter Generation

#### Using get_params_dependent_on_data

This method provides access to image shape and target data for parameter generation:

```python
def get_params_dependent_on_data(
    self,
    params: dict[str, Any],
    data: dict[str, Any]
) -> dict[str, Any]:
    # Access image shape - always available
    height, width = params["shape"][:2]

    # Access targets if they were passed to transform
    image = data.get("image")  # Original image
    mask = data.get("mask")    # Segmentation mask
    bboxes = data.get("bboxes")  # Bounding boxes
    keypoints = data.get("keypoints")  # Keypoint coordinates

    # Example: Calculate parameters based on image size
    crop_size = min(height, width) // 2
    center_x = width // 2
    center_y = height // 2

    return {
        "crop_size": crop_size,
        "center": (center_x, center_y)
    }
```

The method receives:

- `params`: Dictionary containing image metadata, where `params["shape"]` is always available
- `data`: Dictionary containing all targets passed to the transform

Use this method when you need to:

- Calculate parameters based on image dimensions
- Access target data for parameter generation
- Ensure transform parameters are appropriate for the input data

### Parameter Validation with `InitSchema`

Each transform must include an `InitSchema` class that inherits from `BaseTransformInitSchema`. This class is responsible for:

- Validating input parameters before `__init__` execution
- Converting parameter types if needed
- Ensuring consistent parameter handling

  ```python
  # Correct - full parameter validation
  class RandomGravel(ImageOnlyTransform):
      class InitSchema(BaseTransformInitSchema):
        slant_range: Annotated[tuple[float, float], AfterValidator(nondecreasing)]
        brightness_coefficient: float = Field(gt=0, le=1)


    def __init__(self, slant_range: tuple[float, float], brightness_coefficient: float, p: float = 0.5):
        super().__init__(p=p)
        self.slant_range = slant_range
        self.brightness_coefficient = brightness_coefficient
  ```

  ```python
  # Incorrect - missing InitSchema
  class RandomGravel(ImageOnlyTransform):
      def __init__(self, slant_range: tuple[float, float], brightness_coefficient: float, p: float = 0.5):
          super().__init__(p=p)
          self.slant_range = slant_range
          self.brightness_coefficient = brightness_coefficient
  ```

### Coordinate Systems

#### Image Center Calculations

The center point calculation differs slightly between targets:

- For images, masks, and keypoints:

  ```python
  # Correct - using helper function
  from albumentations.augmentations.geometric.functional import center
  center_x, center_y = center(image_shape)  # Returns ((width-1)/2, (height-1)/2)

  # Incorrect - manual calculation might miss the -1
  center_x = width / 2  # Wrong!
  center_y = height / 2  # Wrong!
  ```

- For bounding boxes:

  ```python
  # Correct - using helper function
  from albumentations.augmentations.geometric.functional import center_bbox
  center_x, center_y = center_bbox(image_shape)  # Returns (width/2, height/2)

  # Incorrect - using wrong center calculation
  center_x, center_y = center(image_shape)  # Wrong for bboxes!
  ```

This small difference is crucial for pixel-perfect accuracy. Always use the appropriate helper functions:

- `center()` for image, mask, and keypoint transformations
- `center_bbox()` for bounding box transformations

### Serialization Compatibility

- Ensure transforms work with both tuples and lists for range parameters
- Test serialization/deserialization with JSON and YAML formats

## Documentation

### Docstrings

- Use Google-style docstrings
- Include type information, parameter descriptions, and examples:

  ```python
  def transform(self, image: np.ndarray) -> np.ndarray:
      """Apply brightness transformation to the image.

      Args:
          image: Input image in RGB format.

      Returns:
          Transformed image.

      Examples:
          >>> transform = Brightness(brightness_range=(-0.2, 0.2))
          >>> transformed = transform(image=image)
      """
  ```

### Examples in Docstrings

Every transform class that is a descendant of `ImageOnlyTransform`, `DualTransform`, or `Transform3D` **must** include a comprehensive Examples section in its docstring. The examples should follow these guidelines:

1. **Section Naming**: The section should be titled "Examples" (not "Example").

2. **Jupyter Notebook Format**: Examples should mimic Jupyter notebook format, using `>>>` for code lines and no prefix for output lines.

3. **Comprehensiveness**: Examples should be fully reproducible, including:
   - Initialization of sample data
   - Creation of transform(s)
   - Application of transform(s)
   - Retrieving results from the transform

4. **Target-Specific Requirements**:
   - For `ImageOnlyTransform`: Pass and demonstrate transformation of image data. Including how to get all transformed targets.
   - For `DualTransform`: Pass and demonstrate transformation of image, mask, bboxes, keypoints, bbox_labels, class_labels (where supported). Including how to get all transformed targets including bbox_labels and keypoints_labels
   - For `Transform3D`: Pass and demonstrate transformation of volume and mask3d data. Including how to get all transformed targets.
    Including keypoint_labels

5. **For Base Classes**: Examples for base classes should show:
   - How to initialize a custom transform that inherits from the base class
   - How to use the custom transform as part of a Compose pipeline

6. **Parameter Examples**: When a parameter accepts both a single value and a tuple of values (to be sampled from), always use a tuple in the example.

Here's an example for a `DualTransform`:

```python
"""
Examples:
    >>> import numpy as np
    >>> import albumentations as A
    >>> # Prepare sample data
    >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
    >>> bbox_labels = [1, 2]
    >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
    >>> keypoint_labels = [0, 1]
    >>>
    >>> # Define transform with parameters as tuples when possible
    >>> transform = A.Compose([
    ...     A.HorizontalFlip(p=1.0),
    ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
    ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
    >>>
    >>> # Apply the transform
    >>> transformed = transform(
    ...     image=image,
    ...     mask=mask,
    ...     bboxes=bboxes,
    ...     bbox_labels=bbox_labels,
    ...     keypoints=keypoints,
    ...     keypoint_labels=keypoint_labels
    ... )
    >>>
    >>> # Get the transformed data
    >>> transformed_image = transformed['image']  # Horizontally flipped image
    >>> transformed_mask = transformed['mask']    # Horizontally flipped mask
    >>> transformed_bboxes = transformed['bboxes']  # Horizontally flipped bounding boxes
    >>> transformed_keypoints = transformed['keypoints']  # Horizontally flipped keypoints
"""
```

Examples for a base class showing custom implementation:

```python
"""
Examples:
    # Example of a custom distortion subclass
    >>> import numpy as np
    >>> import albumentations as A
    >>>
    >>> class CustomDistortion(A.BaseDistortion):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         # Add custom parameters here
    ...
    ...     def get_params_dependent_on_data(self, params, data):
    ...         height, width = params["shape"][:2]
    ...         # Generate distortion maps
    ...         map_x = np.zeros((height, width), dtype=np.float32)
    ...         map_y = np.zeros((height, width), dtype=np.float32)
    ...         # Apply your custom distortion logic here
    ...         # ...
    ...         return {"map_x": map_x, "map_y": map_y}
    >>>
    >>> # Prepare sample data
    >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
    >>> bbox_labels = [1, 2]
    >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
    >>> keypoint_labels = [0, 1]
    >>>
    >>> # Apply the custom distortion
    >>> transform = A.Compose([
    ...     CustomDistortion(
    ...         interpolation=A.cv2.INTER_LINEAR,
    ...         mask_interpolation=A.cv2.INTER_NEAREST,
    ...         keypoint_remapping_method="mask",
    ...         p=1.0
    ...     )
    ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
    ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
    >>>
    >>> # Apply the transform
    >>> transformed = transform(
    ...     image=image,
    ...     mask=mask,
    ...     bboxes=bboxes,
    ...     bbox_labels=bbox_labels,
    ...     keypoints=keypoints,
    ...     keypoint_labels=keypoint_labels
    ... )
    >>>
    >>> # Get results
    >>> transformed_image = transformed['image']
    >>> transformed_mask = transformed['mask']
    >>> transformed_bboxes = transformed['bboxes']
    >>> transformed_keypoints = transformed['keypoints']
"""
```

### Comments

- Add comments for complex logic
- Explain why, not what (the code shows what)
- Keep comments up to date with code changes

### Updating Transform Documentation

When adding a new transform or modifying the targets of an existing one, you must update the transforms documentation in the README:

1. Generate the updated documentation by running:

   ```bash
   python -m tools.make_transforms_docs make
   ```

2. This will output a formatted list of all transforms and their supported targets

3. Update the relevant section in README.md with the new information

4. Ensure the documentation accurately reflects which targets (image, mask, bboxes, keypoints, etc.) are supported by each transform

This helps maintain accurate and up-to-date documentation about transform capabilities.

## Testing

### Test Coverage

- Write tests for all new functionality
- Include edge cases and error conditions
- Ensure reproducibility with fixed random seeds

### Test Organization

- Place tests in the appropriate module under `tests/`
- Follow existing test patterns and naming conventions
- Use pytest fixtures when appropriate

## Code Review Guidelines

Before submitting your PR:

1. Run all tests
2. Run pre-commit hooks
3. Check type hints
4. Update documentation if needed
5. Ensure code follows these guidelines

## Getting Help

If you have questions about these guidelines:

1. Join our [Discord community](https://discord.gg/e6zHCXTvaN)
2. Open a GitHub [issue](https://github.com/albumentations-team/albumentations/issues)
3. Ask in your [pull request](https://github.com/albumentations-team/albumentations/pulls)
