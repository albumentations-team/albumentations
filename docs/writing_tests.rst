=============
Writing tests
=============

*************
A first test.
*************

We use ``pytest`` to run tests for albumentations. Python files with tests should be placed inside the ``albumentations/tests`` directory, filenames should start with ``test_``, for example ``test_bbox.py``. Names of test functions should also start with ``test_``, for example, ``def test_random_brightness():``.

Let's say that we want to test the ``brightness_contrast_adjust`` function. The purpose of this function is to take a NumPy array as input and multiply all the values of this array by a value specified in the argument ``alpha``.

We will write a first test for this function that will check that if you pass a NumPy array with all values equal to 128 and a parameter ``alpha`` that equals to 1.5 as inputs the function should produce a NumPy array with all values equal to 192 as output (that's because 128 * 1.5 = 192).

In the directory ``albumentations/tests`` we will create a new file and name it ``test_example.py``

Let's add all the necessary imports:

.. code-block:: python

    import numpy as np

    import albumentations.augmentations.functional as F

Then let's add the test itself:

.. code-block:: python

    def test_random_contrast():
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = F.brightness_contrast_adjust(img, alpha=1.5)
        expected_brightness = 192
        expected = np.ones((100, 100, 3), dtype=np.uint8) * expected_multiplier
        assert np.array_equal(img, expected)

We can run tests from ``test_example.py`` (right now it contains only one test) by executing the following command: ``pytest tests/test_example.py -v``. The ``-v`` flag tells pytest to produce a more verbose output.

``pytest`` will show that the test has been completed successfully::

    tests/test_example.py::test_random_brightness PASSED

****************************************************************
Test parametrization and the @pytest.mark.parametrize decorator.
****************************************************************

Let's say that we also want to test that the function ``brightness_contrast_adjust`` correctly handles a situation in which after multiplying an input array by ``alpha`` some output values exceed 255. Because when we a pass a NumPy array with the data type ``np.uint8`` as input we expect that we will also get an array with the ``np.uint8`` data type as output and that means that output values should not exceed 255 (which is the maximum value for this data type). We also want to check that values don't overflow, so if inside the function we get a value 256 we should clip it to 255 and not overflow to 0.

Let's write a test:

.. code-block:: python

    def test_random_contrast_2():
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = F.brightness_contrast_adjust(img, alpha=3)
        expected_multiplier = 255
        expected = np.ones((100, 100, 3), dtype=np.uint8) * expected_multiplier
        assert np.array_equal(img, expected)

Next, we will run the tests from ``test_example.py``: ``pytest tests/test_example.py -v``

Output::

    tests/test_example.py::test_random_brightness PASSED
    tests/test_example.py::test_random_brightness_2 PASSED

As we see functions ``test_random_brightness`` and ``test_random_brightness_2`` looks almost the same, the only difference is the values of ``alpha`` and ``expected_multiplier``. To get rid of code duplication we can use the ``@pytest.mark.parametrize`` decorator. With this decorator we can describe which values should be passed as arguments to the test and the pytest will run the test multiple times, each time passing the next value from the decorator.

We can rewrite two previous tests as a one test using parametrization:

.. code-block:: python

    import pytest

    @pytest.mark.parametrize(['alpha', 'expected_multiplier'], [(1.5, 192), (3, 255)])
    def test_random_brightness(alpha, expected_multiplier):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = F.brightness_contrast_adjust(img, alpha=alpha)
        expected = np.ones((100, 100, 3), dtype=np.uint8) * expected_multiplier
        assert np.array_equal(img, expected)

This test will run two times, in the first run the ``alpha`` argument will be equal to 1.5 and the ``expected_multiplier`` argument will be equal to 192. In the second run the ``alpha`` argument will be equal to 3 and the ``expected_multiplier`` argument will be equal to 255.

Let's run this test::

    tests/test_example.py::test_random_brightness[1.5-192] PASSED
    tests/test_example.py::test_random_brightness[3-255] PASSED

As we see pytest prints arguments values at each run.

***********************************************************************************************
Simplifying tests for functions that work with both images and masks by using helper functions.
***********************************************************************************************
Let's say that we want to test the ``hflip`` function. This function vertically flips an image or mask that passed as input to it.

We will start with a test that checks that this function works correctly with masks, that is with two-dimensional NumPy arrays that have shape ``(height, width)``.

.. code-block:: python

    def test_vflip_mask():
        mask = np.array(
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected_mask = np.array(
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]], dtype=np.uint8)
        flipped_mask = F.vflip(mask)
        assert np.array_equal(flipped_mask, expected_mask)

Test running result::

    tests/test_example.py::test_vflip_mask PASSED

Next, we will make a test that checks how the same function works with RGB-images, that is with three-dimensional NumPy arrays that have shape ``(height, width, 3)``.

.. code-block:: python

    def test_vflip_img():
        img = np.array(
            [[[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]],
             [[0, 0, 0],
              [1, 1, 1],
              [1, 1, 1]],
             [[0, 0, 0],
              [0, 0, 0],
              [1, 1, 1]]], dtype=np.uint8)
        expected_img = np.array(
            [[[0, 0, 0],
              [0, 0, 0],
              [1, 1, 1]],
             [[0, 0, 0],
              [1, 1, 1],
              [1, 1, 1]],
             [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]], dtype=np.uint8)
        flipped_img = F.vflip(img)
        assert np.array_equal(flipped_img, expected_img)

In this test, the value of ``img`` is the same NumPy array that was assigned to the ``mask`` variable in ``test_vflip_mask``, but this time it is repeated three times (one time for each of the three channels). And ``expected_img`` is also a repeated three times NumPy array that was assigned to the ``expected_mask`` variable in ``test_vflip_mask``.

Let's run the test::

    tests/test_example.py::test_vflip_img PASSED

In ``test_vflip_img`` we manually defined values of ``img`` and ``expected_img`` that equal to repeated three times values of ``mask`` and ``expected_mask`` respectively. To avoid unnecessary and duplicate code we can make a helper function that takes a NumPy array with shape ``(height, width)`` as input and repeats this value 3 times along a new axis to produce a NumPy array with shape ``(height, width, 3)``:

.. code-block:: python

    def convert_2d_to_3d(array, num_channels=3):
        return np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2)

Next, we can use this function to rewrite ``test_vflip_img`` as follows:

.. code-block:: python

    def test_vflip_img_2():
        mask = np.array(
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected_mask = np.array(
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]], dtype=np.uint8)
        img = convert_2d_to_3d(mask)
        expected_img = convert_2d_to_3d(expected_mask)
        flipped_img = F.vflip(img)
        assert np.array_equal(flipped_img, expected_img)

Let's run the test::

    tests/test_example.py::test_vflip_img_2 PASSED

**********************************************************************************************
Simplifying tests for functions that work with both images and masks by using parametrization.
**********************************************************************************************

In the previous section we wrote two separate tests for ``vflip``, the first one checked how ``vflip`` works with masks, the second one checked how ``vflip`` works with images.

Those tests share a large amount of the same code between them, so we can move common parts to a single function and use parametrization to pass information about input type as an argument to the test:

.. code-block:: python

    @pytest.mark.parametrize('target', ['mask', 'image'])
    def test_vflip_img_and_mask(target):
        img = np.array(
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected = np.array(
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]], dtype=np.uint8)
        if target == 'image':
            img = convert_2d_to_3d(img)
            expected = convert_2d_to_3d(expected)
        flipped_img = F.vflip(img)
        assert np.array_equal(flipped_img, expected)

This test will run two times, in the first run the ``target`` argument will be equal to ``'mask'``, the condition ``if target == 'image':`` will not be executed and the test will check how ``vflip`` works with masks. In the second run the ``target`` argument will be equal to ``'image'``, the condition ``if target == 'image':`` will be executed and the test will check how ``vflip`` works with images::

    tests/test_example.py::test_vflip_img_and_mask[mask] PASSED
    tests/test_example.py::test_vflip_img_and_mask[image] PASSED

We can reduce the amount of code even further by moving logic under ``if target == 'image'`` to a separate function:

.. code-block:: python

    def convert_2d_to_target_format(*arrays, target=None):
        if target == 'mask':
            return arrays[0] if len(arrays) == 1 else arrays
        elif target == 'image':
            return tuple(convert_2d_to_3d(array, num_channels=3) for array in arrays)
        else:
            raise ValueError('Unknown target {}'.format(target))

This function will take NumPy arrays with shape ``(height, width)`` as inputs and depending on the value of ``target`` will either return them as is or convert them to NumPy arrays with shape ``(height, width, 3)``.

Using this helper function we can rewrite the test as follows:

.. code-block:: python

    @pytest.mark.parametrize('target', ['mask', 'image'])
    def test_vflip_img_and_mask(target):
        img = np.array(
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected = np.array(
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]], dtype=np.uint8)
        img, expected = convert_2d_to_target_format(img, expected, target=target)
        flipped_img = F.vflip(img)
        assert np.array_equal(flipped_img, expected)

pytest output::

    tests/test_example.py::test_vflip_img_and_mask[mask] PASSED
    tests/test_example.py::test_vflip_img_and_mask[image] PASSED

---------------------
Implementation notes:
---------------------

Implementations of ``convert_2d_to_target_format`` and ``convert_2d_to_3d`` in albumentations slightly differ from implementations described above. We need to support both Python 2.7 and Python 3, so we can't use a function declaration like ``def convert_2d_to_target_format(*arrays, target=None)`` because it produces ``SyntaxError`` in Python 2 and only valid in Python 3 (see `PEP3102 <https://www.python.org/dev/peps/pep-3102/>`_ for more details). Because of this we use the following function declaration: ``def convert_2d_to_target_format(arrays, target)`` where the  ``arrays`` argument should contain a list of NumPy arrays.

The test can be rewritten as follows to be compatible with the current albumentations' test suite (note an updated call to ``convert_2d_to_target_format``, we pass ``img`` and ``expected`` arguments inside a single list):

.. code-block:: python

    @pytest.mark.parametrize('target', ['mask', 'image'])
    def test_vflip_img_and_mask(target):
        img = np.array(
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected = np.array(
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]], dtype=np.uint8)
        img, expected = convert_2d_to_target_format([img, expected], target=target)
        flipped_img = F.vflip(img)
        assert np.array_equal(flipped_img, expected)

***************
Using fixtures.
***************

Let's say that we want to test a situation in which we pass an image and mask with the ``np.uint8`` data type to the ``VerticalFlip`` augmentation and we expect that it won’t change data types of inputs and will produce an image and mask with the ``np.uint8`` data type as output.

Such a test can be written as follows:

.. code-block:: python

    from albumentations import VerticalFlip

    def test_vertical_flip_dtype():
        aug = VerticalFlip(p=1)
        image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
        data = aug(image=image, mask=mask)
        assert data['image'].dtype == np.uint8
        assert data['mask'].dtype == np.uint8

We generate a random image and a random mask, then we pass them as inputs to the augmentation and then we check a data type of output values.

If we want to perform this check for other augmentations as well, we will have to write code to generate a random image and mask at the beginning of each test:

.. code-block:: python

    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)

To avoid this duplication we can move code that generates random values to a fixture. Fixtures work as follows:

1. In the ``tests/conftest.py`` file we create functions that are wrapped with the ``@pytest.fixture`` decorator:

.. code-block:: python

    @pytest.fixture
    def image():
        return np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def mask():
        return np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)

2. In our test we use fixture names as accepted arguments:

.. code-block:: python

    def test_vertical_flip_dtype(image, mask):
        ...

3. pytest will use arguments' names to find fixtures with the same names, then it will execute those fixture functions and will pass the outputs of this functions as arguments to the test function.

We can rewrite ``test_vertical_flip_dtype`` using fixtures as follows:

.. code-block:: python

    def test_vertical_flip_dtype(image, mask):
        aug = VerticalFlip(p=1)
        data = aug(image=image, mask=mask)
        assert data['image'].dtype == np.uint8
        assert data['mask'].dtype == np.uint8

*************************************************
Simultaneous use of fixtures and parametrization.
*************************************************

Let's say that besides ``VerticalFlip`` we also want to test that ``HorizontalFlip`` also returns values with the ``np.uint8`` data type if we passed a ``np.uint8`` input to it.

We can write test like this:

.. code-block:: python

    from albumentations import HorizontalFlip

    def test_horizontal_flip_dtype(image, mask):
        aug = HorizontalFlip(p=1)
        data = aug(image=image, mask=mask)
        assert data['image'].dtype == np.uint8
        assert data['mask'].dtype == np.uint8

But this test is almost completely identical to ``test_vertical_flip_dtype``. And to check each new augmentation we will have to copy practically almost the whole code from ``test_vertical_flip_dtype`` and change the value of the ``aug`` variable, so the test will use a new augmentation. However it would be great to get rid of unnecessary copying of code in tests. For this, we could use parametrization and pass a class as a parameter.

A test that checks both VerticalFlip and HorizontalFlip can be written as follows:

.. code-block:: python

    from albumentations import VerticalFlip, HorizontalFlip

    @pytest.mark.parametrize('augmentation_cls', [
        VerticalFlip,
        HorizontalFlip,
    ])
    def test_multiple_augmentations(augmentation_cls, image, mask):
        aug = augmentation_cls(p=1)
        data = aug(image=image, mask=mask)
        assert data['image'].dtype == np.uint8
        assert data['mask'].dtype == np.uint8

This test will run two times, in the first run the ``augmentation_cls`` argument will be equal to ``VerticalFlip``. In the second run the ``augmentation_cls`` argument will be equal to ``HorizontalFlip``.

pytest output::

    tests/test_example.py::test_multiple_augmentations[VerticalFlip] PASSED
    tests/test_example.py::test_multiple_augmentations[HorizontalFlip] PASSED
