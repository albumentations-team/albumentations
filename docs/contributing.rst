Contributing
============
All development is done on GitHub: https://github.com/albumentations-team/albumentations

If you find a bug or have a feature request file an issue at https://github.com/albumentations-team/albumentations/issues

To create a pull request:
=======================

1. Fork the repository.
2. Clone it.
3. Install pre-commit hook:

.. code-block:: bash

    pip install pre-commit black flake8

4. Initialize it from the folder with the repo:

.. code-block:: bash

    pre-commit install


4. Make desired changes to the code.
5. Install the library in development mode:


.. code-block:: bash

    pip install -e .[tests]


6. Run tests:

.. code-block:: bash

    pytest


7. Push code to your forked repo.
8. Create pull request.
