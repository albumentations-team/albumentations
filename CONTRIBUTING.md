# Contributing to Albumentations

Thank you for your interest in contributing to [Albumentations](https://albumentations.ai/)! This guide is designed to make it easier for you to get involved and help us build a powerful, efficient, and easy-to-use image augmentation library.

For small changes (e.g., bug fixes), feel free to submit a PR.

For larger changes, consider
creating an [**issue**](https://github.com/albumentations-team/albumentations/issues) outlining your proposed change.
You can also join us on [**Discord**](https://discord.gg/e6zHCXTvaN) to discuss your idea with the
community.

## Getting Started

### Setting Up Your Development Environment

#### Fork and clone the repository

Start by forking the project repository to your GitHub account, then clone your fork to your local machine:

```bash
git clone https://github.com/albumentations/albumentations.git
cd albumentations
```

#### Create a virtual environment

We recommend using a virtual environment to isolate project dependencies. Ensure you have Python 3.8 or higher installed on your machine, as it is the minimum supported version for Albumentations. To create and activate a virtual environment, run the following commands:

#### Linux / macOS

```bash
python3 -m venv env
source env/bin/activate
```

#### Windows cmd.exe

```bash
python -m venv env
env\Scripts\activate.bat
```

#### Windows PowerShell

```bash
python -m venv env
env\Scripts\activate.ps1
```

#### Install development dependencies

Install the project's dependencies by running:

```bash
pip install -e .
```

Additionally, to ensure you have all the necessary tools for code formatting, linting, and additional development utilities, install the requirements from `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

## Navigating the Project

* The main source code is located in the `albumentations/` directory.
* Tests are located in the `tests/` directory. Every pull request should include tests for new features or bug fixes.

## How to Contribute

### Types of Contributions

* **Code Contributions**: Whether it's fixing a bug, adding a new feature, or improving performance, your code contributions are valuable.
* **Documentation:** Help us improve the project's documentation for better usability and accessibility.
* **Bug Reports and Feature Requests**: Use GitHub Issues to report bugs, request features, or suggest improvements.

### Contribution Process

1. **Find an issue to work on**: Look for open issues or propose a new one. For newcomers, look for issues labeled "good first issue."
2. **Fork the repository** (if you haven't already).
3. **Create a new branch for your changes**: `git checkout -b feature/my-new-feature`.
4. **Implement your changes**: Write clean, readable, and well-documented code. Note that we do not use np.random module directly but call corresponding functions from the albumentations/random_utils.py module to ensure consistency and control over randomness.
5. **Add or update tests** as necessary.
6. **Ensure all tests pass** and your code adheres to the existing style guidelines.
7. **Submit a Pull Request (PR):** Open a PR from your forked repository to the main Albumentations repository. Provide a clear description of the changes and any relevant issue numbers.

### Code Review Process

* Once you submit a PR, the Albumentations maintainers will review your contribution.
* Engage in the review process if the maintainers have feedback or questions.
* Once your PR is approved, a maintainer will merge it into the main codebase.

## Coding Guidelines

### Using Pre-commit Hooks

To maintain code quality and consistency, we use pre-commit hooks. Follow these steps to set up pre-commit hooks in your local repository:

**Install pre-commit:** If you haven't already, you need to install pre-commit on your machine. You can do this using pip:

```bash
pip install pre-commit
```

**Initialize pre-commit:**

Navigate to the root of your cloned repository and run:

```bash
pre-commit install
```

This command sets up the pre-commit hooks based on the configurations found in `.pre-commit-config.yaml` at the root of the repository.

**Running pre-commit hooks:**

Pre-commit will automatically run the configured hooks on each commit. You can also manually run the hooks on all files in the repository with:

```bash
pre-commit run --all-files
```

Ensure to fix any issues detected by the pre-commit hooks before submitting your pull request.

### Running Tests

Before submitting your contributions, it's important to ensure that all tests pass. This helps maintain the stability and reliability of Albumentations. Here's how you can run the tests:

Install test dependencies:

If you haven't installed the development dependencies, make sure to do so. These dependencies include `pytest`, which is required to run the tests.

```bash
pip install -e .
```

```bash
pip install -r requirements-dev.txt
```

Run the tests:

With `pytest` installed, you can run all tests using the following command from the root of the repository:

```bash
pytest
```

This will execute all the tests and display the results, indicating whether each test passed or failed.

**Tip**: If you've made changes to a specific area of the library, you can run a subset of the tests related to your changes. This can save time and make it easier to debug issues. Use the `pytest` documentation to learn more about running specific tests.

### Ensuring Your Contribution is Ready

* After setting up pre-commit hooks and ensuring all tests pass, your contribution is nearly ready for submission.
* Review your changes one last time, ensuring they meet the project's coding guidelines and documentation standards.
* If your changes affect how users interact with Albumentations, update the documentation accordingly.

### Adding or Modifying Transforms

#### Validation with InitSchema

Each transform includes an `InitSchema` class responsible for validating and modifying input parameters before the execution of the `__init__` method. This step ensures that all parameter manipulations, such as converting a single value into a range, are handled consistently and appropriately.

#### Simplifying Parameter Types

Historically, our transforms have used `Union` types to allow flexibility in parameter input—accepting either a single value or a tuple. While flexible, this approach can lead to confusion about how parameters are interpreted and used within the transform. For example, when a single value is provided, it is unclear whether and how it will be expanded into a tuple, which can lead to unpredictable behavior and difficult-to-understand code.

To improve clarity and predictability:

**Explicit Definitions**: Parameters should be explicitly defined as either a single value or a tuple. This change avoids ambiguity and ensures that the intent and behavior of the transform are clear to all users.

#### Serialization Considerations

Even if a parameter defined as `Tuple`, the transform should work correctly with a `List` that have similar values. This is required as `JSON` and `YAML` serialization formats do not distinguish between lists and tuples and if you serialize and then deserialize it back, you will get a list instead of a tuple. If it is not the case test will fail, but, just in case, keep this in mind while creating or modifying the transform.

**List Compatibility**: Because these formats do not distinguish between lists and tuples, using List in type definitions ensures that transforms work correctly post-serialization, which treats tuples as lists.

#### Probability Handling

To maintain determinism and reproducibility, handle all probability calculations within the `get_params` or `get_params_dependent_on_targets` methods. These calculations should not occur in the `apply_xxx` or `__init__` methods, as it is crucial to separate configuration from execution in our codebase.

#### Using Random Number Generators

When you need to use random number generation in your contributions:

* Prefer `random` from the standard library: Use `random` whenever possible as it generally offers faster performance compared to `np.random`.
* Use `random_utils` for functions from `np.random`: When you need specific functionality provided by `np.random`, use the corresponding functions from `albumentations/random_utils.py` to ensure consistency and control over randomness.

By following this approach, we maintain the efficiency and consistency of random operations across the codebase.

### Specific Guidelines for Method Definitions

#### Handling `apply_xxx` Methods

When contributing code related to transformation methods, specifically methods that start with `apply_` (e.g., `apply_to_mask`, `apply_to_bbox`), please adhere to the following guidelines:

**No Default Arguments**: Do not use default arguments in `apply_xxx` methods. Every parameter should be explicitly required, promoting clarity and reducing hidden behaviors that can arise from default values.

### Examples

Here are a few examples to illustrate these guidelines:

**Incorrect** method definition:

```python
def apply_to_mask(self, mask, fill_value=0):  # Default value not allowed
    # implementation
```

**Correct** method definition:

```python
def apply_to_mask(self, mask, fill_value):  # No default values
    # implementation
```

## Guidelines for Modifying Existing Code

Maintaining the stability and usability of Albumentations for all users is a priority. When contributing, it's important to follow these guidelines for modifying existing code:

### Transform Modifications

**Transform Interfaces**: Changes to transform interfaces or the removal of old transforms should be handled delicately. Introduce changes through a deprecation warning phase that lasts several months. This provides users ample time to adapt to new changes.

### Custom Transformations

**Support for Customization**: We highly value the ability to create custom transformations based on ImageOnlyTransform and DualTransform. Significant changes, like the removal of methods such as get_params_depend_on_targets, should also proceed through a deprecation phase to preserve backward compatibility.

### Helper Functions

**Flexibility with Helpers**: Helper functions can be modified more freely. While these functions may be directly used by some users, typically they are power users who are capable of handling such changes. Thus, adding, removing, or moving helper functions can be done with relative freedom.

### Private Methods and Internal Logic

**Internal Changes**: Private methods and functions within transform_interface and Compose that do not affect inheritance from ImageOnlyTransform, DualTransform, or alter the behavior of transformations can be changed or optimized boldly. These modifications do not require a deprecation phase.

### Handling Broken Features

**Rapid Response**: If it becomes evident that a transformation or feature is fundamentally broken, take decisive action to fix or overhaul it. This may involve substantial changes or relocations within the codebase to correct the issue efficiently.

### Application of Changes

* Always document your changes thoroughly, especially if they affect how users interact with the library.
* Use the pull request description to explain your changes and the reasons behind them. This helps maintainers understand your decisions and facilitates the review process.
By adhering to these guidelines, contributors can ensure that their enhancements and fixes are integrated smoothly and maintain the high standards of the Albumentations library.

## Additional Resources

[Albumentations Documentation](https://albumentations.ai/docs/)

## Acknowledgements

Your contributions are appreciated and recognized. Contributors who have significantly impacted the project will be mentioned in our documentation and releases.

## Contact Information

For any questions or concerns about contributing, please reach out to the maintainers via GitHub Issues.
