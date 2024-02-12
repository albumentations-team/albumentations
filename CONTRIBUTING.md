# Contributing to Albumentations

Thank you for your interest in contributing to Albumentations! This guide is designed to make it easier for you to get involved and help us build a powerful, efficient, and easy-to-use image augmentation library.

## Getting Started

### Setting Up Your Development Environment

#### Fork and clone the repository

Start by forking the project repository to your GitHub account, then clone your fork to your local machine:

```bash
git clone https://github.com/your-username/albumentations.git
cd albumentations
```

#### Create a virtual environment

We recommend using a virtual environment to isolate project dependencies:

```bash
python3 -m venv env
source env/bin/activate
```

#### Install development dependencies:

Install the project's development dependencies, which include testing and documentation tools:

```bash
pip install -e .[develop]
```

## Navigating the Project

* The main source code is located in the `albumentations/` directory.
* Tests are located in the `tests/` directory. Every pull request should include tests for new features or bug fixes.
* Documentation can be found in the docs/ directory. Contributions to improve or expand the documentation are always welcome.

## How to Contribute

### Types of Contributions

* **Code Contributions**: Whether it's fixing a bug, adding a new feature, or improving performance, your code contributions are valuable.
* **Documentation:** Help us improve the project's documentation for better usability and accessibility.
* **Bug Reports and Feature Requests**: Use GitHub Issues to report bugs, request features, or suggest improvements.

### Contribution Process

1. **Find an issue to work on**: Look for open issues or propose a new one. For newcomers, look for issues labeled "good first issue."
2. **Fork the repository** (if you haven't already).
3. **Create a new branch for your changes**: `git checkout -b feature/my-new-feature`.
4. **Implement your changes**: Write clean, readable, and well-documented code.
5. **Add or update tests** as necessary.
6. **Ensure all tests pass** and your code adheres to the existing style guidelines.
7. **Submit a Pull Request (PR):** Open a PR from your forked repository to the main Albumentations repository. Provide a clear description of the changes and any relevant issue numbers.

### Code Review Process

* Once you submit a PR, the Albumentations maintainers will review your contribution.
* Engage in the review process if the maintainers have feedback or questions.
* Once your PR is approved, a maintainer will merge it into the main codebase.

## Coding Guidelines

### Using Pre-commit Hooks

To maintain code quality and consistency, we use pre-commit hooks. These hooks run checks on your commits before they are submitted, helping catch common issues early. To set up pre-commit hooks in your local repository, follow these steps:

1. **Install pre-commit:** If you haven't already, you need to install pre-commit on your machine. You can do this using pip:

```bash
pip install pre-commit
```

2. **Initialize pre-commit:**

Navigate to the root of your cloned repository and run:

```bash
pre-commit install
```

This command sets up the pre-commit hooks based on the configurations found in `.pre-commit-config.yaml` at the root of the repository.

3. **Run pre-commit hooks:**

Pre-commit will now automatically run on each commit. However, you can manually run the checks on all files in the repository with:

```bash
pre-commit run --all-files
```

Ensure to fix any issues detected by the pre-commit hooks before submitting your pull request.

### Running Tests

Before submitting your contributions, it's important to ensure that all tests pass. This helps maintain the stability and reliability of Albumentations. Here's how you can run the tests:

Install test dependencies:

If you haven't installed the development dependencies, make sure to do so. These dependencies include `pytest`, which is required to run the tests.

```bash
pip install -e .[tests]
```

Run the tests:

With `pytest` installed, you can run all tests using the following command from the root of the repository:

bash
Copy code
pytest
This will execute all the tests and display the results, indicating whether each test passed or failed.

**Tip**: If you've made changes to a specific area of the library, you can run a subset of the tests related to your changes. This can save time and make it easier to debug issues. Use the pytest documentation to learn more about running specific tests.

### Ensuring Your Contribution is Ready

* After setting up pre-commit hooks and ensuring all tests pass, your contribution is nearly ready for submission.
* Review your changes one last time, ensuring they meet the project's coding guidelines and documentation standards.
* If your changes affect how users interact with Albumentations, update the documentation accordingly.
*
By following these guidelines, you help ensure that Albumentations remains a high-quality, reliable library. We appreciate your contributions and look forward to your pull request!

## Additional Resources

[Albumentations Documentation](https://albumentations.ai/docs/)

## Acknowledgements

Your contributions are appreciated and recognized. Contributors who have significantly impacted the project will be mentioned in our documentation and releases.

## Contact Information

For any questions or concerns about contributing, please reach out to the maintainers via GitHub Issues.
