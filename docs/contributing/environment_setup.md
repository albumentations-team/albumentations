# Setting Up Your Development Environment

This guide will help you set up your development environment for contributing to Albumentations.

## Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

## Step-by-Step Setup

### 1. Fork and Clone the Repository

1. Fork the [Albumentations repository](https://github.com/albumentations-team/albumentations) on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/albumentations.git
cd albumentations
```

### 2. Create a Virtual Environment

Choose the appropriate commands for your operating system:

#### Linux / macOS

```bash
python3 -m venv env
source env/bin/activate
```

#### Windows (cmd.exe)

```bash
python -m venv env
env\Scripts\activate.bat
```

#### Windows (PowerShell)

```bash
python -m venv env
env\Scripts\activate.ps1
```

### 3. Install Dependencies

1. Install the project in editable mode:

```bash
pip install -e .
```

1. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### 4. Set Up Pre-commit Hooks

Pre-commit hooks help maintain code quality by automatically checking your changes before each commit.

1. Install pre-commit:

```bash
pip install pre-commit
```

1. Set up the hooks:

```bash
pre-commit install
```

1. (Optional) Run hooks manually on all files:

```bash
pre-commit run --files $(find albumentations -type f)
```

## Verifying Your Setup

### Run Tests

Ensure everything is set up correctly by running the test suite:

```bash
pytest
```

### Common Issues and Solutions

#### Permission Errors

- **Linux/macOS**: If you encounter permission errors, try using `sudo` for system-wide installations or consider using `--user` flag with pip
- **Windows**: Run your terminal as administrator if you encounter permission issues

#### Virtual Environment Not Activating

- Ensure you're in the correct directory
- Check that Python is properly installed and in your system PATH
- Try creating the virtual environment with the full Python path

#### Import Errors After Installation

- Verify that you're using the correct virtual environment
- Confirm that all dependencies were installed successfully
- Try reinstalling the package in editable mode

## Next Steps

After setting up your environment:

1. Create a new branch for your work
2. Make your changes
3. Run tests and pre-commit hooks
4. Submit a pull request

For more detailed information about contributing, please refer to [Coding Guidelines](./coding_guidelines.md)

## Getting Help

If you encounter any issues with the setup:

1. Check our [Discord community](https://discord.gg/e6zHCXTvaN)
2. Open an [issue on GitHub](https://github.com/albumentations-team/albumentations/issues)
3. Review existing issues for similar problems and solutions
