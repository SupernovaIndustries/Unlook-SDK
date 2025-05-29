Contributing
============

We welcome contributions to the Unlook SDK! This guide will help you get started with contributing to the project.

Getting Started
--------------

**Repository Structure**

The Unlook SDK repository is organized as follows:

- ``unlook/`` - Main SDK package
  - ``client/`` - Client-side modules (cameras, scanners, etc.)
  - ``server/`` - Server-side hardware control
  - ``core/`` - Core SDK functionality
  - ``examples/`` - Example scripts and demonstrations
- ``docs/`` - Documentation source files
- ``tests/`` - Test suite

**Development Setup**

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/unlook-sdk.git
      cd unlook-sdk

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

Types of Contributions
--------------------

**Bug Reports**

When reporting bugs, please include:

- Clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- System information (OS, Python version, hardware)
- Relevant log output or error messages

**Feature Requests**

For feature requests, please provide:

- Clear description of the proposed feature
- Use case and motivation
- Proposed API design (if applicable)
- Any implementation ideas or references

**Code Contributions**

We accept contributions for:

- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements
- Example scripts and tutorials

Development Guidelines
--------------------

**Code Style**

- Follow PEP 8 style guidelines
- Use type hints for all new code
- Write docstrings for all public functions and classes
- Keep functions focused and reasonably sized

**Testing**

- Write tests for all new functionality
- Ensure existing tests continue to pass
- Aim for good test coverage
- Include both unit tests and integration tests where appropriate

**Documentation**

- Update documentation for any API changes
- Add examples for new features
- Update changelog for significant changes
- Write clear commit messages

**Pull Request Process**

1. Create a feature branch from ``dev``:

   .. code-block:: bash

      git checkout dev
      git pull origin dev
      git checkout -b feature/your-feature-name

2. Make your changes and commit them:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: brief description"

3. Push to your fork and create a pull request:

   .. code-block:: bash

      git push origin feature/your-feature-name

4. Fill out the pull request template with:
   - Description of changes
   - Testing performed
   - Any breaking changes
   - Documentation updates

**Review Process**

- All pull requests require review from maintainers
- Address any feedback or requested changes
- Ensure CI tests pass
- Maintainers will merge approved pull requests

Community Guidelines
------------------

**Communication**

- Be respectful and inclusive in all interactions
- Ask questions if anything is unclear
- Provide constructive feedback
- Help other contributors when possible

**Issue Triage**

Help us manage issues by:

- Reproducing reported bugs
- Adding labels and categorizing issues
- Answering questions from other users
- Identifying duplicate issues

Recognition
----------

Contributors are recognized in several ways:

- Listed in the project contributors
- Mentioned in release notes for significant contributions
- Invited to join the maintainer team for sustained contributions

Getting Help
-----------

If you need help with contributing:

- Check existing documentation and examples
- Search through existing issues and discussions
- Ask questions in new issues with the "question" label
- Reach out to maintainers for guidance

Thank you for your interest in contributing to the Unlook SDK!