# UnLook SDK Documentation

This directory contains the documentation for the UnLook SDK, built using [Sphinx](https://www.sphinx-doc.org/) and hosted on [ReadTheDocs](https://readthedocs.org/).

## Documentation Structure

The documentation is organized as follows:

- `source/`: Source files for the documentation
  - `index.rst`: Main index file
  - `api_reference/`: API reference documentation
  - `user_guide/`: User guides for different features
  - `examples/`: Example code with explanations
  - `_static/`: Static files like images and CSS
  - `_templates/`: Custom templates for Sphinx
  - `conf.py`: Sphinx configuration

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation Locally

```bash
cd docs
make html
```

The output will be in `build/html/`. Open `build/html/index.html` in your browser to view the documentation.

### Build PDF Documentation

```bash
cd docs
make latexpdf
```

The output will be in `build/latex/unlooksdk.pdf`.

## Contributing to Documentation

### Adding New Documentation

1. Create new `.rst` files in the appropriate directories:
   - Add API documentation to `source/api_reference/`
   - Add user guides to `source/user_guide/`
   - Add examples to `source/examples/`

2. Update the relevant index files to include your new documentation.

### Documentation Style Guidelines

- Use title case for section titles
- Use sentence case for subsection titles
- Use reStructuredText syntax for formatting
- Include code examples using the `.. code-block:: python` directive
- Document all functions and classes with proper docstrings
- Include links to related documentation

### Updating Documentation for Code Changes

When making changes to the codebase, please update the relevant documentation:

1. **API Changes**: Update the appropriate files in `api_reference/`
2. **New Features**: Add documentation to the `user_guide/` and update examples
3. **Bug Fixes**: Update any incorrect documentation

## ReadTheDocs Configuration

The ReadTheDocs configuration is in `.readthedocs.yaml` at the root of the repository. This file controls how the documentation is built on ReadTheDocs.

## Regenerating API Reference

To automatically generate API reference documentation:

```bash
sphinx-apidoc -o docs/source/api_reference/ unlook/ -f
```

## Adding Images

Place image files in `source/_static/` and reference them in your RST files:

```rst
.. image:: _static/image_name.png
   :width: 400
   :alt: Description of image
```

## Converting Markdown to RST

When converting Markdown (`.md`) files to reStructuredText (`.rst`), keep in mind these differences:

- **Links**: Change `[text](url)` to `` `text <url>`_ ``
- **Code Blocks**: Change triple backticks to `.. code-block::`
- **Headers**: Use consistent underlines for headers (e.g., `=` for title, `-` for section)
- **Lists**: Ensure proper indentation for nested lists

You can use online converters or tools like `pandoc` to help with the conversion:

```bash
pandoc -f markdown -t rst input.md -o output.rst
```

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [ReadTheDocs Configuration](https://docs.readthedocs.io/en/stable/config-file/v2.html)