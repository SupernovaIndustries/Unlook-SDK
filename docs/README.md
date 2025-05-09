# Unlook SDK Documentation

This directory contains the documentation for the Unlook SDK, set up to be built with Sphinx and hosted on Read the Docs.

## Building the Documentation

To build the documentation locally:

1. Install the required dependencies:

   ```bash
   pip install -r docs/requirements.txt
   ```

2. Build the documentation:

   ```bash
   cd docs
   make html
   ```

3. View the documentation:

   Open `docs/build/html/index.html` in your web browser.

## Documentation Structure

- `source/`: Source files for the documentation
  - `api_reference/`: API reference documentation
  - `examples/`: Code examples
  - `user_guide/`: User guides and tutorials
  - `_static/`: Static files (images, CSS, etc.)
  - `_templates/`: Custom templates
  - `conf.py`: Sphinx configuration
  - Various `.rst` files for main pages

## Updating the Documentation

When making changes to the codebase, please update the relevant documentation:

1. **API Changes**: Update the appropriate files in `api_reference/`
2. **New Features**: Add documentation to the `user_guide/` and update examples if needed
3. **Bug Fixes**: Update any incorrect documentation

## Read the Docs Configuration

This documentation is set up to be built and hosted on Read the Docs. The configuration is in `.readthedocs.yaml` in the project root.

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

## Generating PDF Documentation

To generate PDF documentation:

```bash
cd docs
make latexpdf
```

The PDF will be available in `docs/build/latex/`.