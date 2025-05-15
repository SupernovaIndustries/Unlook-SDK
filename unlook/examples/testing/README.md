# Testing Examples

This directory contains test scripts organized by category:

## Directory Structure

- **imports/**: Test files for import functionality
  - `test_imports.py`: Basic import tests
  - `test_example_imports.py`: Example module import tests

- **camera/**: Camera-related test files
  - `test_camera_import_fix.py`: Tests for camera import fixes
  - `test_client_camera_access.py`: Client camera access tests
  - `test_minimal_camera.py`: Minimal camera functionality tests

- **patterns/**: Pattern generation test files
  - `test_maze_init.py`: Maze pattern initialization tests

- **integration/**: Integration and system tests
  - `test_enum_fixes.py`: Enum handling fix tests
  - `test_fix_verification.py`: Fix verification tests
  - `test_scanner_simulation.py`: Scanner simulation tests

## Running Tests

To run any test:
```bash
python unlook/examples/testing/[category]/[test_file].py
```

For example:
```bash
python unlook/examples/testing/camera/test_camera_import_fix.py
```