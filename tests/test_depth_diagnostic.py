#!/usr/bin/env python3
# Test to see if depth_map_diagnostic.py has any issues

import subprocess
import sys

result = subprocess.run([sys.executable, "-c", "import unlook.examples.scanning.depth_map_diagnostic"], 
                      capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)