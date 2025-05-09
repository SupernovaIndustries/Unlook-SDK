Getting Started
===============

This guide will help you quickly get started with the Unlook SDK for 3D scanning.

Prerequisites
------------

Before you begin, ensure you have the following prerequisites:

- Python 3.7+ (Python 3.9 or 3.10 recommended)
- For server component: Raspberry Pi with Raspberry Pi OS
- Unlook scanner modules (structured light, depth sensor, etc.)

Quick Installation
----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/SupernovaIndustries/unlook.git
   cd unlook

   # Install dependencies
   pip install -r client-requirements.txt

   # For GPU acceleration (NVIDIA)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

Basic Usage
----------

Connect to a Scanner
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from unlook import UnlookClient

   # Create client with auto-discovery
   client = UnlookClient(auto_discover=True)
   client.start_discovery()
   
   # Wait for discovery
   import time
   time.sleep(5)
   
   # Connect to first available scanner
   scanners = client.get_discovered_scanners()
   if scanners:
       client.connect(scanners[0])
       print(f"Connected to: {scanners[0].name}")

Real-time Scanning
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from unlook import UnlookClient
   from unlook.client.realtime_scanner import create_realtime_scanner
   
   # Connect to scanner (as shown above)
   # ...
   
   # Create real-time scanner
   scanner = create_realtime_scanner(
       client=client,
       quality="medium",  # Options: "fast", "medium", "high", "ultra"
   )
   
   # Start scanning
   scanner.start()
   
   # Get point cloud data
   point_cloud = scanner.get_current_point_cloud()
   
   # When done
   scanner.stop()

Next Steps
---------

- Check out the :doc:`installation` guide for detailed installation instructions
- See the :doc:`user_guide/index` for comprehensive usage information
- Explore the :doc:`examples/index` for practical code examples
- Review the :doc:`api_reference/index` for detailed API documentation