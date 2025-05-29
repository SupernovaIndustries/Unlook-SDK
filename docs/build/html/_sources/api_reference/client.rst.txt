Client API
=========

The client API provides functionality for connecting to and controlling Unlook scanners.

UnlookClient
-----------

.. autoclass:: unlook.client.scanner.UnlookClient
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
----------

.. autoclass:: unlook.core.discovery.ScannerInfo
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: unlook.core.events.EventType
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-----------

Basic Connection:

.. code-block:: python

   from unlook import UnlookClient
   
   # Create client
   client = UnlookClient(client_name="ExampleApp")
   
   # Start discovery
   client.start_discovery()
   
   # Wait for discovery
   import time
   time.sleep(5)
   
   # Connect to the first available scanner
   scanners = client.get_discovered_scanners()
   if scanners:
       client.connect(scanners[0])
       print(f"Connected to: {scanners[0].name}")
   
   # Use the scanner
   # ...
   
   # Disconnect when done
   client.disconnect()