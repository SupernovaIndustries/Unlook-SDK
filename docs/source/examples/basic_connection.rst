Basic Connection
===============

This example demonstrates how to establish a basic connection to an UnLook scanner.

Discovering and Connecting to Scanners
-------------------------------------

.. code-block:: python

   from unlook import UnlookClient, EventType
   import time

   # Create client
   client = UnlookClient(client_name="ExampleApp")

   # Register callback for events
   def on_connected(scanner):
       print(f"Connected to: {scanner.name} ({scanner.uuid})")

   client.on(EventType.CONNECTED, on_connected)

   # Start discovery
   client.start_discovery()

   print("Discovering scanners...")
   # Wait for discovery to find scanners
   time.sleep(5)

   # Connect to the first available scanner
   scanners = client.get_discovered_scanners()
   if scanners:
       print(f"Found {len(scanners)} scanners:")
       for i, scanner in enumerate(scanners):
           print(f"  {i+1}. {scanner.name} - {scanner.uuid}")
           
       client.connect(scanners[0])
       
       # Demonstrate that we're connected
       print(f"Client connected: {client.is_connected()}")
       
       # Get basic info
       info = client.get_scanner_info()
       print(f"Scanner info: {info}")
       
       # When done
       time.sleep(3)
       client.disconnect()
   else:
       print("No scanners found.")

Connecting to a Known Scanner
----------------------------

If you already know the scanner details, you can connect directly:

.. code-block:: python

   from unlook import UnlookClient

   # Connect to a known scanner
   client = UnlookClient()
   client.connect("192.168.1.100", 5555)  # IP address and port

   # Check if connected
   if client.is_connected():
       print("Connected directly to scanner")
       # Perform operations...
       client.disconnect()
   else:
       print("Failed to connect to scanner")

Connection Callbacks
------------------

You can register callbacks for connection events:

.. code-block:: python

   from unlook import UnlookClient, EventType

   client = UnlookClient()

   # Connection events
   def on_connected(scanner):
       print(f"Connected to scanner: {scanner.name}")

   def on_disconnected(data):
       print("Disconnected from scanner")

   def on_connection_error(error):
       print(f"Connection error: {error}")

   # Register callbacks
   client.on(EventType.CONNECTED, on_connected)
   client.on(EventType.DISCONNECTED, on_disconnected)
   client.on(EventType.CONNECTION_ERROR, on_connection_error)

   # Use the client
   client.start_discovery()
   # ...

Full Example
-----------

For a complete, runnable example, see the file ``unlook/examples/basic/hello_unlook.py`` in the SDK.