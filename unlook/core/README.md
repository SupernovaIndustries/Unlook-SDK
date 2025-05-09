# Unlook SDK Core Module

This directory contains the core components of the Unlook SDK, providing the foundation for communication, discovery, and event handling between clients and servers.

## Components

- **constants.py** - Core constants and configuration values
- **discovery.py** - Service discovery functionality for finding Unlook scanners on the network
- **events.py** - Event handling system for asynchronous communication
- **protocol.py** - Communication protocol definitions and message formats
- **utils.py** - Utility functions used across the SDK

## Architecture

The core module establishes the communication architecture used throughout the Unlook SDK:

1. **Discovery Protocol**: Uses mDNS/Zeroconf for automatic scanner discovery on local networks
2. **Event System**: Pub/Sub model for event-driven communication
3. **Message Protocol**: Structured message format for reliable client-server communication
4. **ZeroMQ Transport**: Efficient messaging layer for all communication channels

## Usage Examples

The core module is typically used through the UnlookClient and UnlookServer classes, but can also be used directly for advanced use cases:

```python
from unlook.core.discovery import ServiceDiscovery
from unlook.core.events import EventBus, EventType

# Example: Manual discovery
discovery = ServiceDiscovery()
discovery.start()
scanners = discovery.get_discovered_services()

# Example: Event system
event_bus = EventBus()
event_bus.on(EventType.SCANNER_DISCOVERED, lambda scanner: print(f"Found: {scanner.name}"))
```

## Extension Points

When extending the SDK, these core components provide several integration points:

- **Custom Event Types**: Add new event types in `events.py`
- **Protocol Extensions**: Extend message formats in `protocol.py`
- **Discovery Filters**: Customize service discovery criteria in `discovery.py`