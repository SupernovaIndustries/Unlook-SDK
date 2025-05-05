"""
Modulo common per funzionalità condivise tra client e server.
"""

from .constants import *
from .protocol import MessageType, Message
from .discovery import UnlookDiscovery, UnlookScanner