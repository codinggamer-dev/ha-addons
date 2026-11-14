"""Make the repository root importable so tests can import the integration.

``custom_components`` is an implicit namespace package, so the pure modules
(``tools``, ``needle``, ``audio``) can be imported without Home Assistant
installed. ``custom_components.cactus_home.__init__`` defers every Home
Assistant import to call time, which is what makes that possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
