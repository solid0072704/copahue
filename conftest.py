"""Pytest root hook: make the project importable however pytest is launched.

`pytest -q` (console script) does not put the working directory on sys.path, so
`from trading_agent import ...` and `from tests.trading_helpers import ...` fail
without this. `pytest.ini` also sets `pythonpath = .`; this file covers older
pytest versions and editors that start pytest from another directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
