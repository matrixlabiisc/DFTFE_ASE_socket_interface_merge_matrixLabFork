# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Pytest fixtures. Adds ``src/`` to sys.path so tests run without installing."""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

MOCK = os.path.join(_HERE, "mock_dftfe.py")


@pytest.fixture
def mock_command():
    """Launch command that runs the fake DFT-FE under this interpreter."""
    return f"{sys.executable} {MOCK}"
