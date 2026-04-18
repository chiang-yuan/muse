"""Shared pytest fixtures for the muse test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def nacl_kcl_recipe() -> dict[str, int]:
    """A simple NaCl-KCl binary mixture recipe for testing."""
    return {"NaCl": 2, "KCl": 1}
