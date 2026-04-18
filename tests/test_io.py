"""Tests for muse.io module."""

from __future__ import annotations


class TestPmgtrajToExtxyz:
    """Tests for the pmgtraj_to_extxyz function."""

    def test_import(self):
        """Verify the function can be imported from the public API."""
        from muse.io import pmgtraj_to_extxyz
        assert callable(pmgtraj_to_extxyz)

    def test_import_from_module(self):
        """Verify the function can be imported from the module directly."""
        from muse.io.mptrj import pmgtraj_to_extxyz
        assert callable(pmgtraj_to_extxyz)
