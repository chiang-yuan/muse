"""Tests for muse.transforms.mixture structure generation."""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase import Atoms

_HAS_MP_API_KEY = os.getenv("MP_API_KEY") is not None


@pytest.mark.skipif(not _HAS_MP_API_KEY, reason="MP_API_KEY not set")
class TestMixNumber:
    """Tests for the mix_number function (requires MP_API_KEY)."""

    def test_basic_mixture(self, nacl_kcl_recipe):
        """Test that mix_number returns a valid Atoms object."""
        from muse.transforms.mixture import mix_number

        atoms = mix_number(recipe=nacl_kcl_recipe, seed=42, tolerance=2.0, scale=1.1)

        assert isinstance(atoms, Atoms)
        assert len(atoms) > 0
        assert all(atoms.pbc)

    def test_mixture_has_correct_elements(self, nacl_kcl_recipe):
        """Test that the mixture contains the expected chemical elements."""
        from muse.transforms.mixture import mix_number

        atoms = mix_number(recipe=nacl_kcl_recipe, seed=42, tolerance=2.0, scale=1.1)

        symbols = set(atoms.get_chemical_symbols())
        expected = {"Na", "Cl", "K"}
        assert symbols == expected

    def test_seed_reproducibility(self, nacl_kcl_recipe):
        """Test that the same seed produces the same structure."""
        from muse.transforms.mixture import mix_number

        atoms1 = mix_number(
            recipe=dict(nacl_kcl_recipe), seed=42, tolerance=2.0, scale=1.1
        )
        atoms2 = mix_number(
            recipe=dict(nacl_kcl_recipe), seed=42, tolerance=2.0, scale=1.1
        )

        assert len(atoms1) == len(atoms2)
        np.testing.assert_allclose(atoms1.cell[:], atoms2.cell[:])


class TestMixImports:
    """Smoke tests for mixture module imports."""

    def test_import_mix_number(self):
        """Verify mix_number can be imported from the public API."""
        from muse import mix_number

        assert callable(mix_number)

    def test_import_mix_cell(self):
        """Verify mix_cell can be imported from the public API."""
        from muse import mix_cell

        assert callable(mix_cell)

    def test_import_from_submodule(self):
        """Verify imports work from the transforms subpackage."""
        from muse.transforms import mix_cell, mix_number

        assert callable(mix_number)
        assert callable(mix_cell)
