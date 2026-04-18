"""Tests for muse.plots shared utilities."""

from __future__ import annotations

import numpy as np
import pytest

from muse.plots._utils import EPS, redlich_kister_model


class TestRedlichKisterModel:
    """Tests for the Redlich-Kister expansion model."""

    def test_pure_component_zero(self):
        """Excess property should be zero at pure component endpoints."""
        params = [1.0, 0.01, -0.5, 0.005]
        assert redlich_kister_model(0.0, 1000, *params) == pytest.approx(0.0)
        assert redlich_kister_model(1.0, 1000, *params) == pytest.approx(0.0)

    def test_symmetric_single_term(self):
        """With one symmetric term (n=1), result should be symmetric around x=0.5."""
        params = [2.0, 0.0]  # A1=2, B1=0 → temperature independent
        result_03 = redlich_kister_model(0.3, 1000, *params)
        result_07 = redlich_kister_model(0.7, 1000, *params)
        assert result_03 == pytest.approx(result_07)

    def test_maximum_at_half(self):
        """Single symmetric term should peak at x=0.5."""
        params = [4.0, 0.0]
        result_half = redlich_kister_model(0.5, 1000, *params)
        result_quarter = redlich_kister_model(0.25, 1000, *params)
        assert result_half > result_quarter

    def test_temperature_dependence(self):
        """Result should change with temperature when B_n != 0."""
        params = [1.0, 0.01]  # B_n = 0.01
        result_500 = redlich_kister_model(0.5, 500, *params)
        result_1500 = redlich_kister_model(0.5, 1500, *params)
        assert result_500 != pytest.approx(result_1500)

    def test_vectorized(self):
        """Should work with numpy arrays."""
        x = np.linspace(0, 1, 100)
        params = [1.0, 0.0]
        result = redlich_kister_model(x, 1000, *params)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)

    def test_eps_is_small(self):
        """EPS constant should be a small positive number."""
        assert 0 < EPS < 1e-6
