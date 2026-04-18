"""Shared plotting utilities for thermodynamic diagram construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

#: Small epsilon to avoid log(0) in entropy calculations.
EPS = 1e-10


def redlich_kister_model(
    x: float | np.ndarray,
    T: float,
    *params: float,
) -> float | np.ndarray:
    """Redlich–Kister polynomial expansion for excess thermodynamic properties.

    Computes the excess property as:

        Y_ex = x(1-x) * Σ_n L_n * (2x - 1)^(n-1)

    where L_n = A_n + B_n * T allows linear temperature dependence.

    Args:
        x: Mole fraction of one component (0 to 1).
        T: Temperature in Kelvin.
        *params: Pairs of (A_n, B_n) coefficients for each Redlich–Kister term.
            The number of terms N = len(params) // 2.

    Returns:
        The excess property value at the given composition and temperature.
    """
    N = len(params) // 2
    rho_ex: float | np.ndarray = 0
    for n in range(1, N + 1):
        A_n = params[2 * (n - 1)]
        B_n = params[2 * (n - 1) + 1]
        L_n = A_n + B_n * T
        rho_ex += L_n * (2 * x - 1) ** (n - 1)
    return x * (1 - x) * rho_ex
