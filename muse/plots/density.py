"""Binary density–composition diagram with Redlich–Kister curve fitting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from scipy.optimize import curve_fit

from muse.plots._utils import redlich_kister_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from matplotlib.figure import Figure

__author__ = "Yuan Chiang"
__date__ = "2023-11-06"

logger = logging.getLogger(__name__)


class BinaryDXDiagram(Axes):
    """Custom Matplotlib Axes for binary density–composition (D–x) diagrams.

    Processes MD trajectories at various compositions to compute density
    and molar volume, then plots the results with an optional
    Redlich–Kister polynomial fit for the excess property.
    """

    def __init__(
        self,
        fig: Figure,
        *args: Any,
        facecolor=None,
        frameon: bool = True,
        sharex: Axes | None = None,
        sharey: Axes | None = None,
        label: str = "",
        xscale: float | None = None,
        yscale: float | None = None,
        box_aspect: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            fig,
            args,
            facecolor=facecolor,
            frameon=frameon,
            sharex=sharex,
            sharey=sharey,
            label=label,
            xscale=xscale,
            yscale=yscale,
            box_aspect=box_aspect,
            **kwargs,
        )

    def process(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
    ) -> None:
        """Process MD trajectories to extract density and volume statistics.

        Computes mass density (g/cm³) and molar volume (ų/formula unit)
        for each trajectory, storing the results sorted by composition.

        Args:
            trajectories: List of MD trajectories, each a sequence of Atoms.
            phases: Two-element list of phase formulas defining the binary system.
        """
        # Change strings to Formula objects and sort symbols
        for phase in phases:
            phase = Formula.from_list(phase) if isinstance(phase, str) else sort(phase)

        x, denavgs, denstds, volavgs, volstds = [], [], [], [], []
        for traj in trajectories:
            atoms = traj[0]
            formula = atoms.symbols.formula

            portions = {}
            for phase in phases:
                portions[phase] = formula // phase

            total_units = sum(portions.values())

            fractions = {}
            for phase in phases:
                fractions[phase] = portions[phase] / total_units

            x.append(fractions[phases[-1]])

            densities = []
            volumes = []
            for atoms in traj:
                densities.append(
                    atoms.get_masses().sum()
                    * 1.66054e-24
                    / (atoms.get_volume() * 1e-24)
                )
                volumes.append(atoms.get_volume() / total_units)

            denavgs.append(np.mean(densities))
            denstds.append(np.std(densities))
            volavgs.append(np.mean(volumes))
            volstds.append(np.std(volumes))

        x = np.array(x)
        idx = np.argsort(x)
        self.x = x[idx]

        self.y = {} if getattr(self, "y", None) is None else self.y
        self.y["density.avg"] = np.array(denavgs)[idx]
        self.y["density.std"] = np.array(denstds)[idx]
        self.y["volume.avg"] = np.array(volavgs)[idx]
        self.y["volume.std"] = np.array(volstds)[idx]

    def from_trajectories(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
        temperature: float | None = None,
        label: str | None = None,
        rk: int = 2,
        **kwargs,
    ) -> None:
        """Plot a binary density–composition diagram from MD trajectories.

        Computes densities from the trajectories, plots them with error bars,
        and overlays a Redlich–Kister polynomial fit.

        Args:
            trajectories: List of MD trajectories at different compositions.
            phases: Two-element list of phase formulas defining the binary system.
            temperature: Temperature in Kelvin for the Redlich–Kister fit. Defaults to 1000 K.
            label: Label prefix for the legend entries.
            rk: Number of Redlich–Kister terms to use in the fit. Defaults to 2.
            **kwargs: Additional keyword arguments passed to ``errorbar``.
        """
        assert len(phases) == 2

        self.process(trajectories, phases)

        color = kwargs.pop("color", "k")

        self.errorbar(
            self.x,
            self.y["density.avg"],
            yerr=self.y["density.std"],
            label=f"{label}: $\\rho_m$" if label else "$\\rho_m$",
            color=color,
            fmt="o",
            **kwargs,
        )

        T = temperature or 1000

        # Fitting the Redlich-Kister expansion model
        initial_guess = [0.0] * (2 * rk)

        y = self.y["density.avg"]

        params_opt, params_cov = curve_fit(
            lambda x_T, *params: redlich_kister_model(x_T[0], x_T[1], *params),
            (self.x, np.ones_like(self.x) * T),
            y - (y[0] + self.x * (y[-1] - y[0])),
            p0=initial_guess,
        )

        fitted_params = params_opt.reshape(-1, 2)
        logger.info("Fitted Redlich-Kister parameters (A_n, B_n): %s", fitted_params)

        xs = np.linspace(self.x.min(), self.x.max(), int(1e3))
        ys = redlich_kister_model(xs, T, *params_opt)
        ys += y[0] + xs * (y[-1] - y[0])

        self.plot(
            xs,
            ys,
            label=f"{label}: Redlich-Kister Fit" if label else "Redlich-Kister Fit",
            linestyle="--",
            lw=kwargs.get("lw", 1),
            color=color,
        )

    def plot_volume(
        self,
        label: str | None = None,
        **kwargs,
    ):
        """Plot molar volume on a secondary y-axis.

        Must be called after ``from_trajectories`` or ``process`` so that
        ``self.x`` and ``self.y`` are populated.

        Args:
            label: Legend label for the volume curve.
            **kwargs: Additional keyword arguments passed to ``errorbar``.

        Returns:
            Axes: The secondary y-axis Axes.
        """
        self.vol_ax = self.twinx()
        self.vol_ax.errorbar(
            self.x,
            self.y["volume.avg"],
            yerr=self.y["volume.std"],
            label="$\\bar{V}$" if label else "$\\bar{V}$",
            **kwargs,
        )
        return self.vol_ax