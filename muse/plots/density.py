from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

__author__ = "Yuan Chiang"
__date__ = "2023-11-06"

eps = 1e-10


class BinaryDXDiagram(Axes):
    """Binary density-composition diagram plotter."""

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

    def from_trajectories(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
        temperature: float | None = None,
        label: str | None = None,
        **kwargs,
    ) -> None:
        """Plot a binary phase diagram from a list of trajectories."""
        assert len(phases) == 2

        # Change strings to Formula objects and sort symbols
        for phase in phases:
            phase = Formula.from_list(phase) if isinstance(phase, str) else sort(phase)

        x, denavgs, denstds = [], [], []
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
            for atoms in traj:
                densities.append(
                    atoms.get_masses().sum()
                    * 1.66054e-24
                    / (atoms.get_volume() * 1e-24)
                )
            denavgs.append(np.mean(densities))
            denstds.append(np.std(densities))

        x = np.array(x)
        idx = np.argsort(x)
        x = x[idx]

        denavgs = np.array(denavgs)[idx]
        denstds = np.array(denstds)[idx]

        self.errorbar(
            x,
            denavgs,
            yerr=denstds,
            label="$\\rho_m$ " + label if label else "$\\rho_m$",
            **kwargs,
        )
