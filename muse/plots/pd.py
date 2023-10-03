from typing import Sequence

import numpy as np
from ase import Atoms, units
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

eps = 1e-10

class BinaryGXDiagram(Axes):
    """Binary G-x diagram plotter"""

    def __init__(
        self,
        fig: Figure,
        rect: Sequence[float],
        *,
        facecolor = None,
        frameon: bool = True,
        sharex: Axes | None = None,
        sharey: Axes | None = None,
        label: str = "",
        xscale: float | None = None,
        yscale: float | None = None,
        box_aspect: float | None = None,
        **kwargs
    ) -> None:
        super().__init__(
            fig,
            rect,
            facecolor=facecolor,
            frameon=frameon,
            sharex=sharex,
            sharey=sharey,
            label=label,
            xscale=xscale,
            yscale=yscale,
            box_aspect=box_aspect,
            **kwargs
        )

    def from_trajectories(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
        temperature: float | None = None,
        label: str | None = None,
        **kwargs
    ) -> None:
        """Plot a binary phase diagram from a list of trajectories"""

        assert len(phases) == 2

        # Change strings to Formula objects and sort symbols
        for phase in phases:
            phase = Formula.from_list(phase) if isinstance(phase, str) else sort(phase)
            

        x, ergavgs, ergstds = [], [], []
        for traj in trajectories:

            atoms = traj[0]
            formula = atoms.symbols.formula

            portions = {}
            for phase in phases:
                portions[phase] = formula//phase

            total_units = sum(portions.values())

            fractions = {}
            for phase in phases:
                fractions[phase] = portions[phase] / total_units

            x.append(fractions[phases[-1]])

            energies = []
            for atoms in traj:
                energies.append(atoms.get_potential_energy())
            ergavgs.append(np.mean(energies) / total_units)
            ergstds.append(np.std(energies) / total_units)


        x = np.array(x)
        idx = np.argsort(x)
        x = x[idx]

        ergavgs = np.array(ergavgs)[idx]
        ergstds = np.array(ergstds)[idx]

        dH = ergavgs - (ergavgs[0] + x * (ergavgs[-1] - ergavgs[0]))
        dS = - units.kB * (x*np.log(x + eps) + (1-x)*np.log(1-x + eps))

        self.plot(x, dH, label='$\\Delta H$ ' + label if label else '$\\Delta H$', **kwargs)

        if temperature is not None:
            dG = dH - temperature * dS
            self.plot(x, - temperature * dS, label='$-T\\Delta S$ ' + label if label else '$-T\\Delta S$', **kwargs)
            self.plot(x, dG, label='$\\Delta G$ ' + label if label else '$\\Delta G$', **kwargs)
        
            
        