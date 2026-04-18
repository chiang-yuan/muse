# Quick Start

## Building a Mixture

The simplest way to build a mixture is with `mix_number`, which takes a recipe
of chemical formulas and formula unit counts:

```python
from muse.transforms.mixture import mix_number

# Build a NaCl–KCl mixture (3:1 ratio)
atoms = mix_number(
    recipe={"NaCl": 3, "KCl": 1},
    tolerance=2.0,
    scale=1.05,
    seed=42,
)
print(atoms)
print(f"Cell volume: {atoms.get_volume():.1f} ų")
```

The function queries Materials Project for the ground-state crystal structures,
then uses Packmol to pack the desired number of formula units into a cubic
simulation cell.

## Key Parameters

| Parameter   | Description                                      | Default |
| ----------- | ------------------------------------------------ | ------- |
| `recipe`    | Dict of formula → count                          | —       |
| `tolerance` | Min distance between packed molecules (Å)        | 2.0     |
| `scale`     | Multiplicative factor for estimated cell edge    | 1.0     |
| `rattle`    | Gaussian noise σ added to positions (Å)          | 0.5     |
| `seed`      | Random seed for reproducibility                  | 1       |
| `density`   | Target mass density (amu/ų) for cell rescaling   | None    |

## Density Equilibration

After building an initial structure, use `DensityCalc` to equilibrate the
density through NVT → NPT molecular dynamics:

```python
import numpy as np
from ase import units
from muse.calcs.density import DensityCalc

# Attach your ML calculator (e.g., MACE, CHGNet)
# calculator = MACECalculator(model_paths="...", device="cpu")

density_calc = DensityCalc(
    calculator=calculator,
    optimizer="FIRE",
    steps=500,
    mask=np.eye(3),
    rtol=1e-3,
    atol=5e-4,
)

results = density_calc.calc(
    atoms=atoms,
    temperature=1100,       # K
    externalstress=0.0,     # eV/ų
)

print(f"Density: {results['mass_density']:.4f} amu/ų")
print(f"Volume:  {results['volume_avg']:.1f} ų")
```

## Plotting Thermodynamic Diagrams

Muse includes custom Matplotlib Axes for binary thermodynamic diagrams:

- **`BinaryDXDiagram`** — density–composition with Redlich–Kister fit
- **`BinaryGXDiagram`** — Gibbs energy–composition (mixing enthalpy)
- **`MixingVolumeDiagram`** — excess mixing volume
