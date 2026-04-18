# Muse

**Mixture builder for simulation environments**

Build amorphous solids and liquid mixtures from relaxed solid-state structures on [Materials Project](https://materialsproject.org/) using [Packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml) and machine learning interatomic potentials.

---

## Features

- 🧱 **Structure generation** — Build binary/multicomponent amorphous mixtures from Materials Project crystal structures
- ⚖️ **Density equilibration** — Run NVT → NPT molecular dynamics to compute equilibrium densities
- 📊 **Thermodynamic analysis** — Plot binary mixing diagrams with Redlich–Kister fits
- 📁 **Trajectory I/O** — Convert pymatgen trajectories to extended XYZ format
- 🖥️ **HPC integration** — Submit SLURM batch jobs programmatically

## Installation

=== "pip"

    ```bash
    pip install muse-xtal
    ```

=== "uv"

    ```bash
    uv pip install muse-xtal
    ```

=== "From source"

    ```bash
    git clone https://github.com/chiang-yuan/muse.git
    cd muse
    pip install -e ".[dev]"
    ```

### Optional extras

```bash
# MACE calculator support
pip install "muse-xtal[mace]"

# Development tools
pip install "muse-xtal[dev]"
```

### Prerequisites

- [Packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml) must be installed and on your `PATH`
- A [Materials Project API key](https://materialsproject.org/api) set as the `MP_API_KEY` environment variable

## Quick example

```python
from muse.transforms.mixture import mix_number

atoms = mix_number(
    recipe={"NaCl": 3, "KCl": 1},
    tolerance=2.0,
    scale=1.05,
    seed=42,
)
print(atoms)
```

## Citation

If you use Muse in your research, please cite:

```bibtex
@software{chiang2023muse,
    title  = {muse},
    author = {Yuan Chiang},
    year   = {2023},
    url    = {https://github.com/chiang-yuan/muse}
}
```
