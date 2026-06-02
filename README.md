# Cascaide
Cascaide: Cascaded Generative Modeling with Diffusion.

## What's in here

The pipeline has four pluggable pieces, each independently swappable:

- **Encoders** — turn raw 3D coordinates into 2D images. Three implementations are included: a baseline raster-packed encoder and two Hilbert-curve-sorted variants (3-channel and 4-channel) that preserve 3D spatial locality in the 2D layout.
- **Diffusion** — Gaussian DDPM with pluggable noise schedules (linear, cosine, sigmoid) and parameterizations (eps, x0, v).
- **UNet** — channel-and-resolution-adaptive 2D UNet with timestep + energy conditioning.
- **Aux losses** — composable auxiliary losses on top of the standard diffusion MSE: occupancy, count, classification, and multi-axis projection.

Everything is configured through YAML and trained with a single command.

## Installation

Cascaide is a regular Python package. Clone, then install in editable mode:

```bash
git clone https://github.com/vigsam-coder/Cascaide
cd Cascaide
pip install -e .
```

Requires Python 3.10+. The dependency on `ovito` is for reading LAMMPS-style `.dump` files; if you only need the diffusion components and have your own data loader, it can be skipped.

## Data layout

The dataset loader expects this directory structure:

- data_root/
  - 0-10keV/
    - metadata.json
    - 0001_min_vac.dump
    - 0001_min_sia.dump
    - 0002_min_vac.dump
    - ...
  - 10-30keV/
    - ...
  - 100keV/
    - ...
## Quick start

### 1. Preprocess (optional but recommended)

Convert the LAMMPS dump files into a compact `.npz` cache so that training
does not depend on OVITO and does not re-parse dump files every epoch:

```bash
python -m cascaide.data.preprocess --data_root <path/to/data_root>
```

This writes a `{cascade_id}.npz` next to each `*_min_vac.dump` /
`*_min_sia.dump` pair (containing the vacancy/SIA coordinates, energy, and
per-cascade local centroid + scale) and a top-level
`<data_root>/.cache/manifest.json` carrying the dataset-wide global centroid
and a summary entry for every cascade. The pass is idempotent — re-running
reuses existing `.npz` files; pass `--force` to rebuild them. `CascadeDataset`
detects the manifest automatically and skips OVITO entirely when it is present.

### 2. Train

```bash
python cascaide/training/train.py --config cascaide/configs/config.yaml

```

### 3. Inference

```bash
python infer.py --config cascaide/configs/config.yaml --checkpoint runs/hilbert4ch_v1/checkpoints/best.pt --output_dir runs/hilbert4ch_v1/figures_best

```
