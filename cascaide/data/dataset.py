"""Datasets for Cascaide.

This module exposes two ``torch.utils.data.Dataset`` classes:

- :class:`CascadeDataset` returns raw vacancy/SIA coordinates and PKA energy
  for each cascade. It transparently reads from a `.npz` cache produced by
  :mod:`cascaide.data.preprocess` when one is present, and falls back to
  reading LAMMPS dump files via OVITO otherwise. Each item additionally
  carries ``local_centroid`` and ``local_scale`` — the per-cascade centroid
  of (vac ∪ sia) and the maximum distance from it — which downstream encoders
  and conditioners may consume.
- :class:`EncodedCascadeDataset` wraps a raw dataset with one or more
  coordinate encoders and surfaces both the encoded tensors and the source
  point clouds for use by auxiliary losses.

Backwards compatibility: the legacy ``(vac_file, sia_file, energy)`` tuple
structure of ``CascadeDataset.samples`` is preserved, and existing code that
ignores the new dictionary keys continues to work unchanged.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cascaide.encoding.base import CoordinateEncoder
from cascaide.data.preprocess import (
    cache_path_for_dump,
    compute_local_stats,
    load_manifest,
    scan_data_root,
)


class CascadeDataset(Dataset):
    """Lazy reader over the cascade dataset.

    Parameters
    ----------
    data_root : str
        Directory containing ``*keV/`` subdirectories with paired
        ``*_min_vac.dump`` / ``*_min_sia.dump`` files and per-dir ``metadata.json``.
    compute_centroid : bool, default True
        Compute (or load from manifest) the dataset-wide global centroid.
    max_samples : int, optional
        Cap on the number of cascades, applied after scanning.
    use_cache : bool, default True
        If True and a manifest is present under ``data_root/.cache/``, read
        cascades from `.npz` files; otherwise fall back to OVITO dump reading.
    """

    def __init__(
        self,
        data_root: str,
        compute_centroid: bool = True,
        max_samples: Optional[int] = None,
        use_cache: bool = True,
    ):
        self.data_root = data_root
        self.use_cache = use_cache

        manifest = load_manifest(data_root) if use_cache else None
        if manifest is not None:
            self._init_from_manifest(manifest, max_samples)
            self.global_centroid = (
                np.array(manifest["global_centroid"], dtype=np.float64)
                if compute_centroid else None
            )
            print(
                f"[dataset] loaded {len(self.samples)} cached cascades from "
                f"{data_root} (manifest v{manifest['version']})"
            )
        else:
            self._init_from_filesystem(data_root, max_samples)
            self.global_centroid = None
            if compute_centroid:
                self._compute_global_centroid()

        if compute_centroid and self.global_centroid is not None:
            print(f"Global centroid: {self.global_centroid}")

    # ----- init paths --------------------------------------------------------

    def _init_from_manifest(self, manifest: Dict, max_samples: Optional[int]) -> None:
        entries = manifest["entries"]
        if max_samples is not None:
            entries = entries[:max_samples]

        self.samples: List[Tuple[str, str, float]] = []
        self.cache_paths: List[Optional[str]] = []
        self._manifest_entries: List[Dict] = entries

        for e in entries:
            energy_dir = os.path.join(self.data_root, e["energy_dir"])
            cascade_id = int(e["cascade_id"])
            vac_file = os.path.join(energy_dir, f"{cascade_id:04d}_min_vac.dump")
            sia_file = os.path.join(energy_dir, f"{cascade_id:04d}_min_sia.dump")
            self.samples.append((vac_file, sia_file, float(e["energy_eV"])))
            self.cache_paths.append(os.path.join(self.data_root, e["npz_path"]))

    def _init_from_filesystem(self, data_root: str, max_samples: Optional[int]) -> None:
        self.samples = scan_data_root(data_root)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.cache_paths = [
            cache_path_for_dump(vac_file) if os.path.exists(cache_path_for_dump(vac_file))
            else None
            for vac_file, _, _ in self.samples
        ]
        self._manifest_entries = []
        print(
            f"[dataset] no manifest under {data_root}; using filesystem scan "
            f"({len(self.samples)} cascades, "
            f"{sum(p is not None for p in self.cache_paths)} with .npz cache)"
        )

    # ----- coordinate loading ------------------------------------------------

    @lru_cache(maxsize=5000)
    def _load_coordinates(self, filepath: str) -> np.ndarray:
        """Read coordinates from a LAMMPS dump file via OVITO.

        Used only in the legacy (no-cache) code path. The cache version of
        :meth:`__getitem__` bypasses this entirely by reading `.npz` directly.
        """
        try:
            from ovito.io import import_file
            pipeline = import_file(filepath)
            data = pipeline.compute()
            if data.particles.count == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return np.array(data.particles.positions, dtype=np.float32)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.zeros((0, 3), dtype=np.float32)

    def _compute_global_centroid(self) -> None:
        print("Computing global centroid (streaming)...")
        total_sum = np.zeros(3, dtype=np.float64)
        total_count = 0
        for vac_file, sia_file, _ in self.samples:
            vac = self._load_coordinates(vac_file)
            sia = self._load_coordinates(sia_file)
            if len(vac) > 0:
                total_sum += vac.sum(axis=0)
                total_count += len(vac)
            if len(sia) > 0:
                total_sum += sia.sum(axis=0)
                total_count += len(sia)
        self.global_centroid = (
            total_sum / total_count if total_count > 0 else np.zeros(3)
        )

    def compute_required_norm_factor(self) -> float:
        """Return a generous global norm factor covering all defect displacements."""
        if self.global_centroid is None:
            self._compute_global_centroid()
        print("Scanning for max spatial extent...")
        max_dist = 0.0
        for i in range(len(self.samples)):
            vac, sia, _, _, _ = self._read_cascade(i)
            for coords in (vac, sia):
                if len(coords) > 0:
                    d = float(np.linalg.norm(coords - self.global_centroid, axis=1).max())
                    if d > max_dist:
                        max_dist = d
        print(f"Maximum atom distance found: {max_dist:.2f} Å")
        return float(np.ceil(max_dist * 1.1 / 10) * 10)

    # ----- accessors ---------------------------------------------------------

    @property
    def energies(self) -> np.ndarray:
        """All PKA energies as a float32 array (eV)."""
        return np.array([s[2] for s in self.samples], dtype=np.float32)

    @property
    def has_cache(self) -> bool:
        """True iff every cascade has a `.npz` cache file backing it."""
        return bool(self.cache_paths) and all(p is not None for p in self.cache_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_cascade(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
        """Return ``(vac, sia, energy, local_centroid, local_scale)`` for one item.

        Reads from `.npz` when cached; otherwise from dump files via OVITO and
        computes the per-cascade stats on the fly.
        """
        vac_file, sia_file, energy = self.samples[idx]
        cache_path = self.cache_paths[idx] if self.cache_paths else None

        if cache_path is not None and os.path.exists(cache_path):
            with np.load(cache_path) as d:
                vac = np.asarray(d["vac"], dtype=np.float32)
                sia = np.asarray(d["sia"], dtype=np.float32)
                local_centroid = np.asarray(d["local_centroid"], dtype=np.float32)
                local_scale = float(d["local_scale"])
        else:
            vac = self._load_coordinates(vac_file)
            sia = self._load_coordinates(sia_file)
            local_centroid, local_scale = compute_local_stats(vac, sia)

        return vac, sia, float(energy), local_centroid, local_scale

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vac, sia, energy, local_centroid, local_scale = self._read_cascade(idx)
        return {
            "vac_coords": torch.from_numpy(vac),
            "sia_coords": torch.from_numpy(sia),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "local_centroid": torch.from_numpy(local_centroid),
            "local_scale": torch.tensor(local_scale, dtype=torch.float32),
        }


class EncodedCascadeDataset(Dataset):
    """Apply one or more :class:`CoordinateEncoder`s to a :class:`CascadeDataset`."""

    def __init__(
        self,
        raw_dataset: "CascadeDataset",
        encoders: Dict[str, CoordinateEncoder],
        energy_norm_factor: float = 1000.0,
    ):
        self.raw_dataset = raw_dataset
        self.encoders = encoders
        self.energy_norm_factor = energy_norm_factor
        self.global_centroid = raw_dataset.global_centroid

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.raw_dataset[idx]
        vac = sample["vac_coords"]
        sia = sample["sia_coords"]
        energy = sample["energy"]

        out: Dict[str, torch.Tensor] = {}
        for name, enc in self.encoders.items():
            out[name] = enc.encode(vac, sia)

        out["energy"] = energy / self.energy_norm_factor
        out["n_vac"] = len(vac)
        out["n_sia"] = len(sia)
        out["vac_coords"] = vac
        out["sia_coords"] = sia
        # Propagate per-cascade stats so downstream conditioners can consume them.
        if "local_centroid" in sample:
            out["local_centroid"] = sample["local_centroid"]
        if "local_scale" in sample:
            out["local_scale"] = sample["local_scale"]
        return out


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of dataset items into a batched dict.

    Variable-length ``vac_coords`` / ``sia_coords`` are kept as Python lists so
    downstream aux losses can iterate over individual cascades; everything
    else is stacked into a tensor.
    """
    out: Dict = {}
    for k in ("vac_coords", "sia_coords"):
        if k in batch[0]:
            out[k] = [item[k] for item in batch]

    for k in batch[0]:
        if k in ("vac_coords", "sia_coords"):
            continue
        v = batch[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([item[k] for item in batch])
        else:
            out[k] = torch.tensor([item[k] for item in batch])
    return out
