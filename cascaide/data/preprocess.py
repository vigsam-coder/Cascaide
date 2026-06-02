"""
Preprocess Cascaide cascade dump files into a `.npz` cache.

Each cascade pair ``{cascade_id}_min_vac.dump`` / ``{cascade_id}_min_sia.dump``
is converted into a compact ``{cascade_id}.npz`` next to the originals,
containing:

    vac             (N_v, 3) float32 — vacancy coordinates (Å)
    sia             (N_s, 3) float32 — self-interstitial coordinates (Å)
    energy          ()      float32 — PKA energy (eV)
    local_centroid  (3,)    float32 — mean of (vac ∪ sia)
    local_scale     ()      float32 — max distance from local_centroid

A top-level manifest is written to ``{data_root}/.cache/manifest.json`` and
includes the dataset-wide global centroid, so neither OVITO nor a streaming
pass over the dump files is required at training time.

CLI:

    python -m cascaide.data.preprocess --data_root <path> [--force] [--max_samples N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


CACHE_DIRNAME = ".cache"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


def manifest_path(data_root: str) -> str:
    """Return the canonical manifest path for a dataset root."""
    return os.path.join(data_root, CACHE_DIRNAME, MANIFEST_FILENAME)


def cache_path_for_dump(vac_file: str) -> str:
    """Return the `.npz` cache path that pairs with a ``*_min_vac.dump`` file."""
    energy_dir = os.path.dirname(vac_file)
    base = os.path.basename(vac_file)
    cascade_prefix = base.split("_")[0]
    return os.path.join(energy_dir, f"{cascade_prefix}.npz")


def scan_data_root(data_root: str) -> List[Tuple[str, str, float]]:
    """Walk ``data_root`` for paired vacancy/SIA dump files plus per-dir metadata.

    Returns a list of ``(vac_file, sia_file, energy_eV)`` tuples — the same
    structure historically produced by ``CascadeDataset._scan_data_root``.
    """
    samples: List[Tuple[str, str, float]] = []
    for energy_dir in sorted(glob.glob(os.path.join(data_root, "*keV"))):
        meta_file = os.path.join(energy_dir, "metadata.json")
        if not os.path.exists(meta_file):
            continue
        with open(meta_file, "r") as f:
            meta = json.load(f)
        energy_map = {c["cascade_id"]: c["pka_energy_eV"] for c in meta["cascades"]}

        for vac_file in sorted(glob.glob(os.path.join(energy_dir, "*_min_vac.dump"))):
            base = os.path.basename(vac_file)
            try:
                cascade_id = int(base.split("_")[0])
            except ValueError:
                continue
            sia_file = vac_file.replace("_min_vac.dump", "_min_sia.dump")
            if os.path.exists(sia_file) and cascade_id in energy_map:
                samples.append((vac_file, sia_file, float(energy_map[cascade_id])))
    return samples


def _load_coordinates_from_dump(filepath: str) -> np.ndarray:
    """Load atom positions from a LAMMPS-style dump file via OVITO.

    Returns ``(N, 3) float32``. Errors are caught and reported, returning an
    empty array — matching the historical behavior of the dataset loader.
    """
    try:
        from ovito.io import import_file
        pipeline = import_file(filepath)
        data = pipeline.compute()
        if data.particles.count == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(data.particles.positions, dtype=np.float32)
    except Exception as e:  # pragma: no cover — OVITO failure surface is wide
        print(f"[preprocess] error loading {filepath}: {e}")
        return np.zeros((0, 3), dtype=np.float32)


def compute_local_stats(vac: np.ndarray, sia: np.ndarray) -> Tuple[np.ndarray, float]:
    """Per-cascade centroid and scale.

    The centroid is the mean over (vac ∪ sia); the scale is the maximum
    Euclidean distance from any defect to that centroid. Empty cascades
    return ``(zeros(3), 0.0)``.
    """
    parts = []
    if len(vac) > 0:
        parts.append(np.asarray(vac, dtype=np.float64))
    if len(sia) > 0:
        parts.append(np.asarray(sia, dtype=np.float64))
    if not parts:
        return np.zeros(3, dtype=np.float32), 0.0
    pts = np.concatenate(parts, axis=0)
    centroid = pts.mean(axis=0)
    scale = float(np.linalg.norm(pts - centroid, axis=1).max())
    return centroid.astype(np.float32), scale


def write_cascade_npz(
    out_path: str,
    vac: np.ndarray,
    sia: np.ndarray,
    energy: float,
) -> Dict[str, Any]:
    """Write one cascade `.npz` and return its per-cascade stats dict."""
    centroid, scale = compute_local_stats(vac, sia)
    np.savez(
        out_path,
        vac=np.asarray(vac, dtype=np.float32),
        sia=np.asarray(sia, dtype=np.float32),
        energy=np.float32(energy),
        local_centroid=centroid.astype(np.float32),
        local_scale=np.float32(scale),
    )
    return {
        "n_vac": int(len(vac)),
        "n_sia": int(len(sia)),
        "local_centroid": centroid.tolist(),
        "local_scale": scale,
    }


def preprocess_dataset(
    data_root: str,
    *,
    force: bool = False,
    max_samples: Optional[int] = None,
    loader: Callable[[str], np.ndarray] = _load_coordinates_from_dump,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Convert every paired cascade in ``data_root`` to `.npz` and write a manifest.

    Idempotent: existing `.npz` files are reused unless ``force=True``. The
    function always (re)writes the manifest reflecting the on-disk cache.

    The ``loader`` indirection makes the function unit-testable without an
    OVITO dependency; in production it defaults to the dump-file reader.
    """
    samples = scan_data_root(data_root)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(f"No cascade pairs found under {data_root}")

    cache_dir = os.path.join(data_root, CACHE_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    total_sum = np.zeros(3, dtype=np.float64)
    total_count = 0
    n_written = 0
    n_reused = 0

    for i, (vac_file, sia_file, energy) in enumerate(samples):
        npz_path = cache_path_for_dump(vac_file)

        if not force and os.path.exists(npz_path):
            with np.load(npz_path) as d:
                vac = d["vac"]
                sia = d["sia"]
                stats = {
                    "n_vac": int(vac.shape[0]),
                    "n_sia": int(sia.shape[0]),
                    "local_centroid": d["local_centroid"].tolist(),
                    "local_scale": float(d["local_scale"]),
                }
            n_reused += 1
        else:
            vac = loader(vac_file)
            sia = loader(sia_file)
            stats = write_cascade_npz(npz_path, vac, sia, energy)
            n_written += 1

        if len(vac) > 0:
            total_sum += np.asarray(vac, dtype=np.float64).sum(axis=0)
            total_count += len(vac)
        if len(sia) > 0:
            total_sum += np.asarray(sia, dtype=np.float64).sum(axis=0)
            total_count += len(sia)

        base = os.path.basename(vac_file)
        cascade_id = int(base.split("_")[0])
        entries.append({
            "cascade_id": cascade_id,
            "energy_dir": os.path.relpath(os.path.dirname(vac_file), data_root),
            "energy_eV": float(energy),
            "npz_path": os.path.relpath(npz_path, data_root),
            **stats,
        })

        if verbose and (i + 1) % 500 == 0:
            print(f"[preprocess] {i + 1}/{len(samples)} cascades processed")

    global_centroid = (
        (total_sum / total_count).tolist() if total_count > 0 else [0.0, 0.0, 0.0]
    )

    manifest = {
        "version": MANIFEST_VERSION,
        "data_root": os.path.abspath(data_root),
        "n_samples": len(entries),
        "global_centroid": global_centroid,
        "total_atoms": int(total_count),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }

    out = manifest_path(data_root)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(
            f"[preprocess] wrote manifest: {out}\n"
            f"[preprocess]   n_samples = {len(entries)}  "
            f"(new: {n_written}, reused: {n_reused})\n"
            f"[preprocess]   global_centroid = {global_centroid}"
        )

    return manifest


def load_manifest(data_root: str) -> Optional[Dict[str, Any]]:
    """Load a manifest if present and schema-compatible; otherwise return None."""
    path = manifest_path(data_root)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        manifest = json.load(f)
    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise RuntimeError(
            f"Manifest at {path} has version {version}; expected {MANIFEST_VERSION}. "
            f"Re-run `python -m cascaide.data.preprocess --data_root {data_root} --force`."
        )
    return manifest


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to the dataset root containing *keV/ subdirectories.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing .npz files instead of reusing them.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional cap on the number of cascades to preprocess.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    preprocess_dataset(
        data_root=args.data_root,
        force=args.force,
        max_samples=args.max_samples,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
