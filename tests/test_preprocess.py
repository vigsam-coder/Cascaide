"""Tests for ``cascaide.data.preprocess`` and the cached read path of
:class:`cascaide.data.dataset.CascadeDataset`.

The tests avoid the OVITO dependency by injecting a stub ``loader`` callable
that returns synthetic coordinates keyed by file path.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
import torch

from cascaide.data import preprocess as pp
from cascaide.data.dataset import CascadeDataset


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

Cascade = Tuple[str, int, float, np.ndarray, np.ndarray]


def _make_fake_dataset(
    tmp_path,
    cascades: List[Cascade],
) -> Tuple[str, Callable[[str], np.ndarray], Dict[str, np.ndarray]]:
    """Create a directory tree mimicking the production layout.

    Returns ``(data_root, loader, coords_map)``. ``loader`` is suitable to
    pass to :func:`preprocess_dataset` and bypasses OVITO entirely.
    """
    data_root = tmp_path / "data"
    coords_map: Dict[str, np.ndarray] = {}

    by_dir: Dict[str, List[Cascade]] = {}
    for c in cascades:
        by_dir.setdefault(c[0], []).append(c)

    for energy_dir_name, items in by_dir.items():
        d = data_root / energy_dir_name
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "cascades": [
                {"cascade_id": cid, "pka_energy_eV": e}
                for _, cid, e, _, _ in items
            ]
        }
        (d / "metadata.json").write_text(json.dumps(meta))
        for _, cid, _, vac, sia in items:
            vac_path = d / f"{cid:04d}_min_vac.dump"
            sia_path = d / f"{cid:04d}_min_sia.dump"
            vac_path.write_text("")
            sia_path.write_text("")
            coords_map[str(vac_path)] = np.asarray(vac, dtype=np.float32)
            coords_map[str(sia_path)] = np.asarray(sia, dtype=np.float32)

    def loader(path: str) -> np.ndarray:
        return coords_map[path]

    return str(data_root), loader, coords_map


def _default_cascades() -> List[Cascade]:
    return [
        ("0-10keV", 1, 5000.0,
         np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
         np.array([[0.0, 1.0, 0.0]], dtype=np.float32)),
        ("0-10keV", 2, 8000.0,
         np.array([[10.0, 10.0, 10.0]], dtype=np.float32),
         np.array([[11.0, 10.0, 10.0], [10.0, 11.0, 10.0]], dtype=np.float32)),
        ("10-30keV", 5, 20000.0,
         np.array([[-5.0, -5.0, 0.0], [-5.0, 5.0, 0.0]], dtype=np.float32),
         np.array([[5.0, 0.0, 0.0]], dtype=np.float32)),
    ]


# --------------------------------------------------------------------------- #
# Pure-function tests
# --------------------------------------------------------------------------- #

def test_compute_local_stats_basic():
    vac = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    sia = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    centroid, scale = pp.compute_local_stats(vac, sia)

    np.testing.assert_allclose(centroid, [1.0, 1.0 / 3.0, 0.0], rtol=1e-5)
    # Furthest point is one of the vacs at distance sqrt(1 + 1/9).
    expected_scale = float(np.linalg.norm(np.array([0.0, 0.0, 0.0]) - centroid))
    assert scale == pytest.approx(expected_scale, rel=1e-5)


def test_compute_local_stats_empty():
    centroid, scale = pp.compute_local_stats(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(centroid, [0.0, 0.0, 0.0])
    assert scale == 0.0


def test_scan_data_root_finds_pairs(tmp_path):
    data_root, _, _ = _make_fake_dataset(tmp_path, _default_cascades())
    samples = pp.scan_data_root(data_root)
    energies = sorted(e for _, _, e in samples)
    assert energies == [5000.0, 8000.0, 20000.0]
    for vac_file, sia_file, _ in samples:
        assert vac_file.endswith("_min_vac.dump")
        assert sia_file.endswith("_min_sia.dump")
        assert os.path.exists(vac_file)
        assert os.path.exists(sia_file)


# --------------------------------------------------------------------------- #
# preprocess_dataset end-to-end
# --------------------------------------------------------------------------- #

def test_preprocess_writes_npz_and_manifest(tmp_path):
    cascades = _default_cascades()
    data_root, loader, coords_map = _make_fake_dataset(tmp_path, cascades)

    manifest = pp.preprocess_dataset(data_root, loader=loader, verbose=False)

    # Manifest shape.
    assert manifest["version"] == pp.MANIFEST_VERSION
    assert manifest["n_samples"] == len(cascades)
    assert len(manifest["entries"]) == len(cascades)
    assert os.path.exists(pp.manifest_path(data_root))

    # Global centroid = mean over all defect atoms across the dataset.
    all_atoms = np.concatenate(
        [v for v in coords_map.values() if len(v) > 0], axis=0
    )
    expected_global = all_atoms.mean(axis=0).tolist()
    np.testing.assert_allclose(manifest["global_centroid"], expected_global, rtol=1e-5)

    # Each cascade has a .npz next to its dump files with the right contents.
    for entry in manifest["entries"]:
        npz_path = os.path.join(data_root, entry["npz_path"])
        assert os.path.exists(npz_path)
        with np.load(npz_path) as d:
            assert d["vac"].dtype == np.float32
            assert d["sia"].dtype == np.float32
            assert d["vac"].shape[1] == 3
            assert d["sia"].shape[1] == 3
            # local_centroid matches what compute_local_stats produces from vac+sia.
            exp_c, exp_s = pp.compute_local_stats(d["vac"], d["sia"])
            np.testing.assert_allclose(d["local_centroid"], exp_c, rtol=1e-5)
            assert float(d["local_scale"]) == pytest.approx(exp_s, rel=1e-5)


def test_preprocess_idempotent_reuses_existing(tmp_path):
    cascades = _default_cascades()
    data_root, loader, _ = _make_fake_dataset(tmp_path, cascades)

    pp.preprocess_dataset(data_root, loader=loader, verbose=False)

    # Mark every .npz as untouchable; a second call must not rewrite them.
    npz_files = []
    for energy_dir in os.listdir(data_root):
        full = os.path.join(data_root, energy_dir)
        if not os.path.isdir(full):
            continue
        npz_files.extend(
            os.path.join(full, f) for f in os.listdir(full) if f.endswith(".npz")
        )
    assert npz_files, "expected .npz files to exist after first pass"
    mtimes_before = {p: os.path.getmtime(p) for p in npz_files}

    def fail_loader(path: str) -> np.ndarray:  # pragma: no cover — must not run
        raise AssertionError(f"loader should not be called for {path}")

    pp.preprocess_dataset(data_root, loader=fail_loader, verbose=False)
    mtimes_after = {p: os.path.getmtime(p) for p in npz_files}
    assert mtimes_before == mtimes_after


def test_preprocess_force_overwrites(tmp_path):
    data_root, loader, _ = _make_fake_dataset(tmp_path, _default_cascades())
    pp.preprocess_dataset(data_root, loader=loader, verbose=False)

    one_npz = os.path.join(data_root, "0-10keV", "0001.npz")
    # Corrupt the cache so we can detect a rewrite.
    np.savez(one_npz, vac=np.zeros((0, 3), dtype=np.float32),
             sia=np.zeros((0, 3), dtype=np.float32),
             energy=np.float32(0.0),
             local_centroid=np.zeros(3, dtype=np.float32),
             local_scale=np.float32(0.0))

    pp.preprocess_dataset(data_root, loader=loader, force=True, verbose=False)

    with np.load(one_npz) as d:
        # After force-rewrite, content matches the source cascade again.
        assert d["vac"].shape[0] == 2
        assert d["sia"].shape[0] == 1
        assert float(d["energy"]) == 5000.0


def test_preprocess_respects_max_samples(tmp_path):
    data_root, loader, _ = _make_fake_dataset(tmp_path, _default_cascades())
    manifest = pp.preprocess_dataset(
        data_root, loader=loader, max_samples=2, verbose=False
    )
    assert manifest["n_samples"] == 2
    assert len(manifest["entries"]) == 2


def test_load_manifest_returns_none_when_absent(tmp_path):
    data_root, _, _ = _make_fake_dataset(tmp_path, _default_cascades())
    assert pp.load_manifest(data_root) is None


def test_load_manifest_rejects_unknown_version(tmp_path):
    data_root, loader, _ = _make_fake_dataset(tmp_path, _default_cascades())
    pp.preprocess_dataset(data_root, loader=loader, verbose=False)
    path = pp.manifest_path(data_root)
    with open(path) as f:
        manifest = json.load(f)
    manifest["version"] = 999
    with open(path, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(RuntimeError, match="version"):
        pp.load_manifest(data_root)


# --------------------------------------------------------------------------- #
# CascadeDataset cache integration
# --------------------------------------------------------------------------- #

def test_dataset_uses_manifest_when_present(tmp_path):
    cascades = _default_cascades()
    data_root, loader, _ = _make_fake_dataset(tmp_path, cascades)
    manifest = pp.preprocess_dataset(data_root, loader=loader, verbose=False)

    ds = CascadeDataset(data_root, compute_centroid=True, use_cache=True)
    assert len(ds) == len(cascades)
    assert ds.has_cache
    np.testing.assert_allclose(
        ds.global_centroid, manifest["global_centroid"], rtol=1e-5
    )

    # Each sample carries point clouds, energy, and per-cascade stats.
    item = ds[0]
    assert isinstance(item["vac_coords"], torch.Tensor)
    assert isinstance(item["sia_coords"], torch.Tensor)
    assert item["vac_coords"].dtype == torch.float32
    assert item["sia_coords"].shape[1] == 3
    assert isinstance(item["energy"].item(), float)
    assert item["local_centroid"].shape == (3,)
    assert item["local_scale"].ndim == 0  # scalar

    # local_centroid matches the cascade content.
    exp_c, exp_s = pp.compute_local_stats(
        item["vac_coords"].numpy(), item["sia_coords"].numpy()
    )
    np.testing.assert_allclose(item["local_centroid"].numpy(), exp_c, rtol=1e-5)
    assert item["local_scale"].item() == pytest.approx(exp_s, rel=1e-5)


def test_dataset_energies_property_matches_samples(tmp_path):
    cascades = _default_cascades()
    data_root, loader, _ = _make_fake_dataset(tmp_path, cascades)
    pp.preprocess_dataset(data_root, loader=loader, verbose=False)
    ds = CascadeDataset(data_root, compute_centroid=False)
    np.testing.assert_array_equal(
        ds.energies,
        np.array([s[2] for s in ds.samples], dtype=np.float32),
    )
    assert ds.samples[0][2] == ds.energies[0]  # legacy infer.py path


def test_dataset_legacy_path_when_no_manifest(tmp_path, monkeypatch):
    cascades = _default_cascades()
    data_root, loader, coords_map = _make_fake_dataset(tmp_path, cascades)
    # Deliberately do NOT run preprocess; dataset should fall back.

    # Stub OVITO out of the legacy reader.
    monkeypatch.setattr(
        CascadeDataset, "_load_coordinates",
        lambda self, path: coords_map[path],
    )

    ds = CascadeDataset(data_root, compute_centroid=True, use_cache=True)
    assert len(ds) == len(cascades)
    assert not ds.has_cache  # no .npz, no manifest

    item = ds[0]
    assert item["vac_coords"].shape[1] == 3
    # local stats are computed on the fly in the legacy path.
    exp_c, exp_s = pp.compute_local_stats(
        item["vac_coords"].numpy(), item["sia_coords"].numpy()
    )
    np.testing.assert_allclose(item["local_centroid"].numpy(), exp_c, rtol=1e-5)
    assert item["local_scale"].item() == pytest.approx(exp_s, rel=1e-5)


def test_dataset_max_samples_with_manifest(tmp_path):
    data_root, loader, _ = _make_fake_dataset(tmp_path, _default_cascades())
    pp.preprocess_dataset(data_root, loader=loader, verbose=False)

    ds = CascadeDataset(data_root, compute_centroid=False, max_samples=2)
    assert len(ds) == 2
