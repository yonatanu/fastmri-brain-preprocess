"""Loader shapes, normalization, imports, and storage validation."""

import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastmri_preprocess import FastMRIBrain


def test_native_shapes_and_normalization(dataset_root):
    ds = FastMRIBrain(dataset_root, "train")
    wide, narrow = ds[0], ds[1]
    assert wide["ksp"].shape == (2, 220, 220)
    assert narrow["ksp"].shape == (4, 220, 180)
    assert narrow["gt"].shape == (220, 180)
    assert narrow["mask"].all()
    assert not narrow["ksp"].flags.writeable
    assert narrow["sigma"] == 2.5
    np.testing.assert_array_equal(ds.normalize(narrow["gt"], narrow), 2.5)
    np.testing.assert_array_equal(
        ds.denormalize(ds.normalize(narrow["gt"], narrow), narrow), narrow["gt"]
    )
    np.testing.assert_array_equal(ds.select("snr_std > 5"), [0])
    assert ds.volume(1)["gt"].shape == (1, 220, 180)
    assert ds.boxes(0).label.tolist() == ["Mass"]
    assert ds.boxes(1).empty


def test_fixed_shapes_require_explicit_narrow_padding(dataset_root):
    ds = FastMRIBrain(dataset_root, "train", fixed_shape=True)
    assert ds[0]["ksp"].shape == (8, 220, 220)
    assert not ds[0]["ksp"][2:].any()
    with pytest.raises(ValueError, match="phase width"):
        ds[1]
    padded = FastMRIBrain(dataset_root, "train", fixed_shape=True, pad_narrow=True)
    with pytest.warns(UserWarning, match="storage padding"):
        item = padded[1]
    assert item["padded"]
    assert item["gt"].shape == (220, 220)
    assert not item["gt"][:, :20].any()
    assert not item["gt"][:, 200:].any()


def test_original_import_path_and_api(dataset_root, monkeypatch):
    code_dir = Path(__file__).resolve().parents[1] / "code"
    monkeypatch.syspath_prepend(str(code_dir))
    legacy = importlib.import_module("fastmri_brain")
    assert Path(legacy.__file__) == code_dir / "fastmri_brain.py"
    assert inspect.signature(legacy.FastMRIBrain) == inspect.signature(FastMRIBrain)
    old = legacy.FastMRIBrain(dataset_root, "train")
    new = FastMRIBrain(dataset_root, "train")
    assert len(old) == len(new)
    for slot in range(len(old)):
        before, after = old[slot], new[slot]
        assert before.keys() == after.keys()
        for key in before:
            if isinstance(before[key], pd.Series):
                pd.testing.assert_series_equal(before[key], after[key])
            else:
                np.testing.assert_equal(before[key], after[key])
        np.testing.assert_array_equal(
            old.normalize(before["gt"], before), new.normalize(after["gt"], after)
        )
    pd.testing.assert_frame_equal(old.boxes(0), new.boxes(0))
    pd.testing.assert_frame_equal(old.plus_study, new.plus_study)
    np.testing.assert_array_equal(old.select("snr_std > 5"), new.select("snr_std > 5"))
    np.testing.assert_array_equal(old.volume(1)["gt"], new.volume(1)["gt"])
    for key in ("cov", "W", "U"):
        np.testing.assert_array_equal(old.noise(1)[key], new.noise(1)[key])


def test_original_import_does_not_require_package_install():
    import subprocess
    import sys

    code_dir = Path(__file__).resolve().parents[1] / "code"
    script = """
import sys
class BlockPackage:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('fastmri_preprocess'):
            raise ImportError('package intentionally unavailable')
sys.meta_path.insert(0, BlockPackage())
sys.path.insert(0, sys.argv[1])
from fastmri_brain import FastMRIBrain
assert FastMRIBrain.__module__ == 'fastmri_brain'
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(code_dir)], check=True)


def test_gt_in_memory_allocates_an_independent_array(dataset_root):
    ds = FastMRIBrain(dataset_root, "train", gt_in_memory=True)
    assert not isinstance(ds._arrays["gt"], np.memmap)
    assert ds._arrays["gt"].flags.owndata
    ds._arrays["gt"][0] = 42
    fresh = FastMRIBrain(dataset_root, "train")
    assert not np.array_equal(ds[0]["gt"], fresh[0]["gt"])


def test_loader_rejects_missing_slice_rows(dataset_root):
    path = dataset_root / "train/slices.parquet"
    pd.read_parquet(path).iloc[:1].to_parquet(path)
    with pytest.raises(ValueError, match="full storage layout"):
        FastMRIBrain(dataset_root, "train")


def test_loader_rejects_wrong_binary_size(dataset_root):
    with (dataset_root / "train/gt.bin").open("ab") as stream:
        stream.write(b"extra")
    with pytest.raises(ValueError, match="Binary size"):
        FastMRIBrain(dataset_root, "train")
