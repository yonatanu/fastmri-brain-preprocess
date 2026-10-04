"""Temporary two-scan dataset fixtures."""

import json

import numpy as np
import pandas as pd
import pytest

from fastmri_preprocess.fastmri_brain import LAYOUT


@pytest.fixture
def dataset_root(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    scans = pd.DataFrame(
        [
            dict(scan_idx=0, slot0=0, n_slices=1, n_coils=2, my=220, col0=0),
            dict(scan_idx=1, slot0=1, n_slices=1, n_coils=4, my=180, col0=20),
        ]
    )
    scans["stem"] = ["wide", "narrow"]
    scans.to_parquet(split / "scans.parquet")
    pd.DataFrame(
        dict(slot=[0, 1], scan_idx=[0, 1], scale=[2.0, 4.0], snr_std=[10, 2])
    ).to_parquet(split / "slices.parquet")
    pd.DataFrame(dict(slot=[0], label=["Mass"])).to_parquet(
        split / "plus_boxes.parquet"
    )
    pd.DataFrame(columns=["stem", "scan_idx", "label"]).to_parquet(
        split / "plus_study.parquet"
    )
    (tmp_path / "stats.json").write_text(json.dumps(dict(norm_const=10.0)))
    for name, (shape, dtype) in LAYOUT.items():
        data = np.zeros((2, *shape), dtype=dtype)
        for row in scans.itertuples():
            cols = slice(row.col0, row.col0 + row.my)
            if name in ("ksp", "mps"):
                data[row.scan_idx, : row.n_coils, :, cols] = 1 + 2j
            else:
                data[row.scan_idx, :, cols] = 1
        data.tofile(split / f"{name}.bin")
    mask = np.zeros((2, 220), dtype=np.uint8)
    mask[0] = 1
    mask[1, 20:200] = 1
    mask.tofile(split / "pe_mask.bin")
    np.savez(
        split / "noise.npz",
        **{
            f"{stem}/{key}": np.eye(nc, dtype=np.complex64)
            for stem, nc in (("wide", 2), ("narrow", 4))
            for key in ("cov", "W", "U")
        },
    )
    return tmp_path
