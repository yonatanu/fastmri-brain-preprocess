"""Read processed fastMRI scans and slices through NumPy memory maps."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

N_OUT = 220
MAX_COILS = 8
LAYOUT = {
    "ksp": ((MAX_COILS, N_OUT, N_OUT), np.complex64),
    "mps": ((MAX_COILS, N_OUT, N_OUT), np.complex64),
    "gt": ((N_OUT, N_OUT), np.complex64),
    "eig": ((N_OUT, N_OUT), np.float16),
}


def _memmap(path, dtype, shape):
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if os.path.getsize(path) != expected:
        raise ValueError(f"Binary size does not match the tables: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


class FastMRIBrain:
    def __init__(
        self,
        root: str,
        split: str,
        gt_in_memory: bool = False,
        fixed_shape: bool = False,
        pad_narrow: bool = False,
    ):
        """Open a processed split.

        Native shapes are returned by default. fixed_shape pads the coil axis and
        rejects narrow phase FOVs unless pad_narrow is enabled. Such padded k-space
        is storage only and must not be used as a 220 mm acquisition.
        """
        self.root, self.split = root, split
        self.fixed_shape, self.pad_narrow = fixed_shape, pad_narrow
        self._warned = False
        d = os.path.join(root, split)
        self.scans = (
            pd.read_parquet(os.path.join(d, "scans.parquet"))
            .sort_values("scan_idx")
            .reset_index(drop=True)
        )
        self.slices = pd.read_parquet(os.path.join(d, "slices.parquet")).set_index(
            "slot", drop=False
        )
        self._validate_tables()
        self.plus_boxes = pd.read_parquet(os.path.join(d, "plus_boxes.parquet"))
        self.plus_study = pd.read_parquet(os.path.join(d, "plus_study.parquet"))
        with open(os.path.join(root, "stats.json")) as f:
            self.stats = json.load(f)
        self.norm_const = self.stats["norm_const"]
        if not np.isfinite(self.norm_const) or self.norm_const <= 0:
            raise ValueError("The dataset normalization constant must be positive.")
        n = int(self.scans.n_slices.sum())
        self._arrays = {
            name: _memmap(
                os.path.join(d, f"{name}.bin"), dtype=dtype, shape=(n, *shape)
            )
            for name, (shape, dtype) in LAYOUT.items()
        }
        self.pe_mask = _memmap(
            os.path.join(d, "pe_mask.bin"),
            dtype=np.uint8,
            shape=(len(self.scans), N_OUT),
        )
        if not np.isin(self.pe_mask, [0, 1]).all():
            raise ValueError("Acquisition masks must contain only zero and one.")
        if gt_in_memory:
            self._arrays["gt"] = np.array(self._arrays["gt"], copy=True)
        self._noise = None

    def _validate_tables(self):
        scans, slices = self.scans, self.slices
        fields = ["scan_idx", "slot0", "n_slices", "n_coils", "my", "col0"]
        values = scans[fields].to_numpy(dtype=float)
        if scans.empty or not np.isfinite(values).all():
            raise ValueError("Scan storage metadata must be nonempty and finite.")
        if not (values == np.floor(values)).all():
            raise ValueError("Scan storage dimensions must be integers.")
        if scans.stem.isna().any() or scans.stem.duplicated().any():
            raise ValueError("Scan names must be present and unique.")
        if not np.array_equal(scans.scan_idx, np.arange(len(scans))):
            raise ValueError("Scan indices must be contiguous and start at zero.")
        if (scans.n_slices <= 0).any() or not np.array_equal(
            scans.slot0, scans.n_slices.cumsum().shift(fill_value=0)
        ):
            raise ValueError("Scan slice ranges must be contiguous and nonempty.")
        if not scans.n_coils.between(1, MAX_COILS).all():
            raise ValueError("Invalid stored coil count.")
        if (
            not scans.my.between(1, N_OUT).all()
            or not (scans.col0 == N_OUT // 2 - scans.my // 2).all()
        ):
            raise ValueError("Invalid native phase columns.")
        n = int(scans.n_slices.sum())
        if not slices.index.is_unique or not np.array_equal(
            np.sort(slices.index.to_numpy()), np.arange(n)
        ):
            raise ValueError("Slice slots must cover the full storage layout once.")
        expected = np.repeat(scans.scan_idx.to_numpy(), scans.n_slices.astype(int))
        if not np.array_equal(slices.sort_index().scan_idx, expected):
            raise ValueError("Slice scan indices do not match the scan table.")
        if not np.isfinite(slices.scale).all() or (slices.scale <= 0).any():
            raise ValueError("Slice normalization scales must be finite and positive.")
        if "processed" in scans and not scans.processed.fillna(False).eq(True).all():
            raise ValueError("The split contains unprocessed scans.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, slot: int) -> dict:
        if not isinstance(slot, (int, np.integer)):
            raise TypeError(
                "index with one integer slot; for several use [ds[s] for s in slots] "
                "or ds.volume(scan_idx)"
            )
        s = self.slices.loc[slot]
        scan = self.scans.iloc[int(s.scan_idx)]
        c0, w, nc = int(scan.col0), int(scan.my), int(scan.n_coils)
        padded = False
        if self.fixed_shape:
            if w != N_OUT:
                if not self.pad_narrow:
                    raise ValueError(
                        f"slot {slot} ({scan.stem}) has phase width {w} < {N_OUT}; "
                        f"exclude with my == {N_OUT}"
                    )
                if not self._warned:
                    warnings.warn(
                        "pad_narrow: narrow-FOV items are zero-padded to 220 columns; "
                        "their gt/eig are valid "
                        "images but ksp/mps/mask are storage padding, not a 220 mm "
                        "k-space (see item['padded'])"
                    )
                    self._warned = True
                padded = True
            c0, w, nc = 0, N_OUT, MAX_COILS
        cols = slice(c0, c0 + w)
        return dict(
            slot=int(slot),
            padded=padded,
            ksp=self._arrays["ksp"][slot, :nc, :, cols],
            mps=self._arrays["mps"][slot, :nc, :, cols],
            gt=self._arrays["gt"][slot, :, cols],
            eig=self._arrays["eig"][slot, :, cols],
            mask=self.pe_mask[int(s.scan_idx), cols].astype(bool),
            scale=float(s.scale),
            sigma=self.norm_const / float(s.scale),
            slice_row=s,
            scan_row=scan,
        )

    def volume(self, scan_idx: int) -> dict:
        """Return all slices of one scan as memory-map views in native shape."""
        scan = self.scans.iloc[scan_idx]
        s0, n = int(scan.slot0), int(scan.n_slices)
        c0, w, nc = int(scan.col0), int(scan.my), int(scan.n_coils)
        cols = slice(c0, c0 + w)
        return dict(
            slots=np.arange(s0, s0 + n),
            ksp=self._arrays["ksp"][s0 : s0 + n, :nc, :, cols],
            mps=self._arrays["mps"][s0 : s0 + n, :nc, :, cols],
            gt=self._arrays["gt"][s0 : s0 + n, :, cols],
            eig=self._arrays["eig"][s0 : s0 + n, :, cols],
            mask=self.pe_mask[scan_idx, cols].astype(bool),
            scan_row=scan,
        )

    def boxes(self, slot: int) -> pd.DataFrame:
        return self.plus_boxes[self.plus_boxes.slot == slot]

    def noise(self, scan_idx: int) -> dict:
        """Return native-coil covariance, whitening, and compression matrices."""
        if self._noise is None:
            self._noise = np.load(os.path.join(self.root, self.split, "noise.npz"))
        stem = self.scans.iloc[scan_idx].stem
        return {k: self._noise[f"{stem}/{k}"] for k in ("cov", "W", "U")}

    def select(self, query: str) -> np.ndarray:
        """Slot indices of slices matching a pandas query over the slices table."""
        return self.slices.query(query).slot.to_numpy()

    def normalize(self, x, item):
        """Scale stored images or k-space to the dataset normalization."""
        return x * (self.norm_const / item["scale"])

    def denormalize(self, x, item):
        return x * (item["scale"] / self.norm_const)
