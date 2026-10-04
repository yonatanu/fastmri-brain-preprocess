"""Build complete scan, slice, noise, and annotation tables from processing logs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

from .fastmri_brain import MAX_COILS
from .paths import add_output_argument
from .pipeline import ESPIRIT, N_OUT, VOXEL_MM, map_box
from .storage import (
    atomic_output,
    read_qc_records,
    split_lock,
    validate_scan_layout,
    validate_storage,
    write_json,
    write_parquet,
)

SIGMA_DATA = 0.5  # EDM convention: per-element std of the normalised training images


def slug(label: str) -> str:
    return "plus_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load_qc(out_dir):
    records = read_qc_records(out_dir)
    if not records:
        raise ValueError(f"No processing records in {out_dir}.")
    return pd.DataFrame(records)


def validate_qc(scans, qc):
    """Require one successful, complete record for each surveyed scan."""
    if qc.stem.duplicated().any() or set(qc.stem) != set(scans.stem):
        raise ValueError("Processing records do not cover exactly the surveyed scans.")
    if qc.error.isna().any() or (qc.error != "").any():
        raise ValueError("Processing contains failed scans; rerun preprocessing first.")
    by_stem = qc.set_index("stem")
    for scan in scans.itertuples():
        record = by_stem.loc[scan.stem]
        if record.scan_idx != scan.scan_idx or record.slot0 != scan.slot0:
            raise ValueError(
                f"Progress layout differs from the survey for {scan.stem}."
            )
        stats = record.get("slices")
        required = {
            "scale",
            "std_full",
            "std_real",
            "std_imag",
            "gt_p99",
            "gt_max",
            "support_frac",
            "espirit_resid",
            "band_energy_frac",
        }
        if not isinstance(stats, dict) or not required <= stats.keys():
            raise ValueError(f"Missing slice statistics for {scan.stem}.")
        for key, values in stats.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (int(scan.n_slices),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} statistics for {scan.stem}.")
        if np.any(np.asarray(stats["scale"]) <= 0):
            raise ValueError(f"Nonpositive normalization scale for {scan.stem}.")


def build_slices(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in qc.iterrows():
        stats = r.slices
        S = len(stats["scale"])
        for i in range(S):
            rows.append(
                dict(
                    slot=int(r.slot0) + i,
                    scan_idx=int(r.scan_idx),
                    stem=r.stem,
                    slice=i,
                    slice_frac=i / max(S - 1, 1),
                    **{k: float(v[i]) for k, v in stats.items()},
                )
            )
    df = pd.DataFrame(rows)
    df["snr_std"] = df.std_full / np.sqrt(0.5)
    return df


def halo_levels(out_dir, slices: pd.DataFrame, chunk=32):
    """Measure exterior signal and support holes from stored reference images."""
    n = int(slices.slot.max()) + 1
    gt = np.memmap(
        os.path.join(out_dir, "gt.bin"),
        dtype=np.complex64,
        mode="r",
        shape=(n, N_OUT, N_OUT),
    )
    maps = np.memmap(
        os.path.join(out_dir, "mps.bin"),
        dtype=np.complex64,
        mode="r",
        shape=(n, MAX_COILS, N_OUT, N_OUT),
    )
    halo = np.full(n, np.nan, dtype=np.float32)
    holes = np.full(n, np.nan, dtype=np.float32)
    for s0 in range(0, n, chunk):
        m2 = np.abs(np.asarray(gt[s0 : s0 + chunk])) ** 2
        supp = np.any(maps[s0 : s0 + chunk] != 0, axis=1)
        thr = 0.05 * np.percentile(m2.reshape(len(m2), -1), 99.5, axis=1)
        for i in range(len(m2)):
            head = binary_fill_holes(m2[i] > thr[i])
            ext = supp[i] & ~head
            halo[s0 + i] = m2[i][ext].mean() if ext.any() else 0.0
            holes[s0 + i] = (head & ~supp[i]).sum() / max(int(head.sum()), 1)
    return halo, holes


def fastmri_plus(scans: pd.DataFrame, slices: pd.DataFrame, annotation_dir):
    """Map fastMRI+ boxes and study labels, retaining review status."""
    ann = pd.read_csv(os.path.join(annotation_dir, "brain.csv"))
    reviewed = set(
        pd.read_csv(os.path.join(annotation_dir, "brain_file_list.csv"), header=None)[0]
    )
    scans = scans.copy()
    slices = slices.copy()
    labels = sorted(ann.label.dropna().unique())
    scans["plus_reviewed"] = scans.stem.isin(reviewed)
    ann = ann[ann.file.isin(set(scans.stem))]
    by_stem = scans.set_index("stem")
    if not by_stem.index.is_unique:
        raise ValueError("Scan names must be unique for annotation mapping.")

    study = ann[ann.study_level == "Yes"][["file", "label"]].rename(
        columns={"file": "stem"}
    )
    study = study.merge(scans[["stem", "scan_idx"]], on="stem", how="left").reset_index(
        drop=True
    )
    study = study[["stem", "scan_idx", "label"]]

    boxes = ann[ann.study_level != "Yes"].copy()
    out = []
    for r in boxes.itertuples(index=False):
        s = by_stem.loc[r.file]
        voxel_x = float(s.get("voxel_x", VOXEL_MM))
        voxel_y = float(s.get("voxel_y", VOXEL_MM))
        if (
            not np.isfinite(r.slice)
            or r.slice != int(r.slice)
            or not 0 <= r.slice < s.n_slices
        ):
            raise ValueError(f"Invalid annotation slice {r.slice} for {r.file}.")
        coordinates = np.asarray([r.x, r.y, r.width, r.height], dtype=float)
        if not np.isfinite(coordinates).all() or r.width <= 0 or r.height <= 0:
            raise ValueError(f"Invalid annotation box for {r.file}.")
        r0, c0, h, w = map_box(
            r.x,
            r.y,
            r.width,
            r.height,
            (int(s.rss_h), int(s.rss_w)),
            (s.rec_fov_x, s.rec_fov_y),
            (N_OUT, int(s.my)),
            voxel=(voxel_x, voxel_y),
        )
        out.append(
            dict(
                stem=r.file,
                scan_idx=int(s.scan_idx),
                slice=int(r.slice),
                slot=int(s.slot0) + int(r.slice),
                label=r.label,
                r0=r0,
                c0=c0,
                h=h,
                w=w,
                c0_slot=c0 + int(s.col0),
                row_mm=(r0 + h / 2 - N_OUT // 2) * voxel_x,
                col_mm=(c0 + w / 2 - int(s.my) // 2) * voxel_y,
                h_mm=h * voxel_x,
                w_mm=w * voxel_y,
                x_orig=r.x,
                y_orig=r.y,
                w_orig=r.width,
                h_orig=r.height,
            )
        )
    box_cols = [
        "stem",
        "scan_idx",
        "slice",
        "slot",
        "label",
        "r0",
        "c0",
        "h",
        "w",
        "c0_slot",
        "row_mm",
        "col_mm",
        "h_mm",
        "w_mm",
        "x_orig",
        "y_orig",
        "w_orig",
        "h_orig",
    ]
    # Keep the schema when no boxes match the split.
    boxes = pd.DataFrame(out, columns=box_cols)

    # Per-slice label indicators and counts.
    for lab in labels:
        col = slug(lab)
        hit = boxes[boxes.label == lab].slot if len(boxes) else pd.Series([], dtype=int)
        slices[col] = slices.slot.isin(set(hit))
    counts = boxes.groupby("slot").size() if len(boxes) else pd.Series(dtype=int)
    slices["n_plus_boxes"] = slices.slot.map(counts).fillna(0).astype(int)
    art = (
        boxes[boxes.label.str.contains("artifact", case=False)].groupby("slot").size()
        if len(boxes)
        else counts
    )
    slices["plus_artifact_boxes"] = slices.slot.map(art).fillna(0).astype(int)
    slices["plus_reviewed"] = slices.stem.isin(reviewed)

    scans["n_plus_boxes"] = (
        scans.scan_idx.map(boxes.groupby("scan_idx").size() if len(boxes) else counts)
        .fillna(0)
        .astype(int)
    )
    scans["n_plus_study_labels"] = (
        scans.scan_idx.map(study.groupby("scan_idx").size()).fillna(0).astype(int)
    )
    return scans, slices, boxes, study


def pack_noise(out_dir, scans):
    """Pack all noise matrices, failing before replacement if any are missing."""
    packed = {}
    for scan in scans.itertuples():
        stem = scan.stem
        path = Path(out_dir) / "noise" / f"{stem}.npz"
        with np.load(path, allow_pickle=False) as source:
            matrices = {key: source[key] for key in ("cov", "W", "U")}
        cov, whitening, compression = (matrices[key] for key in ("cov", "W", "U"))
        native_coils, stored_coils = int(scan.n_coils_native), int(scan.n_coils)
        compression_shape = (
            (native_coils, stored_coils) if native_coils > stored_coils else (0, 0)
        )
        if (
            cov.shape != (native_coils, native_coils)
            or whitening.shape != cov.shape
            or compression.shape != compression_shape
            or not all(np.isfinite(value).all() for value in matrices.values())
        ):
            raise ValueError(f"Invalid noise matrices for {stem}.")
        for key, value in matrices.items():
            packed[f"{stem}/{key}"] = value
    with atomic_output(Path(out_dir) / "noise.npz") as temporary:
        np.savez(temporary, **packed)
    return len(scans)


def summarize(scans, slices):
    """Update scan summaries from the current slice statistics."""
    scans, slices = scans.copy(), slices.copy()
    slices["snr_median_scan"] = slices.groupby("stem").snr_std.transform("median")
    scans["snr_median"] = scans.stem.map(slices.groupby("stem").snr_std.median())
    scans["espirit_resid_mean"] = scans.stem.map(
        slices.groupby("stem").espirit_resid.mean()
    )
    return scans, slices


def normalization_stats(slices):
    ratio = slices.std_full / slices.scale
    if not np.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError("Normalization requires finite positive image scales.")
    return dict(
        sigma_data=SIGMA_DATA,
        norm_const=SIGMA_DATA / float(ratio.median()),
        std_over_scale_median=float(ratio.median()),
        std_over_scale_cv=float(
            ratio.std(ddof=1 if len(ratio) > 1 else 0) / ratio.mean()
        ),
        espirit=ESPIRIT,
        n_train_slices=len(slices),
        snr_std_percentiles={
            str(p): float(np.percentile(slices.snr_std, p))
            for p in (1, 5, 10, 50, 90, 99)
        },
    )


def finalize(out_dir, annotation_dir, split, overwrite=False):
    out_dir = Path(out_dir)
    outputs = [
        "scans.parquet",
        "slices.parquet",
        "plus_boxes.parquet",
        "plus_study.parquet",
    ]
    if not overwrite and any((out_dir / name).exists() for name in outputs):
        raise FileExistsError(
            "Finalized tables already exist; use --overwrite only to rebuild "
            "from matching logs."
        )
    geom = pd.read_parquet(out_dir / "scans_geom.parquet")
    geom = geom[geom.selected].sort_values("scan_idx").reset_index(drop=True)
    validate_scan_layout(geom)
    validate_storage(out_dir, int(geom.n_slices.sum()), len(geom))
    qc = load_qc(out_dir)
    validate_qc(geom, qc)
    qc_columns = ["stem"] + [key for key in qc if key not in geom and key != "slices"]
    scans = geom.merge(qc[qc_columns], on="stem", validate="one_to_one")
    scans["processed"] = True
    slices = build_slices(qc)
    columns = [
        "stem",
        "contrast",
        "model",
        "scanner_code",
        "n_coils",
        "my",
        "col0",
        "zero_filled",
    ]
    slices = slices.merge(scans[columns], on="stem", validate="many_to_one")
    scans, slices, boxes, study = fastmri_plus(scans, slices, annotation_dir)
    halo, holes = halo_levels(out_dir, slices)
    slices["halo_level"] = halo[slices.slot.to_numpy()]
    if "support_holes_frac" not in slices:
        slices["support_holes_frac"] = holes[slices.slot.to_numpy()]
    if "espirit_thresh" not in slices:
        slices["espirit_thresh"] = ESPIRIT["thresh"]
    scans, slices = summarize(scans, slices)
    stats = normalization_stats(slices) if split == "train" else None
    pack_noise(out_dir, scans)
    for table, name in zip((scans, slices.sort_values("slot"), boxes, study), outputs):
        write_parquet(table, out_dir / name)
    if stats is not None:
        write_json(stats, out_dir.parent / "stats.json")
    print(
        f"{split}: finalized {len(scans)} scans, {len(slices)} slices, "
        f"{len(boxes)} boxes"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument(
        "--annotations", help="fastMRI+ CSV directory (default: <out>/annotations)"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing tables from processing logs",
    )
    add_output_argument(ap)
    args = ap.parse_args()
    out_dir = Path(args.out) / args.split
    annotations = args.annotations or Path(args.out) / "annotations"
    with split_lock(out_dir):
        finalize(out_dir, annotations, args.split, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
