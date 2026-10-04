"""Read raw scan headers and assign contiguous storage slots."""

from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import h5py
import pandas as pd

from .paths import add_output_argument
from .pipeline import MAX_COILS, MIN_FIELD_T, parse_header, plan_geometry
from .storage import split_lock, validate_scan_layout, write_parquet

SPLIT_DIRS = {
    "train": "multicoil_train",
    "val": "multicoil_val",
    "test": "multicoil_test_full",
}


def survey_one(path: str) -> dict:
    stem = os.path.basename(path)[: -len(".h5")]
    row = dict(stem=stem, path=str(Path(path).resolve()), field_T=float("nan"))
    try:
        row["scanner_code"] = int(stem.split("_")[3])
        with h5py.File(path, "r") as hf:
            raw = hf["ismrmrd_header"][()]
            S, C, KX, KY = hf["kspace"].shape
            rss = (
                hf["reconstruction_rss"].shape
                if "reconstruction_rss" in hf
                else (None, None, None)
            )
            attrs = {k: hf.attrs[k] for k in hf.attrs}
        xml = (
            raw.decode()
            if isinstance(raw, (bytes, bytearray))
            else raw.tobytes().decode()
        )
        hdr = parse_header(xml)
        h = asdict(hdr)
        for key in ("enc_mat", "enc_fov", "rec_mat", "rec_fov"):
            v = h.pop(key)
            h[f"{key}_x"], h[f"{key}_y"] = v[0], v[1]
        row.update(h)
        row.update(
            n_slices=int(S),
            n_coils_native=int(C),
            KX=int(KX),
            KY=int(KY),
            rss_h=rss[1],
            rss_w=rss[2],
            acquisition=str(attrs.get("acquisition", "")),
            patient_id=str(attrs.get("patient_id", "")),
            attr_max=float(attrs.get("max", float("nan"))),
            attr_norm=float(attrs.get("norm", float("nan"))),
            pe_frac=(hdr.pe_max + 1) / KY,
        )
        if hdr.field_T >= MIN_FIELD_T:
            geom = plan_geometry(hdr, KX, KY)
            g = asdict(geom)
            g.pop("KX")
            g.pop("KY")
            row.update(
                g,
                n_acq=geom.n_acq,
                zero_filled=geom.zero_filled,
                n_coils=min(C, MAX_COILS),
            )
        row["error"] = ""
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def write_survey(table, out_dir):
    """Keep the original slot assignment once binary storage exists."""
    selected = table[table.selected].sort_values("scan_idx").reset_index(drop=True)
    validate_scan_layout(selected)
    path = Path(out_dir) / "scans_geom.parquet"
    if any(Path(out_dir).glob("*.bin")):
        if not path.is_file():
            raise ValueError(
                "Binaries exist without a survey; use a new output directory."
            )
        previous = pd.read_parquet(path)
        previous = (
            previous[previous.selected].sort_values("scan_idx").reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(previous, selected, check_dtype=False)
        except AssertionError as exc:
            raise ValueError(
                "The survey changed after storage was created; "
                "use a new output directory."
            ) from exc
        return
    write_parquet(table, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLIT_DIRS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--raw", required=True, help="Root containing the raw split directories"
    )
    add_output_argument(ap)
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be positive")

    files = sorted(glob.glob(os.path.join(args.raw, SPLIT_DIRS[args.split], "*.h5")))
    if not files:
        ap.error(
            f"no HDF5 files found in {os.path.join(args.raw, SPLIT_DIRS[args.split])}"
        )
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(survey_one, files, chunksize=4))
    df = pd.DataFrame(rows)
    df["split"] = args.split
    df["contrast"] = df.stem.str.split("_").str[2]

    bad = df[df.error != ""]
    if len(bad):
        print(f"{len(bad)} files with errors (kept in the table, excluded from slots):")
        print(bad[["stem", "error"]].to_string())
    keep = (df.field_T >= MIN_FIELD_T) & (df.error == "")
    if not keep.any():
        raise ValueError("No valid 3 T scans were found; no survey was written.")
    df["selected"] = keep
    df["scan_idx"] = -1
    df["slot0"] = -1
    sel = df[keep].sort_values("stem")
    df.loc[sel.index, "scan_idx"] = range(len(sel))
    df.loc[sel.index, "slot0"] = sel.n_slices.cumsum().shift(fill_value=0).values

    out_dir = os.path.join(args.out, args.split)
    with split_lock(out_dir):
        write_survey(df, out_dir)
    s = df[keep]
    print(
        f"{args.split}: {len(df)} files, {len(s)} selected at >= {MIN_FIELD_T} T, "
        f"{int(s.n_slices.sum())} slices, coils "
        f"{s.n_coils.value_counts().sort_index().to_dict()}, "
        f"PE width {s.my.value_counts().sort_index().to_dict()}, zero-filled "
        f"{int(s.zero_filled.sum())}"
    )


if __name__ == "__main__":
    main()
