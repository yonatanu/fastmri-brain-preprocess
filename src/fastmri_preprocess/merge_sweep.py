"""Merge complete per-slot QC results into the slice table."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from .paths import add_output_argument
from .storage import split_lock, validate_slot_coverage, write_parquet


def merge_sweep(slices, sweep):
    validate_slot_coverage(sweep, slices.slot)
    result = slices.copy()
    indexed = sweep.set_index("slot").reindex(slices.slot)
    for target, source in {
        "n_support_components": "n_components",
        "sweep_resid": "resid",
        "head_frac": "head_frac",
    }.items():
        result[target] = indexed[source].to_numpy()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    add_output_argument(ap)
    args = ap.parse_args()
    d = os.path.join(args.out, args.split)
    with split_lock(d):
        slices = pd.read_parquet(os.path.join(d, "slices.parquet"))
        sw = pd.read_parquet(os.path.join(d, "sweep.parquet"))
        slices = merge_sweep(slices, sw)
        write_parquet(slices, os.path.join(d, "slices.parquet"))
    fragmented = int((slices.n_support_components > 1).sum())
    inconsistent = int((abs(slices.sweep_resid - slices.espirit_resid) > 5e-3).sum())
    print(
        f"{args.split}: merged; components > 1 on {fragmented} slices, "
        f"|sweep_resid - espirit_resid| > 5e-3 on {inconsistent}"
    )


if __name__ == "__main__":
    main()
