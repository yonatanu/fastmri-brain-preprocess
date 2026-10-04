"""Recompute sensitivity maps for slices with support holes or fallback thresholds."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes

from .fastmri_brain import FastMRIBrain
from .finalize import normalization_stats, summarize
from .paths import add_output_argument
from .pipeline import (
    ESPIRIT,
    ESPIRIT_FALLBACK_THRESH,
    HOLE_CUTOFF,
    SCALE_LINES,
    espirit_maps,
    fft2c,
    ifft2c,
    support_holes,
)
from .run import to_slot
from .storage import (
    SlotWriter,
    split_lock,
    validate_slot_coverage,
    write_json,
    write_parquet,
)


def slice_stats(ksp, mps, gt, eig, mask):
    """Compute statistics for a repaired native-grid slice."""
    prediction = fft2c(mps * gt[None]) * mask
    center = ksp.shape[-1] // 2
    magnitude_squared = gt.abs().square()
    threshold = 0.05 * torch.quantile(magnitude_squared.flatten(), 0.995)
    support = mps.abs().square().sum(0).cpu().numpy() > 0
    head = binary_fill_holes((magnitude_squared > threshold).cpu().numpy())
    exterior = support & ~head
    return dict(
        scale=float(
            ksp[..., center - SCALE_LINES // 2 : center + SCALE_LINES // 2].norm()
        ),
        std_full=float(torch.view_as_real(gt).std()),
        std_real=float(gt.real.std()),
        std_imag=float(gt.imag.std()),
        gt_p99=float(gt.abs().flatten().quantile(0.99)),
        gt_max=float(gt.abs().max()),
        support_frac=float(support.mean()),
        espirit_resid=float((prediction - ksp).norm() / ksp.norm()),
        halo_level=(
            float(magnitude_squared.cpu().numpy()[exterior].mean())
            if exterior.any()
            else 0.0
        ),
    )


def repair(
    root,
    split,
    cutoff=HOLE_CUTOFF,
    threshold=ESPIRIT_FALLBACK_THRESH,
    dry_run=False,
    device="cuda:0",
):
    ds = FastMRIBrain(root, split)
    directory = Path(root) / split
    slices = ds.slices.copy()
    stale = pd.Series(False, index=slices.index)
    sweep_path = directory / "sweep.parquet"
    binary_mtime = max(
        (directory / f"{name}.bin").stat().st_mtime_ns for name in ("mps", "gt", "eig")
    )
    if sweep_path.exists() and sweep_path.stat().st_mtime_ns >= binary_mtime:
        sweep = pd.read_parquet(sweep_path)
        validate_slot_coverage(sweep, slices.slot)
        sweep = sweep.set_index("slot").reindex(slices.index)
        slices["support_holes_frac"] = sweep.holes
        stale[:] = ~np.isclose(sweep.resid, slices.espirit_resid, atol=5e-3, rtol=0)
    journal = directory / "repair_pending.json"
    pending = set()
    if journal.exists():
        pending = set(json.loads(journal.read_text())["slots"])
        if not pending <= set(slices.index):
            raise ValueError("Repair journal contains unknown slots.")
        stale.loc[list(pending)] = True
    fallback = np.isclose(slices.espirit_thresh, threshold)
    candidates = slices[(slices.support_holes_frac > cutoff) | fallback | stale]
    print(f"{split}: {len(candidates)} slices selected for repair")
    if candidates.empty:
        return
    device = torch.device(device)
    calibration_device = -1 if device.type == "cpu" else (device.index or 0)
    writer_context = (
        nullcontext(None)
        if dry_run
        else SlotWriter(directory, names=("mps", "gt", "eig"))
    )
    rewritten = 0
    with writer_context as writer:
        for row in candidates.itertuples():
            item = ds[int(row.slot)]
            scan = item["scan_row"]
            kspace = torch.tensor(np.asarray(item["ksp"]), device=device)
            mask = torch.tensor(item["mask"], device=device)
            coil_images = ifft2c(kspace)
            rss = coil_images.abs().square().sum(0).sqrt()
            coils = kspace.shape[0]
            used = ESPIRIT["thresh"]
            maps, eigenvalues = espirit_maps(kspace, calibration_device, thresh=used)
            holes = support_holes(rss, eigenvalues, coils)
            if holes > cutoff:
                used = threshold
                maps, eigenvalues = espirit_maps(
                    kspace, calibration_device, thresh=used
                )
                holes = support_holes(rss, eigenvalues, coils)
            image = (maps.conj() * coil_images).sum(0)
            if not all(
                torch.isfinite(value).all() for value in (maps, image, eigenvalues)
            ):
                raise ValueError(f"Nonfinite repair arrays for slot {row.slot}.")
            stats = slice_stats(kspace, maps, image, eigenvalues, mask)
            stats.update(
                espirit_thresh=used,
                support_holes_frac=holes,
                snr_std=stats["std_full"] / np.sqrt(0.5),
            )
            if not all(np.isfinite(value) for value in stats.values()):
                raise ValueError(f"Nonfinite repair statistics for slot {row.slot}.")
            if writer is not None:
                pending.add(int(row.slot))
                write_json({"slots": sorted(pending)}, journal)
                for name, values in (
                    ("mps", maps),
                    ("gt", image),
                    ("eig", eigenvalues),
                ):
                    writer.write(
                        name,
                        int(row.slot),
                        to_slot(
                            values.cpu().numpy()[None],
                            int(scan.col0),
                            int(scan.my),
                            coils if name == "mps" else None,
                        ),
                    )
                writer.sync()
            for name, value in stats.items():
                slices.at[row.slot, name] = value
            rewritten += 1
    if not dry_run:
        scans, slices = summarize(ds.scans, slices)
        stats = normalization_stats(slices) if split == "train" else None
        write_parquet(slices.reset_index(drop=True), directory / "slices.parquet")
        write_parquet(scans, directory / "scans.parquet")
        if stats is not None:
            write_json(stats, Path(root) / "stats.json")
        journal.unlink(missing_ok=True)
    print(f"{'Would rewrite' if dry_run else 'Rewrote'} {rewritten} slices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--cutoff", type=float, default=HOLE_CUTOFF)
    parser.add_argument("--thresh", type=float, default=ESPIRIT_FALLBACK_THRESH)
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute without writing"
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    add_output_argument(parser)
    args = parser.parse_args()
    if not 0 <= args.cutoff <= 1 or not 0 < args.thresh < 1:
        parser.error("--cutoff must be in [0, 1] and --thresh in (0, 1)")
    context = nullcontext() if args.dry_run else split_lock(Path(args.out) / args.split)
    with context:
        repair(
            args.out, args.split, args.cutoff, args.thresh, args.dry_run, args.device
        )


if __name__ == "__main__":
    main()
