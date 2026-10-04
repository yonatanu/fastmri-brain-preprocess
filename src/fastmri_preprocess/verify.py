"""Verify sampled slices through the public loader and export QC figures."""

from __future__ import annotations

import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

from .fastmri_brain import FastMRIBrain
from .paths import add_output_argument
from .pipeline import ESPIRIT_FALLBACK_THRESH


def fft2c(x):
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
    )


def check_slots(ds: FastMRIBrain, slots, tol=2e-3):
    """Check storage, map normalization, SENSE images, and forward residuals."""
    bad, stds = [], []
    for slot in slots:
        item = ds[int(slot)]
        kspace, maps, image = (item[key] for key in ("ksp", "mps", "gt"))
        scan = item["scan_row"]
        start, width, coils = int(scan.col0), int(scan.my), int(scan.n_coils)
        finite = all(np.isfinite(array[slot]).all() for array in ds._arrays.values())
        padding = True
        for name, array in ds._arrays.items():
            raw = array[slot]
            padding &= (
                not raw[..., :start].any() and not raw[..., start + width :].any()
            )
            if name in ("ksp", "mps"):
                padding &= not raw[coils:].any()
        raw_mask = ds.pe_mask[int(scan.scan_idx)]
        padding &= not raw_mask[:start].any() and not raw_mask[start + width :].any()
        mask_ok = not kspace[..., ~item["mask"]].any()
        squared_norm = (np.abs(maps) ** 2).sum(0)
        support = np.any(maps != 0, axis=0)
        norm_ok = bool(np.all(np.abs(squared_norm[support] - 1) < tol))
        coil_images = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
            axes=(-2, -1),
        )
        combined = (maps.conj() * coil_images).sum(0)
        gt_error = np.linalg.norm(image - combined) / max(np.linalg.norm(image), 1e-12)
        predicted = fft2c(maps * image[None]) * item["mask"]
        data_norm = np.linalg.norm(kspace)
        residual = (
            np.linalg.norm(predicted - kspace) / data_norm if data_norm else np.nan
        )
        residual_ok = bool(
            np.isfinite(residual)
            and abs(residual - item["slice_row"].espirit_resid) < 5e-3
        )
        normalized = ds.normalize(image, item)
        stds.append(
            np.concatenate([normalized.real.ravel(), normalized.imag.ravel()]).std()
        )
        if not (
            finite
            and padding
            and mask_ok
            and norm_ok
            and gt_error < tol
            and residual_ok
        ):
            bad.append(
                dict(
                    slot=int(slot),
                    finite=finite,
                    pad=bool(padding),
                    mask=mask_ok,
                    norm=norm_ok,
                    gt_error=float(gt_error),
                    resid=residual_ok,
                    r=float(residual),
                )
            )
    return bad, np.asarray(stds)


@plt.style.context("dark_background")
def gallery(ds, slots, path, title):
    n = len(slots)
    if n == 0:
        return
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for j, slot in enumerate(slots):
        it = ds[int(slot)]
        s, sc = it["slice_row"], it["scan_row"]
        m = np.abs(it["gt"])
        ax[0, j].imshow(np.flipud(m), cmap="gray", vmin=0, vmax=np.percentile(m, 99.5))
        ax[0, j].set_title(
            f"{sc.stem[11:]}\nsl {int(s.slice)} SNR {s.snr_std:.0f} th "
            f"{s.espirit_thresh}",
            fontsize=7,
        )
        ax[1, j].imshow(
            np.flipud(np.angle(it["gt"])), cmap="twilight", vmin=-np.pi, vmax=np.pi
        )
        for b in ds.boxes(int(slot)).itertuples():
            # Flip native-frame boxes with the displayed image.
            r0 = it["gt"].shape[0] - b.r0 - b.h
            ax[0, j].add_patch(
                plt.Rectangle((b.c0, r0), b.w, b.h, fill=False, ec="yellow", lw=0.8)
            )
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


@plt.style.context("dark_background")
def repaired_gallery(ds, slots, path):
    slots = slots[:16]
    n = len(slots)
    if n == 0:
        return
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for j, slot in enumerate(slots):
        it = ds[int(slot)]
        s = it["slice_row"]
        m = np.abs(it["gt"])
        ax[0, j].imshow(np.flipud(m), cmap="gray", vmin=0, vmax=np.percentile(m, 99.5))
        ax[0, j].set_title(
            f"{it['scan_row'].stem[11:]} sl {int(s.slice)}\nholes "
            f"{s.support_holes_frac:.2%}",
            fontsize=7,
        )
        ax[1, j].imshow(
            np.flipud(it["eig"].astype(np.float32)), cmap="viridis", vmin=0.8, vmax=1
        )
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(
        f"slices repaired with thresh {ESPIRIT_FALLBACK_THRESH}: |GT| and eigenvalue "
        f"map"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figdir", help="QC figures (default: <out>/qc)")
    parser.add_argument("--no-figures", action="store_true")
    add_output_argument(parser)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    ds = FastMRIBrain(args.out, args.split)
    rng = np.random.default_rng(args.seed)
    slots = rng.choice(ds.slices.slot.to_numpy(), min(args.n, len(ds)), replace=False)
    bad, stds = check_slots(ds, slots)
    print(f"{args.split}: checked {len(slots)} slices; {len(bad)} failures")
    for failure in bad[:20]:
        print(failure)
    low, high = np.percentile(stds, [5, 95])
    print(
        f"Normalized image std: median {np.median(stds):.3f}, "
        f"5–95% {low:.3f}–{high:.3f}"
    )
    if bad:
        raise SystemExit(1)
    if args.no_figures:
        return
    plt.switch_backend("Agg")
    figdir = args.figdir or os.path.join(args.out, "qc")
    os.makedirs(figdir, exist_ok=True)
    good = ds.slices[ds.slices.snr_std > 5]
    selected = []
    for _, group in good.groupby("contrast"):
        selected.extend(
            rng.choice(group.slot.to_numpy(), min(3, len(group)), replace=False)
        )
    gallery(ds, selected, os.path.join(figdir, f"{args.split}_gallery.png"), args.split)
    fallback = ds.slices[np.isclose(ds.slices.espirit_thresh, ESPIRIT_FALLBACK_THRESH)]
    repaired_gallery(
        ds, fallback.slot.to_numpy(), os.path.join(figdir, f"{args.split}_repaired.png")
    )
    print(f"Figures: {figdir}")


if __name__ == "__main__":
    main()
