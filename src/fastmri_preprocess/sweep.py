"""Check every stored slice and write QC tables and contact sheets."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import label

from .fastmri_brain import FastMRIBrain
from .paths import add_output_argument
from .pipeline import ESPIRIT, fft2c, head_mask, ifft2c, support_holes
from .storage import split_lock, validate_slot_coverage, write_parquet


def sweep(ds: FastMRIBrain, device, chunk=64, limit=0, slots=None):
    if chunk < 1 or limit < 0:
        raise ValueError("Chunk size must be positive and limit nonnegative.")
    rows = []
    if slots is None:
        n = limit or len(ds.slices)
        slots = ds.slices.slot.to_numpy()[:n]
    slots = np.asarray(slots)
    n = len(slots)
    if n == 0:
        raise ValueError("No slices selected for the sweep.")
    if len(np.unique(slots)) != n or not np.isin(slots, ds.slices.slot).all():
        raise ValueError("Sweep slots must be unique and present in the dataset.")
    scans = ds.scans
    for s0 in range(0, n, chunk):
        sl = slots[s0 : s0 + chunk]
        idx = sl
        K = torch.from_numpy(np.asarray(ds._arrays["ksp"][idx])).to(device)
        M = torch.from_numpy(np.asarray(ds._arrays["mps"][idx])).to(device)
        G = torch.from_numpy(np.asarray(ds._arrays["gt"][idx])).to(device)
        E = torch.from_numpy(np.asarray(ds._arrays["eig"][idx]).astype(np.float32)).to(
            device
        )
        for j, slot in enumerate(sl):
            row = ds.slices.loc[slot]
            sc = scans.iloc[int(row.scan_idx)]
            c0, w, nc = int(sc.col0), int(sc.my), int(sc.n_coils)
            k, m, g, e = K[j], M[j], G[j], E[j]
            cols = slice(c0, c0 + w)
            raw_mask = ds.pe_mask[int(row.scan_idx)]
            pad_ok = bool(
                not k[nc:].any()
                and not k[:, :, :c0].any()
                and not k[:, :, c0 + w :].any()
                and not m[nc:].any()
                and not m[:, :, :c0].any()
                and not m[:, :, c0 + w :].any()
                and not g[:, :c0].any()
                and not g[:, c0 + w :].any()
                and not e[:, :c0].any()
                and not e[:, c0 + w :].any()
                and not raw_mask[:c0].any()
                and not raw_mask[c0 + w :].any()
            )
            finite = all(bool(torch.isfinite(array).all()) for array in (k, m, g, e))
            k, m, g, e = k[:nc, :, cols], m[:nc, :, cols], g[:, cols], e[:, cols]
            mask = torch.from_numpy(raw_mask[cols].astype(bool)).to(device)
            mask_ok = bool(not k[..., ~mask].any())
            ssq = m.abs().square().sum(0)
            supp = m.abs().sum(0) > 0
            on = ssq[supp]
            norm_err = float((on - 1).abs().max()) if on.numel() else 0.0
            off_max = float(ssq[~supp].max()) if (~supp).any() else 0.0
            eig_consistent = bool(
                ((e > ESPIRIT["crop"]) == supp).float().mean() > 0.995
            )  # f16 rounding at the edge
            coil_img = ifft2c(k)
            rss = coil_img.abs().square().sum(0).sqrt()
            head = head_mask(rss, nc)
            supp_np = supp.cpu().numpy()
            holes = support_holes(rss, e, nc, support=supp_np)
            n_comp = int(label(supp_np)[1])
            A = fft2c(m * g[None]) * mask
            kn = float(k.norm())
            resid = float((A - k).norm() / kn) if kn > 0 else float("nan")
            gt_check = float(
                (g - (m.conj() * coil_img).sum(0)).norm() / max(float(g.norm()), 1e-12)
            )
            rows.append(
                dict(
                    slot=int(slot),
                    pad_ok=pad_ok,
                    finite=finite,
                    mask_ok=mask_ok,
                    norm_err=norm_err,
                    off_max=off_max,
                    eig_consistent=eig_consistent,
                    support_frac=float(supp.float().mean()),
                    n_components=n_comp,
                    head_frac=float(head.mean()),
                    holes=holes,
                    resid=resid,
                    gt_recompute_err=gt_check,
                    gt_max=float(g.abs().max()),
                    ksp_norm=kn,
                )
            )
        if (s0 // chunk) % 50 == 0:
            print(f"  {s0 + len(sl)}/{n}", flush=True)
    return pd.DataFrame(rows)


@plt.style.context("dark_background")
def contact_sheets(ds: FastMRIBrain, figdir, per_sheet=200, cols=20):
    scans = ds.scans
    sheets = []
    for start in range(0, len(scans), per_sheet):
        sub = scans.iloc[start : start + per_sheet]
        rows_n = int(np.ceil(len(sub) / cols))
        fig, ax = plt.subplots(
            2 * rows_n, cols, figsize=(cols * 1.1, 2 * rows_n * 1.15), squeeze=False
        )
        for j, sc in enumerate(sub.itertuples()):
            slot = int(sc.slot0) + int(sc.n_slices) // 2
            it = ds[slot]
            r, c = 2 * (j // cols), j % cols
            g = np.abs(it["gt"])
            ax[r, c].imshow(
                np.flipud(g), cmap="gray", vmin=0, vmax=np.percentile(g, 99.5)
            )
            ax[r, c].set_title(f"{sc.scan_idx} {sc.contrast[2:]}", fontsize=5, pad=1)
            ax[r + 1, c].imshow(
                np.flipud(np.abs(it["mps"][0])), cmap="magma", vmin=0, vmax=1
            )
        for a in ax.ravel():
            a.axis("off")
        fig.suptitle(
            f"{ds.split}: scans {sub.scan_idx.iloc[0]}-{sub.scan_idx.iloc[-1]}, "
            f"middle slice |GT| (top) and |S_0| (bottom)",
            fontsize=8,
        )
        plt.tight_layout(pad=0.2)
        path = os.path.join(figdir, f"{ds.split}_sheet_{start // per_sheet:02d}.png")
        plt.savefig(path, dpi=220)
        plt.close(fig)
        sheets.append(path)
    return sheets


@plt.style.context("dark_background")
def outlier_gallery(ds: FastMRIBrain, sw: pd.DataFrame, figdir):
    sl = ds.slices.set_index("slot")
    picks = {}
    for column, count, largest in [
        ("resid", 4, True),
        ("support_frac", 3, True),
        ("support_frac", 3, False),
        ("n_components", 3, True),
        ("holes", 3, True),
    ]:
        rows = sw.nlargest(count, column) if largest else sw.nsmallest(count, column)
        for row in rows.itertuples():
            picks.setdefault(int(row.slot), f"{column}: {getattr(row, column):.3g}")
    for slot in sl.halo_level.nlargest(3).index:
        picks.setdefault(int(slot), f"halo: {sl.loc[slot, 'halo_level']:.1f}")
    for slot in sl[sl.slice_frac < 0.8].snr_std.nsmallest(3).index:
        picks.setdefault(int(slot), f"SNR: {sl.loc[slot, 'snr_std']:.1f}")
    n = len(picks)
    if not n:
        return None
    fig, ax = plt.subplots(3, n, figsize=(2.3 * n, 7.2), squeeze=False)
    for j, (slot, name) in enumerate(picks.items()):
        it = ds[slot]
        g = np.abs(it["gt"])
        ax[0, j].imshow(np.flipud(g), cmap="gray", vmin=0, vmax=np.percentile(g, 99.5))
        ax[0, j].set_title(
            f"{name}\n{it['scan_row'].stem[11:]} sl{int(it['slice_row'].slice)}",
            fontsize=6,
        )
        ax[1, j].imshow(
            np.flipud(it["eig"].astype(np.float32)), cmap="viridis", vmin=0.8, vmax=1
        )
        ax[2, j].imshow(np.flipud(np.abs(it["mps"][0])), cmap="magma", vmin=0, vmax=1)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"{ds.split}: statistical outliers  (|GT|, eigenvalue map, |S_0|)")
    plt.tight_layout()
    path = os.path.join(figdir, f"{ds.split}_outliers.png")
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path


def hard_failures(results):
    """Separate storage or reconstruction failures from image-quality flags."""
    good = results.pad_ok & results.finite & results.mask_ok & results.eig_consistent
    good &= results.norm_err.le(1e-3) & results.off_max.eq(0)
    good &= results.gt_recompute_err.le(2e-3)
    good &= np.isfinite(results.resid)
    return results[~good]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--figdir", help="QC figures (default: <out>/qc)")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--no-figures", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit", type=int, default=0, help="Check N slots without saving"
    )
    selection.add_argument(
        "--slots-query", default="", help="Update matching rows in an existing sweep"
    )
    add_output_argument(parser)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    out_dir = os.path.join(args.out, args.split)
    with split_lock(out_dir):
        ds = FastMRIBrain(args.out, args.split)
        path = os.path.join(out_dir, "sweep.parquet")
        if args.slots_query:
            slots = ds.select(args.slots_query)
            if not len(slots):
                print("No slices match the query; existing sweep unchanged")
                return
            existing = pd.read_parquet(path)
            validate_slot_coverage(existing, ds.slices.slot)
            part = sweep(ds, torch.device(args.device), slots=slots).set_index("slot")
            results = existing.set_index("slot")
            for column in part:
                results.loc[part.index, column] = part[column]
            results = results.reset_index()
        else:
            results = sweep(ds, torch.device(args.device), limit=args.limit)
        if not args.limit:
            write_parquet(results, path)
        failures = hard_failures(results)
        print(
            f"{args.split}: checked {len(results)} slices; "
            f"{len(failures)} integrity failures"
        )
        print(f"Support holes above 0.5%: {int((results.holes > 0.005).sum())}")
        if not failures.empty:
            print(failures.head(20).to_string(index=False))
            raise SystemExit(1)
        if args.limit or args.slots_query or args.no_figures:
            return
        plt.switch_backend("Agg")
        figdir = args.figdir or os.path.join(args.out, "qc")
        os.makedirs(figdir, exist_ok=True)
        contact_sheets(ds, figdir)
        outlier_gallery(ds, results, figdir)
        print(f"Figures: {figdir}")


if __name__ == "__main__":
    main()
