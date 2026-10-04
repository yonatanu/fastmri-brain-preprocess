"""Process scans in parallel and record resumable progress per worker."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from . import pipeline
from .fastmri_brain import MAX_COILS, N_OUT
from .paths import add_output_argument
from .pipeline import Header, preprocess
from .storage import (
    SlotWriter,
    atomic_output,
    preallocate,
    progress_log,
    read_qc_records,
    split_lock,
    validate_scan_layout,
    validate_storage,
    write_json,
)


def run_configuration(seed):
    names = (
        "VOXEL_MM",
        "FOV_MM",
        "MAX_COILS",
        "MIN_FIELD_T",
        "NOISE_BAND_MM",
        "NOISE_TIP_MM",
        "NOISE_SLICE_TOL",
        "ESPIRIT",
        "ESPIRIT_FALLBACK_THRESH",
        "HOLE_CUTOFF",
        "HEAD_REL",
        "HEAD_ERODE_PX",
        "HEAD_MIN_FRAC",
        "SCALE_LINES",
        "OBJECT_SNR",
    )
    parameters = {}
    for name in names:
        value = getattr(pipeline, name)
        parameters[name] = list(value) if isinstance(value, tuple) else value
    return {
        "seed": seed,
        "parameters": parameters,
        "versions": {name: version(name) for name in ("numpy", "torch", "sigpy")},
    }


def validate_configuration(path, configuration):
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError(
            "Processing configuration changed; use a new output directory."
        )


def header_from_row(row) -> Header:
    values = {key: row[key] for key in Header.__dataclass_fields__ if key in row}
    for key in ("enc_mat", "enc_fov", "rec_mat", "rec_fov"):
        values[key] = (row[f"{key}_x"], row[f"{key}_y"])
    return Header(**values)


class ScanReader(Dataset):
    """Read a scan or return its read error for the progress log."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            with h5py.File(self.paths[index], "r") as stream:
                raw = stream["kspace"][()]
            if raw.ndim != 4 or not np.iscomplexobj(raw):
                raise ValueError("kspace must be a complex [slice, coil, x, y] array")
            raw = np.ascontiguousarray(raw, dtype=np.complex64)
            # Real views can be passed through DataLoader shared memory.
            tensor = torch.from_numpy(raw.view(np.float32).reshape(*raw.shape, 2))
            return index, tensor, ""
        except Exception as exc:
            return index, None, f"{type(exc).__name__}: {exc}"


def to_slot(arr, col0, my, n_coils=None):
    """Place native arrays into zero-padded storage slots."""
    if col0 < 0 or my <= 0 or col0 + my > N_OUT:
        raise ValueError("Invalid phase columns for storage slot.")
    shape = (N_OUT, my) if n_coils is None else (n_coils, N_OUT, my)
    if arr.shape[1:] != shape:
        raise ValueError(f"Expected native shape {shape}, got {arr.shape[1:]}.")
    if n_coils is not None and not 1 <= n_coils <= MAX_COILS:
        raise ValueError("Invalid coil count for storage slot.")
    if n_coils is None:
        result = np.zeros((len(arr), N_OUT, N_OUT), dtype=arr.dtype)
        result[:, :, col0 : col0 + my] = arr
    else:
        result = np.zeros((len(arr), MAX_COILS, N_OUT, N_OUT), dtype=arr.dtype)
        result[:, :n_coils, :, col0 : col0 + my] = arr
    return result


def pending_scans(scans, records, out_dir):
    """Resume successful scans regardless of their previous worker assignment."""
    completed = set()
    by_stem = scans.set_index("stem")
    for record in records:
        if record["stem"] not in by_stem.index:
            raise ValueError(f"Progress log contains an unknown scan: {record['stem']}")
        row = by_stem.loc[record["stem"]]
        if record["scan_idx"] != row.scan_idx or record["slot0"] != row.slot0:
            raise ValueError("Progress logs do not match the survey slot layout.")
        if record["error"]:
            continue
        stats = record.get("slices", {})
        if len(stats.get("scale", [])) != row.n_slices:
            raise ValueError(f"Incomplete successful record for {row.name}.")
        if (Path(out_dir) / "noise" / f"{row.name}.npz").is_file():
            completed.add(row.name)
    return scans[~scans.stem.isin(completed)].reset_index(drop=True)


def worker(rank, gpu, shard: pd.DataFrame, out_dir: str, readers: int, seed=0):
    import cupy

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    noise_dir = Path(out_dir) / "noise"
    noise_dir.mkdir(exist_ok=True)
    options = {}
    if readers:
        options.update(prefetch_factor=2, multiprocessing_context="spawn")
    loader = DataLoader(
        ScanReader(shard.path.tolist()), batch_size=None, num_workers=readers, **options
    )
    failures = 0
    with (
        SlotWriter(out_dir) as writer,
        progress_log(Path(out_dir) / f"qc_rank{rank}.jsonl") as log,
    ):
        previous = time.monotonic()
        for index, raw, read_error in loader:
            row = shard.iloc[int(index)]
            scan_seed = (seed + int(row.scan_idx)) % (2**32)
            np.random.seed(scan_seed)
            torch.manual_seed(scan_seed)
            with cupy.cuda.Device(gpu):
                cupy.random.seed(scan_seed)
            started = time.monotonic()
            record = dict(
                stem=row.stem,
                scan_idx=int(row.scan_idx),
                slot0=int(row.slot0),
                error="",
                seed=scan_seed,
            )
            output = kspace = None
            try:
                if read_error:
                    raise OSError(read_error)
                kspace = torch.view_as_complex(raw).to(device)
                output = preprocess(kspace, header_from_row(row), device_id=gpu)
                geometry = output["geom"]
                coils = output["ksp"].shape[1]
                if (
                    geometry.my != row.my
                    or geometry.col0 != row.col0
                    or coils != row.n_coils
                    or len(output["ksp"]) != row.n_slices
                ):
                    raise ValueError("Processed geometry differs from the survey.")
                for name in ("ksp", "mps", "gt", "eig"):
                    values = output[name].cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite {name} for {row.stem}.")
                    writer.write(
                        name,
                        int(row.slot0),
                        to_slot(
                            values,
                            geometry.col0,
                            geometry.my,
                            coils if name in ("ksp", "mps") else None,
                        ),
                    )
                mask = np.zeros(N_OUT, dtype=np.uint8)
                mask[geometry.col0 : geometry.col0 + geometry.my] = (
                    output["mask"].cpu().numpy()
                )
                writer.write_mask(int(row.scan_idx), mask)
                with atomic_output(noise_dir / f"{row.stem}.npz") as temporary:
                    np.savez(
                        temporary,
                        cov=output["cov"],
                        W=output["W"],
                        U=(
                            np.empty((0, 0), np.complex64)
                            if output["U"] is None
                            else output["U"]
                        ),
                    )
                writer.sync()
                record.update(output["scan_qc"])
                record["slices"] = {
                    key: value.tolist() for key, value in output["slice_stats"].items()
                }
            except Exception as exc:
                failures += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                del output, kspace
            record.update(
                t_read=started - previous,
                t_proc=time.monotonic() - started,
                completed_at_ns=time.time_ns(),
            )
            log.write((json.dumps(record) + "\n").encode())
            log.flush()
            os.fsync(log.fileno())
            status = record["error"] or "done"
            print(f"[worker {rank}] {row.stem}: {status}", flush=True)
            previous = time.monotonic()
    if failures:
        raise RuntimeError(f"Worker {rank}: {failures} scans failed; rerun to retry.")


def wait_for_workers(processes):
    for process in processes:
        process.join()
    failed = [process.pid for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"Preprocessing workers failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--gpus", default="0", help="Visible GPU indices, e.g. 0,1")
    parser.add_argument("--readers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process the first N scans"
    )
    add_output_argument(parser)
    args = parser.parse_args()
    if args.readers < 0 or args.limit < 0:
        parser.error("--readers and --limit must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must be a comma-separated list of integers")
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        parser.error("GPU indices must be distinct and nonnegative")
    out_dir = Path(args.out) / args.split
    with split_lock(out_dir):
        scans = pd.read_parquet(out_dir / "scans_geom.parquet")
        scans = scans[scans.selected].sort_values("scan_idx").reset_index(drop=True)
        validate_scan_layout(scans)
        configuration = run_configuration(args.seed)
        config_path = out_dir / "run_config.json"
        validate_configuration(config_path, configuration)
        records = read_qc_records(out_dir)
        if records:
            validate_storage(out_dir, int(scans.n_slices.sum()), len(scans))
        todo = pending_scans(scans, records, out_dir)
        if args.limit:
            todo = todo[todo.scan_idx < args.limit]
        if todo.empty:
            print(f"{args.split}: all requested scans are complete")
            return
        if records and not config_path.exists():
            raise ValueError(
                "Legacy partial output has no run configuration; use a new directory."
            )
        if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
            parser.error(
                "requested GPUs are not available through CUDA_VISIBLE_DEVICES"
            )
        preallocate(out_dir, int(scans.n_slices.sum()), len(scans))
        if not config_path.exists():
            write_json(configuration, config_path)
        context = mp.get_context("spawn")
        processes = []
        try:
            for rank, gpu in enumerate(gpus):
                shard = todo.iloc[rank :: len(gpus)].reset_index(drop=True)
                if shard.empty:
                    continue
                process = context.Process(
                    target=worker,
                    args=(rank, gpu, shard, str(out_dir), args.readers, args.seed),
                )
                process.start()
                processes.append(process)
            wait_for_workers(processes)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
    print(f"{args.split}: preprocessing complete")


if __name__ == "__main__":
    main()
