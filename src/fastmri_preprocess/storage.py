"""Binary storage, progress logs, and atomic metadata writes."""

import fcntl
import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .fastmri_brain import LAYOUT, MAX_COILS, N_OUT


def validate_scan_layout(scans):
    """Validate the contiguous scan and slice ranges used by the binary files."""
    required = {"stem", "scan_idx", "slot0", "n_slices", "n_coils", "my", "col0"}
    if scans.empty or not required <= set(scans.columns):
        raise ValueError("The survey has no selected scans or lacks storage columns.")
    numeric = scans[list(required - {"stem"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not (numeric == np.floor(numeric)).all():
        raise ValueError("Storage dimensions and indices must be finite integers.")
    if scans.stem.duplicated().any() or scans.stem.isna().any():
        raise ValueError("Scan names must be present and unique.")
    if not np.array_equal(scans.scan_idx, np.arange(len(scans))):
        raise ValueError("Scan indices must be contiguous and start at zero.")
    offsets = scans.n_slices.cumsum().shift(fill_value=0)
    if not np.array_equal(scans.slot0, offsets) or (scans.n_slices <= 0).any():
        raise ValueError("Slice slots must be contiguous and nonempty per scan.")
    if (
        not scans.n_coils.between(1, MAX_COILS).all()
        or not scans.my.between(1, N_OUT).all()
    ):
        raise ValueError("Coil counts or phase widths exceed the storage layout.")
    if not (scans.col0 == N_OUT // 2 - scans.my // 2).all():
        raise ValueError("Native phase columns must be centered in storage slots.")


def binary_path(out_dir, name):
    return os.path.join(out_dir, f"{name}.bin")


def validate_slot_coverage(table, expected):
    if table.slot.duplicated().any() or not np.array_equal(
        np.sort(table.slot.to_numpy()), np.sort(np.asarray(expected))
    ):
        raise ValueError("QC rows must cover each dataset slot exactly once.")


def storage_sizes(n_slices, n_scans):
    if n_slices <= 0 or n_scans <= 0:
        raise ValueError("Storage requires at least one scan and one slice.")
    sizes = {
        name: n_slices * int(np.prod(shape)) * np.dtype(dtype).itemsize
        for name, (shape, dtype) in LAYOUT.items()
    }
    sizes["pe_mask"] = n_scans * N_OUT
    return sizes


def validate_storage(out_dir, n_slices, n_scans):
    for name, size in storage_sizes(n_slices, n_scans).items():
        path = Path(binary_path(out_dir, name))
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or incorrectly sized binary: {path}")


def preallocate(out_dir, n_slices, n_scans):
    """Create a new set of binaries, or validate a complete existing set."""
    sizes = storage_sizes(n_slices, n_scans)
    paths = {name: Path(binary_path(out_dir, name)) for name in sizes}
    if any(path.exists() for path in paths.values()):
        try:
            validate_storage(out_dir, n_slices, n_scans)
        except ValueError as exc:
            raise ValueError(
                "Refusing to resize or replace incomplete existing storage. "
                "Use a separate output directory."
            ) from exc
        return
    for name, path in paths.items():
        with path.open("xb") as stream:
            stream.truncate(sizes[name])


class SlotWriter:
    """Write complete arrays into preallocated storage slots."""

    def __init__(self, out_dir, names=None):
        self.fds = {}
        try:
            for name in names if names is not None else [*LAYOUT, "pe_mask"]:
                self.fds[name] = os.open(binary_path(out_dir, name), os.O_WRONLY)
        except BaseException:
            self.close()
            raise

    def _write(self, name, offset, array):
        data = memoryview(array).cast("B")
        fd = self.fds[name]
        if offset < 0 or offset + len(data) > os.fstat(fd).st_size:
            raise ValueError(f"Write exceeds the allocated {name} storage.")
        while data:
            written = os.pwrite(fd, data, offset)
            if written <= 0:
                raise OSError(f"No progress writing {name}.")
            offset += written
            data = data[written:]

    def write(self, name, slot0, arr):
        shape, dtype = LAYOUT[name]
        arr = np.ascontiguousarray(arr, dtype=dtype)
        if arr.ndim != len(shape) + 1 or arr.shape[1:] != shape:
            raise ValueError(f"Invalid {name} shape: {arr.shape}")
        offset = slot0 * int(np.prod(shape)) * np.dtype(dtype).itemsize
        self._write(name, offset, arr)

    def write_mask(self, scan_idx, mask):
        mask = np.asarray(mask)
        if mask.shape != (N_OUT,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("The acquisition mask must contain 220 binary values.")
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        self._write("pe_mask", scan_idx * N_OUT, mask)

    def sync(self):
        for fd in self.fds.values():
            os.fsync(fd)

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@contextmanager
def split_lock(out_dir):
    """Prevent concurrent writers from modifying the same split."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with (Path(out_dir) / ".write.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer is using {out_dir}.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def atomic_output(path):
    """Replace one output only after its temporary file has been written."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        yield Path(temporary)
        permissions = path.stat().st_mode & 0o777 if path.exists() else 0o644
        os.chmod(temporary, permissions)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_parquet(table, path):
    with atomic_output(path) as temporary:
        table.to_parquet(temporary, index=False)


def write_json(value, path):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_qc_records(out_dir):
    """Read the latest record per scan across all workers, including failures."""
    records = []
    for path in sorted(Path(out_dir).glob("qc_rank*.jsonl")):
        modified_at = path.stat().st_mtime_ns
        lines = path.read_bytes().splitlines(keepends=True)
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if number == len(lines) and not line.endswith(b"\n"):
                    warnings.warn(f"Ignoring interrupted final record in {path}.")
                    continue
                raise ValueError(f"Invalid progress record: {path}:{number}") from exc
            required = {"stem", "scan_idx", "slot0", "error"}
            if not isinstance(record, dict) or not required <= record.keys():
                raise ValueError(f"Incomplete progress record: {path}:{number}")
            timestamp = record.get("completed_at_ns", modified_at)
            records.append((timestamp, str(path), number, record))
    latest = {}
    for _, _, _, record in sorted(records, key=lambda row: row[:3]):
        latest[record["stem"]] = record
    return list(latest.values())


@contextmanager
def progress_log(path):
    """Open an append log, discarding an interrupted final JSON fragment."""
    with open(path, "a+b") as stream:
        stream.seek(0)
        data = stream.read()
        if data and not data.endswith(b"\n"):
            start = data.rfind(b"\n") + 1
            try:
                json.loads(data[start:])
            except json.JSONDecodeError:
                stream.truncate(start)
            else:
                stream.write(b"\n")
        stream.seek(0, os.SEEK_END)
        yield stream
