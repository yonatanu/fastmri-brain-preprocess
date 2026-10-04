"""Storage failures, interrupted logs, and resumable processing."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from fastmri_preprocess.run import (
    ScanReader,
    pending_scans,
    run_configuration,
    validate_configuration,
    wait_for_workers,
)
from fastmri_preprocess.storage import (
    SlotWriter,
    atomic_output,
    preallocate,
    progress_log,
    read_qc_records,
    split_lock,
)


def record(stem="scan", error="", timestamp=1):
    return dict(
        stem=stem,
        scan_idx=0,
        slot0=0,
        error=error,
        slices={"scale": [1.0]},
        completed_at_ns=timestamp,
    )


def test_writer_retries_short_writes_and_closes_files(tmp_path, monkeypatch):
    preallocate(tmp_path, 1, 1)
    import os

    original = os.pwrite
    monkeypatch.setattr(
        os, "pwrite", lambda fd, data, offset: original(fd, data[:31], offset)
    )
    with SlotWriter(tmp_path, names=["pe_mask"]) as writer:
        writer.write_mask(0, np.ones(220))
        with pytest.raises(ValueError, match="exceeds"):
            writer.write_mask(1, np.ones(220))
    assert not writer.fds
    assert (tmp_path / "pe_mask.bin").read_bytes() == bytes([1]) * 220


def test_atomic_output_keeps_previous_file_on_error(tmp_path):
    path = tmp_path / "table.json"
    path.write_text("previous")
    with pytest.raises(RuntimeError):
        with atomic_output(path) as temporary:
            temporary.write_text("incomplete")
            raise RuntimeError("interrupted")
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("value", [0.5, 256, -1, np.nan])
def test_writer_rejects_invalid_masks_before_casting(tmp_path, value):
    preallocate(tmp_path, 1, 1)
    with SlotWriter(tmp_path, names=["pe_mask"]) as writer:
        with pytest.raises(ValueError, match="binary values"):
            writer.write_mask(0, np.full(220, value))
    assert (tmp_path / "pe_mask.bin").read_bytes() == bytes(220)


def test_split_lock_rejects_second_writer(tmp_path):
    with split_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another writer"):
            with split_lock(tmp_path):
                pass
    with split_lock(tmp_path):
        pass


def test_logs_use_latest_record_across_workers_and_retry_errors(tmp_path):
    (tmp_path / "qc_rank9.jsonl").write_text(json.dumps(record(timestamp=1)) + "\n")
    (tmp_path / "qc_rank0.jsonl").write_text(
        json.dumps(record(error="failed", timestamp=2)) + "\n"
    )
    scans = pd.DataFrame([dict(stem="scan", scan_idx=0, slot0=0, n_slices=1)])
    latest = read_qc_records(tmp_path)
    assert latest[0]["error"] == "failed"
    assert len(pending_scans(scans, latest, tmp_path)) == 1
    (tmp_path / "noise").mkdir()
    (tmp_path / "noise/scan.npz").touch()
    assert pending_scans(scans, [record()], tmp_path).empty


def test_interrupted_log_tail_is_repaired_before_append(tmp_path):
    path = tmp_path / "qc_rank0.jsonl"
    path.write_text(json.dumps(record()) + '\n{"stem":')
    with pytest.warns(UserWarning, match="interrupted"):
        assert len(read_qc_records(tmp_path)) == 1
    with progress_log(path) as stream:
        stream.write((json.dumps(record(error="retry", timestamp=2)) + "\n").encode())
    assert read_qc_records(tmp_path)[0]["error"] == "retry"


def test_log_corruption_inside_file_is_not_ignored(tmp_path):
    (tmp_path / "qc_rank0.jsonl").write_text("invalid\n" + json.dumps(record()))
    with pytest.raises(ValueError, match="Invalid progress"):
        read_qc_records(tmp_path)


def test_reader_handles_missing_files_and_complex128(tmp_path):
    path = tmp_path / "raw.h5"
    values = np.full((1, 2, 8, 8), 1 + 2j, dtype=np.complex128)
    with h5py.File(path, "w") as stream:
        stream["kspace"] = values
    reader = ScanReader([path, tmp_path / "missing.h5"])
    _, tensor, error = reader[0]
    assert not error
    np.testing.assert_array_equal(tensor.numpy()[..., 0], 1)
    np.testing.assert_array_equal(tensor.numpy()[..., 1], 2)
    assert reader[1][1] is None and reader[1][2]


def test_worker_failure_is_propagated():
    failed = SimpleNamespace(pid=123, exitcode=1, join=lambda: None)
    with pytest.raises(RuntimeError, match="workers failed"):
        wait_for_workers([failed])


def test_resume_rejects_changed_configuration(tmp_path):
    config = run_configuration(seed=42)
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(config))
    validate_configuration(path, run_configuration(seed=42))
    with pytest.raises(ValueError, match="configuration changed"):
        validate_configuration(path, run_configuration(seed=43))
