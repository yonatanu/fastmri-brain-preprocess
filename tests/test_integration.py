"""CPU processing through storage, finalization, QC, and repair."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch
from test_pipeline import header

from fastmri_preprocess import FastMRIBrain
from fastmri_preprocess.finalize import finalize
from fastmri_preprocess.merge_sweep import merge_sweep
from fastmri_preprocess.pipeline import preprocess
from fastmri_preprocess.repair_holes import repair
from fastmri_preprocess.run import to_slot
from fastmri_preprocess.storage import SlotWriter, preallocate
from fastmri_preprocess.sweep import hard_failures, sweep
from fastmri_preprocess.verify import check_slots, gallery, repaired_gallery


@pytest.fixture
def processed_root(tmp_path):
    np.random.seed(42)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    x, y = torch.arange(480) - 240, torch.arange(240) - 120
    image = 80 * torch.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / (2 * 30**2))
    from fastmri_preprocess.pipeline import fft2c

    coils = torch.stack([image, image * (0.6 + 0.8j)])[None] / np.sqrt(2)
    raw = fft2c(coils) + torch.randn(
        coils.shape, dtype=torch.complex64, generator=generator
    )
    output = preprocess(raw, header(), device_id=-1)
    directory = tmp_path / "train"
    directory.mkdir()
    stem = "file_brain_AXT2_200_1"
    metadata = asdict(header())
    for key in ("enc_mat", "enc_fov", "rec_mat", "rec_fov"):
        values = metadata.pop(key)
        metadata[f"{key}_x"], metadata[f"{key}_y"] = values
    metadata.update(asdict(output["geom"]))
    metadata.update(
        stem=stem,
        scan_idx=0,
        slot0=0,
        n_slices=1,
        n_coils=2,
        n_coils_native=2,
        selected=True,
        error="",
        contrast="AXT2",
        scanner_code=200,
        rss_h=240,
        rss_w=240,
        zero_filled=False,
    )
    pd.DataFrame([metadata]).to_parquet(directory / "scans_geom.parquet")
    preallocate(directory, 1, 1)
    with SlotWriter(directory) as writer:
        for name in ("ksp", "mps", "gt", "eig"):
            writer.write(
                name,
                0,
                to_slot(
                    output[name].numpy(), 0, 220, 2 if name in ("ksp", "mps") else None
                ),
            )
        writer.write_mask(0, output["mask"].numpy())
    (directory / "noise").mkdir()
    np.savez(
        directory / "noise" / f"{stem}.npz",
        cov=output["cov"],
        W=output["W"],
        U=np.empty((0, 0), dtype=np.complex64),
    )
    record = dict(
        output["scan_qc"],
        stem=stem,
        scan_idx=0,
        slot0=0,
        error="",
        slices={key: value.tolist() for key, value in output["slice_stats"].items()},
    )
    (directory / "qc_rank0.jsonl").write_text(json.dumps(record) + "\n")
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "brain.csv").write_text(
        "file,slice,study_level,x,y,width,height,label\n"
    )
    (annotations / "brain_file_list.csv").write_text("other_scan\n")
    finalize(directory, annotations, "train")
    return tmp_path


def test_cpu_pipeline_storage_and_qc(processed_root):
    ds = FastMRIBrain(processed_root, "train")
    assert not check_slots(ds, [0])[0]
    results = sweep(ds, torch.device("cpu"), slots=[0])
    assert hard_failures(results).empty
    merged = merge_sweep(ds.slices, results)
    assert np.isfinite(merged.sweep_resid).all()
    with pytest.raises(ValueError, match="exactly once"):
        merge_sweep(ds.slices, results.iloc[:0])
    with pytest.raises(FileExistsError):
        finalize(processed_root / "train", processed_root / "annotations", "train")


def test_verifier_and_sweep_detect_corrupted_reference(processed_root, monkeypatch):
    path = processed_root / "train/gt.bin"
    image = np.memmap(path, dtype=np.complex64, mode="r+", shape=(1, 220, 220))
    image[0] *= 2
    image.flush()
    ds = FastMRIBrain(processed_root, "train")
    assert check_slots(ds, [0])[0]
    results = sweep(ds, torch.device("cpu"), slots=[0])
    assert not hard_failures(results).empty
    from fastmri_preprocess import verify

    monkeypatch.setattr(
        "sys.argv",
        ["verify", "--split", "train", "--out", str(processed_root), "--no-figures"],
    )
    with pytest.raises(SystemExit) as error:
        verify.main()
    assert error.value.code == 1


def test_single_slice_galleries(processed_root):
    ds = FastMRIBrain(processed_root, "train")
    gallery(ds, [], processed_root / "empty.png", "empty")
    assert not (processed_root / "empty.png").exists()
    gallery(ds, [0], processed_root / "one.png", "single slice")
    repaired_gallery(ds, [0], processed_root / "repair.png")
    assert (processed_root / "one.png").stat().st_size > 0
    assert (processed_root / "repair.png").stat().st_size > 0


def test_sweep_detects_mask_padding(dataset_root):
    ds = FastMRIBrain(dataset_root, "train")
    assert sweep(ds, "cpu", slots=[1]).pad_ok.all()
    mask = np.memmap(
        dataset_root / "train/pe_mask.bin", dtype=np.uint8, mode="r+", shape=(2, 220)
    )
    mask[1, 0] = 1
    mask.flush()
    assert not sweep(ds, "cpu", slots=[1]).pad_ok.any()


def prepare_repair(root, monkeypatch):
    path = root / "train/slices.parquet"
    slices = pd.read_parquet(path)
    slices["espirit_thresh"] = 0.02
    slices.to_parquet(path)
    item = FastMRIBrain(root, "train")[0]
    maps = torch.tensor(np.asarray(item["mps"])) * np.exp(0.2j)
    eigenvalues = torch.tensor(item["eig"].astype(np.float32))
    monkeypatch.setattr(
        "fastmri_preprocess.repair_holes.espirit_maps",
        lambda *args, **kwargs: (maps, eigenvalues),
    )


def fingerprint(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_repair_dry_run_never_opens_writer_or_changes_files(
    processed_root, monkeypatch
):
    prepare_repair(processed_root, monkeypatch)

    def reject_writer(*args, **kwargs):
        raise AssertionError("dry-run opened a writer")

    monkeypatch.setattr("fastmri_preprocess.repair_holes.SlotWriter", reject_writer)
    before = fingerprint(processed_root)
    repair(processed_root, "train", dry_run=True, device="cpu")
    assert fingerprint(processed_root) == before


def test_repair_refreshes_statistics_and_resumes_journal(processed_root, monkeypatch):
    prepare_repair(processed_root, monkeypatch)
    directory = processed_root / "train"
    (directory / "repair_pending.json").write_text(json.dumps({"slots": [0]}))
    kspace_before = (directory / "ksp.bin").read_bytes()
    repair(processed_root, "train", device="cpu")
    ds = FastMRIBrain(processed_root, "train")
    assert not check_slots(ds, [0])[0]
    assert ds.slices.espirit_thresh.iloc[0] == 0.05
    assert ds.scans.snr_median.iloc[0] == ds.slices.snr_std.iloc[0]
    expected = 0.5 / (ds.slices.std_full / ds.slices.scale).median()
    assert ds.norm_const == expected
    assert not (directory / "repair_pending.json").exists()
    assert (directory / "ksp.bin").read_bytes() == kspace_before


def test_repair_stores_new_maps_even_when_threshold_is_unchanged(
    processed_root, monkeypatch
):
    prepare_repair(processed_root, monkeypatch)
    path = processed_root / "train/slices.parquet"
    slices = pd.read_parquet(path)
    slices["espirit_thresh"] = 0.05
    slices["support_holes_frac"] = 0.1
    slices.to_parquet(path)
    before = FastMRIBrain(processed_root, "train")[0]["mps"].copy()
    repair(processed_root, "train", device="cpu")
    ds = FastMRIBrain(processed_root, "train")
    assert ds.slices.espirit_thresh.iloc[0] == 0.05
    np.testing.assert_allclose(ds[0]["mps"], before * np.exp(0.2j), atol=1e-7)
    assert not check_slots(ds, [0])[0]
