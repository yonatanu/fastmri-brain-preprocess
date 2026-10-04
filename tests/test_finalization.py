"""Reject incomplete metadata before publishing dataset tables."""

import numpy as np
import pandas as pd
import pytest

from fastmri_preprocess.finalize import pack_noise, validate_qc
from fastmri_preprocess.survey import survey_one, write_survey


def complete_record():
    keys = [
        "scale",
        "std_full",
        "std_real",
        "std_imag",
        "gt_p99",
        "gt_max",
        "support_frac",
        "espirit_resid",
        "band_energy_frac",
    ]
    return dict(
        stem="scan",
        scan_idx=0,
        slot0=0,
        error="",
        slices={key: [1.0, 2.0] for key in keys},
    )


@pytest.mark.parametrize(
    "problem", ["missing", "failed", "short", "nonfinite", "layout"]
)
def test_finalization_rejects_incomplete_records(problem):
    scans = pd.DataFrame([dict(stem="scan", scan_idx=0, slot0=0, n_slices=2)])
    record = complete_record()
    if problem == "failed":
        record["error"] = "read failed"
    elif problem == "short":
        record["slices"]["std_full"] = [1.0]
    elif problem == "nonfinite":
        record["slices"]["scale"] = [1.0, np.nan]
    elif problem == "layout":
        record["slot0"] = 5
    qc = pd.DataFrame([record])
    if problem == "missing":
        qc = qc.iloc[:0]
    with pytest.raises(ValueError):
        validate_qc(scans, qc)


def test_missing_noise_file_preserves_existing_archive(tmp_path):
    target = tmp_path / "noise.npz"
    target.write_bytes(b"previous archive")
    with pytest.raises(FileNotFoundError):
        pack_noise(tmp_path, pd.DataFrame(dict(stem=["missing"])))
    assert target.read_bytes() == b"previous archive"


def test_noise_matrices_must_match_scan_coil_counts(tmp_path):
    (tmp_path / "noise").mkdir()
    np.savez(tmp_path / "noise/scan.npz", cov=np.eye(4), W=np.eye(4), U=np.eye(4))
    scans = pd.DataFrame([dict(stem="scan", n_coils_native=4, n_coils=2)])
    with pytest.raises(ValueError, match="Invalid noise matrices"):
        pack_noise(tmp_path, scans)
    assert not (tmp_path / "noise.npz").exists()


def test_survey_reports_malformed_filename(tmp_path):
    result = survey_one(str(tmp_path / "bad.h5"))
    assert result["error"]
    assert np.isnan(result["field_T"])


def test_survey_cannot_reassign_existing_slots(tmp_path):
    table = pd.DataFrame(
        [
            dict(
                stem="scan",
                scan_idx=0,
                slot0=0,
                n_slices=2,
                n_coils=2,
                my=220,
                col0=0,
                selected=True,
            )
        ]
    )
    write_survey(table, tmp_path)
    original = (tmp_path / "scans_geom.parquet").read_bytes()
    (tmp_path / "gt.bin").touch()
    write_survey(table.copy(), tmp_path)
    table.loc[0, "stem"] = "different_scan"
    with pytest.raises(ValueError, match="survey changed"):
        write_survey(table, tmp_path)
    assert (tmp_path / "scans_geom.parquet").read_bytes() == original
