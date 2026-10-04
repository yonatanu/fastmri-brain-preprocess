"""fastMRI+ geometry and reviewed/unreviewed semantics."""

import numpy as np
import pandas as pd

from fastmri_preprocess.finalize import fastmri_plus
from fastmri_preprocess.pipeline import map_box


def test_annotations_use_requested_source_and_preserve_review_status(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    pd.DataFrame(
        [
            dict(
                file="positive",
                label="Possible artifact",
                study_level="No",
                slice=0,
                x=80,
                y=40,
                width=30,
                height=40,
            ),
            dict(
                file="positive",
                label="Mass",
                study_level="Yes",
                slice=0,
                x=0,
                y=0,
                width=0,
                height=0,
            ),
        ]
    ).to_csv(annotations / "brain.csv", index=False)
    (annotations / "brain_file_list.csv").write_text("positive\nnegative\n")
    scans = pd.DataFrame(
        dict(
            stem=["positive", "negative", "unreviewed"],
            scan_idx=[0, 1, 2],
            n_slices=[1, 1, 1],
            slot0=[0, 1, 2],
            rss_h=[220] * 3,
            rss_w=[220] * 3,
            rec_fov_x=[220] * 3,
            rec_fov_y=[220] * 3,
            my=[180] * 3,
            col0=[20] * 3,
        )
    )
    slices = pd.DataFrame(dict(slot=[0, 1, 2], stem=scans.stem))
    sc, sl, boxes, study = fastmri_plus(scans, slices, annotations)
    assert sc.plus_reviewed.tolist() == [True, True, False]
    assert sl.plus_reviewed.tolist() == [True, True, False]
    assert sl.n_plus_boxes.tolist() == [1, 0, 0]
    assert sl.plus_artifact_boxes.tolist() == [1, 0, 0]
    assert sc.n_plus_study_labels.tolist() == [1, 0, 0]
    np.testing.assert_allclose(
        boxes.iloc[0][["r0", "c0", "h", "w", "c0_slot"]].astype(float),
        [140, 60, 40, 30, 80],
    )
    assert study.label.tolist() == ["Mass"]
    _, unreviewed, empty_boxes, empty_study = fastmri_plus(
        scans.iloc[2:], slices.iloc[2:].copy(), annotations
    )
    assert empty_boxes.empty and empty_study.empty
    assert not unreviewed.plus_reviewed.any()


def test_box_mapping_uses_actual_pixel_spacing():
    box = map_box(80, 40, 30, 40, (220, 220), (220, 220), (220, 180), voxel=(1, 1.25))
    np.testing.assert_allclose(box, [140, 66, 40, 24])
