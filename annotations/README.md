# fastMRI+ annotations

| File | Contents |
|---|---|
| `brain.csv` | Brain bounding boxes and study-level labels |
| `brain_file_list.csv` | Reviewed brain scans |
| `knee.csv` | Knee bounding boxes and study-level labels |
| `knee_file_list.csv` | Reviewed knee scans |

The brain pipeline reads the two brain CSVs:

```bash
python -m fastmri_preprocess.finalize --split train --out /path/to/processed \
    --annotations /path/to/annotations
```

Finalization maps boxes to the processed image grid and writes
`plus_boxes.parquet`, `plus_study.parquet`, and annotation fields in the scan and
slice tables. `plus_reviewed` identifies reviewed scans; missing labels on an
unreviewed scan do not indicate normal anatomy.
