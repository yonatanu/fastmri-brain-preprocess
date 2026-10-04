# fastMRI preprocessing

Preprocess 3 T fastMRI brain scans into whitened, coil-compressed k-space,
ESPIRiT maps, SENSE reference images, and QC tables. Map fastMRI+ boxes and
study labels onto the output grid. The target is 1 mm in-plane resolution and
220 mm FOV; slice thickness and spacing are unchanged.

## Installation

Run from the repository root:

```bash
mamba env create -f environment.yml
mamba activate fastmri-preprocess
```

The environment installs the package in editable mode. GPU preprocessing uses
PyTorch and CuPy and requires a compatible NVIDIA driver.

## Data

- [fastMRI](https://fastmri.med.nyu.edu/): raw brain multicoil MRI data. Follow
  the dataset access instructions to obtain the training, validation, and fully
  sampled test releases.
- [fastMRI+](https://github.com/microsoft/fastmri-plus): pathology annotations and
  reviewed-scan lists in the `Annotations/` directory. This pipeline uses
  `brain.csv` and `brain_file_list.csv`.

## Preprocessing

Place the raw fastMRI brain releases in `multicoil_train`, `multicoil_val`, and
`multicoil_test_full` under the raw-data directory.

```bash
set -euo pipefail
export RAW=/path/to/raw/fastMRI
export OUT=/path/to/processed
export ANNOTATIONS="$PWD/annotations"
export CUDA_VISIBLE_DEVICES=0

for split in train val test; do
    python -m fastmri_preprocess.survey --split "$split" --raw "$RAW" --out "$OUT"
    python -m fastmri_preprocess.run --split "$split" --out "$OUT" --gpus 0 --seed 0
    python -m fastmri_preprocess.finalize --split "$split" --out "$OUT" \
        --annotations "$ANNOTATIONS"
    python -m fastmri_preprocess.sweep --split "$split" --out "$OUT"
    python -m fastmri_preprocess.merge_sweep --split "$split" --out "$OUT"
    python -m fastmri_preprocess.verify --split "$split" --out "$OUT" --n 500
done
```

Train runs first to produce the normalization constants in `stats.json`. GPU
indices in `--gpus` refer to devices visible through `CUDA_VISIBLE_DEVICES`.

Every command accepts `--help`. `--out` defaults to `FASTMRI_OUT_ROOT` or the current
directory; annotations default to `<out>/annotations`, figures to `<out>/qc`.

Rerun `run` with the same output directory to resume processing. Completed scans
are skipped and failed scans are retried. Parameters and library versions must
match `run_config.json`. `--readers 0` reads scans in the processing worker.

`finalize` builds tables from completed processing logs and noise matrices.
`--overwrite` rebuilds existing tables from those logs, replacing subsequent
metadata edits. Complete preprocessing and maintenance before loading the dataset.

## Load data

```python
from fastmri_preprocess import FastMRIBrain

ds = FastMRIBrain("/path/to/processed", "train")
slots = ds.select("snr_std > 5 and my == 220")
item = ds[int(slots[0])]
ksp, mps, gt = item["ksp"], item["mps"], item["gt"]
x = ds.normalize(gt, item)
boxes = ds.boxes(int(slots[0]))
```

For use without installation, add the repository's `code/` directory to
`PYTHONPATH` and import `FastMRIBrain` from `fastmri_brain`.

Items use native coil counts and phase widths. For fixed shapes, use
`fixed_shape=True` and select `my == 220`. `pad_narrow=True` permits padded image
inputs, but their padded k-space is not a valid 220 mm acquisition.

`annotations/` contains the fastMRI+ brain and knee CSVs; this pipeline uses the
brain annotations. Use `plus_reviewed` to distinguish reviewed scans from scans
without annotation coverage. Missing labels do not imply normal anatomy.

See [dataset format and processing details](docs/dataset.md) for array layouts,
annotation coordinates, normalization, and QC fields.

## Quality control

`verify --no-figures` checks sampled slices on CPU. `sweep --device cpu --no-figures`
checks every slice. Integrity failures return a nonzero exit status; support holes
and low SNR are reported as image-quality flags.

`repair_holes` recalibrates slices with support holes or fallback thresholds and
updates maps, reference images, statistics, and training normalization. Use
`--dry-run` to compute repairs without writing. Rerunning an interrupted repair
resumes the pending slices. Run `sweep`, `merge_sweep`, and `verify` after repair.

## Development

```bash
pre-commit install
python -m pytest
pre-commit run --all-files
```

Tests use synthetic data and run on CPU.
