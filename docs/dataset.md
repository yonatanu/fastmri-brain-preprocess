# Dataset format

The pipeline processes fastMRI brain train/validation and the fully sampled test
release. Scans below 2 T are excluded. Slice thickness and spacing are not resampled.

## Processing

1. Read complex multicoil k-space and the ISMRMRD header.
2. Estimate coil noise covariance from readout air strips at 110–120 mm from the
   image center, using acquired phase lines. Exclude slices whose strip variance
   exceeds 1.1 times the volume median.
3. Whiten with the inverse Cholesky factor and compress to at most eight coils
   using the leading eigenvectors of the whitened signal covariance.
4. Center-crop k-space to the nominal 1 mm grid, inverse FFT, crop the image FOV
   to 220 mm, and FFT back. Preserve narrower phase FOVs instead of padding them.
5. Derive the phase mask from the acquired band and set unacquired samples to zero.
6. Estimate ESPIRiT maps with threshold 0.05, crop 0.95, calibration width 24, and
   kernel width 6. Retry at threshold 0.02 if more than 0.1% of the head mask lacks
   map support. The head mask uses coil RSS, hole filling, and 3-pixel erosion.
7. Compute the complex SENSE reference and record scan and slice statistics.

FFTs are centered and orthonormal. Index `N // 2` is DC or the image center;
center crops preserve this convention. Array axes are `[readout, phase]`.
Use `flipud` for the display orientation used by fastMRI+.

Whitening gives a nominal noise variance of one per complex sample before masking.
Acquisition masks and map support affect image-domain noise. Reference images retain
acquisition noise, artifacts, and sensitivity-model errors.

## Binary files

Binaries have no headers. `N` is the slice count and `V` the scan count.

| File | Shape | Type |
|---|---|---|
| `ksp.bin` | `(N, 8, 220, 220)` | complex64 |
| `mps.bin` | `(N, 8, 220, 220)` | complex64 |
| `gt.bin` | `(N, 220, 220)` | complex64 |
| `eig.bin` | `(N, 220, 220)` | float16 |
| `pe_mask.bin` | `(V, 220)` | uint8 |

Slots are ordered by scan, then slice. Each scan occupies
`[slot0, slot0 + n_slices)`. Valid coils are `[:n_coils]`; valid phase columns are
`[col0, col0 + my)`, with `col0 = 110 - my // 2`. The remainder is zero storage
padding. Narrow-FOV padding is not part of the physical acquisition.

Maps satisfy `sum(abs(mps)**2, axis=0) = 1` on their support and zero elsewhere.
Use the maps to determine support; float16 eigenvalues can round across the crop
threshold. The reference is `sum(conj(mps) * ifft2c(ksp), axis=0)`.

`noise/<stem>.npz` stores each scan's native covariance (`cov`), whitening matrix
(`W`), and compression matrix (`U`, empty when compression is unnecessary).
`noise.npz` packs these under `<stem>/<key>` names.

## Tables

| File | Contents |
|---|---|
| `scans_geom.parquet` | Raw paths, header fields, selection, and storage layout |
| `scans.parquet` | Selected scans, geometry, processing status, QC, review status |
| `slices.parquet` | Slot indices, scales, image statistics, QC, and annotation flags |
| `plus_boxes.parquet` | Native-grid boxes, original coordinates, labels, and slots |
| `plus_study.parquet` | Study-level labels keyed by scan |
| `sweep.parquet` | Per-slot storage and reconstruction checks |

Scan geometry includes `scan_idx`, `slot0`, `n_slices`, `n_coils`, `my`, and `col0`.
Acquisition metadata includes contrast, scanner, field strength, TR/TE/TI, FOVs,
matrices, and acquired phase limits.
`voxel_x` and `voxel_y` record the spacing after integer grid rounding and define
the pixel sizes used for annotation mapping.

Slice statistics include `scale`, `std_full`, `snr_std`, `gt_p99`, `gt_max`,
`support_frac`, `espirit_resid`, `espirit_thresh`, `support_holes_frac`, and
`halo_level`. `band_energy_frac` is the retained k-space norm ratio after applying
the phase mask. Merging a sweep adds `n_support_components`, `sweep_resid`, and
`head_frac`.

fastMRI+ boxes use unflipped array coordinates: `r0`, `c0`, `h`, and `w` on the
native grid, plus `c0_slot` on the storage grid. Physical centers and sizes are
stored as `row_mm`, `col_mm`, `h_mm`, and `w_mm`. Original coordinates remain in
`x_orig`, `y_orig`, `w_orig`, and `h_orig`. Slice indices are zero-based.

`plus_reviewed` identifies scans listed in `brain_file_list.csv`, including those
without boxes. Study labels remain separate from slice-level box indicators.

## Normalization

Let $s_i$ be the norm of the central 32 phase lines of a slice's stored k-space,
$c$ the training constant from `stats.json`, $g_i$ the reference image, and $k_i$
the k-space. The normalized image and measurements are

$$x_i = c g_i / s_i, \qquad y_i = c k_i / s_i.$$

The nominal normalized complex noise standard deviation is $c/s_i$. The forward
operator is the acquisition mask times the centered FFT of `mps * x_i`.

The training constant targets a median real-element image standard deviation of
0.5. The 32-line scale can be computed from an undersampled acquisition with a
matching ACS region.
Low-SNR slices remain noise-dominated after normalization; use `snr_std` to filter
them when appropriate.

## Quality control

- `snr_std` measures image standard deviation relative to the nominal noise level.
- `support_holes_frac` measures the head fraction outside sensitivity-map support.
- Use native phase widths and acquisition masks in forward models.
- `espirit_resid` measures the relative forward-model residual. Large values may
  reflect low SNR or acquisition artifacts.
- `halo_level` measures signal outside the filled head mask but inside map support.
- `clipped_x` and `clipped_y` are extent heuristics and can respond to ghosting.

A sweep checks finite arrays, zero storage padding, masks, map normalization,
eigenvalue consistency, and agreement with the recomputed SENSE reference.
Integrity failures fail the command; support-hole counts remain separate QC flags.
