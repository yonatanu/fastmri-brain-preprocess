"""Numerical checks for geometry, whitening, and CPU preprocessing."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from fastmri_preprocess.pipeline import (
    Header,
    center_crop,
    fft2c,
    head_mask,
    ifft2c,
    native_acquired_mask,
    pe_mask,
    plan_geometry,
    preprocess,
    whiten_and_compress,
)


def header(width=240):
    return Header(
        enc_mat=(480, width),
        enc_fov=(480.0, float(width)),
        rec_mat=(240, width),
        rec_fov=(240.0, float(width)),
        pe_center=width // 2,
        pe_max=width - 1,
        field_T=3.0,
        model="synthetic",
        vendor="synthetic",
        institution="test",
        protocol="test",
        sequence="test",
        n_rx=2,
        TR=3000,
        TE=80,
        TI=0,
        flip=90,
    )


@pytest.mark.parametrize("shape", [(8, 9), (9, 8), (8, 8), (9, 9)])
def test_centered_fft_roundtrip_and_energy(shape):
    generator = torch.Generator().manual_seed(42)
    values = torch.randn((2, *shape), dtype=torch.complex64, generator=generator)
    torch.testing.assert_close(ifft2c(fft2c(values)), values)
    torch.testing.assert_close(fft2c(values).norm(), values.norm())
    cropped = center_crop(values, 5, -1)
    torch.testing.assert_close(cropped[..., 2], values[..., shape[-1] // 2])
    with pytest.raises(ValueError, match="Cannot crop"):
        center_crop(values, shape[-1] + 1, -1)


@pytest.mark.parametrize("width", [180, 220, 221, 240])
def test_geometry_preserves_dc_and_counts_acquired_lines(width):
    geometry = plan_geometry(header(width), 480, width)
    mask = pe_mask(geometry)
    assert mask[geometry.my // 2]
    assert int(mask.sum()) == geometry.n_acq == geometry.my
    assert geometry.col0 == 110 - geometry.my // 2


def test_invalid_acquisition_limits_are_rejected():
    with pytest.raises(ValueError, match="Acquisition limits"):
        native_acquired_mask(replace(header(), pe_center=250), 240)
    with pytest.raises(ValueError):
        plan_geometry(replace(header(), enc_fov=(180, 240)), 480, 240)


def test_whitening_and_compression_preserve_white_noise():
    generator = torch.Generator().manual_seed(42)
    white = torch.randn((2, 4, 32, 128), dtype=torch.complex64, generator=generator)
    mixing = torch.tensor(
        [[2, 0, 0, 0], [1j, 1, 0, 0], [0, 0, 3, 0], [0, 0, 1j, 2]],
        dtype=torch.complex128,
    )
    covariance = mixing @ mixing.conj().T
    correlated = torch.einsum("ij,sjxy->sixy", mixing.to(torch.complex64), white)
    result, whitening, compression, _ = whiten_and_compress(
        correlated, covariance, torch.ones(128, dtype=torch.bool), max_coils=2
    )
    torch.testing.assert_close(
        whitening @ covariance @ whitening.conj().T,
        torch.eye(4, dtype=torch.complex128),
    )
    torch.testing.assert_close(
        compression.conj().T @ compression, torch.eye(2, dtype=torch.complex64)
    )
    assert result.shape == (2, 2, 32, 128)
    whitened, _, _, _ = whiten_and_compress(
        correlated, covariance, torch.ones(128, dtype=torch.bool), max_coils=4
    )
    flat = whitened.permute(1, 0, 2, 3).reshape(4, -1)
    empirical = flat @ flat.conj().T / flat.shape[1]
    torch.testing.assert_close(
        empirical, torch.eye(4, dtype=torch.complex64), atol=0.04, rtol=0
    )


def test_zero_erosion_keeps_head_mask():
    rss = torch.zeros(20, 20)
    rss[4:16, 4:16] = 100
    assert head_mask(rss, 1, erode=0).sum() == 144


def test_synthetic_scan_preprocesses_on_cpu():
    np.random.seed(42)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    x = torch.arange(480) - 240
    y = torch.arange(240) - 120
    image = 80 * torch.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / (2 * 30**2))
    coils = torch.stack([image, image * (0.6 + 0.8j)])[None] / np.sqrt(2)
    raw = fft2c(coils) + torch.randn(
        coils.shape, dtype=torch.complex64, generator=generator
    )
    result = preprocess(raw, header(), device_id=-1)
    assert result["ksp"].shape == (1, 2, 220, 220)
    for key in ("ksp", "mps", "gt", "eig"):
        assert torch.isfinite(result[key]).all()
    torch.testing.assert_close(
        result["gt"], (result["mps"].conj() * ifft2c(result["ksp"])).sum(1)
    )
    norm = result["mps"].abs().square().sum(1)
    torch.testing.assert_close(norm[norm > 0], torch.ones_like(norm[norm > 0]))
    assert (result["slice_stats"]["scale"] > 0).all()
