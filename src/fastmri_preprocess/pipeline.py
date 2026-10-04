"""Noise whitening, coil compression, spatial cropping, and ESPIRiT calibration."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import h5py
import numpy as np
import sigpy as sp
import torch
from einops import rearrange
from scipy.ndimage import binary_erosion, binary_fill_holes
from sigpy.mri.app import EspiritCalib

from .fastmri_brain import MAX_COILS, N_OUT

VOXEL_MM = 1.0
FOV_MM = N_OUT * VOXEL_MM
MIN_FIELD_T = 2.0  # keeps the 2.89 T scans, drops 1.5 T
NOISE_BAND_MM = (110.0, 120.0)  # air strips flanking the kept readout FOV
NOISE_TIP_MM = 200.0  # readout tip, used only as a QC ratio
NOISE_SLICE_TOL = 1.1  # maximum strip variance relative to the volume median
ESPIRIT = dict(thresh=0.05, crop=0.95, calib_width=24, kernel_width=6, max_iter=100)
ESPIRIT_FALLBACK_THRESH = 0.02
HOLE_CUTOFF = 0.001  # head fraction outside map support that triggers fallback
HEAD_REL = 0.10  # RSS threshold relative to the slice's 99.5th percentile
HEAD_ERODE_PX = 3
HEAD_MIN_FRAC = 0.01
SCALE_LINES = 32  # central phase lines used for normalization
OBJECT_SNR = 4.0  # object threshold in units of RSS noise


@dataclass
class Header:
    enc_mat: tuple
    enc_fov: tuple
    rec_mat: tuple
    rec_fov: tuple
    pe_center: int
    pe_max: int
    field_T: float
    model: str
    vendor: str
    institution: str
    protocol: str
    sequence: str
    n_rx: int
    TR: float
    TE: float
    TI: float
    flip: float


def _text(root, name, default=None):
    return next(
        (e.text for e in root.iter() if e.tag.rsplit("}", 1)[-1] == name), default
    )


def _space(root, name):
    parent = next(e for e in root.iter() if e.tag.endswith(name))
    mat = next([int(c.text) for c in e] for e in parent if e.tag.endswith("matrixSize"))
    fov = next(
        [float(c.text) for c in e] for e in parent if e.tag.endswith("fieldOfView_mm")
    )
    return tuple(mat), tuple(fov)


def parse_header(xml: str) -> Header:
    root = ET.fromstring(xml)
    lim = next(e for e in root.iter() if e.tag.endswith("kspace_encoding_step_1"))
    lim = {c.tag.split("}")[-1]: int(c.text) for c in lim}
    enc_mat, enc_fov = _space(root, "encodedSpace")
    rec_mat, rec_fov = _space(root, "reconSpace")

    def num(name):
        t = _text(root, name)
        return float(t) if t not in (None, "") else float("nan")

    return Header(
        enc_mat=enc_mat,
        enc_fov=enc_fov,
        rec_mat=rec_mat,
        rec_fov=rec_fov,
        pe_center=lim["center"],
        pe_max=lim["maximum"],
        field_T=num("systemFieldStrength_T"),
        model=_text(root, "systemModel", ""),
        vendor=_text(root, "systemVendor", ""),
        institution=_text(root, "institutionName", ""),
        protocol=_text(root, "protocolName", ""),
        sequence=_text(root, "sequence_type", ""),
        n_rx=int(num("receiverChannels")),
        TR=num("TR"),
        TE=num("TE"),
        TI=num("TI"),
        flip=num("flipAngle_deg"),
    )


def read_header(path: str) -> Header:
    with h5py.File(path, "r") as hf:
        raw = hf["ismrmrd_header"][()]
    xml = (
        raw.decode() if isinstance(raw, (bytes, bytearray)) else raw.tobytes().decode()
    )
    return parse_header(xml)


@dataclass
class Geometry:
    """Native, cropped, and stored grid dimensions for one scan."""

    KX: int  # native k-space size (readout, 2x oversampled)
    KY: int
    nx: int  # after the 1 mm k-space crop, still the full encoded FOV
    ny: int
    mx: int  # stored size after the FOV crop
    my: int
    col0: int  # first valid PE column inside the 220-wide storage slot
    n_below: int  # acquired lines below / above DC on the stored grid
    n_above: int
    voxel_x: float
    voxel_y: float

    @property
    def n_acq(self) -> int:
        return self.n_below + self.n_above + 1

    @property
    def zero_filled(self) -> bool:
        return self.n_acq < self.my


def plan_geometry(hdr: Header, KX: int, KY: int) -> Geometry:
    if not all(np.isfinite(value) and value > 0 for value in hdr.enc_fov[:2]):
        raise ValueError("Encoded field of view must be finite and positive.")
    native_acquired_mask(hdr, KY)
    nx = int(round(hdr.enc_fov[0] / VOXEL_MM))
    ny = int(round(hdr.enc_fov[1] / VOXEL_MM))
    if nx > KX or ny > KY or nx < N_OUT or ny < SCALE_LINES:
        raise ValueError(
            f"Cannot produce the target grid from KX={KX}, KY={KY}, "
            f"encoded dimensions {nx} x {ny} mm."
        )
    mx = N_OUT
    my = min(ny, N_OUT)  # crop to 220 mm when there is FOV to spare, else keep native

    # Acquired lines on each side of DC, clipped to the cropped k-space grid.
    n_below = hdr.pe_center
    n_above = hdr.pe_max - hdr.pe_center
    n_below = min(n_below, ny // 2)
    n_above = min(n_above, ny - ny // 2 - 1)
    # Rescale the band for the cropped FOV, rounding to the nearest measured line.
    if my < ny:
        n_below = int(np.floor(n_below * my / ny + 0.5))
        n_above = int(np.floor(n_above * my / ny + 0.5))
    n_below = min(n_below, my // 2)
    n_above = min(n_above, my - my // 2 - 1)
    return Geometry(
        KX=KX,
        KY=KY,
        nx=nx,
        ny=ny,
        mx=mx,
        my=my,
        col0=N_OUT // 2 - my // 2,
        n_below=n_below,
        n_above=n_above,
        voxel_x=hdr.enc_fov[0] / nx,
        voxel_y=hdr.enc_fov[1] / ny,
    )


def pe_mask(geom: Geometry, device=None) -> torch.Tensor:
    m = torch.zeros(geom.my, dtype=torch.bool, device=device)
    c = geom.my // 2
    m[c - geom.n_below : c + geom.n_above + 1] = True
    return m


def native_acquired_mask(hdr: Header, KY: int, device=None) -> torch.Tensor:
    pad_l = KY // 2 - hdr.pe_center
    if not 0 <= hdr.pe_center <= hdr.pe_max or pad_l < 0 or pad_l + hdr.pe_max >= KY:
        raise ValueError("Acquisition limits exceed the native phase grid.")
    m = torch.zeros(KY, dtype=torch.bool, device=device)
    m[pad_l : pad_l + hdr.pe_max + 1] = True
    return m


def ifft2c(x):
    return torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1),
    )


def fft2c(x):
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1),
    )


def ifft1c(x, dim):
    return torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim
    )


def center_crop(x, n, dim):
    """Crop `dim` to length `n`, keeping index K // 2 (DC / image centre) at n // 2."""
    k = x.shape[dim]
    if n < 1 or n > k:
        raise ValueError(f"Cannot crop axis of length {k} to {n}.")
    if n == k:
        return x
    return x.narrow(dim, k // 2 - n // 2, n)


def noise_covariance(ksp, acq, fov_x_mm):
    """Estimate coil covariance from acquired samples in the readout air strips.

    Reject slices whose strip variance exceeds the volume median by the configured
    factor. Subtract the sample mean before estimating the covariance.
    """
    hyb = ifft1c(ksp, -2)
    KX = hyb.shape[-2]
    x_mm = (torch.arange(KX, device=ksp.device) - KX // 2).abs() * (fov_x_mm / KX)
    band = (x_mm >= NOISE_BAND_MM[0]) & (x_mm < NOISE_BAND_MM[1])
    tip = x_mm >= NOISE_TIP_MM
    if not band.any() or not acq.any():
        raise ValueError("No acquired samples are available in the noise strips.")

    n = hyb[:, :, band][..., acq]  # S C X Y
    per_slice = n.abs().square().mean(dim=(1, 2, 3))
    if not torch.isfinite(per_slice).all() or per_slice.median() <= 0:
        raise ValueError("Noise-strip variance must be finite and positive.")
    keep = per_slice <= NOISE_SLICE_TOL * per_slice.median()
    n = rearrange(n[keep], "s c x y -> c (s x y)").to(torch.complex128)
    if n.shape[1] <= n.shape[0]:
        raise ValueError("Not enough noise samples to estimate coil covariance.")
    n = n - n.mean(1, keepdim=True)
    cov = (n @ n.conj().T) / n.shape[1]
    cov = (cov + cov.conj().T) / 2

    band_var = float(per_slice[keep].mean())
    tip_var = (
        float(hyb[:, :, tip][..., acq][keep].abs().square().mean())
        if tip.any()
        else float("nan")
    )
    qc = dict(
        noise_band_var=band_var,
        noise_tip_ratio=tip_var / band_var,
        noise_slices_used=int(keep.sum()),
        noise_slice_var_max=float(per_slice.max() / per_slice.median()),
        cov_cond=float(torch.linalg.cond(cov).real),
    )
    return cov, qc


def whiten_and_compress(ksp, cov, acq, max_coils=MAX_COILS):
    """Whiten coil noise and retain the leading signal eigenvectors.

    Return compressed k-space, whitening matrix, compression matrix, and retained
    signal fraction after subtracting the whitened noise floor.
    """
    L = torch.linalg.cholesky(cov)
    W = torch.linalg.inv(L)
    ksp = torch.einsum("oc,scxy->soxy", W.to(torch.complex64), ksp)

    flat = rearrange(ksp[..., acq], "s c x y -> c (s x y)")
    gram = (
        flat.to(torch.complex128) @ flat.conj().T.to(torch.complex128)
    ) / flat.shape[1]
    lam, U = torch.linalg.eigh(gram)  # ascending
    lam, U = lam.flip(0).real, U.flip(1)
    signal = (lam - 1.0).clamp_min(0)
    c = ksp.shape[1]
    if c <= max_coils:
        return ksp, W, None, 1.0
    kept = float(signal[:max_coils].sum() / signal.sum().clamp_min(1e-30))
    U = U[:, :max_coils].to(torch.complex64)
    return torch.einsum("sixy,io->soxy", ksp, U.conj()), W, U, kept


def object_extent(img, fov_x_mm, fov_y_mm, noise_band_mm=NOISE_BAND_MM):
    """Measure object extents, offsets, and phase-edge signal in the full FOV."""
    rss = img.abs().square().sum(1).sqrt()  # S X Y
    S, X, Y = rss.shape
    x_mm = (torch.arange(X, device=img.device) - X // 2).abs() * (fov_x_mm / X)
    strip = (x_mm >= noise_band_mm[0]) & (x_mm < noise_band_mm[1])
    thr = OBJECT_SNR * rss[:, strip].median()
    occ = rss > thr

    def span(prof, n, fov):
        idx = torch.nonzero(prof > 0.01).flatten()
        if idx.numel() == 0:
            return 0.0, 0.0
        lo, hi = int(idx[0]), int(idx[-1])
        extent = (hi - lo + 1) * fov / n
        offset = ((lo + hi) / 2 - n // 2) * fov / n
        return float(extent), float(offset)

    ext_x, off_x = span(occ.float().mean(dim=(0, 2)), X, fov_x_mm)
    ext_y, off_y = span(occ.float().mean(dim=(0, 1)), Y, fov_y_mm)
    edge = torch.maximum(rss[:, :, :2].max(), rss[:, :, -2:].max()) / rss.max()
    return dict(
        extent_x_mm=ext_x,
        extent_y_mm=ext_y,
        offset_x_mm=off_x,
        offset_y_mm=off_y,
        pe_edge_rel=float(edge),
    )


def head_mask(rss_slice, n_coils, rel=HEAD_REL, erode=HEAD_ERODE_PX):
    """Threshold coil RSS, fill internal cavities, and erode the boundary."""
    r = rss_slice
    thr = max(
        float(rel * torch.quantile(r.flatten(), 0.995)),
        OBJECT_SNR * float(np.sqrt(n_coils)),
    )
    head = binary_fill_holes((r > thr).cpu().numpy())
    return binary_erosion(head, iterations=erode) if erode > 0 else head


def support_holes(rss_slice, eig_slice, n_coils, *, support=None):
    """Return the fraction of head pixels outside the sensitivity-map support."""
    head = head_mask(rss_slice, n_coils)
    if head.mean() < HEAD_MIN_FRAC:
        return 0.0
    supp = (
        (eig_slice > ESPIRIT["crop"]).cpu().numpy()
        if support is None
        else np.asarray(support, dtype=bool)
    )
    return float((head & ~supp).sum() / head.sum())


def espirit_maps(ksp_slice, device_id, thresh=None):
    """Estimate sensitivity maps and eigenvalues for one [coil, x, y] slice."""
    dev = sp.Device(device_id)
    k = sp.from_pytorch(torch.view_as_real(ksp_slice.contiguous()), iscomplex=True)
    cfg = dict(ESPIRIT, thresh=ESPIRIT["thresh"] if thresh is None else thresh)
    mps, eig = EspiritCalib(
        k, device=dev, show_pbar=False, output_eigenvalue=True, **cfg
    ).run()
    xp = sp.get_device(mps).xp
    mps = torch.view_as_complex(
        sp.to_pytorch(xp.ascontiguousarray(mps)).contiguous()
    ).detach()
    eig = (
        sp.to_pytorch(xp.ascontiguousarray(eig)).detach().reshape(ksp_slice.shape[-2:])
    )
    return mps, eig


@torch.no_grad()
def preprocess(ksp: torch.Tensor, hdr: Header, device_id: int = 0):
    """Process one complex [slice, coil, x, y] scan on its current device.

    Return native-grid arrays, noise matrices, and scan and slice statistics.
    """
    if ksp.ndim != 4 or not ksp.is_complex() or min(ksp.shape) < 1:
        raise ValueError("Expected nonempty complex [slice, coil, x, y] k-space.")
    if not torch.isfinite(ksp).all():
        raise ValueError("Input k-space contains nonfinite values.")
    ksp = ksp.to(torch.complex64)
    S, C, KX, KY = ksp.shape
    geom = plan_geometry(hdr, KX, KY)
    acq = native_acquired_mask(hdr, KY, ksp.device)

    cov, noise_qc = noise_covariance(ksp, acq, hdr.enc_fov[0])
    ksp, W, U, kept = whiten_and_compress(ksp, cov, acq)

    ksp = center_crop(center_crop(ksp, geom.nx, -2), geom.ny, -1)  # -> 1 mm
    img = ifft2c(ksp)
    extent_qc = object_extent(img, hdr.enc_fov[0], hdr.enc_fov[1])
    img = center_crop(center_crop(img, geom.mx, -2), geom.my, -1)  # -> 220 mm
    ksp = fft2c(img)
    mask = pe_mask(geom, ksp.device)
    # Remove leakage outside the acquired band after cropping the phase FOV.
    band_energy_frac = (ksp * mask).flatten(1).norm(dim=1) / ksp.flatten(1).norm(dim=1)
    ksp = ksp * mask

    coil_img = ifft2c(ksp)
    rss = coil_img.abs().square().sum(1).sqrt()
    n_coils = ksp.shape[1]
    mps = torch.empty_like(ksp)
    eig = torch.empty(S, geom.mx, geom.my, device=ksp.device)
    thresh_used = np.full(S, ESPIRIT["thresh"], dtype=np.float64)
    holes = np.zeros(S, dtype=np.float64)
    for i in range(S):
        mps[i], eig[i] = espirit_maps(ksp[i], device_id)
        holes[i] = support_holes(rss[i], eig[i], n_coils)
        if holes[i] > HOLE_CUTOFF:  # fall back to the looser calibration threshold
            mps[i], eig[i] = espirit_maps(
                ksp[i], device_id, thresh=ESPIRIT_FALLBACK_THRESH
            )
            thresh_used[i] = ESPIRIT_FALLBACK_THRESH
            holes[i] = support_holes(rss[i], eig[i], n_coils)

    gt = (mps.conj() * coil_img).sum(1)

    A_gt = fft2c(mps * gt[:, None]) * mask
    resid = (A_gt - ksp).flatten(1).norm(dim=1) / ksp.flatten(1).norm(dim=1)
    c = geom.my // 2
    scale = ksp[..., c - SCALE_LINES // 2 : c + SCALE_LINES // 2].flatten(1).norm(dim=1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Calibration scales must be finite and positive.")
    re, im = gt.real.flatten(1), gt.imag.flatten(1)
    support = (eig > ESPIRIT["crop"]).flatten(1).float().mean(1)
    slice_stats = dict(
        scale=scale,
        std_full=torch.view_as_real(gt).flatten(1).std(dim=1),
        std_real=re.std(dim=1),
        std_imag=im.std(dim=1),
        gt_p99=gt.abs().flatten(1).quantile(0.99, dim=1),
        gt_max=gt.abs().flatten(1).max(dim=1).values,
        support_frac=support,
        espirit_resid=resid,
        band_energy_frac=band_energy_frac,
    )
    slice_stats = {k: v.cpu().numpy() for k, v in slice_stats.items()}
    slice_stats["espirit_thresh"] = thresh_used
    slice_stats["support_holes_frac"] = holes

    half_y = geom.my * geom.voxel_y / 2
    scan_qc = dict(
        **noise_qc,
        **extent_qc,
        compress_energy_kept=kept,
        clipped_x=bool(
            extent_qc["extent_x_mm"] / 2 + abs(extent_qc["offset_x_mm"]) > FOV_MM / 2
        ),
        clipped_y=bool(
            extent_qc["extent_y_mm"] / 2 + abs(extent_qc["offset_y_mm"]) > half_y
        ),
        n_coils=int(ksp.shape[1]),
        n_coils_native=int(C),
        **asdict(geom),
    )
    return dict(
        ksp=ksp,
        mps=mps,
        gt=gt,
        eig=eig,
        mask=mask,
        cov=cov.cpu().numpy(),
        W=W.cpu().numpy(),
        U=None if U is None else U.cpu().numpy(),
        slice_stats=slice_stats,
        scan_qc=scan_qc,
        geom=geom,
    )


def map_box(x, y, w, h, rss_shape, rec_fov, out_shape, voxel=VOXEL_MM):
    """Map a fastMRI+ box from flipped RSS coordinates to the native output grid."""
    H, W = rss_shape
    spacing = np.broadcast_to(np.asarray(voxel, dtype=float), (2,))
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Output voxel sizes must be finite and positive.")
    voxel_x, voxel_y = spacing
    pr, pc = rec_fov[0] / H, rec_fov[1] / W
    r0, c0 = H - y - h, x
    out_r = ((r0 - H // 2) * pr) / voxel_x + out_shape[0] // 2
    out_c = ((c0 - W // 2) * pc) / voxel_y + out_shape[1] // 2
    return out_r, out_c, h * pr / voxel_x, w * pc / voxel_y
