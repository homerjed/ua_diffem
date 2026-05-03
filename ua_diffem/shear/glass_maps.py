from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ua_diffem.diffem import CorruptionBatch


def _require_glass_healpy():
    try:
        import glass
        import healpy as hp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Spherical shear training requires `glass` and `healpy`. "
            "Install them directly or use the Singularity definition in this repo."
        ) from exc
    return glass, hp


def default_lmax(nside: int) -> int:
    return 3 * int(nside) - 1


def glass_kappa_cls(
    lmax: int,
    *,
    amplitude: float = 0.9,
    spectral_index: float = 1.25,
    damping: float = 0.006,
) -> np.ndarray:
    ell = np.arange(int(lmax) + 1, dtype=np.float64)
    cl = amplitude * (ell + 1.0) ** (-spectral_index) * np.exp(
        -damping * ell * (ell + 1.0)
    )
    cl[:2] = 0.0
    return cl.astype(np.float64)


def smail_redshift_distribution(
    *,
    z0: float = 0.5,
    alpha: float = 2.0,
    beta: float = 1.5,
    zmax: float = 3.0,
    nz: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth source distribution commonly used for weak-lensing forecasts."""

    if z0 <= 0.0 or alpha <= -1.0 or beta <= 0.0 or zmax <= 0.0 or nz < 8:
        raise ValueError("Invalid Smail redshift distribution parameters.")

    z = np.linspace(0.0, float(zmax), int(nz), dtype=np.float64)
    dndz = np.power(z, alpha) * np.exp(-np.power(z / z0, beta))
    norm = np.trapezoid(dndz, z)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Smail redshift distribution normalization failed.")
    return z, dndz / norm


def camb_lensing_kappa_cls(
    lmax: int,
    *,
    h0: float = 67.66,
    ombh2: float = 0.02242,
    omch2: float = 0.11933,
    as_scalar: float = 2.105e-9,
    ns: float = 0.9665,
    mnu: float = 0.06,
    nonlinear: bool = True,
    source_z0: float = 0.5,
    source_alpha: float = 2.0,
    source_beta: float = 1.5,
    source_zmax: float = 3.0,
    source_nz: int = 256,
) -> np.ndarray:
    """Compute a cosmological weak-lensing convergence spectrum with CAMB."""

    try:
        import camb
        from camb import model, sources
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`--kappa_power_spectrum camb` requires the optional `camb` package. "
            "Install it directly or rebuild the Singularity image from this repo."
        ) from exc

    lmax = int(lmax)
    z, dndz = smail_redshift_distribution(
        z0=source_z0,
        alpha=source_alpha,
        beta=source_beta,
        zmax=source_zmax,
        nz=source_nz,
    )
    pars = camb.set_params(
        H0=float(h0),
        ombh2=float(ombh2),
        omch2=float(omch2),
        As=float(as_scalar),
        ns=float(ns),
        mnu=float(mnu),
        NonLinear=model.NonLinear_both if nonlinear else model.NonLinear_none,
    )
    pars.set_for_lmax(lmax)
    pars.SourceWindows = [
        sources.SplinedSourceWindow(z=z, W=dndz, source_type="lensing"),
    ]
    cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
    cl = np.asarray(cls["W1xW1"], dtype=np.float64)[: lmax + 1]
    cl = np.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0)
    cl[:2] = 0.0
    return np.maximum(cl, 0.0)


def make_kappa_cls(
    lmax: int,
    *,
    power_spectrum: str = "toy",
    amplitude: float = 0.9,
    spectral_index: float = 1.25,
    damping: float = 0.006,
    camb_h0: float = 67.66,
    camb_ombh2: float = 0.02242,
    camb_omch2: float = 0.11933,
    camb_as: float = 2.105e-9,
    camb_ns: float = 0.9665,
    camb_mnu: float = 0.06,
    camb_nonlinear: bool = True,
    source_z0: float = 0.5,
    source_alpha: float = 2.0,
    source_beta: float = 1.5,
    source_zmax: float = 3.0,
    source_nz: int = 256,
) -> np.ndarray:
    if power_spectrum == "toy":
        return glass_kappa_cls(
            lmax,
            amplitude=amplitude,
            spectral_index=spectral_index,
            damping=damping,
        )
    if power_spectrum == "camb":
        return camb_lensing_kappa_cls(
            lmax,
            h0=camb_h0,
            ombh2=camb_ombh2,
            omch2=camb_omch2,
            as_scalar=camb_as,
            ns=camb_ns,
            mnu=camb_mnu,
            nonlinear=camb_nonlinear,
            source_z0=source_z0,
            source_alpha=source_alpha,
            source_beta=source_beta,
            source_zmax=source_zmax,
            source_nz=source_nz,
        )
    raise ValueError(f"Unknown kappa power spectrum {power_spectrum!r}.")


def validate_kappa_cls(cl_kappa: np.ndarray, *, lmax: int) -> np.ndarray:
    cl_kappa = np.asarray(cl_kappa, dtype=np.float64)
    if cl_kappa.ndim != 1 or cl_kappa.shape[0] < int(lmax) + 1:
        raise ValueError("`cl_kappa` must be one-dimensional with length at least lmax + 1.")
    cl_kappa = cl_kappa[: int(lmax) + 1].copy()
    cl_kappa = np.nan_to_num(cl_kappa, nan=0.0, posinf=0.0, neginf=0.0)
    cl_kappa[:2] = 0.0
    return np.maximum(cl_kappa, 0.0)


def _readonly_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    array.setflags(write=False)
    return array


@lru_cache(maxsize=None)
def _alm_ell_indices(lmax: int) -> np.ndarray:
    _, hp = _require_glass_healpy()
    ell_alm, _ = hp.Alm.getlm(int(lmax))
    return _readonly_array(np.asarray(ell_alm, dtype=np.int32))


@lru_cache(maxsize=None)
def _observable_alm_keep_mask(lmax: int, ell_min: int) -> np.ndarray:
    keep = _alm_ell_indices(int(lmax)) >= int(ell_min)
    return _readonly_array(keep)


@lru_cache(maxsize=None)
def _ks_response(nside: int, lmax: int) -> np.ndarray:
    _, hp = _require_glass_healpy()
    ell_alm = _alm_ell_indices(int(lmax)).astype(np.float64, copy=False)
    response = np.zeros(ell_alm.shape, dtype=np.float64)
    good = ell_alm >= 2.0
    response[good] = -np.sqrt(
        (ell_alm[good] * (ell_alm[good] + 1.0))
        / ((ell_alm[good] - 1.0) * (ell_alm[good] + 2.0))
    )

    if int(nside) > 0:
        pw0, pw2 = hp.pixwin(int(nside), lmax=int(lmax), pol=True)
        pw0 = np.asarray(pw0, dtype=np.float64)
        pw2 = np.asarray(pw2, dtype=np.float64)
        pw_ratio = np.ones(int(lmax) + 1, dtype=np.float64)
        valid = np.abs(pw2) > 0.0
        pw_ratio[valid] = pw0[valid] / pw2[valid]
        response *= pw_ratio[np.asarray(_alm_ell_indices(int(lmax)), dtype=np.intp)]

    return _readonly_array(response)


@lru_cache(maxsize=None)
def _reorder_healpix_indices(nside: int, input_nest: bool, output_nest: bool) -> np.ndarray:
    if bool(input_nest) == bool(output_nest):
        raise ValueError("Reorder indices are only needed when the ordering changes.")

    _, hp = _require_glass_healpy()
    nside = int(nside)
    npix = int(hp.nside2npix(nside))
    pix = np.arange(npix, dtype=np.int64)
    if bool(input_nest):
        indices = hp.ring2nest(nside, pix)
    else:
        indices = hp.nest2ring(nside, pix)
    return _readonly_array(np.asarray(indices, dtype=np.int64))


@lru_cache(maxsize=None)
def _pixel_vectors(nside: int, nest: bool) -> np.ndarray:
    _, hp = _require_glass_healpy()
    nside = int(nside)
    npix = int(hp.nside2npix(nside))
    pix = np.arange(npix, dtype=np.int64)
    vec = np.asarray(hp.pix2vec(nside, pix, nest=bool(nest)), dtype=np.float64).T
    return _readonly_array(vec)


def reorder_healpix_array(
    maps: np.ndarray,
    *,
    nside: int,
    input_nest: bool,
    output_nest: bool,
) -> np.ndarray:
    if input_nest == output_nest:
        return np.asarray(maps)

    maps = np.asarray(maps)
    indices = _reorder_healpix_indices(int(nside), bool(input_nest), bool(output_nest))
    return np.take(maps, indices, axis=-1)


def project_observable_kappa_numpy(
    kappa: np.ndarray,
    *,
    nside: int,
    lmax: int,
    input_nest: bool = True,
    output_nest: bool = True,
    ell_min: int = 2,
) -> np.ndarray:
    """Project convergence maps into the shear-observable harmonic subspace.

    Shear does not constrain the monopole or dipole of convergence. To make
    the latent targets comparable to spherical shear reconstructions, we remove
    all modes below `ell_min` from the final kappa maps.
    """

    _, hp = _require_glass_healpy()
    kappa = np.asarray(kappa, dtype=np.float32)
    if kappa.ndim == 1:
        kappa = kappa[None, :]

    keep_alm = _observable_alm_keep_mask(int(lmax), int(ell_min))
    projected = []
    for kappa_map in kappa:
        kappa_ring = (
            reorder_healpix_array(
                kappa_map,
                nside=nside,
                input_nest=True,
                output_nest=False,
            )
            if input_nest
            else kappa_map
        )
        alm = hp.map2alm(kappa_ring, lmax=int(lmax), pol=False, use_pixel_weights=False)
        alm = np.asarray(alm)
        alm[~keep_alm] = 0.0
        kappa_proj = np.asarray(hp.alm2map(alm, nside, lmax=int(lmax)), dtype=np.float32)
        if output_nest:
            kappa_proj = reorder_healpix_array(
                kappa_proj,
                nside=nside,
                input_nest=False,
                output_nest=True,
            )
        projected.append(kappa_proj)

    return np.asarray(projected, dtype=np.float32)


def generate_glass_lognormal_kappa_dataset(
    *,
    n_samples: int,
    nside: int,
    lmax: int | None = None,
    seed: int = 0,
    amplitude: float = 0.9,
    spectral_index: float = 1.25,
    damping: float = 0.006,
    cl_kappa: np.ndarray | None = None,
    gaussian_sigma: float = 0.8,
    nest: bool = True,
    progress_every: int = 0,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Draw log-normal convergence maps with GLASS on a HEALPix sphere.

    The GLASS notebook in this folder samples a Gaussian convergence field and
    normalizes it before visualizing. This helper keeps the same GLASS draw but
    applies the planar shear example's log-normal transform so the spherical
    dataset has the same positive-skewed latent character.
    """

    if n_samples < 1:
        raise ValueError("`n_samples` must be at least 1.")
    if nside < 1:
        raise ValueError("`nside` must be positive.")

    glass, hp = _require_glass_healpy()
    lmax = default_lmax(nside) if lmax is None else int(lmax)
    rng = np.random.default_rng(seed)
    if cl_kappa is None:
        cl_kappa = glass_kappa_cls(
            lmax,
            amplitude=amplitude,
            spectral_index=spectral_index,
            damping=damping,
        )
    else:
        cl_kappa = validate_kappa_cls(cl_kappa, lmax=lmax)
    gls = glass.discretized_cls([cl_kappa], lmax=lmax, nside=nside)

    fields = []
    for sample_idx in range(n_samples):
        field = np.asarray(
            next(glass.generate([glass.grf.Normal()], gls, nside, rng=rng)),
            dtype=np.float32,
        )
        field = hp.remove_dipole(field, fitval=False)
        field = np.ma.filled(field, 0.0).astype(np.float32)
        field = field - np.mean(field)
        field = field / (np.std(field) + 1e-6)
        kappa = np.exp(gaussian_sigma * field - 0.5 * gaussian_sigma**2) - 1.0
        kappa = project_observable_kappa_numpy(
            kappa,
            nside=nside,
            lmax=lmax,
            input_nest=False,
            output_nest=nest,
            ell_min=2,
        )[0]
        if nest:
            kappa = np.asarray(kappa, dtype=np.float32)
        fields.append(np.asarray(kappa, dtype=np.float32))
        if progress is not None and progress_every > 0:
            current = sample_idx + 1
            if current == n_samples or current % progress_every == 0:
                progress(f"generated {current}/{n_samples} GLASS kappa maps")

    return np.asarray(fields, dtype=np.float32)[:, None, :]


def generate_glass_gaussian_kappa_dataset(
    *,
    n_samples: int,
    nside: int,
    lmax: int | None = None,
    seed: int = 0,
    amplitude: float = 0.9,
    spectral_index: float = 1.25,
    damping: float = 0.006,
    cl_kappa: np.ndarray | None = None,
    remove_dipole: bool = True,
    nest: bool = True,
    progress_every: int = 0,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Draw Gaussian convergence maps with GLASS on a HEALPix sphere.

    Unlike `generate_glass_lognormal_kappa_dataset`, this preserves the Gaussian
    random field sampled from the input angular power spectrum. The optional
    dipole removal is a linear projection, so the resulting latent prior stays
    Gaussian in the subspace used for reconstruction.
    """

    if n_samples < 1:
        raise ValueError("`n_samples` must be at least 1.")
    if nside < 1:
        raise ValueError("`nside` must be positive.")

    glass, hp = _require_glass_healpy()
    lmax = default_lmax(nside) if lmax is None else int(lmax)
    rng = np.random.default_rng(seed)
    if cl_kappa is None:
        cl_kappa = glass_kappa_cls(
            lmax,
            amplitude=amplitude,
            spectral_index=spectral_index,
            damping=damping,
        )
    else:
        cl_kappa = validate_kappa_cls(cl_kappa, lmax=lmax)
    gls = glass.discretized_cls([cl_kappa], lmax=lmax, nside=nside)

    fields = []
    for sample_idx in range(n_samples):
        kappa = np.asarray(
            next(glass.generate([glass.grf.Normal()], gls, nside, rng=rng)),
            dtype=np.float32,
        )
        if remove_dipole:
            kappa = hp.remove_dipole(kappa, fitval=False)
        kappa = np.ma.filled(kappa, 0.0).astype(np.float32)
        kappa = project_observable_kappa_numpy(
            kappa,
            nside=nside,
            lmax=lmax,
            input_nest=False,
            output_nest=nest,
            ell_min=2,
        )[0]
        fields.append(np.asarray(kappa, dtype=np.float32))
        if progress is not None and progress_every > 0:
            current = sample_idx + 1
            if current == n_samples or current % progress_every == 0:
                progress(f"generated {current}/{n_samples} Gaussian GLASS kappa maps")

    return np.asarray(fields, dtype=np.float32)[:, None, :]


def spherical_shear_numpy(
    kappa: np.ndarray,
    *,
    nside: int,
    lmax: int,
    input_nest: bool = True,
    output_nest: bool = True,
    progress_every: int = 0,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    glass, hp = _require_glass_healpy()
    kappa = np.asarray(kappa, dtype=np.float32)
    if kappa.ndim == 1:
        kappa = kappa[None, :]

    # Reorder the whole batch once so the per-map loop only pays for the
    # spherical harmonic transform itself.
    kappa_ring_batch = (
        reorder_healpix_array(
            kappa,
            nside=nside,
            input_nest=True,
            output_nest=False,
        )
        if input_nest
        else kappa
    )

    shears = []
    for sample_idx, kappa_ring in enumerate(kappa_ring_batch):
        gamma = glass.from_convergence(kappa_ring, lmax=int(lmax), shear=True)[0]
        gamma1 = np.asarray(gamma.real, dtype=np.float32)
        gamma2 = np.asarray(gamma.imag, dtype=np.float32)
        shears.append(np.stack([gamma1, gamma2], axis=0))
        if progress is not None and progress_every > 0:
            current = sample_idx + 1
            if current == kappa_ring_batch.shape[0] or current % progress_every == 0:
                progress(f"computed {current}/{kappa_ring_batch.shape[0]} spherical shear maps")

    shears = np.asarray(shears, dtype=np.float32)
    if output_nest:
        shears = reorder_healpix_array(
            shears,
            nside=nside,
            input_nest=False,
            output_nest=True,
        )
    return np.asarray(shears, dtype=np.float32)


def spherical_kaiser_squires_numpy(
    gamma: np.ndarray,
    mask: np.ndarray | None,
    *,
    nside: int,
    lmax: int,
    input_nest: bool = True,
    output_nest: bool = True,
    bootstrap_iterations: int = 3,
) -> np.ndarray:
    """Inverse of GLASS shear-from-convergence in harmonic space.

    This matches GLASS's `from_convergence(..., shear=True)` convention,
    including the default discretized pixel-window correction. When a mask is
    present, missing shear pixels are iteratively refilled from the forward
    projection of the current kappa estimate instead of being treated as
    observed zero shear.
    """

    glass, hp = _require_glass_healpy()
    gamma = np.asarray(gamma, dtype=np.float32)
    if gamma.ndim == 2:
        gamma = gamma[None, ...]
    if gamma.ndim != 3 or gamma.shape[1] != 2:
        raise ValueError(f"`gamma` must have shape (B,2,npix), got {gamma.shape}.")

    if mask is None:
        mask = np.ones((gamma.shape[0], 1, gamma.shape[-1]), dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[:, None, :]

    # Reorder the whole batch once so the reconstruction loop can focus on the
    # harmonic inverse and masked bootstrap updates.
    gamma_ring_batch = (
        reorder_healpix_array(
            gamma,
            nside=nside,
            input_nest=True,
            output_nest=False,
        )
        if input_nest
        else gamma
    )
    mask_ring_batch = (
        reorder_healpix_array(
            mask,
            nside=nside,
            input_nest=True,
            output_nest=False,
        )
        if input_nest
        else mask
    )
    mask_ring_batch = (np.asarray(mask_ring_batch, dtype=np.float32) > 0.5).astype(np.float32)

    response = _ks_response(int(nside), int(lmax))
    bootstrap_iterations = max(0, int(bootstrap_iterations))

    def ks_inverse_ring(gamma1_ring: np.ndarray, gamma2_ring: np.ndarray) -> np.ndarray:
        alm_e, _ = hp.map2alm_spin([gamma1_ring, gamma2_ring], spin=2, lmax=int(lmax))
        kappa_alm = alm_e * response
        kappa_rec_ring = np.asarray(
            hp.alm2map(kappa_alm, nside, lmax=int(lmax)),
            dtype=np.float32,
        )
        return project_observable_kappa_numpy(
            kappa_rec_ring,
            nside=nside,
            lmax=lmax,
            input_nest=False,
            output_nest=False,
            ell_min=2,
        )[0]

    reconstructions = []
    for gamma_map, mask_map in zip(gamma_ring_batch, mask_ring_batch):
        gamma1, gamma2 = gamma_map
        mask_ring = np.asarray(mask_map[0], dtype=np.float32)

        observed_gamma1 = np.asarray(gamma1, dtype=np.float32)
        observed_gamma2 = np.asarray(gamma2, dtype=np.float32)
        filled_gamma1 = observed_gamma1.copy()
        filled_gamma2 = observed_gamma2.copy()

        if bootstrap_iterations > 0 and not np.all(mask_ring > 0.5):
            for _ in range(bootstrap_iterations):
                kappa_ring = ks_inverse_ring(filled_gamma1, filled_gamma2)
                gamma_pred = glass.from_convergence(kappa_ring, lmax=int(lmax), shear=True)[0]
                gamma1_pred = np.asarray(gamma_pred.real, dtype=np.float32)
                gamma2_pred = np.asarray(gamma_pred.imag, dtype=np.float32)
                filled_gamma1 = mask_ring * observed_gamma1 + (1.0 - mask_ring) * gamma1_pred
                filled_gamma2 = mask_ring * observed_gamma2 + (1.0 - mask_ring) * gamma2_pred

        reconstructions.append(ks_inverse_ring(filled_gamma1, filled_gamma2).astype(np.float32))

    reconstructions = np.asarray(reconstructions, dtype=np.float32)
    if output_nest:
        reconstructions = reorder_healpix_array(
            reconstructions,
            nside=nside,
            input_nest=False,
            output_nest=True,
        )
    return reconstructions[:, None, :]


def generate_spherical_hole_masks(
    *,
    batch_size: int,
    nside: int,
    rng: np.random.Generator,
    num_holes: int = 22,
    radius_deg: float = 5.5,
    hole_probability: float = 1.0,
    nest: bool = True,
) -> np.ndarray:
    _, hp = _require_glass_healpy()
    npix = hp.nside2npix(int(nside))
    vec = _pixel_vectors(int(nside), bool(nest))
    cos_radius = np.cos(np.deg2rad(radius_deg))
    masks = []

    for _ in range(batch_size):
        mask = np.ones(npix, dtype=bool)
        if num_holes > 0 and radius_deg > 0.0 and hole_probability > 0.0:
            centers = rng.normal(size=(int(num_holes), 3))
            centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
            keep_hole = rng.random(int(num_holes)) < hole_probability
            for center, apply_hole in zip(centers, keep_hole):
                if apply_hole:
                    mask &= (vec @ center) < cos_radius
        masks.append(mask.astype(np.float32))

    return np.asarray(masks, dtype=np.float32)[:, None, :]


def spherical_corruption_from_shear_numpy(
    *,
    gamma_true: np.ndarray,
    rng: np.random.Generator,
    nside: int,
    noise_std: float,
    gamma_scale: float,
    mask_fraction: float,
    hole_radius_deg: float,
    num_holes: int,
    nest: bool = True,
) -> CorruptionBatch:
    gamma_true = np.asarray(gamma_true, dtype=np.float32)
    if gamma_true.ndim != 3 or gamma_true.shape[1] != 2:
        raise ValueError(f"`gamma_true` must have shape (B,2,npix), got {gamma_true.shape}.")

    noise = noise_std * rng.normal(size=gamma_true.shape).astype(np.float32)
    mask = generate_spherical_hole_masks(
        batch_size=int(gamma_true.shape[0]),
        nside=nside,
        rng=rng,
        num_holes=num_holes,
        radius_deg=hole_radius_deg,
        hole_probability=mask_fraction,
        nest=nest,
    )
    gamma_obs = (gamma_true + noise) * mask
    condition = np.concatenate([gamma_obs / float(gamma_scale), mask], axis=1)
    condition = jnp.asarray(condition, dtype=jnp.float32)
    return CorruptionBatch(
        condition=condition,
        observed=condition,
        mask=jnp.asarray(mask, dtype=jnp.float32),
    )


def _seed_from_key(key: jax.Array) -> int:
    words = np.asarray(jax.device_get(jr.key_data(key)), dtype=np.uint32).reshape(-1)
    if words.size == 1:
        return int(words[0])
    return (int(words[0]) << 32) ^ int(words[1])


@dataclass(frozen=True)
class SphericalShearObservationChannel:
    nside: int
    lmax: int
    noise_std: float
    mask_fraction: float
    hole_radius_deg: float
    num_holes: int
    target_mean: float
    target_std: float
    gamma_scale: float
    bootstrap_iterations: int = 3
    nest: bool = True
    progress_every: int = 0
    progress: Callable[[str], None] | None = None

    @property
    def condition_channels(self) -> int:
        return 3

    @property
    def npix(self) -> int:
        _, hp = _require_glass_healpy()
        return int(hp.nside2npix(int(self.nside)))

    def sample(self, key: jax.Array, x: jax.Array) -> CorruptionBatch:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"`x` must have shape (B,1,npix), got {x.shape}.")

        x_np = np.asarray(jax.device_get(x), dtype=np.float32)
        rng = np.random.default_rng(_seed_from_key(key))
        kappa = x_np[:, 0] * float(self.target_std) + float(self.target_mean)
        gamma_true = spherical_shear_numpy(
            kappa,
            nside=self.nside,
            lmax=self.lmax,
            input_nest=self.nest,
            output_nest=self.nest,
            progress_every=self.progress_every,
            progress=self.progress,
        )
        return spherical_corruption_from_shear_numpy(
            gamma_true=gamma_true,
            rng=rng,
            nside=self.nside,
            noise_std=self.noise_std,
            gamma_scale=self.gamma_scale,
            mask_fraction=self.mask_fraction,
            hole_radius_deg=self.hole_radius_deg,
            num_holes=self.num_holes,
            nest=self.nest,
        )

    def sample_from_shear(self, key: jax.Array, gamma_true: np.ndarray) -> CorruptionBatch:
        rng = np.random.default_rng(_seed_from_key(key))
        return spherical_corruption_from_shear_numpy(
            gamma_true=gamma_true,
            rng=rng,
            nside=self.nside,
            noise_std=self.noise_std,
            gamma_scale=self.gamma_scale,
            mask_fraction=self.mask_fraction,
            hole_radius_deg=self.hole_radius_deg,
            num_holes=self.num_holes,
            nest=self.nest,
        )

    def bootstrap_reconstruction(self, condition: jax.Array) -> jax.Array:
        condition_np = np.asarray(jax.device_get(condition), dtype=np.float32)
        if condition_np.ndim != 3 or condition_np.shape[1] != self.condition_channels:
            raise ValueError(
                f"`condition` must have shape (B,{self.condition_channels},npix), "
                f"got {condition_np.shape}."
            )
        gamma = condition_np[:, :2] * float(self.gamma_scale)
        mask = condition_np[:, 2:3]
        kappa_ks = spherical_kaiser_squires_numpy(
            gamma,
            mask,
            nside=self.nside,
            lmax=self.lmax,
            input_nest=self.nest,
            output_nest=self.nest,
            bootstrap_iterations=self.bootstrap_iterations,
        )
        standardized = (kappa_ks - float(self.target_mean)) / float(self.target_std)
        return jnp.asarray(standardized, dtype=jnp.float32)
