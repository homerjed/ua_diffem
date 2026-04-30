from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


def reorder_healpix_array(
    maps: np.ndarray,
    *,
    nside: int,
    input_nest: bool,
    output_nest: bool,
) -> np.ndarray:
    if input_nest == output_nest:
        return np.asarray(maps)

    _, hp = _require_glass_healpy()
    maps = np.asarray(maps)
    flat = maps.reshape((-1, maps.shape[-1]))
    reordered = [
        hp.reorder(m, n2r=True) if input_nest else hp.reorder(m, r2n=True)
        for m in flat
    ]
    return np.asarray(reordered, dtype=maps.dtype).reshape(maps.shape)


def generate_glass_lognormal_kappa_dataset(
    *,
    n_samples: int,
    nside: int,
    lmax: int | None = None,
    seed: int = 0,
    amplitude: float = 0.9,
    spectral_index: float = 1.25,
    damping: float = 0.006,
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
    cl_kappa = glass_kappa_cls(
        lmax,
        amplitude=amplitude,
        spectral_index=spectral_index,
        damping=damping,
    )
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
        if nest:
            kappa = hp.reorder(kappa, r2n=True)
        fields.append(np.asarray(kappa, dtype=np.float32))
        if progress is not None and progress_every > 0:
            current = sample_idx + 1
            if current == n_samples or current % progress_every == 0:
                progress(f"generated {current}/{n_samples} GLASS kappa maps")

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

    shears = []
    for sample_idx, kappa_map in enumerate(kappa):
        kappa_ring = hp.reorder(kappa_map, n2r=True) if input_nest else kappa_map
        gamma = glass.from_convergence(kappa_ring, lmax=int(lmax), shear=True)[0]
        gamma1 = np.asarray(gamma.real, dtype=np.float32)
        gamma2 = np.asarray(gamma.imag, dtype=np.float32)
        if output_nest:
            gamma1 = hp.reorder(gamma1, r2n=True)
            gamma2 = hp.reorder(gamma2, r2n=True)
        shears.append(np.stack([gamma1, gamma2], axis=0))
        if progress is not None and progress_every > 0:
            current = sample_idx + 1
            if current == kappa.shape[0] or current % progress_every == 0:
                progress(f"computed {current}/{kappa.shape[0]} spherical shear maps")
    return np.asarray(shears, dtype=np.float32)


def spherical_kaiser_squires_numpy(
    gamma: np.ndarray,
    mask: np.ndarray | None,
    *,
    nside: int,
    lmax: int,
    input_nest: bool = True,
    output_nest: bool = True,
) -> np.ndarray:
    """Spin-2 spherical Kaiser-Squires-style inversion for HEALPix shear maps."""

    _, hp = _require_glass_healpy()
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

    ell_alm, _ = hp.Alm.getlm(int(lmax))
    response = np.zeros_like(ell_alm, dtype=np.float64)
    good = ell_alm >= 2
    response[good] = 2.0 / np.sqrt((ell_alm[good] - 1) * (ell_alm[good] + 2))

    reconstructions = []
    for gamma_map, mask_map in zip(gamma, mask):
        gamma1, gamma2 = gamma_map
        local_mask = mask_map[0] > 0.5
        if input_nest:
            gamma1 = hp.reorder(gamma1, n2r=True)
            gamma2 = hp.reorder(gamma2, n2r=True)
            local_mask = hp.reorder(local_mask.astype(np.float32), n2r=True) > 0.5

        alm_e, _ = hp.map2alm_spin([gamma1, gamma2], spin=2, lmax=int(lmax))
        kappa_alm = alm_e * response
        kappa_rec = np.asarray(hp.alm2map(kappa_alm, nside, lmax=int(lmax)), dtype=np.float32)
        if np.any(local_mask):
            kappa_rec = kappa_rec - np.mean(kappa_rec[local_mask])
        else:
            kappa_rec = kappa_rec - np.mean(kappa_rec)
        if output_nest:
            kappa_rec = hp.reorder(kappa_rec, r2n=True)
        reconstructions.append(kappa_rec.astype(np.float32))

    return np.asarray(reconstructions, dtype=np.float32)[:, None, :]


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
    pix = np.arange(npix)
    vec = np.asarray(hp.pix2vec(int(nside), pix, nest=nest)).T
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
        )
        standardized = (kappa_ks - float(self.target_mean)) / float(self.target_std)
        return jnp.asarray(standardized, dtype=jnp.float32)
