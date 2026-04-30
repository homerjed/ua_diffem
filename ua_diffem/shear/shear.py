from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ua_diffem.diffem import CorruptionBatch


def ks_kernel_numpy(image_size: int) -> tuple[np.ndarray, np.ndarray]:
    freq = np.fft.fftfreq(image_size) * 2.0 * np.pi
    ky, kx = np.meshgrid(freq, freq, indexing="ij")
    denom = kx**2 + ky**2
    safe = denom > 0.0
    d1 = np.zeros_like(denom, dtype=np.float32)
    d2 = np.zeros_like(denom, dtype=np.float32)
    d1[safe] = ((kx[safe] ** 2 - ky[safe] ** 2) / denom[safe]).astype(np.float32)
    d2[safe] = ((2.0 * kx[safe] * ky[safe]) / denom[safe]).astype(np.float32)
    return d1, d2


def ks_kernel_jax(image_size: int) -> tuple[jax.Array, jax.Array]:
    freq = jnp.fft.fftfreq(image_size) * 2.0 * jnp.pi
    ky, kx = jnp.meshgrid(freq, freq, indexing="ij")
    denom = kx**2 + ky**2
    safe = denom > 0.0
    d1 = jnp.where(safe, (kx**2 - ky**2) / jnp.where(safe, denom, 1.0), 0.0)
    d2 = jnp.where(safe, (2.0 * kx * ky) / jnp.where(safe, denom, 1.0), 0.0)
    return d1.astype(jnp.float32), d2.astype(jnp.float32)


def kaiser_squires_shear_numpy(kappa: np.ndarray) -> np.ndarray:
    image_size = int(kappa.shape[-1])
    d1, d2 = ks_kernel_numpy(image_size)
    kappa_hat = np.fft.fft2(kappa, axes=(-2, -1))
    gamma1 = np.fft.ifft2(kappa_hat * d1, axes=(-2, -1)).real
    gamma2 = np.fft.ifft2(kappa_hat * d2, axes=(-2, -1)).real
    return np.stack([gamma1, gamma2], axis=1).astype(np.float32)


def kaiser_squires_shear_jax(kappa: jax.Array) -> jax.Array:
    image_size = int(kappa.shape[-1])
    d1, d2 = ks_kernel_jax(image_size)
    kappa_hat = jnp.fft.fft2(kappa, axes=(-2, -1))
    gamma1 = jnp.fft.ifft2(kappa_hat * d1, axes=(-2, -1)).real
    gamma2 = jnp.fft.ifft2(kappa_hat * d2, axes=(-2, -1)).real
    return jnp.stack([gamma1, gamma2], axis=1).astype(jnp.float32)


def kaiser_squires_inverse_jax(gamma: jax.Array) -> jax.Array:
    image_size = int(gamma.shape[-1])
    d1, d2 = ks_kernel_jax(image_size)
    gamma1_hat = jnp.fft.fft2(gamma[:, 0], axes=(-2, -1))
    gamma2_hat = jnp.fft.fft2(gamma[:, 1], axes=(-2, -1))
    kappa_hat = d1 * gamma1_hat + d2 * gamma2_hat
    return jnp.fft.ifft2(kappa_hat, axes=(-2, -1)).real[:, None].astype(jnp.float32)


def generate_lognormal_kappa_dataset(
    *,
    n_samples: int,
    image_size: int,
    spectral_index: float,
    gaussian_sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(image_size)
    ky, kx = np.meshgrid(freq, freq, indexing="ij")
    k = np.sqrt(kx**2 + ky**2)
    k_min = 1.0 / float(image_size)
    power = (k + k_min) ** (-spectral_index)
    power[0, 0] = 0.0
    sqrt_power = np.sqrt(power).astype(np.float32)

    fields = []
    for _ in range(n_samples):
        white = rng.normal(size=(image_size, image_size)).astype(np.float32)
        colored = np.fft.ifft2(np.fft.fft2(white) * sqrt_power).real
        colored = colored - colored.mean()
        colored = colored / (colored.std() + 1e-6)
        g = gaussian_sigma * colored
        kappa = np.exp(g - 0.5 * gaussian_sigma**2) - 1.0
        fields.append(kappa.astype(np.float32))

    return np.asarray(fields, dtype=np.float32)[:, None]


def standardize_targets(kappa: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mean = float(kappa.mean())
    std = float(kappa.std() + 1e-6)
    return ((kappa - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def apply_standardization(x: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return ((x - stats["mean"]) / stats["std"]).astype(np.float32)


def reverse_standardize(x: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return x * stats["std"] + stats["mean"]


@dataclass(frozen=True)
class ShearObservationChannel:
    image_size: int
    noise_std: float
    mask_fraction: float
    mask_size: int
    num_masks: int
    target_mean: float
    target_std: float
    gamma_scale: float

    @property
    def condition_channels(self) -> int:
        return 2

    def _to_physical_kappa(self, x: jax.Array) -> jax.Array:
        return x * self.target_std + self.target_mean

    def _make_mask(self, key: jax.Array, batch_size: int) -> jax.Array:
        mask = jnp.ones((batch_size, 1, self.image_size, self.image_size), dtype=jnp.float32)

        if self.mask_fraction <= 0.0 or self.num_masks <= 0:
            return mask

        yy = jnp.arange(self.image_size)[None, :, None]
        xx = jnp.arange(self.image_size)[None, None, :]
        mask_size = max(1, min(self.image_size, int(self.mask_size)))

        for mask_idx in range(self.num_masks):
            key_top, key_left, key_apply = jr.split(jr.fold_in(key, mask_idx), 3)
            top = jr.randint(
                key_top,
                (batch_size,),
                minval=0,
                maxval=max(1, self.image_size - mask_size + 1),
            )
            left = jr.randint(
                key_left,
                (batch_size,),
                minval=0,
                maxval=max(1, self.image_size - mask_size + 1),
            )
            apply_mask = jr.bernoulli(key_apply, p=self.mask_fraction, shape=(batch_size,))
            rect = (
                (yy >= top[:, None, None])
                & (yy < (top + mask_size)[:, None, None])
                & (xx >= left[:, None, None])
                & (xx < (left + mask_size)[:, None, None])
                & apply_mask[:, None, None]
            )
            mask = mask * (1.0 - rect[:, None].astype(mask.dtype))
        return mask

    def sample(self, key: jax.Array, x: jax.Array) -> CorruptionBatch:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"`x` must have shape (B,1,H,W), got {x.shape}.")

        key_noise, key_mask = jr.split(key)
        kappa = self._to_physical_kappa(x)[:, 0]
        gamma_true = kaiser_squires_shear_jax(kappa)
        gamma_noise = self.noise_std * jr.normal(key_noise, shape=gamma_true.shape, dtype=gamma_true.dtype)
        mask = self._make_mask(key_mask, int(x.shape[0]))
        gamma_obs = (gamma_true + gamma_noise) * mask
        condition = gamma_obs / self.gamma_scale
        return CorruptionBatch(condition=condition, observed=condition, mask=mask)

    def bootstrap_reconstruction(self, condition: jax.Array) -> jax.Array:
        gamma_obs = condition * self.gamma_scale
        kappa_ks = kaiser_squires_inverse_jax(gamma_obs)
        return ((kappa_ks - self.target_mean) / self.target_std).astype(jnp.float32)
