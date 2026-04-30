from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linked_flow.src.models.unet import (
    RandomOrLearnedSinusoidalPosEmb,
    SinusoidalPosEmb,
    UNet as BaseUNet,
)
from ua_flow import UAFlow


class SpatialConditionedUNet2D(eqx.Module):
    """UNet posterior network for UA flow matching.

    The model predicts a velocity mean and log standard deviation. The corrupted
    observation is passed as spatial conditioning channels `q`, which is the
    most direct fit for image inverse problems such as MNIST inpainting.
    """

    backbone: BaseUNet
    time_embed: RandomOrLearnedSinusoidalPosEmb | SinusoidalPosEmb
    time_in: eqx.nn.Linear
    time_out: eqx.nn.Linear
    cond_channels: int
    time_dim: int

    def __init__(
        self,
        *,
        channels: int,
        cond_channels: int,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2),
        time_dim: int = 128,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10_000,
        dropout: float = 0.0,
        attn_dim_head: int = 32,
        attn_heads: int = 2,
        full_attn: bool = False,
        flash_attn: bool = False,
        key: PRNGKeyArray,
    ):
        key_time, key_backbone = jr.split(key)
        key_time_embed, key_time_in, key_time_out = jr.split(key_time, 3)

        if learned_sinusoidal_cond or random_fourier_features:
            self.time_embed = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim,
                is_random=random_fourier_features,
                key=key_time_embed,
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            self.time_embed = SinusoidalPosEmb(time_dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = time_dim

        self.time_in = eqx.nn.Linear(fourier_dim, time_dim, key=key_time_in)
        self.time_out = eqx.nn.Linear(time_dim, time_dim, key=key_time_out)
        self.cond_channels = cond_channels
        self.time_dim = time_dim

        self.backbone = BaseUNet(
            dim=dim,
            out_dim=channels * 2,
            dim_mults=dim_mults,
            channels=channels,
            q_channels=cond_channels,
            a_dim=time_dim,
            learned_variance=False,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            random_fourier_features=random_fourier_features,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            sinusoidal_pos_emb_theta=sinusoidal_pos_emb_theta,
            dropout=dropout,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            full_attn=full_attn,
            flash_attn=flash_attn,
            key=key_backbone,
        )

    def _time_embedding(self, time: Array) -> Array:
        features = jnp.ravel(self.time_embed(jnp.asarray(time, dtype=jnp.float32)))
        hidden = jax.nn.gelu(self.time_in(features))
        return self.time_out(hidden)

    def make_null_condition(self, cond: Array | None) -> Array | None:
        if cond is None:
            return None
        return jnp.zeros_like(jnp.asarray(cond))

    def __call__(
        self,
        x: Array,
        *,
        time: Array,
        cond: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array]:
        if cond is None:
            cond = jnp.zeros((self.cond_channels, *x.shape[-2:]), dtype=x.dtype)
        else:
            cond = jnp.asarray(cond, dtype=x.dtype)

        if cond.ndim != x.ndim:
            raise ValueError(f"`cond` must have shape (C,H,W), got {cond.shape}.")
        if cond.shape[0] != self.cond_channels:
            raise ValueError(
                f"Expected {self.cond_channels} conditioning channels, got {cond.shape[0]}."
            )

        output = self.backbone(x, q=cond, a=self._time_embedding(time), key=key)
        mean, log_sigma = jnp.split(output, 2, axis=0)
        return mean, log_sigma


@dataclass(frozen=True)
class UAFlowConfig:
    image_size: int = 28
    nside: int = 16
    channels: int = 1
    cond_channels: int = 2
    model_name: str = "unet"
    model_dim: int = 32
    dim_mults: tuple[int, ...] = (1, 2)
    time_dim: int = 128
    dropout: float = 0.0
    attn_dim_head: int = 32
    attn_heads: int = 2
    dit_patch_size: int = 4
    dit_depth: int = 4
    dit_hidden_ratio: int = 4
    spherical_chebyshev_order: int = 3
    max_timesteps: int = 100
    beta_nll: float = 1.0
    covariance_probes: int = 1
    covariance_mode: str = "zero"
    top_k_uncertainty_ratio: float = 0.1


def build_ua_flow(config: UAFlowConfig, *, key: PRNGKeyArray) -> UAFlow:
    """Build the uncertainty-aware posterior flow used inside DiffEM."""

    if config.model_name == "unet":
        model = SpatialConditionedUNet2D(
            channels=config.channels,
            cond_channels=config.cond_channels,
            dim=config.model_dim,
            dim_mults=config.dim_mults,
            time_dim=config.time_dim,
            dropout=config.dropout,
            attn_dim_head=config.attn_dim_head,
            attn_heads=config.attn_heads,
            key=key,
        )
    elif config.model_name == "dit":
        from .dit import BasicDiT

        model = BasicDiT(
            image_size=config.image_size,
            channels=config.channels,
            cond_channels=config.cond_channels,
            patch_size=config.dit_patch_size,
            embed_dim=config.model_dim,
            depth=config.dit_depth,
            n_heads=config.attn_heads,
            hidden_ratio=config.dit_hidden_ratio,
            dropout=config.dropout,
            key=key,
        )
    elif config.model_name == "spherical_unet":
        from .spherical_unet import SphericalDeepSphereUNet

        model = SphericalDeepSphereUNet(
            nside=config.nside,
            channels=config.channels,
            cond_channels=config.cond_channels,
            dim=config.model_dim,
            dim_mults=config.dim_mults,
            time_dim=config.time_dim,
            chebyshev_order=config.spherical_chebyshev_order,
            dropout=config.dropout,
            key=key,
        )
    else:
        raise ValueError(f"Unknown UA posterior model {config.model_name!r}.")

    return UAFlow(
        model=model,
        max_timesteps=config.max_timesteps,
        beta_nll=config.beta_nll,
        covariance_probes=config.covariance_probes,
        covariance_mode=config.covariance_mode,
        top_k_uncertainty_ratio=config.top_k_uncertainty_ratio,
    )
