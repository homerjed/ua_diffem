from __future__ import annotations

from dataclasses import dataclass
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .dit import timestep_embedding


def _require_healpy():
    try:
        import healpy as hp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The spherical DeepSphere model requires `healpy` to build the "
            "HEALPix graph. Install `healpy` or use the Singularity definition."
        ) from exc
    return hp


@dataclass(frozen=True)
class HealpixGraph:
    nside: int
    neighbors: np.ndarray
    laplacian_weights: np.ndarray


def build_healpix_graph(nside: int, *, nest: bool = True) -> HealpixGraph:
    """Build the normalized HEALPix neighbor graph used by DeepSphere layers."""

    hp = _require_healpy()
    npix = hp.nside2npix(int(nside))
    pixels = np.arange(npix, dtype=np.int64)
    neighbors = np.asarray(hp.get_all_neighbours(int(nside), pixels, nest=nest)).T
    valid = neighbors >= 0
    safe_neighbors = np.where(valid, neighbors, 0).astype(np.int32)

    degree = np.maximum(valid.sum(axis=1).astype(np.float32), 1.0)
    neighbor_degree = degree[safe_neighbors]
    weights = np.where(
        valid,
        1.0 / np.sqrt(degree[:, None] * neighbor_degree),
        0.0,
    ).astype(np.float32)
    return HealpixGraph(
        nside=int(nside),
        neighbors=safe_neighbors,
        laplacian_weights=weights,
    )


class DeepSphereConv(eqx.Module):
    """Chebyshev graph convolution on a HEALPix DeepSphere graph."""

    weight: Array
    bias: Array
    neighbors: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    laplacian_weights: tuple[tuple[float, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        graph: HealpixGraph,
        *,
        kernel_size: int = 3,
        key: PRNGKeyArray,
    ):
        if kernel_size < 1:
            raise ValueError("`kernel_size` must be at least 1.")

        scale = 1.0 / math.sqrt(max(1, in_channels * kernel_size))
        self.weight = scale * jr.normal(
            key,
            (int(kernel_size), int(out_channels), int(in_channels)),
            dtype=jnp.float32,
        )
        self.bias = jnp.zeros((int(out_channels), 1), dtype=jnp.float32)
        self.neighbors = tuple(tuple(int(value) for value in row) for row in graph.neighbors)
        self.laplacian_weights = tuple(
            tuple(float(value) for value in row) for row in graph.laplacian_weights
        )

    def _scaled_laplacian(self, x: Array) -> Array:
        neighbors = jnp.asarray(self.neighbors, dtype=jnp.int32)
        weights = jnp.asarray(self.laplacian_weights, dtype=x.dtype)
        gathered = jnp.take(x, neighbors, axis=1)
        return -jnp.sum(gathered * weights[None, :, :], axis=-1)

    def __call__(self, x: Array) -> Array:
        if x.ndim != 2:
            raise ValueError(f"`x` must have shape (C,npix), got {x.shape}.")

        cheb_terms = [x]
        if self.weight.shape[0] > 1:
            cheb_terms.append(self._scaled_laplacian(x))
        for _ in range(2, self.weight.shape[0]):
            cheb_terms.append(2.0 * self._scaled_laplacian(cheb_terms[-1]) - cheb_terms[-2])

        features = jnp.stack(cheb_terms, axis=0)
        return jnp.einsum("koi,kin->on", self.weight, features) + self.bias


class SphericalResBlock(eqx.Module):
    conv_in: DeepSphereConv
    conv_out: DeepSphereConv
    time_proj: eqx.nn.Linear
    skip: DeepSphereConv | None
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        graph: HealpixGraph,
        *,
        time_dim: int,
        kernel_size: int,
        dropout: float,
        key: PRNGKeyArray,
    ):
        key_in, key_time, key_out, key_skip = jr.split(key, 4)
        self.conv_in = DeepSphereConv(
            in_channels,
            out_channels,
            graph,
            kernel_size=kernel_size,
            key=key_in,
        )
        self.conv_out = DeepSphereConv(
            out_channels,
            out_channels,
            graph,
            kernel_size=kernel_size,
            key=key_out,
        )
        self.time_proj = eqx.nn.Linear(time_dim, out_channels, key=key_time)
        self.skip = (
            None
            if in_channels == out_channels
            else DeepSphereConv(in_channels, out_channels, graph, kernel_size=1, key=key_skip)
        )
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        x: Array,
        time_emb: Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        key_dropout = key
        residual = x if self.skip is None else self.skip(x)
        h = self.conv_in(x)
        h = h + self.time_proj(time_emb)[:, None]
        h = jax.nn.gelu(h)
        h = self.dropout(h, key=key_dropout)
        h = self.conv_out(h)
        return jax.nn.gelu(h + residual)


class SphericalDeepSphereUNet(eqx.Module):
    """Spherical UA-flow velocity network using DeepSphere Chebyshev layers.

    Inputs are HEALPix maps in NESTED order with shape `(channels, npix)`.
    Conditions are concatenated as additional graph-signal channels. Pooling and
    unpooling use the HEALPix NESTED hierarchy, where each lower-resolution
    parent pixel owns four contiguous children.
    """

    nside: int
    channels: int
    cond_channels: int
    dims: tuple[int, ...]
    time_dim: int
    input_proj: DeepSphereConv
    time_in: eqx.nn.Linear
    time_out: eqx.nn.Linear
    encoder_blocks: list[SphericalResBlock]
    down_blocks: list[SphericalResBlock]
    mid_block: SphericalResBlock
    decoder_blocks: list[SphericalResBlock]
    output_proj: DeepSphereConv

    def __init__(
        self,
        *,
        nside: int,
        channels: int,
        cond_channels: int,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2),
        time_dim: int = 128,
        chebyshev_order: int = 3,
        dropout: float = 0.0,
        key: PRNGKeyArray,
    ):
        if nside < 1:
            raise ValueError("`nside` must be positive.")
        if not dim_mults:
            raise ValueError("`dim_mults` must contain at least one value.")
        if nside % (2 ** (len(dim_mults) - 1)) != 0:
            raise ValueError("`nside` must be divisible by 2 ** (len(dim_mults) - 1).")

        self.nside = int(nside)
        self.channels = int(channels)
        self.cond_channels = int(cond_channels)
        self.dims = tuple(int(dim * mult) for mult in dim_mults)
        self.time_dim = int(time_dim)

        graphs = [
            build_healpix_graph(self.nside // (2**level), nest=True)
            for level in range(len(self.dims))
        ]
        key_input, key_time, key_enc, key_down, key_mid, key_dec, key_output = jr.split(key, 7)
        key_time_in, key_time_out = jr.split(key_time)

        self.input_proj = DeepSphereConv(
            self.channels + self.cond_channels,
            self.dims[0],
            graphs[0],
            kernel_size=chebyshev_order,
            key=key_input,
        )
        self.time_in = eqx.nn.Linear(self.time_dim, self.time_dim, key=key_time_in)
        self.time_out = eqx.nn.Linear(self.time_dim, self.time_dim, key=key_time_out)
        self.encoder_blocks = [
            SphericalResBlock(
                width,
                width,
                graph,
                time_dim=self.time_dim,
                kernel_size=chebyshev_order,
                dropout=dropout,
                key=block_key,
            )
            for width, graph, block_key in zip(
                self.dims,
                graphs,
                jr.split(key_enc, len(self.dims)),
            )
        ]
        self.down_blocks = [
            SphericalResBlock(
                self.dims[level],
                self.dims[level + 1],
                graphs[level + 1],
                time_dim=self.time_dim,
                kernel_size=chebyshev_order,
                dropout=dropout,
                key=block_key,
            )
            for level, block_key in enumerate(jr.split(key_down, max(1, len(self.dims) - 1))[: len(self.dims) - 1])
        ]
        self.mid_block = SphericalResBlock(
            self.dims[-1],
            self.dims[-1],
            graphs[-1],
            time_dim=self.time_dim,
            kernel_size=chebyshev_order,
            dropout=dropout,
            key=key_mid,
        )

        decoder_blocks = []
        current_width = self.dims[-1]
        for level, block_key in zip(
            reversed(range(len(self.dims))),
            jr.split(key_dec, len(self.dims)),
        ):
            decoder_blocks.append(
                SphericalResBlock(
                    current_width + self.dims[level],
                    self.dims[level],
                    graphs[level],
                    time_dim=self.time_dim,
                    kernel_size=chebyshev_order,
                    dropout=dropout,
                    key=block_key,
                )
            )
            current_width = self.dims[level]
        self.decoder_blocks = decoder_blocks
        self.output_proj = DeepSphereConv(
            self.dims[0],
            self.channels * 2,
            graphs[0],
            kernel_size=1,
            key=key_output,
        )

    def _time_embedding(self, time: Array) -> Array:
        emb = timestep_embedding(time, self.time_dim)
        emb = jax.nn.gelu(self.time_in(emb))
        return self.time_out(emb)

    @staticmethod
    def _pool(x: Array) -> Array:
        channels, npix = x.shape
        if npix % 4 != 0:
            raise ValueError("HEALPix NESTED pooling requires npix divisible by 4.")
        return jnp.reshape(x, (channels, npix // 4, 4)).mean(axis=-1)

    @staticmethod
    def _unpool(x: Array) -> Array:
        return jnp.repeat(x, 4, axis=1)

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
        if x.ndim != 2 or x.shape[0] != self.channels:
            raise ValueError(f"`x` must have shape ({self.channels},npix), got {x.shape}.")
        if cond is None:
            cond = jnp.zeros((self.cond_channels, x.shape[-1]), dtype=x.dtype)
        else:
            cond = jnp.asarray(cond, dtype=x.dtype)
        if cond.ndim != 2 or cond.shape[0] != self.cond_channels:
            raise ValueError(
                f"`cond` must have shape ({self.cond_channels},npix), got {cond.shape}."
            )
        if cond.shape[-1] != x.shape[-1]:
            raise ValueError("`cond` and `x` must have the same HEALPix pixel count.")

        n_keys = len(self.encoder_blocks) + len(self.down_blocks) + len(self.decoder_blocks) + 1
        block_keys = [None] * n_keys if key is None else list(jr.split(key, n_keys))
        key_idx = 0

        time_emb = self._time_embedding(time)
        h = self.input_proj(jnp.concatenate([x, cond], axis=0))
        skips = []

        for level, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb, key=block_keys[key_idx])
            key_idx += 1
            skips.append(h)
            if level < len(self.down_blocks):
                h = self._pool(h)
                h = self.down_blocks[level](h, time_emb, key=block_keys[key_idx])
                key_idx += 1

        h = self.mid_block(h, time_emb, key=block_keys[key_idx])
        key_idx += 1

        for decoder_idx, block in enumerate(self.decoder_blocks):
            level = len(self.dims) - 1 - decoder_idx
            if decoder_idx > 0:
                h = self._unpool(h)
            h = jnp.concatenate([h, skips[level]], axis=0)
            h = block(h, time_emb, key=block_keys[key_idx])
            key_idx += 1

        mean, log_sigma = jnp.split(self.output_proj(h), 2, axis=0)
        return mean, log_sigma
