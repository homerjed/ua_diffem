from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, PRNGKeyArray


def timestep_embedding(time: Array, dim: int, max_period: int = 10_000) -> Array:
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half) / max(half, 1))
    args = jnp.asarray(time, dtype=jnp.float32) * freqs
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        emb = jnp.concatenate([emb, jnp.zeros((1,), dtype=emb.dtype)], axis=-1)
    return emb


class FeedForward(eqx.Module):
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, dim: int, hidden_ratio: int, dropout: float, *, key: PRNGKeyArray):
        key_in, key_out = jr.split(key)
        self.proj_in = eqx.nn.Linear(dim, dim * hidden_ratio, key=key_in)
        self.proj_out = eqx.nn.Linear(dim * hidden_ratio, dim, key=key_out)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        key_hidden, key_out = (None, None) if key is None else jr.split(key)
        x = self.proj_in(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, key=key_hidden)
        x = self.proj_out(x)
        return self.dropout(x, key=key_out)


class TransformerBlock(eqx.Module):
    norm_attn: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    norm_ff: eqx.nn.LayerNorm
    ff: FeedForward

    def __init__(
        self,
        dim: int,
        n_heads: int,
        hidden_ratio: int,
        dropout: float,
        *,
        key: PRNGKeyArray,
    ):
        key_attn, key_ff = jr.split(key)
        self.norm_attn = eqx.nn.LayerNorm(dim)
        self.attention = eqx.nn.MultiheadAttention(n_heads, dim, key=key_attn)
        self.norm_ff = eqx.nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_ratio, dropout, key=key_ff)

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        key_ff = None if key is None else key
        attn_in = jax.vmap(self.norm_attn)(x)
        x = x + self.attention(attn_in, attn_in, attn_in)
        ff_in = jax.vmap(self.norm_ff)(x)
        x = x + jax.vmap(lambda token: self.ff(token, key=key_ff))(ff_in)
        return x


class BasicDiT(eqx.Module):
    """Small image-conditioned DiT-style posterior network.

    This is intentionally compact. It concatenates the flow state and corrupted
    observation channels before patchifying, adds a learned positional embedding
    plus a time embedding, and predicts UA-flow velocity mean/log-sigma fields.
    """

    image_size: int
    patch_size: int
    channels: int
    cond_channels: int
    embed_dim: int
    patch_embed: eqx.nn.Conv2d
    pos_embedding: Array
    time_in: eqx.nn.Linear
    time_out: eqx.nn.Linear
    blocks: list[TransformerBlock]
    norm: eqx.nn.LayerNorm
    unpatch: eqx.nn.ConvTranspose2d

    def __init__(
        self,
        *,
        image_size: int,
        channels: int,
        cond_channels: int,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        hidden_ratio: int = 4,
        dropout: float = 0.0,
        key: PRNGKeyArray,
    ):
        if image_size % patch_size != 0:
            raise ValueError("`image_size` must be divisible by `patch_size`.")

        key_patch, key_pos, key_time, key_blocks, key_unpatch = jr.split(key, 5)
        key_time_in, key_time_out = jr.split(key_time)

        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.cond_channels = cond_channels
        self.embed_dim = embed_dim
        self.patch_embed = eqx.nn.Conv2d(
            channels + cond_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            key=key_patch,
        )

        n_tokens = (image_size // patch_size) ** 2
        self.pos_embedding = 0.02 * jr.normal(key_pos, (n_tokens, embed_dim))
        self.time_in = eqx.nn.Linear(embed_dim, embed_dim, key=key_time_in)
        self.time_out = eqx.nn.Linear(embed_dim, embed_dim, key=key_time_out)
        self.blocks = [
            TransformerBlock(
                embed_dim,
                n_heads,
                hidden_ratio,
                dropout,
                key=block_key,
            )
            for block_key in jr.split(key_blocks, depth)
        ]
        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.unpatch = eqx.nn.ConvTranspose2d(
            embed_dim,
            channels * 2,
            kernel_size=patch_size,
            stride=patch_size,
            key=key_unpatch,
        )

    def make_null_condition(self, cond: Array | None) -> Array | None:
        if cond is None:
            return None
        return jnp.zeros_like(jnp.asarray(cond))

    def _time_embedding(self, time: Array) -> Array:
        emb = timestep_embedding(time, self.embed_dim)
        emb = jax.nn.gelu(self.time_in(emb))
        return self.time_out(emb)

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

        x = jnp.concatenate([x, cond], axis=0)
        x = self.patch_embed(x)
        x = rearrange(x, "c h w -> (h w) c")
        x = x + self.pos_embedding + self._time_embedding(time)

        block_keys = [None] * len(self.blocks) if key is None else list(jr.split(key, len(self.blocks)))
        for block, block_key in zip(self.blocks, block_keys):
            x = block(x, key=block_key)

        x = jax.vmap(self.norm)(x)
        x = rearrange(
            x,
            "(h w) c -> c h w",
            h=self.image_size // self.patch_size,
            w=self.image_size // self.patch_size,
        )
        mean, log_sigma = jnp.split(self.unpatch(x), 2, axis=0)
        return mean, log_sigma
