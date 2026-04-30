from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray

from ua_flow import UAFlow
from ua_diffem.utils import DataParallelSharding, shard_batch, shard_replicated_tree


class CorruptionBatch(NamedTuple):
    condition: Array
    observed: Array
    mask: Array


class ReconstructionBatch(NamedTuple):
    samples: Array
    variance: Array
    uncertainty_score: Array


class MStepResult(NamedTuple):
    state: "DiffEMState"
    train_losses: list[float]
    validation_steps: list[int]
    validation_losses: list[float]
    stopped_early: bool


@dataclass(frozen=True)
class InpaintGaussianChannel:
    """Known forward channel Q(y | x) for a small MNIST DiffEM example.

    `condition` is the tensor that the posterior model sees. It concatenates
    the corrupted image and its binary mask, so it has `2 * n_channels`.
    """

    keep_prob: float = 0.35
    noise_std: float = 0.05
    fill_value: float = 0.0
    n_channels: int = 1

    @property
    def condition_channels(self) -> int:
        return 2 * self.n_channels

    def condition_from_observation(self, observed: Array, mask: Array) -> Array:
        return jnp.concatenate([observed, mask.astype(observed.dtype)], axis=1)

    def split_condition(self, condition: Array) -> tuple[Array, Array]:
        observed = condition[:, : self.n_channels]
        mask = condition[:, self.n_channels : self.condition_channels]
        return observed, mask

    def sample(self, key: PRNGKeyArray, x: Array) -> CorruptionBatch:
        if x.ndim != 4:
            raise ValueError(f"`x` must have shape (B,C,H,W), got {x.shape}.")

        key_mask, key_noise = jr.split(key)
        mask = jr.bernoulli(key_mask, p=self.keep_prob, shape=x.shape).astype(x.dtype)
        noise = self.noise_std * jr.normal(key_noise, shape=x.shape, dtype=x.dtype)
        observed = mask * (x + noise) + (1.0 - mask) * self.fill_value
        return CorruptionBatch(
            condition=self.condition_from_observation(observed, mask),
            observed=observed,
            mask=mask,
        )

    def bootstrap_reconstruction(self, condition: Array) -> Array:
        observed, mask = self.split_condition(condition)
        return mask * observed + (1.0 - mask) * self.fill_value


@dataclass(frozen=True)
class DiffEMConfig:
    em_steps: int = 500
    m_steps_per_em: int = 500
    batch_size: int = 32
    posterior_batch_size: int = 16
    posterior_sample_steps: int = 20
    posterior_solver: str = "euler"
    cfg_max_scale: float = 0.0
    ucg_scale: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    bootstrap_first_e_step: bool = True
    m_step_early_stopping: bool = False
    m_step_min_steps: int = 50
    m_step_patience: int = 50
    m_step_min_delta: float = 1e-4
    m_step_validation_fraction: float = 0.1
    m_step_validation_freq: int = 10
    m_step_validation_batches: int = 2
    use_ema: bool = True
    ema_rate: float = 0.999


@dataclass
class DiffEMState:
    flow: UAFlow
    opt_state: optax.OptState
    flow_ema: UAFlow | None = None
    em_step: int = 0
    m_step: int = 0

    @property
    def sampling_flow(self) -> UAFlow:
        return self.flow_ema if self.flow_ema is not None else self.flow


def save_training_state(path, state: DiffEMState) -> None:
    payload = (
        jnp.asarray(state.em_step, dtype=jnp.int32),
        state.flow,
        state.opt_state,
    )
    if state.flow_ema is not None:
        payload = payload + (state.flow_ema,)
    eqx.tree_serialise_leaves(path, payload)


def load_training_state(
    path,
    flow_like: UAFlow,
    opt_state_like: optax.OptState,
    *,
    flow_ema_like: UAFlow | None = None,
) -> DiffEMState:
    payload = (
        jnp.asarray(0, dtype=jnp.int32),
        flow_like,
        opt_state_like,
    )

    if flow_ema_like is not None:
        try:
            em_step, flow, opt_state, flow_ema = eqx.tree_deserialise_leaves(
                path,
                payload + (flow_ema_like,),
            )
            return DiffEMState(
                flow=flow,
                opt_state=opt_state,
                flow_ema=flow_ema,
                em_step=int(em_step),
            )
        except Exception:
            pass

    em_step, flow, opt_state = eqx.tree_deserialise_leaves(path, payload)
    return DiffEMState(
        flow=flow,
        opt_state=opt_state,
        flow_ema=None,
        em_step=int(em_step),
    )


@eqx.filter_jit
def train_step(
    flow: UAFlow,
    opt_state: optax.OptState,
    x: Array,
    cond: Array,
    *,
    key: PRNGKeyArray,
    opt: optax.GradientTransformation,
) -> tuple[UAFlow, optax.OptState, Array]:
    def loss_fn(current_flow: UAFlow) -> Array:
        loss, _ = current_flow.loss(key, x, cond=cond, training=True)
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(flow, eqx.is_array))
    flow = eqx.apply_updates(flow, updates)
    return flow, opt_state, loss


@eqx.filter_jit
def validation_step(
    flow: UAFlow,
    x: Array,
    cond: Array,
    *,
    key: PRNGKeyArray,
) -> Array:
    loss, _ = flow.loss(key, x, cond=cond, training=False)
    return loss


@eqx.filter_jit
def update_ema(target: UAFlow, source: UAFlow, decay: float) -> UAFlow:
    target_arrays, target_static = eqx.partition(target, eqx.is_array)
    source_arrays, _ = eqx.partition(source, eqx.is_array)
    blended = jax.tree_util.tree_map(
        lambda ema_value, src_value: decay * ema_value + (1.0 - decay) * src_value,
        target_arrays,
        source_arrays,
    )
    return eqx.combine(blended, target_static)


@eqx.filter_jit
def _sample_reconstruction_batches(
    flow: UAFlow,
    observation_conditions: Array,
    *,
    key: PRNGKeyArray,
    data_shape: tuple[int, int, int],
    batch_size: int,
    sample_steps: int,
    solver: str,
    cfg_max_scale: float,
    ucg_scale: float,
) -> ReconstructionBatch:
    num_batches = observation_conditions.shape[0] // batch_size

    def scan_batch(
        _: None,
        batch_idx: Array,
    ) -> tuple[None, tuple[Array, Array, Array]]:
        start = batch_idx * batch_size
        cond = jax.lax.dynamic_slice_in_dim(
            observation_conditions,
            start_index=start,
            slice_size=batch_size,
            axis=0,
        )
        result = flow.sample(
            jr.fold_in(key, start),
            batch_size=batch_size,
            data_shape=data_shape,
            cond=cond,
            steps=sample_steps,
            solver=solver,
            cfg_max_scale=cfg_max_scale,
            ucg_scale=ucg_scale,
        )
        return None, (result.samples, result.variance, result.uncertainty_score)

    _, (samples, variances, scores) = jax.lax.scan(
        scan_batch,
        init=None,
        xs=jnp.arange(num_batches),
        length=num_batches,
    )

    return ReconstructionBatch(
        samples=jnp.reshape(samples, (num_batches * batch_size, *samples.shape[2:])),
        variance=jnp.reshape(variances, (num_batches * batch_size, *variances.shape[2:])),
        uncertainty_score=jnp.reshape(scores, (num_batches * batch_size,)),
    )


def e_step_reconstruct(
    flow: UAFlow,
    observation_conditions: Array,
    *,
    key: PRNGKeyArray,
    data_shape: tuple[int, int, int],
    batch_size: int,
    sample_steps: int,
    solver: str = "euler",
    cfg_max_scale: float = 0.0,
    ucg_scale: float = 0.0,
    bootstrap_channel: InpaintGaussianChannel | None = None,
) -> ReconstructionBatch:
    """E-step: sample x_hat ~ p_theta(x | y) for each corrupted observation."""

    observation_conditions = jnp.asarray(observation_conditions)

    if bootstrap_channel is not None:
        samples = bootstrap_channel.bootstrap_reconstruction(observation_conditions)
        variance = jnp.zeros_like(samples)
        uncertainty_score = jnp.zeros((samples.shape[0],), dtype=samples.dtype)
        return ReconstructionBatch(samples, variance, uncertainty_score)

    n = int(observation_conditions.shape[0])
    if n < 1:
        raise ValueError("`observation_conditions` must contain at least one observation.")
    if batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")

    num_batches = (n + batch_size - 1) // batch_size
    padded_n = num_batches * batch_size
    pad_n = padded_n - n
    if pad_n:
        pad_width = [(0, pad_n), *[(0, 0) for _ in range(observation_conditions.ndim - 1)]]
        observation_conditions = jnp.pad(observation_conditions, pad_width)

    reconstructions = _sample_reconstruction_batches(
        flow,
        observation_conditions,
        key=key,
        data_shape=data_shape,
        batch_size=batch_size,
        sample_steps=sample_steps,
        solver=solver,
        cfg_max_scale=cfg_max_scale,
        ucg_scale=ucg_scale,
    )
    return ReconstructionBatch(
        samples=reconstructions.samples[:n],
        variance=reconstructions.variance[:n],
        uncertainty_score=reconstructions.uncertainty_score[:n],
    )


def m_step_train(
    state: DiffEMState,
    reconstructions: Array,
    channel: InpaintGaussianChannel,
    config: DiffEMConfig,
    *,
    key: PRNGKeyArray,
    opt: optax.GradientTransformation,
    sharding: DataParallelSharding | None = None,
) -> MStepResult:
    """M-step: retrain p_theta(x | y) on a fixed corrupted x_hat dataset."""

    flow = shard_replicated_tree(state.flow, sharding)
    opt_state = shard_replicated_tree(state.opt_state, sharding)
    flow_ema = shard_replicated_tree(state.flow_ema, sharding)
    losses: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    n = int(reconstructions.shape[0])
    key_corrupted_dataset, key_split, key_loop = jr.split(key, 3)
    fixed_corrupted = channel.sample(key_corrupted_dataset, reconstructions)
    if n < 2:
        raise ValueError("M-step validation requires at least two reconstructions.")

    validation_fraction = float(config.m_step_validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("`m_step_validation_fraction` must be in (0, 1).")

    validation_count = int(round(n * validation_fraction))
    validation_count = min(n - 1, max(1, validation_count))
    permutation = jr.permutation(key_split, n)
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]

    best_validation_loss = float("inf")
    validation_checks_without_improvement = 0
    stopped_early = False
    min_steps = max(1, int(config.m_step_min_steps))
    patience = max(1, int(config.m_step_patience))
    validation_freq = max(1, int(config.m_step_validation_freq))
    validation_batches = max(1, int(config.m_step_validation_batches))

    def run_validation(step_idx: int, validation_key: PRNGKeyArray) -> float:
        batch_losses = []
        for batch_idx in range(validation_batches):
            key_idx, key_loss = jr.split(jr.fold_in(validation_key, batch_idx))
            local_idx = jr.randint(
                key_idx,
                (config.batch_size,),
                minval=0,
                maxval=validation_indices.shape[0],
            )
            idx = validation_indices[local_idx]
            x_val = shard_batch(reconstructions[idx], sharding)
            cond_val = shard_batch(fixed_corrupted.condition[idx], sharding)
            val_loss = validation_step(flow, x_val, cond_val, key=key_loss)
            batch_losses.append(float(jax.device_get(val_loss)))
        validation_steps.append(step_idx)
        validation_losses.append(float(np.mean(batch_losses)))
        return validation_losses[-1]

    for step in range(config.m_steps_per_em):
        key_step = jr.fold_in(key_loop, step)
        key_idx, key_loss, key_validation = jr.split(key_step, 3)
        local_idx = jr.randint(
            key_idx,
            (config.batch_size,),
            minval=0,
            maxval=train_indices.shape[0],
        )
        idx = train_indices[local_idx]
        x_batch = reconstructions[idx]
        condition_batch = fixed_corrupted.condition[idx]
        x_batch = shard_batch(x_batch, sharding)
        condition = shard_batch(condition_batch, sharding)
        flow, opt_state, loss = train_step(
            flow,
            opt_state,
            x_batch,
            condition,
            key=key_loss,
            opt=opt,
        )
        if config.use_ema:
            if flow_ema is None:
                flow_ema = flow
            else:
                flow_ema = update_ema(flow_ema, flow, config.ema_rate)
        loss_value = float(jax.device_get(loss))
        losses.append(loss_value)

        should_validate = (step + 1) % validation_freq == 0 or step == config.m_steps_per_em - 1
        if should_validate:
            validation_loss = run_validation(len(losses) - 1, key_validation)

            if validation_loss < best_validation_loss - config.m_step_min_delta:
                best_validation_loss = validation_loss
                validation_checks_without_improvement = 0
            elif len(losses) >= min_steps:
                validation_checks_without_improvement += 1

            if (
                config.m_step_early_stopping
                and len(losses) >= min_steps
                and validation_checks_without_improvement >= patience
            ):
                stopped_early = True
                break

    next_state = DiffEMState(
        flow=flow,
        opt_state=opt_state,
        flow_ema=flow_ema,
        em_step=state.em_step + 1,
        m_step=state.m_step + len(losses),
    )
    return MStepResult(
        state=next_state,
        train_losses=losses,
        validation_steps=validation_steps,
        validation_losses=validation_losses,
        stopped_early=stopped_early,
    )
