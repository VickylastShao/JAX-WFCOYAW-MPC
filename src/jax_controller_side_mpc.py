#!/usr/bin/env python3
"""Differentiable C1 actuator and LES rollout for Paper 2.

The frozen first-stage runner uses an immediate, horizon-constant yaw.  This
separate module implements the second-stage contract: target commands, exact
per-LES-step rate-limited yaw position, and two move blocks.  Keeping it
separate preserves the first-stage source and evidence hashes.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import FlowState


Array = jax.Array


class MoveBlockedRolloutResult(NamedTuple):
    farm: FlowState
    final_yaw_deg: Array
    objective_mw: Array
    mean_power_mw: Array
    angle_penalty_mw: Array
    turbine_mean_power_watts: Array
    yaw_travel_deg: Array
    saturation_fraction: Array
    maximum_abs_yaw_deg: Array


def rate_limited_target_step(
    yaw_deg: Array,
    target_deg: Array,
    *,
    max_increment_deg: float,
    min_yaw_deg: float,
    max_yaw_deg: float,
) -> Array:
    increment = jnp.clip(
        target_deg - yaw_deg,
        -np.float32(max_increment_deg),
        np.float32(max_increment_deg),
    )
    return jnp.clip(
        yaw_deg + increment,
        np.float32(min_yaw_deg),
        np.float32(max_yaw_deg),
    )


def make_move_blocked_les_rollout(
    case: Any,
    farm_functions: dict[str, Any],
    *,
    horizon_les_steps: int = 1000,
    first_block_les_steps: int = 600,
    checkpoint_block_les_steps: int = 10,
    max_yaw_rate_deg_per_s: float = 0.3,
    min_yaw_deg: float = -30.0,
    max_yaw_deg: float = 30.0,
    angle_penalty_scale_mw: float = 0.5,
    angle_penalty_exponent: int = 26,
) -> Callable[[FlowState, Array, Array, Array], MoveBlockedRolloutResult]:
    """Build a differentiable N-turbine C1 rollout with exact yaw dynamics.

    The returned function accepts `(initial_farm, initial_yaw_deg,
    flat_targets_deg, inlet_planes)`.  Targets have shape `[batch, 2*N]` and
    inlet planes have shape `[batch, horizon_steps, 3, ny, nz]`.
    """

    if horizon_les_steps < 1:
        raise ValueError("horizon_les_steps must be positive")
    if not 0 < first_block_les_steps < horizon_les_steps:
        raise ValueError("first move block must split the horizon")
    if checkpoint_block_les_steps < 1:
        raise ValueError("checkpoint block must be positive")
    if horizon_les_steps % checkpoint_block_les_steps:
        raise ValueError("checkpoint block must divide the horizon")
    if max_yaw_rate_deg_per_s <= 0.0:
        raise ValueError("maximum yaw rate must be positive")
    if min_yaw_deg >= max_yaw_deg:
        raise ValueError("yaw bounds are invalid")

    batch = int(case.batch)
    num_turbines = int(case.num_turbines)
    max_increment_deg = max_yaw_rate_deg_per_s * float(case.dt)
    block_count = horizon_les_steps // checkpoint_block_les_steps

    def rollout(
        initial_farm: FlowState,
        initial_yaw_deg: Array,
        flat_targets_deg: Array,
        inlet_planes: Array,
    ) -> MoveBlockedRolloutResult:
        if initial_yaw_deg.shape != (batch, num_turbines):
            raise ValueError("initial yaw shape does not match the case")
        if flat_targets_deg.shape != (batch, 2 * num_turbines):
            raise ValueError("C1 requires two targets per turbine")
        if inlet_planes.shape != (
            batch,
            horizon_les_steps,
            3,
            int(case.ny),
            int(case.nz),
        ):
            raise ValueError("inlet-plane tensor does not match the rollout")

        targets = jnp.clip(
            flat_targets_deg.reshape((batch, 2, num_turbines)),
            np.float32(min_yaw_deg),
            np.float32(max_yaw_deg),
        )
        initial_power_sum = jnp.zeros(
            (batch, num_turbines), dtype=jnp.float32
        )
        initial_penalty_sum = jnp.zeros((batch,), dtype=jnp.float32)
        initial_travel = jnp.zeros((batch,), dtype=jnp.float32)
        initial_saturation = jnp.zeros((batch,), dtype=jnp.float32)
        initial_maximum = jnp.max(jnp.abs(initial_yaw_deg), axis=-1)

        def advance_checkpoint(
            carry: tuple[FlowState, Array, Array, Array, Array, Array, Array],
            checkpoint_index: Array,
        ) -> tuple[
            tuple[FlowState, Array, Array, Array, Array, Array, Array], None
        ]:
            start_step = checkpoint_index * checkpoint_block_les_steps

            def advance_step(
                step_carry: tuple[
                    FlowState, Array, Array, Array, Array, Array, Array
                ],
                local_step: Array,
            ) -> tuple[
                tuple[FlowState, Array, Array, Array, Array, Array, Array], None
            ]:
                (
                    farm,
                    yaw_deg,
                    power_sum,
                    penalty_sum,
                    yaw_travel,
                    saturation_count,
                    maximum_abs_yaw,
                ) = step_carry
                global_step = start_step + local_step
                target = jnp.where(
                    global_step < first_block_les_steps,
                    targets[:, 0],
                    targets[:, 1],
                )
                next_yaw = rate_limited_target_step(
                    yaw_deg,
                    target,
                    max_increment_deg=max_increment_deg,
                    min_yaw_deg=min_yaw_deg,
                    max_yaw_deg=max_yaw_deg,
                )
                disk_weights = farm_functions["disk_weights"](next_yaw)
                inlet = jax.lax.dynamic_index_in_dim(
                    inlet_planes,
                    global_step,
                    axis=1,
                    keepdims=False,
                )
                next_farm, power_watts = farm_functions["advance"](
                    farm, next_yaw, disk_weights, inlet
                )
                step_penalty = jnp.mean(
                    np.float32(angle_penalty_scale_mw)
                    * (
                        next_yaw / np.float32(max(abs(min_yaw_deg), max_yaw_deg))
                    )
                    ** angle_penalty_exponent,
                    axis=-1,
                )
                at_bound = jnp.mean(
                    (
                        jnp.abs(next_yaw)
                        >= np.float32(max(abs(min_yaw_deg), max_yaw_deg))
                    ).astype(jnp.float32),
                    axis=-1,
                )
                return (
                    next_farm,
                    next_yaw,
                    power_sum + power_watts,
                    penalty_sum + step_penalty,
                    yaw_travel + jnp.sum(jnp.abs(next_yaw - yaw_deg), axis=-1),
                    saturation_count + at_bound,
                    jnp.maximum(maximum_abs_yaw, jnp.max(jnp.abs(next_yaw), axis=-1)),
                ), None

            return jax.lax.scan(
                advance_step,
                carry,
                jnp.arange(checkpoint_block_les_steps, dtype=jnp.int32),
            )[0], None

        final_carry, _ = jax.lax.scan(
            jax.checkpoint(advance_checkpoint, prevent_cse=False),
            (
                initial_farm,
                initial_yaw_deg,
                initial_power_sum,
                initial_penalty_sum,
                initial_travel,
                initial_saturation,
                initial_maximum,
            ),
            jnp.arange(block_count, dtype=jnp.int32),
        )
        (
            final_farm,
            final_yaw,
            power_sum,
            penalty_sum,
            yaw_travel,
            saturation_count,
            maximum_abs_yaw,
        ) = final_carry
        turbine_mean_power_watts = power_sum / np.float32(horizon_les_steps)
        mean_power_mw = jnp.mean(turbine_mean_power_watts, axis=-1) / np.float32(
            1.0e6
        )
        angle_penalty_mw = penalty_sum / np.float32(horizon_les_steps)
        per_member_objective = mean_power_mw - angle_penalty_mw
        return MoveBlockedRolloutResult(
            final_farm,
            final_yaw,
            jnp.mean(per_member_objective),
            jnp.mean(mean_power_mw),
            jnp.mean(angle_penalty_mw),
            turbine_mean_power_watts,
            yaw_travel,
            saturation_count / np.float32(horizon_les_steps),
            maximum_abs_yaw,
        )

    return rollout
