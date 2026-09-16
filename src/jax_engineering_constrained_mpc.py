#!/usr/bin/env python3
"""Differentiable engineering-constrained C2 LES rollout for Paper 2.

The immutable C1 implementation remains in :mod:`jax_controller_side_mpc`.
This module adds only the prospectively frozen C2 terminal-window value and
two-move execution regularizer while preserving raw electrical-power outputs.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_controller_side_mpc import rate_limited_target_step
from jax_mole_compact_sac_benchmark import FlowState


Array = jax.Array
_ALLOWED_TERMINAL_WEIGHTS = frozenset({0.0, 0.25, 0.5})
_ALLOWED_MOVEMENT_PENALTIES_MW = frozenset({0.0, 0.0225, 0.045})


class EngineeringConstrainedRolloutResult(NamedTuple):
    farm: FlowState
    final_yaw_deg: Array
    objective_mw: Array
    mean_power_mw: Array
    tail_power_mw: Array
    terminal_power_mw: Array
    angle_penalty_mw: Array
    movement_cost_dimensionless: Array
    movement_penalty_mw: Array
    turbine_mean_power_watts: Array
    yaw_travel_deg: Array
    saturation_fraction: Array
    maximum_abs_yaw_deg: Array


def _movement_regularizer_per_member(initial_yaw_deg: Array, flat_targets_deg: Array) -> Array:
    if initial_yaw_deg.ndim != 2:
        raise ValueError("initial yaw must have shape [batch, turbine]")
    batch, num_turbines = initial_yaw_deg.shape
    if flat_targets_deg.shape != (batch, 2 * num_turbines):
        raise ValueError("targets must contain two moves per turbine")
    targets = flat_targets_deg.reshape((batch, 2, num_turbines))
    first_delta = (targets[:, 0] - initial_yaw_deg) / np.float32(30.0)
    second_delta = (targets[:, 1] - targets[:, 0]) / np.float32(30.0)
    return np.float32(0.5) * jnp.mean(
        jnp.square(first_delta) + jnp.square(second_delta), axis=-1
    )


def movement_regularizer(initial_yaw_deg: Array, flat_targets_deg: Array) -> Array:
    """Return the batch-mean frozen two-move execution regularizer."""

    return jnp.mean(_movement_regularizer_per_member(initial_yaw_deg, flat_targets_deg))


def make_engineering_constrained_les_rollout(
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
    terminal_tail_seconds: int = 40,
    terminal_weight: float = 0.25,
    movement_penalty_mw: float = 0.0225,
) -> Callable[[FlowState, Array, Array, Array], EngineeringConstrainedRolloutResult]:
    """Build the frozen two-block C2 differentiable LES rollout."""

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
    if terminal_weight not in _ALLOWED_TERMINAL_WEIGHTS:
        raise ValueError("unfrozen terminal weight")
    if movement_penalty_mw not in _ALLOWED_MOVEMENT_PENALTIES_MW:
        raise ValueError("unfrozen movement penalty")
    if terminal_tail_seconds <= 0:
        raise ValueError("terminal-tail duration must be positive")

    tail_steps_float = float(terminal_tail_seconds) / float(case.dt)
    terminal_tail_steps = int(round(tail_steps_float))
    if not np.isclose(tail_steps_float, terminal_tail_steps, rtol=0.0, atol=1.0e-9):
        raise ValueError("terminal-tail duration must contain an integer number of LES steps")
    if terminal_tail_steps > horizon_les_steps:
        raise ValueError("terminal tail exceeds the differentiable horizon")

    batch = int(case.batch)
    num_turbines = int(case.num_turbines)
    max_increment_deg = max_yaw_rate_deg_per_s * float(case.dt)
    block_count = horizon_les_steps // checkpoint_block_les_steps
    terminal_tail_start = horizon_les_steps - terminal_tail_steps

    def rollout(
        initial_farm: FlowState,
        initial_yaw_deg: Array,
        flat_targets_deg: Array,
        inlet_planes: Array,
    ) -> EngineeringConstrainedRolloutResult:
        if initial_yaw_deg.shape != (batch, num_turbines):
            raise ValueError("initial yaw shape does not match the case")
        if flat_targets_deg.shape != (batch, 2 * num_turbines):
            raise ValueError("C2 requires two targets per turbine")
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
        initial_power_sum = jnp.zeros((batch, num_turbines), dtype=jnp.float32)
        initial_tail_power_sum = jnp.zeros((batch, num_turbines), dtype=jnp.float32)
        initial_penalty_sum = jnp.zeros((batch,), dtype=jnp.float32)
        initial_travel = jnp.zeros((batch,), dtype=jnp.float32)
        initial_saturation = jnp.zeros((batch,), dtype=jnp.float32)
        initial_maximum = jnp.max(jnp.abs(initial_yaw_deg), axis=-1)

        def advance_checkpoint(carry, checkpoint_index):
            start_step = checkpoint_index * checkpoint_block_les_steps

            def advance_step(step_carry, local_step):
                (
                    farm,
                    yaw_deg,
                    power_sum,
                    tail_power_sum,
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
                    inlet_planes, global_step, axis=1, keepdims=False
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
                is_tail = global_step >= terminal_tail_start
                return (
                    next_farm,
                    next_yaw,
                    power_sum + power_watts,
                    tail_power_sum + jnp.where(is_tail, power_watts, jnp.zeros_like(power_watts)),
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
                initial_tail_power_sum,
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
            tail_power_sum,
            penalty_sum,
            yaw_travel,
            saturation_count,
            maximum_abs_yaw,
        ) = final_carry

        turbine_mean_power_watts = power_sum / np.float32(horizon_les_steps)
        per_member_mean_power_mw = jnp.mean(turbine_mean_power_watts, axis=-1) / np.float32(1.0e6)
        per_member_tail_power_mw = jnp.mean(
            tail_power_sum / np.float32(terminal_tail_steps), axis=-1
        ) / np.float32(1.0e6)
        per_member_terminal_power_mw = (
            np.float32(1.0 - terminal_weight) * per_member_mean_power_mw
            + np.float32(terminal_weight) * per_member_tail_power_mw
        )
        per_member_angle_penalty_mw = penalty_sum / np.float32(horizon_les_steps)
        per_member_movement_cost = _movement_regularizer_per_member(
            initial_yaw_deg, flat_targets_deg
        )
        per_member_movement_penalty_mw = (
            np.float32(movement_penalty_mw) * per_member_movement_cost
        )
        per_member_objective = (
            per_member_terminal_power_mw
            - per_member_angle_penalty_mw
            - per_member_movement_penalty_mw
        )
        return EngineeringConstrainedRolloutResult(
            final_farm,
            final_yaw,
            jnp.mean(per_member_objective),
            jnp.mean(per_member_mean_power_mw),
            jnp.mean(per_member_tail_power_mw),
            jnp.mean(per_member_terminal_power_mw),
            jnp.mean(per_member_angle_penalty_mw),
            jnp.mean(per_member_movement_cost),
            jnp.mean(per_member_movement_penalty_mw),
            turbine_mean_power_watts,
            yaw_travel,
            saturation_count / np.float32(horizon_les_steps),
            maximum_abs_yaw,
        )

    return rollout
