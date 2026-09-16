#!/usr/bin/env python3
"""Device-resident projected-Adam loop for differentiable LES objectives."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


class DeviceOptimizationResult(NamedTuple):
    initial_value: jax.Array
    best_value: jax.Array
    best_targets: jax.Array
    command_targets: jax.Array
    final_targets: jax.Array
    final_gradient: jax.Array
    update_values: jax.Array
    all_finite: jax.Array


def make_projected_adam_optimizer(
    value_and_grad: Callable[..., Any],
    *,
    update_count: int,
    min_target: float,
    max_target: float,
    command_dimension: int,
) -> Callable[[jax.Array, Any, jax.Array], DeviceOptimizationResult]:
    """Build a pure function whose entire fixed-count update loop can be JITed.

    ``value_and_grad`` must return ``((value, aux), gradient)``.  The equations
    intentionally match the legacy Paper-2 loop; the unit learning-rate scale
    is therefore implicit in the normalized Adam step.
    """

    if update_count < 1:
        raise ValueError("update_count must be positive")
    if not min_target < max_target:
        raise ValueError("min_target must be smaller than max_target")
    if command_dimension < 1:
        raise ValueError("command_dimension must be positive")

    def optimize(initial_targets, farm_state, forecast_planes):
        (initial_value, _), initial_gradient = value_and_grad(
            initial_targets, farm_state, forecast_planes
        )
        zero_moment = jnp.zeros_like(initial_targets)
        initial_carry = (
            initial_targets,
            zero_moment,
            zero_moment,
            initial_value,
            initial_targets,
            initial_gradient,
        )
        min_value = jnp.asarray(min_target, dtype=initial_targets.dtype)
        max_value = jnp.asarray(max_target, dtype=initial_targets.dtype)

        def update(carry, iteration):
            targets, first_moment, second_moment, best_value, best_targets, gradient = carry
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * jnp.square(gradient)
            step = jnp.asarray(iteration + 1, dtype=initial_targets.dtype)
            first_hat = first_moment / (1.0 - 0.9**step)
            second_hat = second_moment / (1.0 - 0.999**step)
            targets = jnp.clip(
                targets + first_hat / (jnp.sqrt(second_hat) + 1.0e-8),
                min_value,
                max_value,
            )
            (value, _), gradient = value_and_grad(
                targets, farm_state, forecast_planes
            )
            choose = value > best_value
            best_value = jnp.where(choose, value, best_value)
            best_targets = jnp.where(choose, targets, best_targets)
            return (
                targets,
                first_moment,
                second_moment,
                best_value,
                best_targets,
                gradient,
            ), value

        final_carry, update_values = jax.lax.scan(
            update,
            initial_carry,
            jnp.arange(update_count, dtype=jnp.int32),
        )
        final_targets, _, _, best_value, best_targets, final_gradient = final_carry
        if command_dimension > initial_targets.shape[-1]:
            raise ValueError("command_dimension exceeds the target dimension")
        command_targets = best_targets[..., :command_dimension]
        all_finite = (
            jnp.all(jnp.isfinite(initial_value))
            & jnp.all(jnp.isfinite(best_value))
            & jnp.all(jnp.isfinite(best_targets))
            & jnp.all(jnp.isfinite(command_targets))
            & jnp.all(jnp.isfinite(final_targets))
            & jnp.all(jnp.isfinite(final_gradient))
            & jnp.all(jnp.isfinite(update_values))
        )
        return DeviceOptimizationResult(
            initial_value=initial_value,
            best_value=best_value,
            best_targets=best_targets,
            command_targets=command_targets,
            final_targets=final_targets,
            final_gradient=final_gradient,
            update_values=update_values,
            all_finite=all_finite,
        )

    return optimize
