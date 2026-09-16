#!/usr/bin/env python3
"""Absolute constraints and rollout assembly for pure LES-MPC."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


Array = jax.Array


def two_move_total_variation(initial_yaw_deg: Array, flat_targets_deg: Array) -> Array:
    """Return absolute scheduled travel for two targets, one value per batch member."""

    if initial_yaw_deg.ndim != 2:
        raise ValueError("initial yaw must have shape [batch, turbine]")
    batch, num_turbines = initial_yaw_deg.shape
    if flat_targets_deg.shape != (batch, 2 * num_turbines):
        raise ValueError("targets must contain two moves per turbine")
    targets = flat_targets_deg.reshape((batch, 2, num_turbines))
    return jnp.sum(jnp.abs(targets[:, 0] - initial_yaw_deg), axis=-1) + jnp.sum(
        jnp.abs(targets[:, 1] - targets[:, 0]), axis=-1
    )


def scale_two_move_targets_to_budget(
    flat_targets_deg: Array,
    initial_yaw_deg: Array,
    *,
    absolute_budget_deg: float,
    min_yaw_deg: float,
    max_yaw_deg: float,
) -> Array:
    """Scale both move increments uniformly to satisfy an absolute L1 budget.

    This is a deterministic feasibility map, not an Euclidean projection.  It
    preserves the proposed two-move direction while imposing yaw bounds and a
    hard, lookup-independent scheduled-travel budget.
    """

    if absolute_budget_deg <= 0.0:
        raise ValueError("absolute yaw budget must be positive")
    if min_yaw_deg >= max_yaw_deg:
        raise ValueError("yaw bounds are invalid")
    if initial_yaw_deg.ndim != 2:
        raise ValueError("initial yaw must have shape [batch, turbine]")
    batch, num_turbines = initial_yaw_deg.shape
    if flat_targets_deg.shape != (batch, 2 * num_turbines):
        raise ValueError("targets must contain two moves per turbine")

    bounded_initial = jnp.clip(initial_yaw_deg, min_yaw_deg, max_yaw_deg)
    bounded_targets = jnp.clip(flat_targets_deg, min_yaw_deg, max_yaw_deg).reshape(
        (batch, 2, num_turbines)
    )
    first_increment = bounded_targets[:, 0] - bounded_initial
    second_increment = bounded_targets[:, 1] - bounded_targets[:, 0]
    proposed_variation = jnp.sum(jnp.abs(first_increment), axis=-1) + jnp.sum(
        jnp.abs(second_increment), axis=-1
    )
    scale = jnp.minimum(
        jnp.ones_like(proposed_variation),
        jnp.asarray(absolute_budget_deg, dtype=flat_targets_deg.dtype)
        / jnp.maximum(proposed_variation, jnp.asarray(1.0e-12, flat_targets_deg.dtype)),
    )
    first = bounded_initial + scale[:, None] * first_increment
    second = first + scale[:, None] * second_increment
    return jnp.concatenate((first, second), axis=-1)


def make_absolute_budget_projector(
    *,
    absolute_budget_deg: float | None,
    min_yaw_deg: float,
    max_yaw_deg: float,
) -> Callable[[Array, Array], Array]:
    """Build the pure target feasibility map used by both candidate solvers."""

    if min_yaw_deg >= max_yaw_deg:
        raise ValueError("yaw bounds are invalid")
    if absolute_budget_deg is not None and absolute_budget_deg <= 0.0:
        raise ValueError("absolute yaw budget must be positive")

    def project(flat_targets_deg: Array, initial_yaw_deg: Array) -> Array:
        bounded = jnp.clip(flat_targets_deg, min_yaw_deg, max_yaw_deg)
        if absolute_budget_deg is None:
            return bounded
        return scale_two_move_targets_to_budget(
            bounded,
            initial_yaw_deg,
            absolute_budget_deg=absolute_budget_deg,
            min_yaw_deg=min_yaw_deg,
            max_yaw_deg=max_yaw_deg,
        )

    return project


def make_pure_terminal_window_les_rollout(
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
    terminal_window_seconds: int = 40,
    terminal_weight: float = 0.25,
    movement_penalty_mw: float = 0.0,
) -> Callable[..., Any]:
    """Assemble a pure LES objective with no movement-reference controller.

    The terminal value is the final physical window *inside* the 200 s LES
    horizon.  It is not a learned terminal model and does not extrapolate the
    causal forecast beyond the supplied inlet planes.
    """

    from jax_engineering_constrained_mpc import make_engineering_constrained_les_rollout

    return make_engineering_constrained_les_rollout(
        case,
        farm_functions,
        horizon_les_steps=horizon_les_steps,
        first_block_les_steps=first_block_les_steps,
        checkpoint_block_les_steps=checkpoint_block_les_steps,
        max_yaw_rate_deg_per_s=max_yaw_rate_deg_per_s,
        min_yaw_deg=min_yaw_deg,
        max_yaw_deg=max_yaw_deg,
        angle_penalty_scale_mw=angle_penalty_scale_mw,
        angle_penalty_exponent=angle_penalty_exponent,
        terminal_tail_seconds=terminal_window_seconds,
        terminal_weight=terminal_weight,
        movement_penalty_mw=movement_penalty_mw,
    )
