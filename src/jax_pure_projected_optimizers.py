#!/usr/bin/env python3
"""Device-resident, lookup-independent optimizers for pure LES-MPC."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class PureOptimizationResult(NamedTuple):
    initial_value: Array
    best_value: Array
    best_targets: Array
    best_aux: Any
    command_targets: Array
    final_targets: Array
    final_gradient: Array
    update_values: Array
    objective_gradient_evaluations: Array
    all_finite: Array


def _tree_all_finite(tree: Any) -> Array:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))


def _tree_select(condition: Array, candidate: Any, current: Any) -> Any:
    def select(candidate_leaf, current_leaf):
        expanded = jnp.asarray(condition)
        if expanded.ndim < candidate_leaf.ndim:
            expanded = jnp.reshape(
                expanded,
                expanded.shape + (1,) * (candidate_leaf.ndim - expanded.ndim),
            )
        return jnp.where(expanded, candidate_leaf, current_leaf)

    return jax.tree.map(select, candidate, current)


def make_projected_adam_optimizer(
    value_and_grad: Callable[..., Any],
    *,
    projector: Callable[[Array, Array], Array],
    update_count: int,
    command_dimension: int,
    step_scale: float,
) -> Callable[[Array, Array, Any, Array], PureOptimizationResult]:
    """Build a fixed-count projected-Adam maximizer fully expressible in JAX."""

    if update_count < 1:
        raise ValueError("update_count must be positive")
    if command_dimension < 1:
        raise ValueError("command_dimension must be positive")
    if step_scale <= 0.0:
        raise ValueError("step scale must be positive")

    def optimize(initial_targets, initial_yaw, farm_state, forecast_planes):
        targets = projector(initial_targets, initial_yaw)
        (initial_value, initial_aux), initial_gradient = value_and_grad(
            targets, farm_state, forecast_planes
        )
        zeros = jnp.zeros_like(targets)
        initial_carry = (
            targets,
            zeros,
            zeros,
            initial_value,
            targets,
            initial_aux,
            initial_gradient,
        )

        def update(carry, iteration):
            (
                targets,
                first_moment,
                second_moment,
                best_value,
                best_targets,
                best_aux,
                gradient,
            ) = carry
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * jnp.square(gradient)
            step = jnp.asarray(iteration + 1, dtype=targets.dtype)
            first_hat = first_moment / (1.0 - 0.9**step)
            second_hat = second_moment / (1.0 - 0.999**step)
            proposed = targets + jnp.asarray(step_scale, targets.dtype) * first_hat / (
                jnp.sqrt(second_hat) + 1.0e-8
            )
            targets = projector(proposed, initial_yaw)
            (value, aux), gradient = value_and_grad(
                targets, farm_state, forecast_planes
            )
            choose = value > best_value
            return (
                targets,
                first_moment,
                second_moment,
                jnp.where(choose, value, best_value),
                jnp.where(choose, targets, best_targets),
                _tree_select(choose, aux, best_aux),
                gradient,
            ), value

        final_carry, update_values = jax.lax.scan(
            update, initial_carry, jnp.arange(update_count, dtype=jnp.int32)
        )
        (
            final_targets,
            _,
            _,
            best_value,
            best_targets,
            best_aux,
            final_gradient,
        ) = final_carry
        if command_dimension > initial_targets.shape[-1]:
            raise ValueError("command_dimension exceeds the target dimension")
        command_targets = best_targets[..., :command_dimension]
        finite = (
            jnp.all(jnp.isfinite(initial_value))
            & _tree_all_finite(initial_aux)
            & jnp.all(jnp.isfinite(best_value))
            & jnp.all(jnp.isfinite(best_targets))
            & _tree_all_finite(best_aux)
            & jnp.all(jnp.isfinite(final_targets))
            & jnp.all(jnp.isfinite(final_gradient))
            & jnp.all(jnp.isfinite(update_values))
        )
        return PureOptimizationResult(
            initial_value,
            best_value,
            best_targets,
            best_aux,
            command_targets,
            final_targets,
            final_gradient,
            update_values,
            jnp.asarray(update_count + 1, dtype=jnp.int32),
            finite,
        )

    return optimize


def make_projected_lbfgs_grid_optimizer(
    value_and_grad: Callable[..., Any],
    *,
    projector: Callable[[Array, Array], Array],
    update_count: int,
    command_dimension: int,
    line_search_scales: Sequence[float],
    history_size: int,
    serial_line_search: bool = False,
) -> Callable[[Array, Array, Any, Array], PureOptimizationResult]:
    """Build a projected limited-memory BFGS maximizer with a fixed step grid.

    The bounded step grid replaces a host-side strong-Wolfe loop so compilation,
    line-search candidate evaluation and command formation remain on device.
    ``serial_line_search`` streams candidates through one device loop instead of
    batching their complete reverse-mode LES tapes with ``vmap``.
    It is intentionally named ``projected_lbfgs_grid`` rather than L-BFGS-B.
    """

    if update_count < 1:
        raise ValueError("update_count must be positive")
    if command_dimension < 1:
        raise ValueError("command_dimension must be positive")
    if history_size < 1:
        raise ValueError("history_size must be positive")
    scales = tuple(float(scale) for scale in line_search_scales)
    if not scales or any(scale <= 0.0 for scale in scales):
        raise ValueError("line-search scales must be positive")

    def optimize(initial_targets, initial_yaw, farm_state, forecast_planes):
        targets = projector(initial_targets, initial_yaw)
        (initial_value, initial_aux), gradient = value_and_grad(
            targets, farm_state, forecast_planes
        )
        batch, dimension = targets.shape
        s_history = jnp.zeros((history_size, batch, dimension), dtype=targets.dtype)
        y_history = jnp.zeros_like(s_history)
        rho_history = jnp.zeros((history_size, batch), dtype=targets.dtype)
        initial_carry = (
            targets,
            gradient,
            initial_value,
            targets,
            initial_aux,
            s_history,
            y_history,
            rho_history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        scale_array = jnp.asarray(scales, dtype=targets.dtype)

        def two_loop(gradient, s_hist, y_hist, rho_hist, cursor, filled):
            cost_gradient = -gradient
            alpha = jnp.zeros((history_size, batch), dtype=targets.dtype)

            def backward(index, state):
                q, alpha_values = state
                slot = jnp.mod(cursor - 1 - index, history_size)
                valid = index < filled
                s = s_hist[slot]
                y = y_hist[slot]
                rho = rho_hist[slot]
                coefficient = rho * jnp.sum(s * q, axis=-1)
                coefficient = jnp.where(valid, coefficient, jnp.zeros_like(coefficient))
                q = q - coefficient[:, None] * y
                alpha_values = alpha_values.at[index].set(coefficient)
                return q, alpha_values

            q, alpha = jax.lax.fori_loop(0, history_size, backward, (cost_gradient, alpha))
            newest = jnp.mod(cursor - 1, history_size)
            newest_s = s_hist[newest]
            newest_y = y_hist[newest]
            sy = jnp.sum(newest_s * newest_y, axis=-1)
            yy = jnp.sum(newest_y * newest_y, axis=-1)
            gamma = jnp.where(
                filled > 0,
                sy / jnp.maximum(yy, jnp.asarray(1.0e-12, targets.dtype)),
                jnp.ones_like(sy),
            )
            r = gamma[:, None] * q

            def forward(reverse_index, current):
                index = history_size - 1 - reverse_index
                slot = jnp.mod(cursor - 1 - index, history_size)
                valid = index < filled
                s = s_hist[slot]
                y = y_hist[slot]
                rho = rho_hist[slot]
                beta = rho * jnp.sum(y * current, axis=-1)
                correction = jnp.where(
                    valid,
                    alpha[index] - beta,
                    jnp.zeros_like(beta),
                )
                return current + correction[:, None] * s

            inverse_cost_gradient = jax.lax.fori_loop(0, history_size, forward, r)
            ascent = -inverse_cost_gradient
            norm = jnp.maximum(
                jnp.max(jnp.abs(ascent), axis=-1, keepdims=True),
                jnp.asarray(1.0e-12, targets.dtype),
            )
            return ascent / norm

        def update(carry, _):
            (
                targets,
                gradient,
                value,
                best_targets,
                best_aux,
                s_hist,
                y_hist,
                rho_hist,
                cursor,
                filled,
            ) = carry
            direction = two_loop(gradient, s_hist, y_hist, rho_hist, cursor, filled)
            candidates = jax.vmap(
                lambda scale: projector(targets + scale * direction, initial_yaw)
            )(scale_array)

            def evaluate(candidate):
                return value_and_grad(candidate, farm_state, forecast_planes)

            if serial_line_search:
                def evaluate_serial(_, candidate):
                    (candidate_value, candidate_aux), candidate_gradient = evaluate(
                        candidate
                    )
                    return None, (candidate_value, candidate_aux, candidate_gradient)

                _, (candidate_values, candidate_aux, candidate_gradients) = jax.lax.scan(
                    evaluate_serial,
                    None,
                    candidates,
                    unroll=1,
                )
            else:
                (candidate_values, candidate_aux), candidate_gradients = jax.vmap(
                    evaluate
                )(candidates)
            best_index = jnp.argmax(candidate_values)
            proposed_value = candidate_values[best_index]
            proposed_targets = candidates[best_index]
            proposed_aux = jax.tree.map(
                lambda value: value[best_index], candidate_aux
            )
            proposed_gradient = candidate_gradients[best_index]
            accept = jnp.isfinite(proposed_value) & (proposed_value > value)
            next_targets = jnp.where(accept, proposed_targets, targets)
            next_gradient = jnp.where(accept, proposed_gradient, gradient)
            next_value = jnp.where(accept, proposed_value, value)
            next_aux = _tree_select(accept, proposed_aux, best_aux)

            s = next_targets - targets
            y = gradient - next_gradient
            curvature = jnp.sum(s * y, axis=-1)
            valid_curvature = accept & jnp.all(jnp.isfinite(s), axis=-1) & (
                curvature > jnp.asarray(1.0e-8, targets.dtype)
            )
            slot = jnp.mod(cursor, history_size)
            old_s = s_hist[slot]
            old_y = y_hist[slot]
            old_rho = rho_hist[slot]
            s_hist = s_hist.at[slot].set(jnp.where(valid_curvature[:, None], s, old_s))
            y_hist = y_hist.at[slot].set(jnp.where(valid_curvature[:, None], y, old_y))
            rho = 1.0 / jnp.maximum(curvature, jnp.asarray(1.0e-12, targets.dtype))
            rho_hist = rho_hist.at[slot].set(jnp.where(valid_curvature, rho, old_rho))
            stored = jnp.any(valid_curvature)
            cursor = jnp.where(stored, cursor + 1, cursor)
            filled = jnp.where(stored, jnp.minimum(filled + 1, history_size), filled)
            improve_best = next_value > value
            best_targets = jnp.where(improve_best, next_targets, best_targets)
            return (
                next_targets,
                next_gradient,
                next_value,
                best_targets,
                next_aux,
                s_hist,
                y_hist,
                rho_hist,
                cursor,
                filled,
            ), next_value

        final_carry, update_values = jax.lax.scan(
            update, initial_carry, xs=None, length=update_count
        )
        final_targets, final_gradient, best_value, best_targets, best_aux, *_ = final_carry
        if command_dimension > initial_targets.shape[-1]:
            raise ValueError("command_dimension exceeds the target dimension")
        command_targets = best_targets[..., :command_dimension]
        finite = (
            jnp.all(jnp.isfinite(initial_value))
            & _tree_all_finite(initial_aux)
            & jnp.all(jnp.isfinite(best_value))
            & jnp.all(jnp.isfinite(best_targets))
            & _tree_all_finite(best_aux)
            & jnp.all(jnp.isfinite(final_gradient))
            & jnp.all(jnp.isfinite(update_values))
        )
        evaluation_count = 1 + update_count * len(scales)
        return PureOptimizationResult(
            initial_value,
            best_value,
            best_targets,
            best_aux,
            command_targets,
            final_targets,
            final_gradient,
            update_values,
            jnp.asarray(evaluation_count, dtype=jnp.int32),
            finite,
        )

    return optimize


def make_projected_lbfgs_serial_grid_optimizer(
    value_and_grad: Callable[..., Any],
    *,
    projector: Callable[[Array, Array], Array],
    update_count: int,
    command_dimension: int,
    line_search_scales: Sequence[float],
    history_size: int,
) -> Callable[[Array, Array, Any, Array], PureOptimizationResult]:
    """Build the same fixed-grid L-BFGS method with serial device evaluation."""

    return make_projected_lbfgs_grid_optimizer(
        value_and_grad,
        projector=projector,
        update_count=update_count,
        command_dimension=command_dimension,
        line_search_scales=line_search_scales,
        history_size=history_size,
        serial_line_search=True,
    )
