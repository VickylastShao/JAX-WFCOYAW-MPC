#!/usr/bin/env python3
"""Pure adjudication helpers for E-COMP and G1M.

The functions in this module intentionally do not import JAX.  They make the
dimension, equivalence, timing, projection, and residency rules independently
testable before GPU evidence is produced.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def derivative_dimensions(num_turbines: int) -> tuple[int, int, int]:
    """Return the one-, two-, and four-move-block control dimensions."""

    if num_turbines < 1:
        raise ValueError("num_turbines must be positive")
    return tuple(num_turbines * blocks for blocks in (1, 2, 4))


def summarize_seconds(values: Iterable[float]) -> dict[str, Any]:
    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("at least one timing sample is required")
    if not np.all(np.isfinite(samples)) or np.any(samples < 0.0):
        raise ValueError("timing samples must be finite and non-negative")
    return {
        "count": int(samples.size),
        "median": float(np.median(samples)),
        "q25": float(np.percentile(samples, 25.0)),
        "q75": float(np.percentile(samples, 75.0)),
        "iqr": float(np.percentile(samples, 75.0) - np.percentile(samples, 25.0)),
        "p95": float(np.percentile(samples, 95.0)),
        "maximum": float(np.max(samples)),
        "minimum": float(np.min(samples)),
    }


def backend_equivalence(
    reference_objective: float,
    candidate_objective: float,
    reference_gradient: Sequence[float],
    candidate_gradient: Sequence[float],
    *,
    objective_relative_tolerance: float = 1.0e-4,
    gradient_relative_tolerance: float = 5.0e-3,
) -> dict[str, Any]:
    reference = np.asarray(reference_gradient, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_gradient, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape or reference.size < 1:
        raise ValueError("gradient arrays must have the same non-empty shape")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("gradient arrays must be finite")
    objective_scale = max(abs(reference_objective), abs(candidate_objective), 1.0e-12)
    objective_relative_error = abs(reference_objective - candidate_objective) / objective_scale
    gradient_scale = max(np.linalg.norm(reference), np.linalg.norm(candidate), 1.0e-12)
    gradient_relative_error = float(np.linalg.norm(reference - candidate) / gradient_scale)
    component_scale = np.maximum(np.maximum(np.abs(reference), np.abs(candidate)), 1.0e-12)
    maximum_component_relative_error = float(np.max(np.abs(reference - candidate) / component_scale))
    objective_passes = bool(objective_relative_error <= objective_relative_tolerance)
    gradient_passes = bool(
        gradient_relative_error <= gradient_relative_tolerance
        or maximum_component_relative_error <= gradient_relative_tolerance
    )
    return {
        "passes": objective_passes and gradient_passes,
        "objective_passes": objective_passes,
        "gradient_passes": gradient_passes,
        "objective_relative_error": float(objective_relative_error),
        "gradient_norm_relative_error": gradient_relative_error,
        "maximum_component_relative_error": maximum_component_relative_error,
        "objective_relative_tolerance": objective_relative_tolerance,
        "gradient_relative_tolerance": gradient_relative_tolerance,
    }


def project_central_fd_time(
    *,
    target_dimension: int,
    forward_call_seconds: float,
    measured_sweeps: Mapping[int, float],
    calibration_relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Project 2*d forward calls only after two dimensions validate the rule."""

    if target_dimension < 1 or not np.isfinite(forward_call_seconds) or forward_call_seconds <= 0.0:
        raise ValueError("target dimension and forward-call time must be positive")
    if len(measured_sweeps) < 2:
        raise ValueError("two complete measured sweeps are required")
    calibration = []
    for dimension, measured_seconds in sorted(measured_sweeps.items()):
        expected = 2.0 * int(dimension) * forward_call_seconds
        relative_error = abs(float(measured_seconds) - expected) / expected
        calibration.append(
            {
                "dimension": int(dimension),
                "measured_seconds": float(measured_seconds),
                "forward_call_projection_seconds": expected,
                "relative_error": relative_error,
            }
        )
    if any(item["relative_error"] > calibration_relative_tolerance for item in calibration):
        raise ValueError("measured finite-difference sweeps do not validate projection")
    return {
        "label": "projected",
        "target_dimension": int(target_dimension),
        "projected_seconds": 2.0 * target_dimension * forward_call_seconds,
        "calibration_relative_tolerance": calibration_relative_tolerance,
        "calibration": calibration,
    }


def paired_ratio_bootstrap(
    numerator_seconds: Sequence[float],
    denominator_seconds: Sequence[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Bootstrap the median of complete paired-sweep timing ratios."""

    numerator = np.asarray(numerator_seconds, dtype=np.float64)
    denominator = np.asarray(denominator_seconds, dtype=np.float64)
    if numerator.ndim != 1 or numerator.shape != denominator.shape or numerator.size < 2:
        raise ValueError("two or more matched one-dimensional timing pairs are required")
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("paired timings must be finite")
    if np.any(numerator < 0.0) or np.any(denominator <= 0.0):
        raise ValueError("numerator timings must be non-negative and denominators positive")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    ratios = numerator / denominator
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        bootstrap[index] = np.median(rng.choice(ratios, size=ratios.size, replace=True))
    return {
        "paired_ratios": ratios.tolist(),
        "median": float(np.median(ratios)),
        "bootstrap_95_interval": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(replicates),
        "resampling_unit": "complete_interleaved_sweep",
    }


def residency_label(transfers: Sequence[Mapping[str, Any]]) -> str:
    """Assign only the two labels allowed by the frozen E-COMP-5 contract."""

    non_command = [item for item in transfers if item.get("purpose") != "command_download"]
    if not non_command:
        return "end-to-end GPU-resident MPC"
    return "GPU-native LES with host-orchestrated MPC"
