#!/usr/bin/env python3
"""Pure numerical contracts used by the corrected V0a grid-response gate.

This module deliberately has no JAX dependency so the physical-coordinate
definitions can be regression-tested outside the pinned GPU environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LinearInterpolationStencil:
    lower_index: int
    upper_index: int
    upper_weight: float


@dataclass(frozen=True)
class PhysicalQuadrature:
    coordinates_m: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class InterpolationStencils:
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    upper_weights: np.ndarray


@dataclass(frozen=True)
class HybridWakeCheck:
    passed: bool
    acceptance_route: str
    absolute_difference_m: float
    relative_difference: float | None
    material_sign_same: bool


def component_statistics(planes: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute per-component statistics without relying on array contiguity."""

    values = np.asarray(planes)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("planes must have shape (time, 3, ny, nz)")
    components = [
        np.asarray(values[:, index], dtype=np.float64) for index in range(3)
    ]
    return (
        [float(component.mean()) for component in components],
        [float(component.std()) for component in components],
    )


def actuator_kernel_thickness(
    diameter_m: float,
    yaw_degrees: float,
    reference_dx_m: float,
    reference_dz_m: float,
) -> float:
    """Return the actuator smoothing thickness on a fixed physical grid."""

    if min(diameter_m, reference_dx_m, reference_dz_m) <= 0.0:
        raise ValueError("diameter and reference spacings must be positive")
    yaw = math.radians(yaw_degrees)
    normal_spacing = math.hypot(
        reference_dx_m * math.cos(yaw),
        reference_dz_m * math.sin(yaw),
    )
    return max(diameter_m / 8.0, 1.5 * normal_spacing)


def linear_interpolation_stencil(
    coordinate_m: float, spacing_m: float, point_count: int
) -> LinearInterpolationStencil:
    """Map a physical coordinate to a bounded cell-centred linear stencil."""

    if spacing_m <= 0.0 or point_count < 2:
        raise ValueError("spacing must be positive and point_count at least two")
    maximum = spacing_m * (point_count - 1)
    if coordinate_m < 0.0 or coordinate_m > maximum:
        raise ValueError(
            f"coordinate {coordinate_m} outside grid interval [0, {maximum}]"
        )
    position = coordinate_m / spacing_m
    lower = min(int(math.floor(position)), point_count - 2)
    upper_weight = position - lower
    if coordinate_m == maximum:
        upper_weight = 1.0
    return LinearInterpolationStencil(
        lower_index=lower,
        upper_index=lower + 1,
        upper_weight=float(upper_weight),
    )


def linear_interpolation_stencils(
    coordinates_m: np.ndarray, spacing_m: float, point_count: int
) -> InterpolationStencils:
    """Vector form of :func:`linear_interpolation_stencil`."""

    coordinates = np.asarray(coordinates_m, dtype=np.float64)
    stencils = [
        linear_interpolation_stencil(float(value), spacing_m, point_count)
        for value in coordinates.flat
    ]
    shape = coordinates.shape
    return InterpolationStencils(
        lower_indices=np.asarray(
            [item.lower_index for item in stencils], dtype=np.int32
        ).reshape(shape),
        upper_indices=np.asarray(
            [item.upper_index for item in stencils], dtype=np.int32
        ).reshape(shape),
        upper_weights=np.asarray(
            [item.upper_weight for item in stencils], dtype=np.float64
        ).reshape(shape),
    )


def periodic_interpolation_stencils(
    coordinates_m: np.ndarray, spacing_m: float, point_count: int
) -> InterpolationStencils:
    """Map physical coordinates to a periodic uniform-grid stencil."""

    if spacing_m <= 0.0 or point_count < 2:
        raise ValueError("spacing must be positive and point_count at least two")
    coordinates = np.asarray(coordinates_m, dtype=np.float64)
    domain_length = spacing_m * point_count
    positions = np.mod(coordinates, domain_length) / spacing_m
    lower = np.floor(positions).astype(np.int32) % point_count
    return InterpolationStencils(
        lower_indices=lower,
        upper_indices=(lower + 1) % point_count,
        upper_weights=positions - np.floor(positions),
    )


def common_rotor_y_quadrature(
    hub_height_m: float, diameter_m: float, point_count: int
) -> PhysicalQuadrature:
    """Define one physical rotor-height band with trapezoidal weights."""

    if diameter_m <= 0.0 or point_count < 2:
        raise ValueError("diameter must be positive and point_count at least two")
    lower = hub_height_m - 0.5 * diameter_m
    upper = hub_height_m + 0.5 * diameter_m
    if lower < 0.0:
        raise ValueError("rotor-height quadrature extends below the domain")
    coordinates = np.linspace(lower, upper, point_count, dtype=np.float64)
    weights = np.ones(point_count, dtype=np.float64)
    weights[[0, -1]] = 0.5
    weights /= np.sum(weights)
    return PhysicalQuadrature(coordinates_m=coordinates, weights=weights)


def hybrid_wake_response_check(
    nominal_response_m: float,
    refined_response_m: float,
    diameter_m: float,
    relative_threshold: float,
    absolute_threshold_d: float,
) -> HybridWakeCheck:
    """Evaluate a response with relative and near-zero absolute tolerances."""

    values = np.asarray(
        [nominal_response_m, refined_response_m, diameter_m], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or diameter_m <= 0.0:
        return HybridWakeCheck(False, "failed", math.inf, None, False)
    absolute = abs(nominal_response_m - refined_response_m)
    absolute_pass = absolute <= absolute_threshold_d * diameter_m
    denominator = abs(refined_response_m)
    relative = None if denominator <= 1.0e-12 else absolute / denominator
    material = max(abs(nominal_response_m), abs(refined_response_m)) > (
        absolute_threshold_d * diameter_m
    )
    sign_same = bool(
        not material
        or (
            nominal_response_m != 0.0
            and refined_response_m != 0.0
            and np.sign(nominal_response_m) == np.sign(refined_response_m)
        )
    )
    relative_pass = bool(
        relative is not None and relative <= relative_threshold and sign_same
    )
    if relative_pass:
        route = "relative"
    elif absolute_pass and sign_same:
        route = "absolute"
    else:
        route = "failed"
    return HybridWakeCheck(
        passed=route != "failed",
        acceptance_route=route,
        absolute_difference_m=float(absolute),
        relative_difference=None if relative is None else float(relative),
        material_sign_same=sign_same,
    )
