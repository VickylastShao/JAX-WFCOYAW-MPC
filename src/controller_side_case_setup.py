#!/usr/bin/env python3
"""Shared, non-outcome case preparation for G1M and G2 runners."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from controller_side_mpc_protocol import ControllerContract
from jax_mole_compact_sac_benchmark import (
    FlowState,
    _apply_velocity_boundaries,
    build_flow_functions,
    make_initial_flow,
    make_mole_training_case,
    make_scaled_mole_training_case,
)
from jax_mole_continuous_sac import (
    ContinuousControlConfig,
    _advance_coupled_interval,
    _time_plane_marker,
)
from jax_mole_layouts import canonical_scale_layouts, rectangular_mole_layout
from mole_plane_archive import PlaneArchive


def block_tree(value: Any) -> Any:
    return jax.tree.map(
        lambda item: item.block_until_ready()
        if hasattr(item, "block_until_ready")
        else item,
        value,
    )


@dataclass
class PreparedControllerCase:
    case: Any
    farm_functions: dict[str, Any]
    delayed_farm: FlowState
    mature_farm: FlowState
    zero_yaw: jax.Array
    delay_host: np.ndarray
    forecast_host: np.ndarray
    forecast_device: jax.Array
    contract: ControllerContract
    archive: PlaneArchive
    source_nz: int
    preparation_timing_seconds: dict[str, Any]
    transfer_inventory: list[dict[str, Any]]
    delay_executable: Any


def _make_case(turbines: int, rhs_assembly: str) -> Any:
    if turbines == 3:
        return make_mole_training_case(1, rhs_assembly=rhs_assembly)
    if turbines == 6:
        return make_scaled_mole_training_case(
            rectangular_mole_layout(3, 2), batch=1, rhs_assembly=rhs_assembly
        )
    if turbines == 9:
        return make_scaled_mole_training_case(
            canonical_scale_layouts()[9], batch=1, rhs_assembly=rhs_assembly
        )
    raise ValueError("G1M/G2 setup supports N=3, N=6, or N=9")


def prepare_controller_case(
    *,
    turbines: int,
    plane_archive: Path,
    seed: int,
    plane_offset: int = 0,
    burnin_interactions: int = 150,
    checkpoint_block_les_steps: int = 10,
    rhs_assembly: str = "tensor",
    case_override: Any | None = None,
) -> PreparedControllerCase:
    contract = ControllerContract(num_turbines=turbines)
    contract.validate()
    case = _make_case(turbines, rhs_assembly) if case_override is None else case_override
    if int(case.num_turbines) != turbines:
        raise ValueError("case override turbine count does not match turbines")
    archive = PlaneArchive(plane_archive)
    if archive.plane_shape[:2] != (3, case.ny):
        raise ValueError("archive components/y grid do not match the case")
    source_nz = int(archive.plane_shape[2])
    if source_nz < int(case.nz):
        raise ValueError("archive z grid is narrower than the case")
    burnin_steps = burnin_interactions * 50
    required_planes = burnin_steps + contract.delay_steps + contract.horizon_steps
    if plane_offset + required_planes > archive.completed_planes:
        raise ValueError("archive does not cover burn-in, delay, and horizon")

    farm_functions = build_flow_functions(case, periodic_x=False, include_turbines=True)
    host_all = archive.read(plane_offset, required_planes)[..., : int(case.nz)]
    initial_host_plane = host_all[:1]
    burnin_host = host_all[:burnin_steps]
    delay_host = np.ascontiguousarray(
        host_all[burnin_steps : burnin_steps + contract.delay_steps],
        dtype=np.float32,
    )
    forecast_host = np.ascontiguousarray(
        host_all[
            burnin_steps + contract.delay_steps : burnin_steps
            + contract.delay_steps
            + contract.horizon_steps
        ],
        dtype=np.float32,
    )
    del host_all

    transfer_inventory: list[dict[str, Any]] = []

    def place(name: str, value: np.ndarray, *, online: bool) -> jax.Array:
        started = time.perf_counter()
        placed = jnp.asarray(value[None], dtype=jnp.float32)
        block_tree(placed)
        transfer_inventory.append(
            {
                "purpose": name,
                "direction": "H2D",
                "shape": list(value[None].shape),
                "bytes": int(value[None].nbytes),
                "inside_optimization_loop": False,
                "inside_online_decision": online,
                "seconds": time.perf_counter() - started,
            }
        )
        return placed

    initial_plane = place("initial_inlet_placement", initial_host_plane, online=False)
    burnin_planes = place("burnin_archive_placement", burnin_host, online=False)
    delay_planes = place("scheduled_delay_forecast_upload", delay_host, online=True)
    forecast_device = place("optimization_forecast_upload", forecast_host, online=True)
    del burnin_host

    initial_farm = make_initial_flow(case, farm_functions, seed)
    initial_farm = initial_farm._replace(
        velocity=_apply_velocity_boundaries(
            initial_farm.velocity, initial_plane[:, 0], periodic_x=False
        )
    )
    marker = _time_plane_marker(case)
    zero_yaw = jnp.zeros((1, case.num_turbines), dtype=jnp.float32)
    zero_weights = farm_functions["disk_weights"](zero_yaw)
    burnin_config = ContinuousControlConfig(
        inlet_mode="time_planes", les_steps_per_interaction=50
    )
    delay_config = ContinuousControlConfig(
        inlet_mode="time_planes",
        les_steps_per_interaction=checkpoint_block_les_steps,
        max_yaw_speed_deg_per_s=contract.max_yaw_rate_deg_per_s,
        max_yaw_angle_deg=contract.max_yaw_deg,
    )

    def burnin(initial: FlowState, planes: jax.Array) -> FlowState:
        def body(current, interaction):
            start = interaction * 50
            chunk = jax.lax.dynamic_slice_in_dim(planes, start, 50, axis=1)
            _, next_farm, _, _, _ = _advance_coupled_interval(
                (marker, current), zero_yaw, zero_weights, case, burnin_config,
                {"case": case}, farm_functions, chunk, current.step_index,
            )
            return next_farm, None

        return jax.lax.scan(
            body, initial, jnp.arange(burnin_interactions, dtype=jnp.int32)
        )[0]

    def scheduled_delay(initial: FlowState, planes: jax.Array) -> FlowState:
        block_count = contract.delay_steps // checkpoint_block_les_steps

        def body(current, block_index):
            start = block_index * checkpoint_block_les_steps
            chunk = jax.lax.dynamic_slice_in_dim(
                planes, start, checkpoint_block_les_steps, axis=1
            )
            _, next_farm, _, _, _ = _advance_coupled_interval(
                (marker, current), zero_yaw, zero_weights, case, delay_config,
                {"case": case}, farm_functions, chunk, current.step_index,
            )
            return next_farm, None

        return jax.lax.scan(
            body, initial, jnp.arange(block_count, dtype=jnp.int32)
        )[0]

    compilation: dict[str, float] = {}
    started = time.perf_counter()
    burnin_executable = jax.jit(burnin).lower(initial_farm, burnin_planes).compile()
    compilation["burnin"] = time.perf_counter() - started
    started = time.perf_counter()
    mature_farm = block_tree(burnin_executable(initial_farm, burnin_planes))
    burnin_seconds = time.perf_counter() - started
    mature_farm = jax.tree.map(jax.lax.stop_gradient, mature_farm)
    started = time.perf_counter()
    delay_executable = jax.jit(scheduled_delay).lower(mature_farm, delay_planes).compile()
    compilation["scheduled_delay"] = time.perf_counter() - started
    started = time.perf_counter()
    delayed_farm = block_tree(delay_executable(mature_farm, delay_planes))
    delay_seconds = time.perf_counter() - started
    delayed_farm = jax.tree.map(jax.lax.stop_gradient, delayed_farm)
    return PreparedControllerCase(
        case=case,
        farm_functions=farm_functions,
        delayed_farm=delayed_farm,
        mature_farm=mature_farm,
        zero_yaw=zero_yaw,
        delay_host=delay_host,
        forecast_host=forecast_host,
        forecast_device=forecast_device,
        contract=contract,
        archive=archive,
        source_nz=source_nz,
        preparation_timing_seconds={
            "compile": compilation,
            "burnin_execute": burnin_seconds,
            "scheduled_delay_execute": delay_seconds,
        },
        transfer_inventory=transfer_inventory,
        delay_executable=delay_executable,
    )
