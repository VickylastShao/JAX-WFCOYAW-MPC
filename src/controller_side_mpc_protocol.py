#!/usr/bin/env python3
"""Pure contracts for the Paper 2 controller-side MPC experiment.

This module deliberately contains no LES implementation.  It freezes the
timing, forecast-packet, actuator, move-blocking, and worker-assignment rules
that every numerical backend must satisfy before formal outcomes are run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ControllerContract:
    num_turbines: int = 9
    les_dt_seconds: float = 0.2
    decision_period_seconds: float = 120.0
    scheduled_delay_seconds: float = 60.0
    prediction_horizon_seconds: float = 200.0
    forecast_coverage_seconds: float = 260.0
    first_block_seconds: float = 120.0
    max_yaw_rate_deg_per_s: float = 0.3
    min_yaw_deg: float = -30.0
    max_yaw_deg: float = 30.0
    controller_deadline_seconds: float = 60.0

    @property
    def horizon_steps(self) -> int:
        return _integer_steps(
            self.prediction_horizon_seconds, self.les_dt_seconds
        )

    @property
    def first_block_steps(self) -> int:
        return _integer_steps(self.first_block_seconds, self.les_dt_seconds)

    @property
    def delay_steps(self) -> int:
        return _integer_steps(self.scheduled_delay_seconds, self.les_dt_seconds)

    @property
    def decision_steps(self) -> int:
        return _integer_steps(self.decision_period_seconds, self.les_dt_seconds)

    @property
    def control_dimension(self) -> int:
        return 2 * self.num_turbines

    def validate(self) -> None:
        if self.num_turbines < 1:
            raise ValueError("num_turbines must be positive")
        if self.les_dt_seconds <= 0.0:
            raise ValueError("les_dt_seconds must be positive")
        if not 0.0 < self.first_block_seconds < self.prediction_horizon_seconds:
            raise ValueError("first block must split the prediction horizon")
        if self.forecast_coverage_seconds < (
            self.scheduled_delay_seconds + self.prediction_horizon_seconds
        ):
            raise ValueError("forecast packet does not cover delay plus horizon")
        if self.max_yaw_rate_deg_per_s <= 0.0:
            raise ValueError("max_yaw_rate_deg_per_s must be positive")
        if self.min_yaw_deg >= self.max_yaw_deg:
            raise ValueError("yaw bounds are reversed or empty")
        for duration in (
            self.decision_period_seconds,
            self.scheduled_delay_seconds,
            self.prediction_horizon_seconds,
            self.first_block_seconds,
        ):
            _integer_steps(duration, self.les_dt_seconds)


def _integer_steps(duration_seconds: float, dt_seconds: float) -> int:
    steps = duration_seconds / dt_seconds
    rounded = int(round(steps))
    if not np.isclose(steps, rounded, rtol=0.0, atol=1.0e-9):
        raise ValueError("duration must be an integer number of LES steps")
    return rounded


def validate_forecast_packet(
    lead_seconds: np.ndarray,
    boundary_parameters: dict[str, np.ndarray],
    contract: ControllerContract,
) -> dict[str, Any]:
    """Validate an issue-relative packet without inventing missing coverage."""

    contract.validate()
    leads = np.asarray(lead_seconds, dtype=np.float64)
    if leads.ndim != 1 or leads.size < 2:
        raise ValueError("lead_seconds must be a one-dimensional time axis")
    if not np.all(np.isfinite(leads)):
        raise ValueError("lead_seconds contains non-finite values")
    if not np.all(np.diff(leads) > 0.0):
        raise ValueError("lead_seconds must be strictly increasing")
    if leads[0] > 0.0 or leads[-1] < contract.forecast_coverage_seconds:
        raise ValueError("forecast packet does not cover leads 0 through 260 s")
    if not boundary_parameters:
        raise ValueError("boundary_parameters must not be empty")
    for name, values in boundary_parameters.items():
        array = np.asarray(values)
        if array.shape[0] != leads.size:
            raise ValueError(f"{name} does not share the packet time axis")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")

    applied_start = contract.scheduled_delay_seconds
    applied_end = applied_start + contract.prediction_horizon_seconds
    return {
        "lead_start_seconds": float(leads[0]),
        "lead_end_seconds": float(leads[-1]),
        "optimization_start_seconds": applied_start,
        "optimization_end_seconds": applied_end,
        "sample_count": int(leads.size),
        "parameter_names": sorted(boundary_parameters),
    }


def split_move_block_targets(
    flat_targets_deg: np.ndarray, contract: ControllerContract
) -> np.ndarray:
    contract.validate()
    targets = np.asarray(flat_targets_deg)
    if targets.shape != (contract.control_dimension,):
        raise ValueError(
            "C1 target vector must contain two targets per turbine: "
            f"expected {(contract.control_dimension,)}, got {targets.shape}"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("C1 targets contain non-finite values")
    return targets.reshape(2, contract.num_turbines)


def project_targets(
    flat_targets_deg: np.ndarray, contract: ControllerContract
) -> np.ndarray:
    targets = split_move_block_targets(flat_targets_deg, contract)
    return np.clip(
        targets, contract.min_yaw_deg, contract.max_yaw_deg
    ).reshape(-1)


def shifted_c1_warm_start(
    previous_flat_targets_deg: np.ndarray, contract: ControllerContract
) -> np.ndarray:
    """Shift old block 2 into both blocks, then enforce yaw bounds."""

    previous = split_move_block_targets(previous_flat_targets_deg, contract)
    terminal = np.clip(
        previous[1], contract.min_yaw_deg, contract.max_yaw_deg
    )
    return np.concatenate((terminal, terminal)).astype(previous.dtype, copy=False)


def rate_limited_yaw_trajectory(
    initial_yaw_deg: np.ndarray,
    flat_targets_deg: np.ndarray,
    contract: ControllerContract,
) -> np.ndarray:
    """Propagate target-tracking yaw at every LES step over the C1 horizon."""

    contract.validate()
    yaw = np.asarray(initial_yaw_deg)
    if yaw.shape != (contract.num_turbines,):
        raise ValueError("initial yaw has the wrong turbine dimension")
    if not np.all(np.isfinite(yaw)):
        raise ValueError("initial yaw contains non-finite values")
    yaw = np.clip(
        yaw.astype(np.float64, copy=True),
        contract.min_yaw_deg,
        contract.max_yaw_deg,
    )
    targets = split_move_block_targets(
        project_targets(flat_targets_deg, contract), contract
    ).astype(np.float64)
    max_increment = contract.max_yaw_rate_deg_per_s * contract.les_dt_seconds
    trajectory = np.empty(
        (contract.horizon_steps, contract.num_turbines), dtype=np.float64
    )
    for step in range(contract.horizon_steps):
        block = 0 if step < contract.first_block_steps else 1
        increment = np.clip(targets[block] - yaw, -max_increment, max_increment)
        yaw = np.clip(
            yaw + increment, contract.min_yaw_deg, contract.max_yaw_deg
        )
        trajectory[step] = yaw
    return trajectory


def worker_gpu_for_unit(unit_ordinal: int, physical_gpus: tuple[int, ...]) -> int:
    """Deterministic whole-unit assignment; arms must never call this separately."""

    if unit_ordinal < 0:
        raise ValueError("unit_ordinal must be non-negative")
    if not physical_gpus or len(set(physical_gpus)) != len(physical_gpus):
        raise ValueError("physical_gpus must be a non-empty unique tuple")
    return physical_gpus[unit_ordinal % len(physical_gpus)]


def watchdog_decision(
    response_seconds: float | None,
    worker_completion_seconds: float | None,
    contract: ControllerContract,
) -> dict[str, Any]:
    """Freeze the no-retrospective-replacement deadline rule."""

    deadline = contract.controller_deadline_seconds
    valid_on_time = (
        response_seconds is not None
        and np.isfinite(response_seconds)
        and 0.0 <= response_seconds <= deadline
    )
    late_completion = (
        worker_completion_seconds is not None
        and np.isfinite(worker_completion_seconds)
        and worker_completion_seconds > deadline
    )
    return {
        "command_policy": (
            "apply_new_target" if valid_on_time else "hold_previous_target"
        ),
        "deadline_miss": not valid_on_time,
        "late_result_discarded": bool((not valid_on_time) and late_completion),
        "response_available_time": response_seconds,
        "worker_completion_time": worker_completion_seconds,
    }


def adjudicate_worker_result(
    *,
    expected_decision_id: str,
    returned_decision_id: str | None,
    response_seconds: float | None,
    worker_completion_seconds: float | None,
    contract: ControllerContract,
) -> dict[str, Any]:
    """Reject late or cross-decision results before command application."""

    if not expected_decision_id:
        raise ValueError("expected_decision_id must be non-empty")
    decision = watchdog_decision(
        response_seconds, worker_completion_seconds, contract
    )
    decision_id_matches = returned_decision_id == expected_decision_id
    if not decision_id_matches:
        decision["command_policy"] = "hold_previous_target"
        decision["deadline_miss"] = True
    decision["expected_decision_id"] = expected_decision_id
    decision["returned_decision_id"] = returned_decision_id
    decision["decision_id_matches"] = decision_id_matches
    decision["cross_decision_result_discarded"] = not decision_id_matches
    return decision


class UnitClaimStore:
    """Filesystem-backed atomic whole-unit ownership for H20 workers."""

    _SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def __init__(
        self,
        root: str | Path,
        *,
        physical_gpus: tuple[int, ...] = (4, 5, 6, 7),
    ) -> None:
        self.root = Path(root)
        self.physical_gpus = physical_gpus
        if not physical_gpus or len(set(physical_gpus)) != len(physical_gpus):
            raise ValueError("physical_gpus must be a non-empty unique tuple")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, unit_id: str) -> Path:
        if not self._SAFE_ID.fullmatch(unit_id):
            raise ValueError("unit_id contains unsafe characters")
        return self.root / f"{unit_id}.json"

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def claim(self, unit_id: str, unit_ordinal: int, worker_id: str) -> dict[str, Any]:
        """Atomically claim a unit or resume ownership by the same worker."""

        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        gpu_id = worker_gpu_for_unit(unit_ordinal, self.physical_gpus)
        record = {
            "unit_id": unit_id,
            "unit_ordinal": unit_ordinal,
            "worker_id": worker_id,
            "physical_gpu": gpu_id,
            "status": "running",
            "completed_arms": [],
        }
        path = self._path(unit_id)
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            existing = self._load(path)
            if (
                existing.get("worker_id") == worker_id
                and existing.get("unit_ordinal") == unit_ordinal
                and existing.get("physical_gpu") == gpu_id
            ):
                return {**existing, "resumed": True}
            raise RuntimeError(f"unit already claimed: {unit_id}") from None
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return {**record, "resumed": False}

    def record_arm(
        self,
        unit_id: str,
        *,
        worker_id: str,
        physical_gpu: int,
        arm_id: str,
    ) -> dict[str, Any]:
        """Persist an arm while enforcing whole-unit colocation."""

        if not arm_id or not self._SAFE_ID.fullmatch(arm_id):
            raise ValueError("arm_id contains unsafe characters")
        path = self._path(unit_id)
        record = self._load(path)
        if record.get("worker_id") != worker_id:
            raise RuntimeError("worker does not own this unit")
        if record.get("physical_gpu") != physical_gpu:
            raise RuntimeError("all arms of one unit must remain on the assigned GPU")
        arms = list(record.get("completed_arms", []))
        if arm_id not in arms:
            arms.append(arm_id)
        record["completed_arms"] = arms
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
        return record
