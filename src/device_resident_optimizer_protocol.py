#!/usr/bin/env python3
"""Pure adjudication rules for the device-resident optimizer experiment."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from computational_method_protocol import backend_equivalence


DEVICE_CORE_LABEL = (
    "GPU-resident differentiable-LES optimization core with host-supervised safety and I/O"
)
HOST_ORCHESTRATED_LABEL = "GPU-native LES with host-orchestrated MPC"
FROZEN_UPDATE_COUNTS = (2, 3, 4)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_compiled_scan_evidence(
    *,
    jaxpr_by_update_count: Mapping[int, str],
    compiler_ir_by_update_count: Mapping[int, str],
    expected_update_counts: Sequence[int] = FROZEN_UPDATE_COUNTS,
) -> dict[str, Any]:
    """Build hash-bound evidence that every frozen optimizer contains a scan."""

    expected = tuple(int(value) for value in expected_update_counts)
    if not expected or expected != tuple(sorted(set(expected))):
        raise ValueError("expected update counts must be unique and increasing")
    jaxprs = {int(key): str(value) for key, value in jaxpr_by_update_count.items()}
    compiler_ir = {
        int(key): str(value) for key, value in compiler_ir_by_update_count.items()
    }
    blockers: list[str] = []
    missing = [value for value in expected if value not in jaxprs or value not in compiler_ir]
    if missing:
        blockers.append(
            "missing update-count evidence: " + ",".join(str(value) for value in missing)
        )
    extras = sorted((set(jaxprs) | set(compiler_ir)) - set(expected))
    if extras:
        blockers.append(
            "unexpected update-count evidence: " + ",".join(str(value) for value in extras)
        )

    by_update_count: dict[str, Any] = {}
    for update_count in expected:
        if update_count not in jaxprs or update_count not in compiler_ir:
            continue
        jaxpr = jaxprs[update_count]
        ir = compiler_ir[update_count]
        contains_scan = "scan[" in jaxpr.replace(" ", "")
        nonempty_ir = bool(ir.strip())
        if not contains_scan:
            blockers.append(f"u={update_count} JAXPR does not contain scan")
        if not nonempty_ir:
            blockers.append(f"u={update_count} compiler IR is empty")
        by_update_count[str(update_count)] = {
            "jaxpr_contains_scan": contains_scan,
            "jaxpr_sha256": _sha256_text(jaxpr),
            "jaxpr_bytes": len(jaxpr.encode("utf-8")),
            "compiler_ir_nonempty": nonempty_ir,
            "compiler_ir_sha256": _sha256_text(ir),
            "compiler_ir_bytes": len(ir.encode("utf-8")),
        }
    return {
        "passes": not blockers,
        "expected_update_counts": list(expected),
        "by_update_count": by_update_count,
        "blockers": blockers,
    }


def adjudicate_formal_runtime_contract(
    *,
    backend: str,
    devices: Sequence[str],
    cuda_visible_devices: str | None,
    physical_gpu: int,
    container_digest: str | None,
    expected_container_digest: str,
    input_hashes: Mapping[str, str],
    expected_input_hashes: Mapping[str, str],
    update_counts: Sequence[int],
    repetitions: int,
) -> dict[str, Any]:
    """Fail closed when the formal H20 runtime or frozen input contract drifts."""

    blockers: list[str] = []
    if backend != "gpu":
        blockers.append(f"backend must be gpu, got {backend!r}")
    if len(devices) != 1:
        blockers.append(f"exactly one JAX device is required, got {len(devices)}")
    if cuda_visible_devices != "0":
        blockers.append(
            f"container CUDA_VISIBLE_DEVICES must be '0', got {cuda_visible_devices!r}"
        )
    if int(physical_gpu) != 5:
        blockers.append(f"formal physical GPU must be 5, got {physical_gpu}")
    if container_digest != expected_container_digest:
        blockers.append("container digest does not match the frozen digest")
    for name, expected in expected_input_hashes.items():
        actual = input_hashes.get(name)
        if actual != expected:
            blockers.append(f"{name} does not match the frozen input hash")
    counts = tuple(int(value) for value in update_counts)
    if counts != FROZEN_UPDATE_COUNTS:
        blockers.append(
            f"update counts must be {FROZEN_UPDATE_COUNTS}, got {counts}"
        )
    if int(repetitions) != 6:
        blockers.append(f"paired repetitions must be 6, got {repetitions}")
    return {
        "passes": not blockers,
        "backend": str(backend),
        "devices": [str(value) for value in devices],
        "cuda_visible_devices": cuda_visible_devices,
        "physical_gpu": int(physical_gpu),
        "container_digest": container_digest,
        "expected_container_digest": expected_container_digest,
        "input_hashes": dict(input_hashes),
        "expected_input_hashes": dict(expected_input_hashes),
        "update_counts": list(counts),
        "repetitions": int(repetitions),
        "blockers": blockers,
    }


def required_telemetry_labels(
    update_counts: Sequence[int] = FROZEN_UPDATE_COUNTS,
) -> list[str]:
    """Return the frozen begin/midpoint/end telemetry schedule."""

    counts = tuple(int(value) for value in update_counts)
    labels = ["preflight", "after_warmups"]
    for update_count in counts:
        labels.extend(
            [
                f"u{update_count}_start",
                f"u{update_count}_midpoint",
                f"u{update_count}_end",
            ]
        )
    labels.append("final")
    return labels


def adjudicate_telemetry_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    update_counts: Sequence[int] = FROZEN_UPDATE_COUNTS,
    expected_cpu_affinity: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Check telemetry completeness and obvious selected-GPU contamination."""

    required = required_telemetry_labels(update_counts)
    observed = [str(item.get("label")) for item in snapshots]
    blockers: list[str] = []
    if observed != required:
        blockers.append(
            f"telemetry labels/order mismatch: expected {required}, got {observed}"
        )
    previous_timestamp = -np.inf
    expected_affinity = (
        tuple(int(value) for value in expected_cpu_affinity)
        if expected_cpu_affinity is not None
        else None
    )
    for index, snapshot in enumerate(snapshots):
        label = str(snapshot.get("label", f"snapshot-{index}"))
        try:
            timestamp = float(snapshot.get("unix_seconds"))
        except (TypeError, ValueError):
            timestamp = np.nan
        if not np.isfinite(timestamp) or timestamp <= previous_timestamp:
            blockers.append(f"{label} timestamp is missing or non-increasing")
        previous_timestamp = timestamp
        affinity = tuple(int(value) for value in snapshot.get("cpu_affinity", ()))
        if not affinity:
            blockers.append(f"{label} CPU affinity is missing")
        elif expected_affinity is not None and affinity != expected_affinity:
            blockers.append(f"{label} CPU affinity drifted from the frozen set")
        gpu_rows = snapshot.get("gpu_rows", ())
        if len(gpu_rows) != 1 or not str(gpu_rows[0].get("uuid", "")).strip():
            blockers.append(f"{label} must contain exactly one identified GPU row")
        process_ids = {
            int(item["pid"])
            for item in snapshot.get("compute_processes", ())
            if item.get("pid") is not None
        }
        if len(process_ids) > 1:
            blockers.append(f"{label} has multiple GPU compute processes: {sorted(process_ids)}")
    return {
        "passes": not blockers,
        "required_labels": required,
        "observed_labels": observed,
        "snapshot_count": len(snapshots),
        "expected_cpu_affinity": list(expected_affinity) if expected_affinity else None,
        "blockers": blockers,
    }


def balanced_path_orders(repetitions: int) -> list[tuple[str, str]]:
    """Return deterministic, exactly balanced AB/BA execution orders."""

    if repetitions < 2 or repetitions % 2:
        raise ValueError("repetitions must be a positive even number of at least two")
    return [
        ("legacy", "device") if index % 2 == 0 else ("device", "legacy")
        for index in range(repetitions)
    ]


def paired_optimizer_equivalence(
    *,
    legacy_objective: float,
    device_objective: float,
    legacy_gradient: Sequence[float],
    device_gradient: Sequence[float],
    legacy_command_deg: Sequence[float],
    device_command_deg: Sequence[float],
    objective_relative_tolerance: float = 1.0e-4,
    gradient_relative_tolerance: float = 5.0e-3,
    command_absolute_tolerance_deg: float = 0.1,
) -> dict[str, Any]:
    """Check all frozen numerical-equivalence dimensions for one pair."""

    backend = backend_equivalence(
        legacy_objective,
        device_objective,
        legacy_gradient,
        device_gradient,
        objective_relative_tolerance=objective_relative_tolerance,
        gradient_relative_tolerance=gradient_relative_tolerance,
    )
    legacy_command = np.asarray(legacy_command_deg, dtype=np.float64).reshape(-1)
    device_command = np.asarray(device_command_deg, dtype=np.float64).reshape(-1)
    if legacy_command.shape != device_command.shape or legacy_command.size < 1:
        raise ValueError("commands must have the same non-empty shape")
    if not np.all(np.isfinite(legacy_command)) or not np.all(np.isfinite(device_command)):
        raise ValueError("commands must be finite")
    maximum_command_absolute_error_deg = float(
        np.max(np.abs(legacy_command - device_command))
    )
    command_passes = bool(
        maximum_command_absolute_error_deg <= command_absolute_tolerance_deg
    )
    return {
        **backend,
        "passes": bool(backend["passes"] and command_passes),
        "command_passes": command_passes,
        "maximum_command_absolute_error_deg": maximum_command_absolute_error_deg,
        "command_absolute_tolerance_deg": command_absolute_tolerance_deg,
    }


def adjudicate_device_resident_core(
    *,
    transfers: Sequence[Mapping[str, Any]],
    equivalence_passes: bool,
    all_results_finite: bool,
    compiled_scan_evidence_passes: bool,
    transfer_guard_passes: bool,
    host_watchdog_preserved: bool,
    runtime_evidence_passes: bool,
    median_time_ratio_device_over_legacy: float,
) -> dict[str, Any]:
    """Adjudicate architecture independently of any observed speed change."""

    if not np.isfinite(median_time_ratio_device_over_legacy) or median_time_ratio_device_over_legacy <= 0:
        raise ValueError("time ratio must be finite and positive")
    in_loop_transfers = [
        dict(item) for item in transfers if bool(item.get("inside_optimization_loop"))
    ]
    architecture_passes = bool(
        not in_loop_transfers
        and equivalence_passes
        and all_results_finite
        and compiled_scan_evidence_passes
        and transfer_guard_passes
        and host_watchdog_preserved
        and runtime_evidence_passes
    )
    return {
        "passes_architecture_gate": architecture_passes,
        "allowed_label": DEVICE_CORE_LABEL if architecture_passes else HOST_ORCHESTRATED_LABEL,
        "in_loop_transfer_count": len(in_loop_transfers),
        "in_loop_transfers": in_loop_transfers,
        "equivalence_passes": bool(equivalence_passes),
        "all_results_finite": bool(all_results_finite),
        "compiled_scan_evidence_passes": bool(compiled_scan_evidence_passes),
        "transfer_guard_passes": bool(transfer_guard_passes),
        "host_watchdog_preserved": bool(host_watchdog_preserved),
        "runtime_evidence_passes": bool(runtime_evidence_passes),
        "median_time_ratio_device_over_legacy": float(
            median_time_ratio_device_over_legacy
        ),
        "speedup_required_for_architecture_gate": False,
        "formal_outcome": False,
    }
