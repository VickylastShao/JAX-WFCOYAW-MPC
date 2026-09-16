#!/usr/bin/env python3
"""Run one independent supplied-input unit of the H20 full-G2 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from computational_method_protocol import summarize_seconds
from controller_side_case_setup import block_tree, prepare_controller_case
from controller_side_mpc_protocol import adjudicate_worker_result
from device_resident_optimizer_protocol import build_compiled_scan_evidence
from full_g2_protocol import bind_precursor_identity
from jax_controller_side_mpc import make_move_blocked_les_rollout
from jax_device_resident_optimizer import make_projected_adam_optimizer
from mole_plane_archive import PlaneArchive, sha256_file


PINNED_CONTAINER_IMAGE = "shao/jax-les-h20:0.9.0.1-cu12-runtime-20260819"
PINNED_CONTAINER_DIGEST = "sha256:d9245902c9a0a282fb7e2cc9320b93eb018350e922e007d3973145cfa4f75b29"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--plane-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--plane-offset", type=int, default=0)
    parser.add_argument("--burnin-interactions", type=int, default=150)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--update-counts", default="2,3,4")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.update_counts = tuple(int(value) for value in args.update_counts.split(","))
    if args.update_counts != tuple(sorted(set(args.update_counts))) or not set(args.update_counts) <= {2, 3, 4}:
        parser.error("update counts must be an increasing subset of 2,3,4")
    if args.smoke:
        if args.repetitions != 1 or args.update_counts != (2,):
            parser.error("smoke mode requires --repetitions 1 --update-counts 2")
    elif args.repetitions != 6 or args.update_counts != (2, 3, 4):
        parser.error("the frozen full-G2 contract requires six repetitions at counts 2,3,4")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_tree(value: Any) -> str:
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree.leaves(value)):
        host = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(index).encode())
        digest.update(str(host.dtype).encode())
        digest.update(json.dumps(list(host.shape)).encode())
        digest.update(host.tobytes())
    return digest.hexdigest()


def query_visible_gpu() -> dict[str, Any]:
    """Return immutable device identity from the single-GPU container."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("single-GPU container did not expose exactly one nvidia-smi row")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError("unexpected nvidia-smi identity row")
    return {
        "name": fields[0],
        "uuid": fields[1],
        "driver_version": fields[2],
        "memory_total_mib": int(fields[3]),
    }


def fault_injection(contract: Any, unit_id: str) -> dict[str, Any]:
    first = adjudicate_worker_result(
        expected_decision_id=f"{unit_id}-fault-k10", returned_decision_id=None,
        response_seconds=None, worker_completion_seconds=65.0, contract=contract,
    )
    second = adjudicate_worker_result(
        expected_decision_id=f"{unit_id}-fault-k11", returned_decision_id=None,
        response_seconds=None, worker_completion_seconds=66.0, contract=contract,
    )
    cross = adjudicate_worker_result(
        expected_decision_id=f"{unit_id}-fault-k12",
        returned_decision_id=f"{unit_id}-fault-k10",
        response_seconds=2.0, worker_completion_seconds=2.0, contract=contract,
    )
    passes = bool(
        all(item["command_policy"] == "hold_previous_target" for item in (first, second, cross))
        and first["late_result_discarded"]
        and second["late_result_discarded"]
        and cross["cross_decision_result_discarded"]
    )
    return {
        "two_consecutive_timeouts": [first, second],
        "cross_decision_late_result": cross,
        "passes": passes,
    }


def main() -> int:
    args = parse_args()
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    archive = PlaneArchive(args.plane_archive)
    input_identity = bind_precursor_identity(
        campaign=campaign,
        actual_archive_manifest_sha256=archive.manifest_sha256,
        archive_source_state_sha256=archive.source_state_sha256,
        archive_payload_sha256=str(archive.manifest.get("archive_payload_sha256")),
        completed_planes=archive.completed_planes,
    )
    if int(campaign["seed"]) != int(args.unit_id.split("-")[-1]):
        raise ValueError("unit ID does not encode the campaign seed")
    actual_digest = os.environ.get("PAPER2_CONTAINER_DIGEST")
    if actual_digest != PINNED_CONTAINER_DIGEST:
        raise RuntimeError("pinned container digest environment binding is absent or mismatched")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(args.physical_gpu), "0"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the declared physical GPU mapping")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("full-G2 unit requires exactly one visible JAX GPU")
    gpu_identity = query_visible_gpu()

    prepared = prepare_controller_case(
        turbines=9,
        plane_archive=args.plane_archive,
        seed=int(campaign["seed"]),
        plane_offset=args.plane_offset,
        burnin_interactions=args.burnin_interactions,
    )
    contract = prepared.contract
    rollout = make_move_blocked_les_rollout(
        prepared.case,
        prepared.farm_functions,
        horizon_les_steps=contract.horizon_steps,
        first_block_les_steps=contract.first_block_steps,
        checkpoint_block_les_steps=10,
        max_yaw_rate_deg_per_s=contract.max_yaw_rate_deg_per_s,
        min_yaw_deg=contract.min_yaw_deg,
        max_yaw_deg=contract.max_yaw_deg,
    )

    def objective_with_aux(targets, farm, planes):
        result = rollout(farm, prepared.zero_yaw, targets, planes)
        return result.objective_mw, (result.mean_power_mw, result.angle_penalty_mw)

    value_and_grad = jax.value_and_grad(objective_with_aux, argnums=0, has_aux=True)
    initial_targets = jnp.zeros((1, contract.control_dimension), dtype=jnp.float32)
    packet_host = np.ascontiguousarray(
        np.concatenate((prepared.delay_host, prepared.forecast_host), axis=0),
        dtype=np.float32,
    )
    packet_sha256 = sha256_array(packet_host)
    derived_mature_state_sha256 = sha256_tree(prepared.mature_farm)
    derived_delayed_state_sha256 = sha256_tree(prepared.delayed_farm)

    executables: dict[int, Any] = {}
    compile_seconds: dict[str, float] = {}
    jaxprs: dict[int, str] = {}
    stablehlo: dict[int, str] = {}
    for update_count in args.update_counts:
        optimizer = make_projected_adam_optimizer(
            value_and_grad,
            update_count=update_count,
            min_target=contract.min_yaw_deg,
            max_target=contract.max_yaw_deg,
            command_dimension=contract.num_turbines,
        )
        jaxprs[update_count] = str(
            jax.make_jaxpr(optimizer)(
                initial_targets, prepared.delayed_farm, prepared.forecast_device
            )
        )
        lowered = jax.jit(optimizer).lower(
            initial_targets, prepared.delayed_farm, prepared.forecast_device
        )
        stablehlo[update_count] = str(lowered.compiler_ir(dialect="stablehlo"))
        started = time.perf_counter()
        executables[update_count] = lowered.compile()
        compile_seconds[str(update_count)] = time.perf_counter() - started
    scan_evidence = build_compiled_scan_evidence(
        jaxpr_by_update_count=jaxprs,
        compiler_ir_by_update_count=stablehlo,
        expected_update_counts=args.update_counts,
    )
    if not scan_evidence["passes"]:
        raise RuntimeError("compiled device optimizer does not satisfy scan evidence")

    source_paths = (
        Path("src/run_full_g2_unit.py"),
        Path("src/full_g2_protocol.py"),
        Path("src/jax_device_resident_optimizer.py"),
        Path("src/device_resident_optimizer_protocol.py"),
        Path("src/controller_side_case_setup.py"),
        Path("src/controller_side_mpc_protocol.py"),
        Path("src/jax_controller_side_mpc.py"),
        Path("src/jax_mole_compact_sac_benchmark.py"),
        Path("src/jax_mole_continuous_sac.py"),
        Path("src/mole_plane_archive.py"),
        Path("environment/requirements-gpu.txt"),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": (
            "paper2_full_g2_integration_smoke"
            if args.smoke
            else "paper2_full_g2_independent_input_unit"
        ),
        "execution_mode": "smoke" if args.smoke else "full",
        "status": "running",
        "formal_outcome": False,
        "unit_id": args.unit_id,
        "input_identity": input_identity,
        "input_runtime": {
            "campaign_path": str(args.campaign),
            "campaign_sha256": sha256_file(args.campaign),
            "archive_path": str(args.plane_archive),
            "packet_shape": list(packet_host.shape),
            "packet_bytes": int(packet_host.nbytes),
            "packet_sha256": packet_sha256,
            "derived_mature_farm_state_sha256": derived_mature_state_sha256,
            "derived_delayed_farm_state_sha256": derived_delayed_state_sha256,
            "derivation": "N=9 farm state deterministically initialized with the campaign seed and burned in from the archive rooted at the bound precursor mature state.",
        },
        "runtime": {
            "raw_argv": list(sys.argv),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": getattr(jax.lib, "__version__", None),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "physical_gpu": args.physical_gpu,
            "cuda_visible_devices": visible,
            "visible_gpu_identity": gpu_identity,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "XLA_FLAGS",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                )
            },
            "container_image": PINNED_CONTAINER_IMAGE,
            "container_digest": actual_digest,
        },
        "contract": {
            **contract.__dict__,
            "horizon_steps": contract.horizon_steps,
            "delay_steps": contract.delay_steps,
            "control_dimension": contract.control_dimension,
            "update_counts": list(args.update_counts),
            "repetitions_per_count": args.repetitions,
        },
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "compilation_seconds": {
            **prepared.preparation_timing_seconds["compile"],
            "device_outer_optimizer_by_update_count": compile_seconds,
        },
        "preparation_timing_seconds": prepared.preparation_timing_seconds,
        "device_contract": {
            "compiled_scan_passes": True,
            "compiled_scan_evidence": scan_evidence,
            "transfer_guard_passes": True,
            "inside_optimization_loop_transfer_count": 0,
        },
        "update_count_results": [],
        "fault_injection": {},
        "claim_boundary": (
            "One supplied independent development input on the named H20 device. "
            "Only the six-unit aggregate can adjudicate full G2; this is not a "
            "control-effect, sensing-to-actuation or field-deployment result."
        ),
    }
    atomic_json(args.output, payload)

    def one_decision(update_count: int, decision_id: str) -> dict[str, Any]:
        total_started = time.perf_counter()
        stages: dict[str, float] = {}
        started = time.perf_counter()
        received = np.array(packet_host, copy=True)
        if not np.all(np.isfinite(received)):
            raise ValueError("forecast packet contains non-finite values")
        observed_packet_sha = sha256_array(received)
        stages["forecast_validation_read"] = time.perf_counter() - started
        started = time.perf_counter()
        packet_device = jnp.asarray(received[None], dtype=jnp.float32)
        block_tree(packet_device)
        stages["host_to_device_transfer"] = time.perf_counter() - started
        delay_device = packet_device[:, : contract.delay_steps]
        forecast_device = packet_device[:, contract.delay_steps :]
        started = time.perf_counter()
        delayed_farm = block_tree(
            prepared.delay_executable(prepared.mature_farm, delay_device)
        )
        stages["scheduled_delay_propagation"] = time.perf_counter() - started
        started = time.perf_counter()
        with jax.transfer_guard("disallow"):
            device_result = executables[update_count](
                initial_targets, delayed_farm, forecast_device
            )
            block_tree(device_result)
        stages["objective_gradient_optimizer"] = time.perf_counter() - started
        started = time.perf_counter()
        command = np.asarray(device_result.command_targets[0], dtype=np.float32).copy()
        update_values = np.asarray(device_result.update_values, dtype=np.float32).reshape(-1)
        selected = float(device_result.best_value)
        initial = float(device_result.initial_value)
        finite = bool(
            device_result.all_finite
            and np.all(np.isfinite(command))
            and np.all(np.isfinite(update_values))
            and np.isfinite(initial)
            and np.isfinite(selected)
        )
        stages["diagnostics_best_candidate_command"] = time.perf_counter() - started
        before_watchdog = time.perf_counter() - total_started
        started = time.perf_counter()
        watchdog = adjudicate_worker_result(
            expected_decision_id=decision_id,
            returned_decision_id=decision_id,
            response_seconds=before_watchdog,
            worker_completion_seconds=before_watchdog,
            contract=contract,
        )
        stages["watchdog"] = time.perf_counter() - started
        total_seconds = time.perf_counter() - total_started
        return {
            "decision_id": decision_id,
            "packet_sha256": observed_packet_sha,
            "stage_seconds": stages,
            "controller_total_seconds": total_seconds,
            "real_time_factor": total_seconds / contract.decision_period_seconds,
            "command_deg": command.tolist(),
            "initial_objective_mw": initial,
            "selected_objective_mw": selected,
            "update_objectives_mw": update_values.tolist(),
            "finite": finite,
            "transfer_guard_passes": True,
            "watchdog": watchdog,
            "transfers": [
                {
                    "purpose": "decision_forecast_upload",
                    "direction": "H2D",
                    "bytes": int(received.nbytes),
                    "inside_optimization_loop": False,
                },
                {
                    "purpose": "final_optimizer_diagnostics_and_command",
                    "direction": "D2H",
                    "bytes": int(command.nbytes + update_values.nbytes + 8),
                    "inside_optimization_loop": False,
                },
            ],
        }

    for update_count in args.update_counts:
        one_decision(update_count, f"{args.unit_id}-warmup-u{update_count}")
        decisions = [
            one_decision(update_count, f"{args.unit_id}-u{update_count}-r{index:02d}")
            for index in range(args.repetitions)
        ]
        seconds = [item["controller_total_seconds"] for item in decisions]
        rt = [item["real_time_factor"] for item in decisions]
        payload["update_count_results"].append(
            {
                "update_count": update_count,
                "decisions": decisions,
                "controller_total_summary_seconds": summarize_seconds(seconds),
                "real_time_factor_summary": summarize_seconds(rt),
            }
        )
        atomic_json(args.output, payload)

    payload["fault_injection"] = fault_injection(contract, args.unit_id)
    payload["status"] = "complete"
    payload["completed_unix_seconds"] = time.time()
    atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
