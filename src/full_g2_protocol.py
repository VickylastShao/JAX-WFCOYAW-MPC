#!/usr/bin/env python3
"""Fail-closed contracts for the six-input Paper-2 full-G2 gate."""

from __future__ import annotations

from typing import Any, Iterable

from computational_method_protocol import summarize_seconds


FROZEN_UPDATE_COUNTS = (2, 3, 4)


def bind_precursor_identity(
    *,
    campaign: dict[str, Any],
    actual_archive_manifest_sha256: str,
    archive_source_state_sha256: str,
    archive_payload_sha256: str,
    completed_planes: int,
) -> dict[str, Any]:
    """Bind one controller input to a complete, unique precursor root."""

    if campaign.get("status") != "complete":
        raise ValueError("precursor campaign is not terminal complete")
    mature = campaign.get("mature_state") or {}
    archive = campaign.get("archive") or {}
    mature_sha = mature.get("npz_sha256")
    if not mature_sha:
        raise ValueError("precursor mature state identity is missing")
    if campaign.get("parent_state_sha256") is not None:
        raise ValueError("precursor is not an independent root state")
    if archive_source_state_sha256 != mature_sha:
        raise ValueError("archive source state does not match mature state")
    if actual_archive_manifest_sha256 != archive.get("manifest_sha256"):
        raise ValueError("archive manifest hash does not match campaign")
    if archive_payload_sha256 != archive.get("archive_payload_sha256"):
        raise ValueError("archive payload hash does not match campaign")
    if int(completed_planes) != 20000 or int(archive.get("completed_planes", -1)) != 20000:
        raise ValueError("archive does not contain exactly 20,000 planes")
    seed = campaign.get("seed")
    lineage_id = campaign.get("lineage_id")
    if seed is None or not lineage_id:
        raise ValueError("seed or lineage identity is missing")
    return {
        "seed": int(seed),
        "lineage_id": str(lineage_id),
        "parent_state_sha256": None,
        "mature_state_sha256": str(mature_sha),
        "archive_manifest_sha256": str(actual_archive_manifest_sha256),
        "archive_payload_sha256": str(archive_payload_sha256),
        "completed_planes": int(completed_planes),
    }


def _unique(rows: Iterable[dict[str, Any]], key: str) -> bool:
    values = [row.get(key) for row in rows]
    return bool(values) and None not in values and len(set(values)) == len(values)


def adjudicate_full_g2_units(
    units: list[dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    worker_recovery_passes: bool,
    execution_checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Aggregate independent unit outputs under the frozen full-G2 rule."""

    identities = [unit.get("input_identity") or {} for unit in units]
    observed_seeds = [identity.get("seed") for identity in identities]
    checks = {
        "expected_unit_count": len(units) == len(expected_seeds),
        "expected_seed_set": sorted(observed_seeds) == sorted(expected_seeds),
        "all_units_terminal_complete": all(unit.get("status") == "complete" for unit in units),
        "all_units_are_full_execution": all(unit.get("execution_mode") == "full" for unit in units),
        "all_root_states": all(identity.get("parent_state_sha256") is None for identity in identities),
        "unique_seed": _unique(identities, "seed"),
        "unique_lineage_id": _unique(identities, "lineage_id"),
        "unique_mature_state_sha256": _unique(identities, "mature_state_sha256"),
        "unique_archive_manifest_sha256": _unique(identities, "archive_manifest_sha256"),
        "unique_archive_payload_sha256": _unique(identities, "archive_payload_sha256"),
        "all_archives_have_20000_planes": all(identity.get("completed_planes") == 20000 for identity in identities),
        "all_unit_fault_injections_pass": all((unit.get("fault_injection") or {}).get("passes") is True for unit in units),
        "all_device_transfer_guards_pass": all((unit.get("device_contract") or {}).get("transfer_guard_passes") is True for unit in units),
        "zero_optimizer_loop_transfers": all((unit.get("device_contract") or {}).get("inside_optimization_loop_transfer_count") == 0 for unit in units),
        "all_compiled_scan_checks_pass": all((unit.get("device_contract") or {}).get("compiled_scan_passes") is True for unit in units),
        "worker_recovery_passes": bool(worker_recovery_passes),
    }
    if execution_checks:
        checks.update({str(key): bool(value) for key, value in execution_checks.items()})

    results_by_unit: list[dict[int, dict[str, Any]]] = []
    for unit in units:
        rows = unit.get("update_count_results") or []
        results_by_unit.append({int(row["update_count"]): row for row in rows})

    summaries: dict[str, dict[str, Any]] = {}
    for update_count in FROZEN_UPDATE_COUNTS:
        decisions = [
            decision
            for by_count in results_by_unit
            for decision in (by_count.get(update_count) or {}).get("decisions", [])
        ]
        seconds = [float(item["controller_total_seconds"]) for item in decisions]
        rt = [float(item["real_time_factor"]) for item in decisions]
        expected_decisions = len(expected_seeds) * 6
        deadline_misses = sum(bool((item.get("watchdog") or {}).get("deadline_miss")) for item in decisions)
        decision_mismatches = sum(not bool((item.get("watchdog") or {}).get("decision_id_matches")) for item in decisions)
        late_results_applied = sum(
            bool((item.get("watchdog") or {}).get("deadline_miss"))
            and (item.get("watchdog") or {}).get("command_policy") == "apply_new_target"
            for item in decisions
        )
        summary_seconds = summarize_seconds(seconds) if seconds else None
        summary_rt = summarize_seconds(rt) if rt else None
        passes = bool(
            len(decisions) == expected_decisions
            and all(bool(item.get("finite")) for item in decisions)
            and deadline_misses == 0
            and decision_mismatches == 0
            and late_results_applied == 0
            and summary_seconds is not None
            and summary_seconds["maximum"] <= 60.0
            and summary_rt is not None
            and summary_rt["p95"] < 0.5
        )
        summaries[str(update_count)] = {
            "update_count": update_count,
            "decision_count": len(decisions),
            "expected_decision_count": expected_decisions,
            "controller_total_summary_seconds": summary_seconds,
            "real_time_factor_summary": summary_rt,
            "deadline_miss_count": int(deadline_misses),
            "decision_id_mismatch_count": int(decision_mismatches),
            "late_result_applied_count": int(late_results_applied),
            "passes": passes,
        }

    passing_counts = [count for count in FROZEN_UPDATE_COUNTS if summaries[str(count)]["passes"]]
    largest = max(passing_counts) if passing_counts else None
    checks["two_update_contract_passes"] = summaries["2"]["passes"]
    non_timing_checks_pass = all(checks.values())
    full_pass = bool(non_timing_checks_pass and largest is not None)
    return {
        "schema_version": 1,
        "purpose": "paper2_six_independent_input_full_g2_adjudication",
        "independent_input_count": len(units),
        "expected_seeds": list(expected_seeds),
        "checks": checks,
        "update_count_summaries": summaries,
        "largest_passing_update_count": largest,
        "full_g2_gate_pass": full_pass,
        "formal_mpc_matrix_authorized": False,
        "claim_boundary": (
            "Controller-side timing and host-supervised fault handling on six "
            "supplied independent development inputs and the named H20 stratum; "
            "not sensing-to-actuation latency, field deployability or a control-effect result."
        ),
    }
