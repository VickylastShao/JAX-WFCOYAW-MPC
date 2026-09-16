#!/usr/bin/env python3
"""Fail-closed pre-experiment alignment gate for the Paper 2 research mainline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PURPOSE = "paper2_pure_les_mpc_preexperiment_alignment"
MAINLINE_PURPOSE = "paper2_pure_les_mpc_research_mainline_freeze"
ALLOWED_ROLES = {
    "pure_mpc_method",
    "pure_mpc_control_effect",
    "computational_support",
    "external_baseline",
}
PURE_MPC_ROLES = {"pure_mpc_method", "pure_mpc_control_effect"}
PROHIBITED_B1PSTAR_FIELDS = (
    "b1pstar_in_objective",
    "b1pstar_in_regularizer",
    "b1pstar_in_warm_start",
    "b1pstar_in_terminal_condition",
    "b1pstar_in_candidate_selection",
    "b1pstar_in_supervisor",
    "b1pstar_in_fallback",
    "b1pstar_in_command_formation",
)
REQUIRED_SCOPE_FIELDS = (
    "state_scope",
    "forecast_scope",
    "controller_scope",
    "source_scope",
    "input_scope",
    "resource_scope",
)
PURE_MPC_FALLBACKS = {
    "hold_last_valid_mpc",
    "hold_active_target",
    "rate_limited_neutral",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one immutable input file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _mainline_violations(mainline: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if mainline.get("schema_version") != 1:
        violations.append("research mainline schema_version must equal 1")
    if mainline.get("purpose") != MAINLINE_PURPOSE:
        violations.append("research mainline purpose is invalid")
    if mainline.get("status") != "controlling":
        violations.append("research mainline status must be controlling")
    if mainline.get("proposed_controller") != "pure_les_mpc":
        violations.append("research mainline proposed_controller must be pure_les_mpc")
    if mainline.get("b1pstar_role") != "external_comparator_only":
        violations.append("research mainline must restrict B1P* to external comparator")
    return violations


def adjudicate_preexperiment_alignment(
    record: Mapping[str, Any], *, mainline_path: Path
) -> dict[str, Any]:
    """Adjudicate one prospective experiment record against the controlling mainline."""

    violations: list[str] = []
    mainline_path = mainline_path.resolve()
    try:
        mainline = json.loads(mainline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        mainline = {}
        violations.append(f"research mainline is unreadable: {type(error).__name__}")
    if isinstance(mainline, Mapping):
        violations.extend(_mainline_violations(mainline))
    else:
        violations.append("research mainline must be a JSON object")

    observed_path = record.get("research_mainline_path")
    if not isinstance(observed_path, str) or not observed_path.strip():
        violations.append("research_mainline_path must be a non-empty string")
    else:
        try:
            if Path(observed_path).resolve() != mainline_path:
                violations.append("research_mainline_path does not match adjudicated file")
        except OSError:
            violations.append("research_mainline_path cannot be resolved")

    observed_sha = record.get("research_mainline_sha256")
    if mainline_path.is_file():
        if observed_sha != sha256_file(mainline_path):
            violations.append("research mainline SHA-256 mismatch")
    else:
        violations.append("research mainline file is absent")

    if record.get("schema_version") != 1:
        violations.append("schema_version must equal 1")
    if record.get("purpose") != PURPOSE:
        violations.append("purpose is invalid")
    if record.get("status") != "prospectively_frozen":
        violations.append("status must be prospectively_frozen")
    if record.get("outcomes_present") is not False:
        violations.append("outcomes_present must be false before launch")
    if not isinstance(record.get("experiment_id"), str) or not str(
        record.get("experiment_id")
    ).strip():
        violations.append("experiment_id must be a non-empty string")
    if not isinstance(record.get("core_thesis_clause_tested"), str) or not str(
        record.get("core_thesis_clause_tested")
    ).strip():
        violations.append("core_thesis_clause_tested must be a non-empty string")
    if not _nonempty_string_list(record.get("expected_pass_evidence")):
        violations.append("expected_pass_evidence must be a non-empty string list")
    if not _nonempty_string_list(record.get("expected_fail_evidence")):
        violations.append("expected_fail_evidence must be a non-empty string list")
    if not isinstance(record.get("evidence_class"), str) or not str(
        record.get("evidence_class")
    ).strip():
        violations.append("evidence_class must be a non-empty string")
    for field in REQUIRED_SCOPE_FIELDS:
        if not isinstance(record.get(field), Mapping) or not record.get(field):
            violations.append(f"{field} must be a non-empty object")

    role = record.get("experiment_role")
    if role not in ALLOWED_ROLES:
        violations.append("experiment_role is not mainline-authorized")

    for field in PROHIBITED_B1PSTAR_FIELDS:
        if record.get(field) is not False:
            violations.append(f"{field} must be false")

    runtime_role = record.get("b1pstar_runtime_role")
    fallback = record.get("fallback_policy")
    if role in PURE_MPC_ROLES:
        if runtime_role != "not_used":
            violations.append("pure MPC experiment must set b1pstar_runtime_role=not_used")
        if fallback not in PURE_MPC_FALLBACKS:
            violations.append("pure MPC fallback_policy is not authorized")
    elif role == "external_baseline":
        if runtime_role != "external_comparator_only":
            violations.append(
                "external baseline must set b1pstar_runtime_role=external_comparator_only"
            )
        if fallback != "not_applicable":
            violations.append("external baseline fallback_policy must be not_applicable")
    elif role == "computational_support":
        if runtime_role != "not_used":
            violations.append("computational support must not use B1P* at runtime")
        if fallback != "not_applicable":
            violations.append("computational support fallback_policy must be not_applicable")

    passed = not violations
    return {
        "schema_version": 1,
        "purpose": "paper2_pure_les_mpc_preexperiment_alignment_adjudication",
        "experiment_id": record.get("experiment_id"),
        "experiment_role": role,
        "research_mainline_path": str(mainline_path),
        "research_mainline_sha256": (
            sha256_file(mainline_path) if mainline_path.is_file() else None
        ),
        "status": "pass" if passed else "fail_closed",
        "authorized_to_launch": passed,
        "violations": violations,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--mainline", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    result = adjudicate_preexperiment_alignment(record, mainline_path=args.mainline)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["authorized_to_launch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
