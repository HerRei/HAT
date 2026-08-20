#!/usr/bin/env python3
"""Emit an immutable accepted-checkpoint record only after all declared gates pass."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import pilot_matrix


GATE_DEFINITION_PATH = Path(__file__).with_name("acceptance_gates.json").resolve()
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AcceptanceGateError(RuntimeError):
    """Raised when metric, evidence, or human-review acceptance fails."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceGateError(f"Cannot read {label} '{path}': {error}") from error
    if not isinstance(value, dict):
        raise AcceptanceGateError(f"{label} '{path}' is not a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AcceptanceGateError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AcceptanceGateError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise AcceptanceGateError(f"{label} must be a finite number")
    return parsed


def _sha256(path: Path) -> str:
    return pilot_matrix._sha256_file(path)


def _resolved_local(path: Path, label: str) -> Path:
    return pilot_matrix._local_path(str(path), label)


def _name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected BUCKET=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("bucket and path must both be nonempty")
    return name, Path(path)


def _mapping(entries: Sequence[tuple[str, Path]], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, path in entries:
        if key in result:
            raise AcceptanceGateError(f"Duplicate {label} bucket '{key}'")
        result[key] = _resolved_local(path, label)
    return result


def load_gate(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    definition = _load_json(GATE_DEFINITION_PATH, "gate definition")
    if definition.get("schema_version") != 1:
        raise AcceptanceGateError("acceptance_gates.json must use schema_version 1")
    stages = definition.get("stages")
    if not isinstance(stages, dict) or stage not in stages or not isinstance(stages[stage], dict):
        raise AcceptanceGateError(f"No declared acceptance gate for stage '{stage}'")
    gate = stages[stage]
    buckets = gate.get("required_buckets")
    constraints = gate.get("constraints")
    if (
        not isinstance(buckets, list)
        or not buckets
        or len(buckets) != len(set(buckets))
        or not isinstance(constraints, list)
        or not constraints
    ):
        raise AcceptanceGateError(f"Malformed gate definition for stage '{stage}'")
    if gate.get("primary_bucket") not in buckets:
        raise AcceptanceGateError(f"Stage '{stage}' primary bucket is not required")
    for constraint in constraints:
        if not isinstance(constraint, dict) or constraint.get("bucket") not in buckets:
            raise AcceptanceGateError(f"Malformed constraint in stage '{stage}'")
        if constraint.get("direction") != "higher":
            raise AcceptanceGateError("Gate v1 supports explicit higher-is-better delta constraints only")
        _finite(constraint.get("threshold"), "constraint threshold")
    return definition, gate


def _parse_utc(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AcceptanceGateError("Attestation reviewed_at must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AcceptanceGateError("Attestation reviewed_at is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AcceptanceGateError("Attestation reviewed_at must have UTC offset +00:00/Z")
    return value


def validate_attestation(
    path: Path, required_buckets: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attestation = _load_json(path, "human attestation")
    if attestation.get("schema_version") != 1:
        raise AcceptanceGateError("Human attestation must use schema_version 1")
    if attestation.get("attested") is not True or attestation.get("decision") != "approved":
        raise AcceptanceGateError("Human attestation must explicitly set attested=true and decision=approved")
    reviewer = attestation.get("reviewer")
    notes = attestation.get("notes")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise AcceptanceGateError("Human attestation requires a nonempty reviewer")
    if not isinstance(notes, str) or not notes.strip():
        raise AcceptanceGateError("Human attestation requires nonempty review notes")
    if attestation.get("protocol") != "side_by_side_fixed_selection_v1":
        raise AcceptanceGateError(
            "Human attestation protocol must be side_by_side_fixed_selection_v1"
        )
    _parse_utc(attestation.get("reviewed_at"))
    sheets = attestation.get("contact_sheets")
    if not isinstance(sheets, list):
        raise AcceptanceGateError("Human attestation requires contact_sheets")
    by_bucket: dict[str, dict[str, Any]] = {}
    for item in sheets:
        if not isinstance(item, dict):
            raise AcceptanceGateError("Contact-sheet attestation entry must be an object")
        bucket = str(item.get("bucket", ""))
        if bucket in by_bucket:
            raise AcceptanceGateError(f"Duplicate attested contact sheet for '{bucket}'")
        sheet_path = _resolved_local(Path(str(item.get("path", ""))), "Contact sheet")
        expected_hash = str(item.get("sha256", ""))
        if not sheet_path.is_file() or not SHA256.fullmatch(expected_hash):
            raise AcceptanceGateError(f"Invalid contact-sheet evidence for '{bucket}'")
        actual_hash = _sha256(sheet_path)
        if actual_hash != expected_hash:
            raise AcceptanceGateError(f"Contact-sheet hash mismatch for '{bucket}'")
        by_bucket[bucket] = {
            "bucket": bucket,
            "path": str(sheet_path),
            "sha256": actual_hash,
        }
    if set(by_bucket) != set(required_buckets):
        raise AcceptanceGateError(
            "Attested contact-sheet buckets must exactly match the stage gate: "
            + ", ".join(required_buckets)
        )
    return attestation, [by_bucket[bucket] for bucket in required_buckets]


def _extract(value: Mapping[str, Any], dotted_path: str, label: str) -> Any:
    current: Any = value
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise AcceptanceGateError(f"Missing {label} field '{dotted_path}'")
        current = current[key]
    return current


def _verify_aggregate_evidence(
    aggregate: Mapping[str, Any],
    aggregate_path: Path,
    completion_paths: Mapping[str, Path],
) -> None:
    if aggregate.get("schema_version") != 2:
        raise AcceptanceGateError(f"Evaluator aggregate must use schema_version 2: '{aggregate_path}'")
    provenance = aggregate.get("provenance")
    if not isinstance(provenance, Mapping):
        raise AcceptanceGateError(f"Evaluator evidence is malformed in '{aggregate_path}'")
    predictions = provenance.get("predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != set(completion_paths):
        raise AcceptanceGateError(f"Evaluator completion evidence set is wrong in '{aggregate_path}'")
    for model_id, completion_path in completion_paths.items():
        item = predictions[model_id]
        if not isinstance(item, Mapping):
            raise AcceptanceGateError(f"Invalid evaluator evidence for '{model_id}'")
        if item.get("status") != "verified":
            raise AcceptanceGateError(f"Evaluator evidence is not verified for '{model_id}'")
        recorded_path = _resolved_local(
            Path(str(item.get("record_path", ""))), "Completion record"
        )
        if recorded_path != completion_path.resolve():
            raise AcceptanceGateError(
                f"Evaluator bound the wrong completion record for '{model_id}'"
            )
        if item.get("record_sha256") != _sha256(completion_path):
            raise AcceptanceGateError(
                f"Evaluator completion-record hash drifted for '{model_id}'"
            )


def evaluate_constraints(
    gate: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
) -> list[dict[str, Any]]:
    comparison_name = f"{candidate_id}_vs_base"
    results: list[dict[str, Any]] = []
    validated_metrics: set[tuple[str, str]] = set()
    for constraint in gate["constraints"]:
        bucket = str(constraint["bucket"])
        metric = str(constraint["metric"])
        statistic = str(constraint["statistic"])
        aggregate = aggregates[bucket]
        if aggregate.get("pairing", {}).get("evaluated_common_count") != gate["required_pair_count"]:
            raise AcceptanceGateError(f"Aggregate '{bucket}' does not contain the required paired count")
        configuration = aggregate.get("configuration")
        if not isinstance(configuration, Mapping):
            raise AcceptanceGateError(f"Aggregate '{bucket}' lacks configuration")
        if configuration.get("baseline") != "base" or configuration.get("pair_policy") != "strict":
            raise AcceptanceGateError(f"Aggregate '{bucket}' is not a strict base comparison")
        if int(configuration.get("bootstrap_samples", 0)) < int(gate["minimum_bootstrap_samples"]):
            raise AcceptanceGateError(f"Aggregate '{bucket}' used too few bootstrap samples")
        if _finite(configuration.get("confidence"), "aggregate confidence") != _finite(
            gate["required_confidence"], "required confidence"
        ):
            raise AcceptanceGateError(f"Aggregate '{bucket}' used the wrong confidence")
        models = aggregate.get("models")
        comparisons = aggregate.get("comparisons")
        if not isinstance(models, Mapping) or set(models) != {"base", candidate_id}:
            raise AcceptanceGateError(f"Aggregate '{bucket}' has the wrong model set")
        if not isinstance(comparisons, Mapping) or comparison_name not in comparisons:
            raise AcceptanceGateError(f"Aggregate '{bucket}' lacks comparison '{comparison_name}'")
        metric_key = (bucket, metric)
        comparison = _extract(comparisons, f"{comparison_name}.{metric}", "comparison")
        if not isinstance(comparison, Mapping):
            raise AcceptanceGateError(f"Comparison metric '{bucket}/{metric}' is malformed")
        if metric_key not in validated_metrics:
            if comparison.get("direction") != "higher_is_better":
                raise AcceptanceGateError(f"Gate metric '{bucket}/{metric}' has the wrong direction")
            if (
                comparison.get("total_pair_count") != gate["required_pair_count"]
                or comparison.get("finite_pair_count") != gate["required_pair_count"]
                or comparison.get("nonfinite_pair_count") != 0
            ):
                raise AcceptanceGateError(f"Gate metric '{bucket}/{metric}' has incomplete pairs")
            ci = comparison.get("candidate_minus_baseline_ci")
            if (
                not isinstance(ci, Mapping)
                or ci.get("method") != "paired_percentile_bootstrap"
                or int(ci.get("samples", 0)) < int(gate["minimum_bootstrap_samples"])
                or _finite(ci.get("confidence"), "metric CI confidence")
                != _finite(gate["required_confidence"], "required confidence")
            ):
                raise AcceptanceGateError(f"Gate metric '{bucket}/{metric}' has invalid paired CI")
            validated_metrics.add(metric_key)
        value = _finite(_extract(comparison, statistic, "statistic"), f"{bucket}/{metric}/{statistic}")
        threshold = _finite(constraint["threshold"], "gate threshold")
        baseline_value = _finite(
            _extract(models, f"base.{metric}.mean", "baseline metric"),
            f"base {bucket}/{metric}",
        )
        candidate_value = _finite(
            _extract(models, f"{candidate_id}.{metric}.mean", "candidate metric"),
            f"candidate {bucket}/{metric}",
        )
        passed = value >= threshold
        results.append(
            {
                "name": f"{metric}.{statistic}",
                "metric": metric,
                "statistic": statistic,
                "bucket": bucket,
                "value": value,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "direction": "higher",
                "threshold": threshold,
                "ci": comparison["candidate_minus_baseline_ci"],
                "passed": passed,
            }
        )
    return results


def _immutable_write(path: Path, value: Mapping[str, Any]) -> tuple[str, Path]:
    if path.name != "accepted_checkpoint.json":
        raise AcceptanceGateError("Acceptance output filename must be accepted_checkpoint.json")
    digest_path = Path(f"{path}.sha256")
    if path.exists() or path.is_symlink() or digest_path.exists() or digest_path.is_symlink():
        raise AcceptanceGateError(f"Acceptance output or digest already exists: '{path}'")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = pilot_matrix._json_bytes(value)
    digest = pilot_matrix._sha256_bytes(content)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
        digest_descriptor = os.open(
            digest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
        with os.fdopen(digest_descriptor, "wb") as handle:
            handle.write(f"{digest}  {path.name}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        digest_path.chmod(0o444)
    except BaseException:
        # Preserve any partial evidence; never unlink an acceptance artifact automatically.
        raise
    return digest, digest_path


def create_acceptance_record(
    *,
    run_id: str,
    stage: str,
    candidate_id: str,
    aggregate_paths: Mapping[str, Path],
    attestation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    definition, gate = load_gate(stage)
    required_buckets = [str(bucket) for bucket in gate["required_buckets"]]
    if set(aggregate_paths) != set(required_buckets):
        raise AcceptanceGateError(
            "Aggregate buckets must exactly match the stage gate: " + ", ".join(required_buckets)
        )
    try:
        manifest = pilot_matrix._prepared_manifest(run_id)
    except pilot_matrix.PilotMatrixError as error:
        raise AcceptanceGateError(str(error)) from error
    if manifest.get("mode") != "explicit_candidate":
        raise AcceptanceGateError("Acceptance records require an explicit-candidate prepared run")
    if manifest["selection"]["models"] != ["base", candidate_id]:
        raise AcceptanceGateError("Prepared run does not contain exactly base plus the candidate")
    if not set(required_buckets).issubset(manifest["selection"]["datasets"]):
        raise AcceptanceGateError("Prepared run lacks one or more required stage buckets")

    matrix, _ = pilot_matrix.load_canonical()
    completion_records: dict[str, dict[str, dict[str, Any]]] = {}
    aggregate_values: dict[str, dict[str, Any]] = {}
    report_entries = []
    evidence_entries: list[dict[str, Any]] = []
    for bucket in required_buckets:
        data_report = pilot_matrix.preflight_data(matrix, {bucket})
        if data_report[bucket]["snapshot"] != manifest["data"][bucket]:
            raise AcceptanceGateError(f"Prepared data provenance drifted for '{bucket}'")
        completions: dict[str, dict[str, Any]] = {}
        completion_paths: dict[str, Path] = {}
        for model_id in ("base", candidate_id):
            try:
                completions[model_id] = pilot_matrix._validated_completion(
                    matrix, manifest, run_id, model_id, bucket, data_report
                )
            except pilot_matrix.PilotMatrixError as error:
                raise AcceptanceGateError(str(error)) from error
            completion_path = (
                pilot_matrix.result_dir(run_id, model_id, bucket) / "completion.json"
            ).resolve()
            completion_paths[model_id] = completion_path
            evidence_entries.append(
                {
                    "kind": f"{bucket}_{model_id}_completion",
                    "path": str(completion_path),
                    "sha256": _sha256(completion_path),
                }
            )
        completion_records[bucket] = completions
        aggregate_path = aggregate_paths[bucket].resolve()
        aggregate = _load_json(aggregate_path, f"{bucket} aggregate")
        _verify_aggregate_evidence(aggregate, aggregate_path, completion_paths)
        aggregate_values[bucket] = aggregate
        aggregate_hash = _sha256(aggregate_path)
        report_entries.append(
            {"bucket": bucket, "path": str(aggregate_path), "sha256": aggregate_hash}
        )
        evidence_entries.append(
            {"kind": f"{bucket}_aggregate", "path": str(aggregate_path), "sha256": aggregate_hash}
        )

    metric_results = evaluate_constraints(gate, aggregate_values, candidate_id)
    failed = [
        f"{item['bucket']}:{item['name']}={item['value']} < {item['threshold']}"
        for item in metric_results
        if not item["passed"]
    ]
    if failed:
        raise AcceptanceGateError("Numeric gate failed; no acceptance record written: " + "; ".join(failed))

    attestation_path = attestation_path.resolve()
    attestation, contact_sheets = validate_attestation(attestation_path, required_buckets)
    evidence_entries.append(
        {
            "kind": "human_attestation",
            "path": str(attestation_path),
            "sha256": _sha256(attestation_path),
        }
    )
    for sheet in contact_sheets:
        evidence_entries.append(
            {
                "kind": f"{sheet['bucket']}_contact_sheet",
                "path": sheet["path"],
                "sha256": sheet["sha256"],
            }
        )
    definition_hash = _sha256(GATE_DEFINITION_PATH)
    evidence_entries.append(
        {
            "kind": "gate_definition",
            "path": str(GATE_DEFINITION_PATH),
            "sha256": definition_hash,
        }
    )
    prepared_manifest_path = (
        pilot_matrix._workspace(run_id) / "run_manifest.json"
    ).resolve()
    evidence_entries.append(
        {
            "kind": "prepared_manifest",
            "path": str(prepared_manifest_path),
            "sha256": _sha256(prepared_manifest_path),
        }
    )
    for bucket in required_buckets:
        data = manifest["data"][bucket]
        evidence_entries.extend(
            (
                {
                    "kind": f"{bucket}_data_manifest",
                    "path": data["path"],
                    "sha256": data["sha256"],
                },
                {
                    "kind": f"{bucket}_data_manifest_metadata",
                    "path": data["metadata_path"],
                    "sha256": data["metadata_sha256"],
                },
            )
        )

    primary_bucket = str(gate["primary_bucket"])
    primary_completion = completion_records[primary_bucket][candidate_id]
    checkpoint = manifest["checkpoints"][candidate_id]
    record = {
        "schema_version": 1,
        "created_at": attestation["reviewed_at"],
        "status": "accepted",
        "source_stage": stage,
        "checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "param_key": checkpoint["param_key"],
            "signature_sha256": checkpoint["signature_sha256"],
        },
        "gate": {
            "name": gate["gate_name"],
            "gate_set_id": definition["gate_set_id"],
            "definition_path": str(GATE_DEFINITION_PATH),
            "definition_sha256": definition_hash,
            "passed": True,
            "metrics": metric_results,
            "report_path": report_entries[required_buckets.index(primary_bucket)]["path"],
            "report_sha256": report_entries[required_buckets.index(primary_bucket)]["sha256"],
            "reports": report_entries,
        },
        "visual_attestation": {
            "attested": True,
            "reviewer": attestation["reviewer"],
            "reviewed_at": attestation["reviewed_at"],
            "protocol": attestation["protocol"],
            "notes": attestation["notes"],
            "attestation_path": str(attestation_path),
            "attestation_sha256": _sha256(attestation_path),
            "contact_sheets": contact_sheets,
        },
        "provenance": {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "config_path": primary_completion["config"]["path"],
            "config_sha256": primary_completion["config"]["sha256"],
            "prepared_manifest_path": str(prepared_manifest_path),
            "prepared_manifest_sha256": _sha256(prepared_manifest_path),
            "environment": manifest["provenance"],
            "manifest_paths": sorted(evidence_entries, key=lambda item: item["kind"]),
        },
    }
    output_path = _resolved_local(output_path, "Acceptance output")
    digest, digest_path = _immutable_write(output_path, record)
    return {
        "accepted_checkpoint": str(output_path),
        "sha256": digest,
        "sha256_file": str(digest_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True, choices=("A", "B"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--aggregate",
        action="append",
        required=True,
        type=_name_path,
        metavar="BUCKET=PATH",
    )
    parser.add_argument("--human-attestation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = create_acceptance_record(
            run_id=args.run_id,
            stage=args.stage,
            candidate_id=pilot_matrix._safe_id(args.candidate_id, "candidate model id"),
            aggregate_paths=_mapping(args.aggregate, "aggregate"),
            attestation_path=_resolved_local(args.human_attestation, "Human attestation"),
            output_path=args.out,
        )
    except (AcceptanceGateError, pilot_matrix.PilotMatrixError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
