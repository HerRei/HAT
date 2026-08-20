from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recovery.inference import acceptance_gate, pilot_matrix


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_aggregate(gate: dict[str, object], candidate_id: str) -> dict[str, object]:
    metric_names = sorted(
        {
            str(constraint["metric"])
            for constraint in gate["constraints"]  # type: ignore[index]
        }
    )
    models = {
        "base": {metric: {"mean": 1.0} for metric in metric_names},
        candidate_id: {metric: {"mean": 1.1} for metric in metric_names},
    }
    comparisons = {}
    for metric in metric_names:
        thresholds = [
            float(constraint["threshold"])
            for constraint in gate["constraints"]  # type: ignore[index]
            if constraint["metric"] == metric
        ]
        passing_value = max(thresholds) + 0.1
        comparisons[metric] = {
            "direction": "higher_is_better",
            "total_pair_count": 512,
            "finite_pair_count": 512,
            "nonfinite_pair_count": 0,
            "mean_improvement": passing_value,
            "candidate_minus_baseline_ci": {
                "method": "paired_percentile_bootstrap",
                "confidence": 0.95,
                "samples": 5000,
                "finite_pair_count": 512,
                "low": passing_value,
                "high": passing_value + 0.1,
            },
        }
    return {
        "schema_version": 2,
        "configuration": {
            "baseline": "base",
            "pair_policy": "strict",
            "bootstrap_samples": 5000,
            "confidence": 0.95,
        },
        "pairing": {"evaluated_common_count": 512},
        "models": models,
        "comparisons": {f"{candidate_id}_vs_base": comparisons},
    }


class AcceptanceGateTests(unittest.TestCase):
    def test_declared_stage_a_constraints_are_atomic_higher_thresholds(self) -> None:
        _, gate = acceptance_gate.load_gate("A")
        aggregate = _passing_aggregate(gate, "stagea_5k")
        results = acceptance_gate.evaluate_constraints(
            gate, {"clean": aggregate}, "stagea_5k"
        )
        self.assertTrue(results)
        self.assertTrue(all(result["passed"] for result in results))
        self.assertTrue(all(result["direction"] == "higher" for result in results))
        self.assertTrue(all(isinstance(result["threshold"], float) for result in results))

    def test_failed_numeric_gate_cannot_be_relabelled_passed(self) -> None:
        _, gate = acceptance_gate.load_gate("A")
        aggregate = _passing_aggregate(gate, "stagea_5k")
        aggregate["comparisons"]["stagea_5k_vs_base"]["psnr"][  # type: ignore[index]
            "mean_improvement"
        ] = -10.0
        results = acceptance_gate.evaluate_constraints(
            gate, {"clean": aggregate}, "stagea_5k"
        )
        failed = [result for result in results if not result["passed"]]
        self.assertEqual(failed[0]["name"], "psnr.mean_improvement")

    def test_human_attestation_requires_explicit_approval_and_hashed_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sheet = root / "contact.png"
            sheet.write_bytes(b"reviewed contact sheet")
            attestation_path = root / "attestation.json"
            attestation = {
                "schema_version": 1,
                "attested": False,
                "decision": "rejected",
                "reviewer": "operator",
                "reviewed_at": "2026-08-20T18:00:00Z",
                "protocol": "side_by_side_fixed_selection_v1",
                "notes": "Rejected during test.",
                "contact_sheets": [
                    {"bucket": "clean", "path": str(sheet), "sha256": _hash(sheet)}
                ],
            }
            _write_json(attestation_path, attestation)
            with self.assertRaisesRegex(acceptance_gate.AcceptanceGateError, "explicitly"):
                acceptance_gate.validate_attestation(attestation_path, ["clean"])
            attestation["attested"] = True
            attestation["decision"] = "approved"
            _write_json(attestation_path, attestation)
            validated, sheets = acceptance_gate.validate_attestation(
                attestation_path, ["clean"]
            )
            self.assertEqual(validated["reviewer"], "operator")
            self.assertEqual(sheets[0]["sha256"], _hash(sheet))

    def test_full_stage_a_record_is_hash_pinned_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "stagea_review_v1"
            candidate_id = "stagea_5k"
            _, gate = acceptance_gate.load_gate("A")
            workspace = root / "orchestration" / run_id
            prepared_manifest = workspace / "run_manifest.json"
            _write_json(prepared_manifest, {"prepared": True})
            data_manifest = root / "clean_pilot.jsonl"
            data_metadata = root / "clean_pilot.jsonl.meta.json"
            data_manifest.write_text("{}\n", encoding="ascii")
            _write_json(data_metadata, {"manifest": True})
            candidate_checkpoint = root / "net_g_5000.pth"
            candidate_checkpoint.write_bytes(b"candidate")
            candidate_config = workspace / "configs" / "stagea_5k__clean.yml"
            candidate_config.parent.mkdir(parents=True)
            candidate_config.write_text("name: candidate\n", encoding="ascii")
            output_root = root / "outputs"
            completion_paths = {}
            completions = {}
            for model_id in ("base", candidate_id):
                completion_path = (
                    output_root
                    / run_id
                    / pilot_matrix.experiment_name(run_id, model_id, "clean")
                    / "completion.json"
                )
                _write_json(completion_path, {"model": model_id})
                completion_paths[model_id] = completion_path.resolve()
                completions[model_id] = {
                    "config": {
                        "path": str(candidate_config),
                        "sha256": _hash(candidate_config),
                    }
                }
            aggregate = _passing_aggregate(gate, candidate_id)
            aggregate["provenance"] = {
                "predictions": {
                    model_id: {
                        "status": "verified",
                        "record_path": str(path),
                        "record_sha256": _hash(path),
                    }
                    for model_id, path in completion_paths.items()
                },
            }
            aggregate["identifiers"] = {
                "protocol_id": "protocol",
                "evaluation_id": "evaluation",
            }
            aggregate_path = root / "aggregate.json"
            _write_json(aggregate_path, aggregate)
            contact_sheet = root / "contact_sheet.png"
            contact_sheet.write_bytes(b"contact sheet")
            attestation_path = root / "attestation.json"
            _write_json(
                attestation_path,
                {
                    "schema_version": 1,
                    "attested": True,
                    "decision": "approved",
                    "reviewer": "operator",
                    "reviewed_at": "2026-08-20T18:00:00Z",
                    "protocol": "side_by_side_fixed_selection_v1",
                    "notes": "Reviewed all fixed selections at native and zoomed scale.",
                    "contact_sheets": [
                        {
                            "bucket": "clean",
                            "path": str(contact_sheet),
                            "sha256": _hash(contact_sheet),
                        }
                    ],
                },
            )
            manifest = {
                "mode": "explicit_candidate",
                "selection": {
                    "models": ["base", candidate_id],
                    "datasets": ["clean"],
                },
                "data": {
                    "clean": {
                        "path": str(data_manifest),
                        "sha256": _hash(data_manifest),
                        "metadata_path": str(data_metadata),
                        "metadata_sha256": _hash(data_metadata),
                    }
                },
                "checkpoints": {
                    candidate_id: {
                        "path": str(candidate_checkpoint),
                        "sha256": _hash(candidate_checkpoint),
                        "param_key": "params_ema",
                        "signature_sha256": "a" * 64,
                    }
                },
                "provenance": {"git": {"head": "0" * 40}},
            }
            data_report = {
                "clean": {"snapshot": manifest["data"]["clean"], "ids": ["face_a"]},
                "pilot_selection": {"snapshot": {}, "ids": ["face_a"]},
            }

            def validated_completion(
                _matrix: object,
                _manifest: object,
                _run_id: str,
                model_id: str,
                _bucket: str,
                _data: object,
            ) -> dict[str, object]:
                return completions[model_id]

            output = root / "acceptance" / "accepted_checkpoint.json"
            with (
                mock.patch.object(pilot_matrix, "OUTPUT_ROOT", output_root),
                mock.patch.object(pilot_matrix, "ORCHESTRATION_ROOT", root / "orchestration"),
                mock.patch.object(pilot_matrix, "_prepared_manifest", return_value=manifest),
                mock.patch.object(pilot_matrix, "load_canonical", return_value=({}, {})),
                mock.patch.object(pilot_matrix, "preflight_data", return_value=data_report),
                mock.patch.object(
                    pilot_matrix, "_validated_completion", side_effect=validated_completion
                ),
            ):
                result = acceptance_gate.create_acceptance_record(
                    run_id=run_id,
                    stage="A",
                    candidate_id=candidate_id,
                    aggregate_paths={"clean": aggregate_path},
                    attestation_path=attestation_path,
                    output_path=output,
                )
            record = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["source_stage"], "A")
            self.assertTrue(record["gate"]["passed"])
            self.assertTrue(record["visual_attestation"]["attested"])
            self.assertTrue(all(item["passed"] for item in record["gate"]["metrics"]))
            self.assertEqual(result["sha256"], _hash(output))
            digest = Path(f"{output}.sha256")
            self.assertEqual(
                digest.read_text(encoding="ascii"),
                f"{_hash(output)}  accepted_checkpoint.json\n",
            )
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(digest.stat().st_mode), 0o444)
            with self.assertRaisesRegex(acceptance_gate.AcceptanceGateError, "already exists"):
                acceptance_gate._immutable_write(output, record)


if __name__ == "__main__":
    unittest.main()
