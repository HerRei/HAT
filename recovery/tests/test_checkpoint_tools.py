from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from recovery.tools.interpolate_checkpoints import extract_state, interpolate_states, write_interpolation
from recovery.tools.verify_run_state import verify_state


class CheckpointToolsTest(unittest.TestCase):
    def test_interpolate_states_blends_float_and_preserves_index(self) -> None:
        base = {
            "weight": torch.tensor([0.0, 2.0]),
            "index": torch.tensor([1, 2], dtype=torch.int64),
        }
        tuned = {
            "weight": torch.tensor([4.0, 6.0]),
            "index": torch.tensor([1, 2], dtype=torch.int64),
        }

        result = interpolate_states(base, tuned, 0.25)

        self.assertTrue(torch.equal(result["weight"], torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.equal(result["index"], base["index"]))

    def test_interpolate_states_rejects_changed_index(self) -> None:
        base = {"index": torch.tensor([1], dtype=torch.int64)}
        tuned = {"index": torch.tensor([2], dtype=torch.int64)}

        with self.assertRaisesRegex(ValueError, "Non-floating tensor"):
            interpolate_states(base, tuned, 0.5)

    def test_extract_state_rejects_empty_and_non_tensor_mappings(self) -> None:
        source = Path("invalid.pth")
        with self.assertRaisesRegex(KeyError, "non-empty mapping"):
            extract_state({"params_ema": {}}, "params_ema", source)
        with self.assertRaisesRegex(TypeError, "string names only to tensors"):
            extract_state({"params_ema": {"weight": "not a tensor"}}, "params_ema", source)
        with self.assertRaisesRegex(TypeError, "string names only to tensors"):
            extract_state({"params_ema": {1: torch.tensor([1.0])}}, "params_ema", source)

    def test_write_interpolation_uses_safe_weights_only_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.pth"
            tuned_path = root / "tuned.pth"
            output_path = root / "output.pth"
            base_path.write_bytes(b"base")
            tuned_path.write_bytes(b"tuned")
            payloads = [
                {"params_ema": {"weight": torch.tensor([0.0])}},
                {"params_ema": {"weight": torch.tensor([2.0])}},
            ]
            with mock.patch(
                "recovery.tools.interpolate_checkpoints.torch.load",
                side_effect=payloads,
            ) as load:
                write_interpolation(
                    base_path, tuned_path, output_path, 0.5, "params_ema"
                )

            self.assertEqual(load.call_count, 2)
            for call in load.call_args_list:
                self.assertEqual(call.kwargs["map_location"], "cpu")
                self.assertIs(call.kwargs["weights_only"], True)
            saved = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(saved["params_ema"]["weight"], torch.tensor([1.0]))
            )

    def test_verify_state_reports_matching_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"checkpoint")
            digest = hashlib.sha256(b"checkpoint").hexdigest()
            state = {
                "failed_run": {
                    "artifacts": {
                        "present": {"path": "artifact.bin", "sha256": digest},
                        "missing": {"path": "missing.bin", "sha256": "0" * 64},
                    }
                }
            }
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            results = verify_state(state_path, root)

            self.assertTrue(results[0]["ok"])
            self.assertFalse(results[1]["ok"])

    def test_verify_state_finds_declared_artifacts_outside_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "metric.pth"
            artifact.write_bytes(b"identity metric")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            state = {
                "failed_run": {"artifacts": {}},
                "workflow": {
                    "arcface_identity_model": {
                        "path": str(artifact),
                        "sha256": digest,
                    }
                },
                "recovery_artifacts": {
                    "interpolations": {
                        "metric_copy": {
                            "path": "metric.pth",
                            "sha256": digest,
                        }
                    }
                },
            }
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            results = verify_state(state_path, root)

            names = {result["name"] for result in results}
            self.assertIn("workflow.arcface_identity_model", names)
            self.assertIn(
                "recovery_artifacts.interpolations.metric_copy", names
            )
            self.assertTrue(all(result["ok"] for result in results))


if __name__ == "__main__":
    unittest.main()
