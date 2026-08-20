from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from recovery.eval.core import EvaluationConfig, PredictionSpec, evaluate_saved_outputs
from recovery.eval.metrics import cosine_similarity_from_embeddings
from recovery.eval.provenance import TREE_DIGEST_ALGORITHM, tree_digest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecoveryEvaluationProvenanceTests(unittest.TestCase):
    def _write_image(self, path: Path, value: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((20, 20, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

    def _completion_record(
        self,
        root: Path,
        prediction_name: str,
        prediction_directory: Path,
        suffix: str,
    ) -> Path:
        checkpoint = root / f"{prediction_name}.pth"
        config = root / f"{prediction_name}.yml"
        checkpoint.write_bytes(f"checkpoint:{prediction_name}".encode("ascii"))
        config.write_text(f"name: {prediction_name}\n", encoding="ascii")
        paths = sorted(prediction_directory.glob("*.png"))
        outputs = tree_digest(prediction_directory, paths)
        record = {
            "schema_version": 3,
            "status": "complete",
            "model": {
                "name": prediction_name,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _sha256(checkpoint),
                },
            },
            "config": {"path": str(config), "sha256": _sha256(config)},
            "outputs": {
                "root": str(prediction_directory),
                "file_count": outputs["file_count"],
                "prediction_suffix": suffix,
                "files_manifest_sha256": outputs["files_manifest_sha256"],
                "tree_sha256": outputs["tree_sha256"],
                "tree_sha256_algorithm": TREE_DIGEST_ALGORITHM,
            },
        }
        record_path = root / f"{prediction_name}_completion.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return record_path

    def test_nonempty_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "evaluation"
            output.mkdir()
            evidence = output / "existing.txt"
            evidence.write_text("preserve me", encoding="ascii")
            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root / "missing_gt",
                        predictions=(PredictionSpec("model", root / "missing_pred"),),
                        output_directory=output,
                    )
                )
            self.assertEqual(evidence.read_text(encoding="ascii"), "preserve me")

    def test_completion_record_digests_are_verified_and_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_image(root / "gt" / "face.png", 100)
            self._write_image(root / "prediction" / "face_model.png", 103)
            record = self._completion_record(root, "model", root / "prediction", "_model")

            aggregate = evaluate_saved_outputs(
                EvaluationConfig(
                    gt_directory=root / "gt",
                    predictions=(
                        PredictionSpec("model", root / "prediction", "_model", record),
                    ),
                    output_directory=root / "evaluation",
                    crop_border=2,
                    bootstrap_samples=10,
                )
            )

            provenance = aggregate["provenance"]["predictions"]["model"]
            self.assertEqual(provenance["status"], "verified")
            self.assertEqual(
                provenance["checkpoint"]["declared_sha256"],
                provenance["checkpoint"]["verified_sha256"],
            )
            self.assertEqual(
                provenance["outputs"]["tree_sha256"],
                provenance["completion_record"]["outputs"]["tree_sha256"],
            )
            self.assertRegex(aggregate["identifiers"]["protocol_id"], r"^[0-9a-f]{64}$")
            self.assertRegex(aggregate["identifiers"]["evaluation_id"], r"^[0-9a-f]{64}$")

    def test_completion_record_rejects_changed_prediction_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_image(root / "gt" / "face.png", 100)
            prediction = root / "prediction" / "face_model.png"
            self._write_image(prediction, 103)
            record = self._completion_record(root, "model", prediction.parent, "_model")
            self._write_image(prediction, 104)

            with self.assertRaisesRegex(ValueError, "output tree hash mismatch"):
                evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root / "gt",
                        predictions=(
                            PredictionSpec("model", prediction.parent, "_model", record),
                        ),
                        output_directory=root / "evaluation",
                        crop_border=2,
                    )
                )

    def test_completion_record_rejects_unbound_output_tree_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_image(root / "gt" / "face.png", 100)
            prediction = root / "prediction" / "face_model.png"
            self._write_image(prediction, 103)
            record = self._completion_record(root, "model", prediction.parent, "_model")
            (prediction.parent / "unrecorded.txt").write_text("drift", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "exactly the indexed prediction files"):
                evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root / "gt",
                        predictions=(
                            PredictionSpec("model", prediction.parent, "_model", record),
                        ),
                        output_directory=root / "evaluation",
                        crop_border=2,
                    )
                )

    def test_completion_records_are_all_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "all-or-none"):
                evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root,
                        predictions=(
                            PredictionSpec("a", root, completion_record=root / "a.json"),
                            PredictionSpec("b", root),
                        ),
                        output_directory=root / "out",
                    )
                )

    def test_arcface_gt_embedding_is_computed_once_per_image(self) -> None:
        class FakeArcFace:
            def __init__(self) -> None:
                self.embedding_calls = 0
                self.batch_calls = 0
                self.metadata = {"model_key": "fake-model-key"}

            def preprocess_key(self, crop_border: int) -> str:
                return f"fake-preprocess-{crop_border}"

            def embedding(self, image: np.ndarray, crop_border: int) -> np.ndarray:
                self.embedding_calls += 1
                return np.array([float(image.mean()), 1.0], dtype=np.float64)

            def embeddings(
                self,
                images: list[np.ndarray],
                crop_border: int,
                batch_size: int,
            ) -> np.ndarray:
                self.batch_calls += 1
                return np.stack(
                    [self.embedding(image, crop_border) for image in images], axis=0
                )

            def similarity_from_embeddings(
                self, prediction: np.ndarray, ground_truth: np.ndarray
            ) -> float:
                return cosine_similarity_from_embeddings(prediction, ground_truth)

            def __call__(self, *_args: object) -> float:
                raise AssertionError("core must use the cacheable embedding API")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for image_id, value in (("a", 80), ("b", 140)):
                self._write_image(root / "gt" / f"{image_id}.png", value)
                self._write_image(root / "base" / f"{image_id}.png", value + 2)
                self._write_image(root / "candidate" / f"{image_id}.png", value + 4)
            arcface_model = root / "arcface.pth"
            arcface_model.write_bytes(b"fake")
            fake_metric = FakeArcFace()
            with mock.patch(
                "recovery.eval.core.create_arcface_metric", return_value=fake_metric
            ):
                aggregate = evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root / "gt",
                        predictions=(
                            PredictionSpec("base", root / "base"),
                            PredictionSpec("candidate", root / "candidate"),
                        ),
                        output_directory=root / "evaluation",
                        crop_border=2,
                        baseline="base",
                        bootstrap_samples=10,
                        arcface_backend="facexlib-pth",
                        arcface_model=arcface_model,
                    )
                )

            self.assertEqual(fake_metric.embedding_calls, 6)
            self.assertEqual(fake_metric.batch_calls, 3)
            cache = aggregate["optional_metrics"]["arcface"][
                "ground_truth_embedding_cache"
            ]
            self.assertEqual(cache["entries"], 2)
            self.assertEqual(cache["ground_truth_embedding_reuses_avoided"], 2)
            self.assertEqual(
                aggregate["optional_metrics"]["arcface"]["preprocess_key"],
                "fake-preprocess-2",
            )


if __name__ == "__main__":
    unittest.main()
