from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from recovery.eval.build_contact_sheet import build_contact_sheet
from recovery.eval.core import EvaluationConfig, PredictionSpec, evaluate_saved_outputs


class RecoverySavedOutputPipelineTests(unittest.TestCase):
    def _write(self, path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(path), image))

    def test_recovery_pipeline_writes_paired_outputs_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rng = np.random.default_rng(901)
            for index in range(3):
                gt = rng.integers(12, 244, size=(32, 32, 3), dtype=np.uint8)
                base = np.clip(gt.astype(np.int16) + 8, 0, 255).astype(np.uint8)
                candidate = np.clip(gt.astype(np.int16) + 3, 0, 255).astype(np.uint8)
                image_id = f"{index:05d}"
                self._write(root / "gt" / f"{image_id}.png", gt)
                self._write(root / "base" / f"{image_id}_base.png", base)
                self._write(
                    root / "candidate" / image_id / f"{image_id}_125000.png",
                    candidate,
                )

            output = root / "evaluation"
            selection = output / "selection.json"
            aggregate = evaluate_saved_outputs(
                EvaluationConfig(
                    gt_directory=root / "gt",
                    predictions=(
                        PredictionSpec("base", root / "base", "_base"),
                        PredictionSpec("candidate", root / "candidate", "_125000"),
                    ),
                    output_directory=output,
                    crop_border=2,
                    color_space="y",
                    baseline="base",
                    bootstrap_samples=100,
                    seed=7,
                    selection_json=selection,
                    selection_count=3,
                )
            )

            self.assertEqual(aggregate["pairing"]["evaluated_common_count"], 3)
            comparison = aggregate["comparisons"]["candidate_vs_base"]["psnr"]
            self.assertEqual(comparison["wins"], 3)
            self.assertGreater(comparison["mean_candidate_minus_baseline"], 0.0)
            self.assertTrue((output / "aggregate.json").is_file())
            with (output / "per_image.jsonl").open(encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), 6)
            with (output / "per_image.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            candidate_rows = [row for row in rows if row["model"] == "candidate"]
            self.assertTrue(candidate_rows[0]["psnr_difference_vs_baseline"])
            with selection.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["selected_count"], 3)
            self.assertEqual(len(set(manifest["ordered_ids"])), 3)
            contact_sheet = output / "contact.png"
            build_contact_sheet(selection, contact_sheet, tile_size=32)
            rendered = cv2.imread(str(contact_sheet))
            self.assertIsNotNone(rendered)
            self.assertEqual(rendered.shape[:2], (3 * (32 + 34), 3 * 32))

    def test_recovery_pipeline_strict_pairing_rejects_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = np.full((16, 16, 3), 100, dtype=np.uint8)
            self._write(root / "gt" / "a.png", image)
            self._write(root / "gt" / "b.png", image)
            self._write(root / "prediction" / "a.png", image)
            with self.assertRaisesRegex(ValueError, "strict pairing failed"):
                evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root / "gt",
                        predictions=(PredictionSpec("model", root / "prediction"),),
                        output_directory=root / "out",
                        crop_border=2,
                    )
                )

    def test_recovery_pipeline_requested_lpips_cannot_be_silently_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "download"):
                evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root,
                        predictions=(PredictionSpec("model", root),),
                        output_directory=root / "out",
                        lpips=True,
                        lpips_allow_model_downloads=False,
                    )
                )

    def test_recovery_pipeline_arcface_backend_is_never_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "backend"):
                evaluate_saved_outputs(
                    EvaluationConfig(
                        gt_directory=root,
                        predictions=(PredictionSpec("model", root),),
                        output_directory=root / "out",
                        arcface_model=root / "recognition_arcface_ir_se50.pth",
                    )
                )


if __name__ == "__main__":
    unittest.main()
