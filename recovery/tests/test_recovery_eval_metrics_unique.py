from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

from recovery.eval.metrics import (
    FacexlibArcFaceMetric,
    edge_correlation,
    facexlib_arcface_input,
    fidelity_metrics,
    validate_arcface_device,
)


class RecoveryEvalMetricCompatibilityTests(unittest.TestCase):
    def test_recovery_fidelity_matches_basicsr_rgb_and_y(self) -> None:
        rng = np.random.default_rng(1834)
        gt = rng.integers(0, 256, size=(32, 35, 3), dtype=np.uint8)
        prediction = np.clip(gt.astype(np.int16) + 3, 0, 255).astype(np.uint8)
        for color_space, test_y in (("rgb", False), ("y", True)):
            with self.subTest(color_space=color_space):
                actual = fidelity_metrics(
                    prediction, gt, crop_border=4, color_space=color_space
                )
                expected_psnr = calculate_psnr(
                    prediction, gt, crop_border=4, test_y_channel=test_y
                )
                expected_ssim = calculate_ssim(
                    prediction, gt, crop_border=4, test_y_channel=test_y
                )
                self.assertAlmostEqual(actual["psnr"], expected_psnr, places=12)
                self.assertAlmostEqual(actual["ssim"], expected_ssim, places=12)

    def test_recovery_edge_correlation_identical_is_one(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
        self.assertAlmostEqual(edge_correlation(image, image, 2), 1.0, places=12)

    def test_recovery_facexlib_arcface_preprocessing_is_explicit_rgb_minus_one_to_one(self) -> None:
        import torch

        bgr = np.empty((16, 18, 3), dtype=np.uint8)
        bgr[..., 0] = 0
        bgr[..., 1] = 128
        bgr[..., 2] = 255
        tensor = facexlib_arcface_input(bgr, crop_border=2, torch_module=torch)
        self.assertEqual(tuple(tensor.shape), (1, 3, 112, 112))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), -1.0, places=6)
        self.assertTrue(torch.all(tensor >= -1.0))
        self.assertTrue(torch.all(tensor <= 1.0))

    def test_recovery_arcface_cuda_requires_explicit_consent(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires both"):
            validate_arcface_device("cuda", confirm_gpu=False)
        with self.assertRaisesRegex(ValueError, "only valid"):
            validate_arcface_device("cpu", confirm_gpu=True)

    def test_recovery_arcface_cuda_refuses_active_hat_before_model_load(self) -> None:
        with mock.patch(
            "recovery.eval.metrics.active_hat_processes",
            return_value=[(1234, "python hat/train.py -opt train.yml")],
        ):
            with self.assertRaisesRegex(RuntimeError, "HAT train/test is active"):
                FacexlibArcFaceMetric(
                    Path("not-even-opened.pth"), device="cuda", confirm_gpu=True
                )

    def test_recovery_arcface_cuda_guard_allows_explicit_idle_request(self) -> None:
        with mock.patch(
            "recovery.eval.metrics.active_hat_processes", return_value=[]
        ):
            validate_arcface_device("cuda", confirm_gpu=True)


if __name__ == "__main__":
    unittest.main()
