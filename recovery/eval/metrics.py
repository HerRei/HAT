"""Full-reference metrics matching BasicSR's NumPy evaluation conventions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.metrics.metric_util import to_y_channel
except ImportError as exc:  # pragma: no cover - exercised by CLI environments, not the project venv
    raise ImportError(
        "BasicSR is required for PSNR/SSIM. Run this tool with the project venv: "
        "/home/hermes/hat-face-training/hat-face/bin/python"
    ) from exc


def read_bgr(path: Path) -> np.ndarray:
    """Read an image exactly as an 8-bit BGR array, as BasicSR validation does."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def crop_image(image: np.ndarray, crop_border: int) -> np.ndarray:
    if crop_border < 0:
        raise ValueError("crop border cannot be negative")
    if crop_border == 0:
        return image
    height, width = image.shape[:2]
    if height <= 2 * crop_border or width <= 2 * crop_border:
        raise ValueError(
            f"crop border {crop_border} removes image with dimensions {width}x{height}"
        )
    return image[crop_border:-crop_border, crop_border:-crop_border, ...]


def validate_metric_shape(prediction: np.ndarray, ground_truth: np.ndarray, crop_border: int) -> None:
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"prediction and GT shapes differ: {prediction.shape} != {ground_truth.shape}"
        )
    cropped = crop_image(prediction, crop_border)
    if min(cropped.shape[:2]) < 11:
        raise ValueError(
            "BasicSR SSIM requires at least 11x11 pixels after border cropping; "
            f"got {cropped.shape[1]}x{cropped.shape[0]}"
        )


def luminance(image: np.ndarray) -> np.ndarray:
    """BasicSR/Matlab-style Y in [0, 255] from an 8-bit BGR image."""
    return to_y_channel(image)[..., 0].astype(np.float64)


def laplacian_variance(image: np.ndarray, crop_border: int) -> float:
    """Variance of the BasicSR Y-channel Laplacian; descriptive, not a quality score."""
    y = crop_image(luminance(image), crop_border)
    return float(cv2.Laplacian(y, cv2.CV_64F).var())


def edge_correlation(prediction: np.ndarray, ground_truth: np.ndarray, crop_border: int) -> float:
    """Pearson correlation between prediction and GT Sobel-gradient magnitudes."""
    pred_y = crop_image(luminance(prediction), crop_border)
    gt_y = crop_image(luminance(ground_truth), crop_border)

    def magnitude(image: np.ndarray) -> np.ndarray:
        x_gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        y_gradient = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(x_gradient, y_gradient).reshape(-1)

    pred_edges = magnitude(pred_y)
    gt_edges = magnitude(gt_y)
    pred_std = float(pred_edges.std())
    gt_std = float(gt_edges.std())
    if pred_std == 0.0 or gt_std == 0.0:
        return 1.0 if np.array_equal(pred_edges, gt_edges) else 0.0
    return float(np.corrcoef(pred_edges, gt_edges)[0, 1])


def fidelity_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    crop_border: int,
    color_space: str,
) -> dict[str, float]:
    validate_metric_shape(prediction, ground_truth, crop_border)
    if color_space not in {"rgb", "y"}:
        raise ValueError(f"unsupported color space: {color_space}")
    test_y_channel = color_space == "y"
    pred_sharpness = laplacian_variance(prediction, crop_border)
    gt_sharpness = laplacian_variance(ground_truth, crop_border)
    return {
        "psnr": float(
            calculate_psnr(
                prediction,
                ground_truth,
                crop_border=crop_border,
                input_order="HWC",
                test_y_channel=test_y_channel,
            )
        ),
        "ssim": float(
            calculate_ssim(
                prediction,
                ground_truth,
                crop_border=crop_border,
                input_order="HWC",
                test_y_channel=test_y_channel,
            )
        ),
        "sharpness_laplacian_variance": pred_sharpness,
        "gt_sharpness_laplacian_variance": gt_sharpness,
        "sharpness_absolute_error": abs(pred_sharpness - gt_sharpness),
        "edge_correlation": edge_correlation(prediction, ground_truth, crop_border),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def normalized_embedding(value: np.ndarray, *, source: str) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float64).reshape(-1)
    if not embedding.size or not np.all(np.isfinite(embedding)):
        raise ValueError(f"{source} produced an empty or non-finite embedding")
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        raise ValueError(f"{source} produced a zero-norm embedding")
    return embedding / norm


def cosine_similarity_from_embeddings(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compare two embeddings after strict finite-value validation and normalization."""
    pred_embedding = normalized_embedding(prediction, source="prediction ArcFace")
    gt_embedding = normalized_embedding(ground_truth, source="ground-truth ArcFace")
    if pred_embedding.shape != gt_embedding.shape:
        raise ValueError(
            "ArcFace embedding shapes differ: "
            f"{pred_embedding.shape} != {gt_embedding.shape}"
        )
    return float(np.clip(np.dot(pred_embedding, gt_embedding), -1.0, 1.0))


def active_hat_processes(proc_root: Path = Path("/proc")) -> list[tuple[int, str]]:
    """Return live HAT training/inference commands without invoking external tools."""
    active: list[tuple[int, str]] = []
    own_pid = os.getpid()
    try:
        entries = proc_root.iterdir()
    except OSError:
        return active
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except (OSError, PermissionError):
            continue
        if "hat/train.py" in command or "hat/test.py" in command:
            active.append((int(entry.name), command.strip()))
    return active


def validate_arcface_device(device: str, confirm_gpu: bool) -> None:
    """Enforce explicit GPU consent and HAT-process exclusion before model transfer."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("ArcFace device must be 'cpu' or 'cuda'")
    if device == "cpu":
        if confirm_gpu:
            raise ValueError("--confirm-arcface-gpu is only valid with --arcface-device cuda")
        return
    if not confirm_gpu:
        raise RuntimeError(
            "ArcFace GPU execution requires both --arcface-device cuda and "
            "--confirm-arcface-gpu"
        )
    active = active_hat_processes()
    if active:
        detail = "; ".join(f"PID {pid}: {command}" for pid, command in active)
        raise RuntimeError(
            "ArcFace refuses GPU initialization while HAT train/test is active: " + detail
        )


class LpipsMetric:
    """Explicit, CPU-only LPIPS adapter.

    LPIPS may ask torchvision to obtain pretrained backbone weights. Construction
    therefore requires explicit download consent even if a local cache may satisfy it.
    """

    def __init__(
        self,
        *,
        network: str,
        calibration_weights: Path | None,
        allow_model_downloads: bool,
    ) -> None:
        if not allow_model_downloads:
            raise RuntimeError(
                "LPIPS was requested but --lpips-allow-model-downloads was not set. "
                "The upstream package can fetch torchvision backbone weights; consent "
                "must be explicit. Set the flag after reviewing network/cache policy."
            )
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS was requested but the 'lpips' and 'torch' packages are not both "
                "available. Install them in a separate environment; this tool never "
                "silently omits a requested metric."
            ) from exc
        if calibration_weights is not None and not calibration_weights.is_file():
            raise FileNotFoundError(f"LPIPS calibration weights not found: {calibration_weights}")
        if network not in {"alex", "vgg", "squeeze"}:
            raise ValueError("LPIPS network must be one of: alex, vgg, squeeze")
        kwargs: dict[str, Any] = {"net": network, "verbose": False}
        if calibration_weights is not None:
            kwargs["model_path"] = str(calibration_weights)
        self._model = lpips.LPIPS(**kwargs).cpu().eval()
        self._torch = torch
        self.metadata = {
            "implementation": "lpips.LPIPS",
            "package_version": getattr(lpips, "__version__", "unknown"),
            "network": network,
            "device": "cpu",
            "backbone_weight_policy": "upstream_lpips_torchvision_downloads_explicitly_allowed",
            "calibration_weights": (
                {
                    "source": "explicit_file",
                    "path": str(calibration_weights.resolve()),
                    "sha256": sha256_file(calibration_weights),
                }
                if calibration_weights is not None
                else {"source": "bundled_by_lpips_package"}
            ),
        }

    def __call__(self, prediction: np.ndarray, ground_truth: np.ndarray, crop_border: int) -> float:
        pred = crop_image(prediction, crop_border)[..., ::-1].copy()
        gt = crop_image(ground_truth, crop_border)[..., ::-1].copy()
        pred_tensor = self._torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
        gt_tensor = self._torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float()
        pred_tensor = pred_tensor / 127.5 - 1.0
        gt_tensor = gt_tensor / 127.5 - 1.0
        with self._torch.no_grad():
            return float(self._model(pred_tensor, gt_tensor).reshape(-1)[0].item())


class ArcFaceMetric:
    """Explicit-device ArcFace ONNX adapter for already-aligned face images."""

    def __init__(
        self, model_path: Path, *, device: str = "cpu", confirm_gpu: bool = False
    ) -> None:
        validate_arcface_device(device, confirm_gpu)
        if not model_path.is_file():
            raise FileNotFoundError(f"ArcFace ONNX weights not found: {model_path}")
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "ArcFace was requested but 'onnxruntime' is unavailable. Provide a CPU "
                "environment with onnxruntime; identity scoring is never silently omitted."
            ) from exc
        model_sha256 = sha256_file(model_path)
        available_providers = set(onnxruntime.get_available_providers())
        if device == "cpu":
            provider = "CPUExecutionProvider"
        elif "ROCMExecutionProvider" in available_providers:
            provider = "ROCMExecutionProvider"
        elif "CUDAExecutionProvider" in available_providers:
            provider = "CUDAExecutionProvider"
        else:
            raise RuntimeError(
                "ArcFace CUDA/ROCm was explicitly requested, but ONNX Runtime exposes "
                f"no GPU provider (available: {sorted(available_providers)})"
            )
        self._session = onnxruntime.InferenceSession(str(model_path), providers=[provider])
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = model_input.shape
        if len(shape) != 4:
            raise ValueError(
                f"ArcFace model input must be NCHW with four dimensions, got {shape}"
            )
        self._height = int(shape[2]) if isinstance(shape[2], int) else 112
        self._width = int(shape[3]) if isinstance(shape[3], int) else 112
        if sha256_file(model_path) != model_sha256:
            raise RuntimeError("ArcFace ONNX model bytes changed while the session was loading")
        self._preprocessing = {
            "schema_version": 1,
            "input_size": [self._width, self._height],
            "alignment": "ffhq_aligned_full_frame_no_landmark_realign",
            "resize": "opencv_bilinear",
            "channel_order": "bgr_to_rgb",
            "normalization": "uint8_div_127p5_minus_1",
            "layout": "nchw",
        }
        self.metadata = {
            "implementation": "onnxruntime_direct_arcface_embedding",
            "onnxruntime_version": getattr(onnxruntime, "__version__", "unknown"),
            "model_path": str(model_path.resolve()),
            "model_sha256": model_sha256,
            "model_key": _stable_key(
                {"schema_version": 1, "backend": "onnx", "model_sha256": model_sha256}
            ),
            "weight_loading": "explicit_local_onnx_file_no_download",
            "device": device,
            "execution_provider": provider,
            "input_size": [self._width, self._height],
            "alignment": (
                "FFHQ-aligned full image after metric border crop, resized directly to "
                "model input; no detector or landmark realignment"
            ),
            "preprocessing": "BGR-to-RGB, bilinear resize, uint8/127.5-1.0, NCHW",
        }

    def preprocess_key(self, crop_border: int) -> str:
        return _stable_key({**self._preprocessing, "crop_border": crop_border})

    def embedding(self, bgr_image: np.ndarray, crop_border: int) -> np.ndarray:
        """Return one normalized embedding so callers can cache GT work."""
        image = crop_image(bgr_image, crop_border)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 127.5,
            size=(self._width, self._height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        embedding = np.asarray(
            self._session.run(None, {self._input_name: blob})[0], dtype=np.float64
        )
        return normalized_embedding(embedding, source="ArcFace ONNX model")

    def embeddings(
        self,
        bgr_images: Sequence[np.ndarray],
        crop_border: int,
        batch_size: int,
    ) -> np.ndarray:
        """Embed a sequence; fixed-batch ONNX models are handled conservatively."""
        if batch_size < 1:
            raise ValueError("ArcFace batch size must be positive")
        if not bgr_images:
            raise ValueError("ArcFace embedding batch cannot be empty")
        return np.stack(
            [self.embedding(image, crop_border) for image in bgr_images], axis=0
        )

    def similarity_from_embeddings(
        self, prediction: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        return cosine_similarity_from_embeddings(prediction, ground_truth)

    def __call__(self, prediction: np.ndarray, ground_truth: np.ndarray, crop_border: int) -> float:
        return self.similarity_from_embeddings(
            self.embedding(prediction, crop_border),
            self.embedding(ground_truth, crop_border),
        )


def facexlib_arcface_input(bgr_image: np.ndarray, crop_border: int, torch_module: Any) -> Any:
    """Prepare an already-aligned BGR image for facexlib's ArcFace IR-SE50."""
    image = crop_image(bgr_image, crop_border)
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = image[..., ::-1].copy()
    tensor = torch_module.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return tensor / 127.5 - 1.0


class FacexlibArcFaceMetric:
    """Strict explicit-device facexlib IR-SE50 adapter; never downloads weights."""

    def __init__(
        self, model_path: Path, *, device: str = "cpu", confirm_gpu: bool = False
    ) -> None:
        validate_arcface_device(device, confirm_gpu)
        if not model_path.is_file():
            raise FileNotFoundError(f"facexlib ArcFace weights not found: {model_path}")
        try:
            import facexlib
            import torch
            from facexlib.recognition.arcface_arch import Backbone
        except ImportError as exc:
            raise RuntimeError(
                "facexlib-pth ArcFace was requested but 'facexlib' and 'torch' are not "
                "both available. Identity scoring is never silently omitted."
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "ArcFace CUDA/ROCm was explicitly requested, but torch.cuda.is_available() "
                "is false"
            )
        torch_device = torch.device(device)
        model_sha256 = sha256_file(model_path)
        model = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se")
        try:
            state_dict = torch.load(
                model_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "failed to load the explicit ArcFace file as facexlib's strict IR-SE50 "
                f"state dict ({model_path}): {exc}"
            ) from exc
        if sha256_file(model_path) != model_sha256:
            raise RuntimeError("facexlib ArcFace model bytes changed while weights were loading")
        self._model = model.to(torch_device).eval()
        self._torch = torch
        self._device = torch_device
        device_metadata: dict[str, Any] = {"device": device}
        if device == "cuda":
            current_device = torch.cuda.current_device()
            device_metadata.update(
                {
                    "torch_device_index": int(current_device),
                    "torch_device_name": torch.cuda.get_device_name(current_device),
                    "torch_cuda_version": getattr(torch.version, "cuda", None),
                    "torch_hip_version": getattr(torch.version, "hip", None),
                }
            )
        self._preprocessing = {
            "schema_version": 1,
            "input_size": [112, 112],
            "alignment": "ffhq_aligned_full_frame_no_landmark_realign",
            "resize": "opencv_bilinear",
            "channel_order": "bgr_to_rgb",
            "normalization": "uint8_div_127p5_minus_1",
            "layout": "nchw",
        }
        self.metadata = {
            "implementation": "facexlib.recognition.arcface_arch.Backbone_ir_se50",
            "facexlib_version": getattr(facexlib, "__version__", "unknown"),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "model_path": str(model_path.resolve()),
            "model_sha256": model_sha256,
            "model_key": _stable_key(
                {
                    "schema_version": 1,
                    "backend": "facexlib-pth",
                    "model_sha256": model_sha256,
                }
            ),
            "weight_loading": "explicit_local_file_strict_state_dict_no_download",
            **device_metadata,
            "input_size": [112, 112],
            "alignment": (
                "FFHQ-aligned full image after metric border crop, resized directly to "
                "112x112; no detector or landmark realignment"
            ),
            "preprocessing": "BGR-to-RGB, bilinear resize, uint8/127.5-1.0, NCHW",
        }

    def preprocess_key(self, crop_border: int) -> str:
        return _stable_key({**self._preprocessing, "crop_border": crop_border})

    def embedding(self, bgr_image: np.ndarray, crop_border: int) -> np.ndarray:
        """Return one normalized embedding so callers can cache GT work."""
        return self.embeddings([bgr_image], crop_border, batch_size=1)[0]

    def embeddings(
        self,
        bgr_images: Sequence[np.ndarray],
        crop_border: int,
        batch_size: int,
    ) -> np.ndarray:
        """Run explicit bounded batches and return normalized row embeddings."""
        if batch_size < 1:
            raise ValueError("ArcFace batch size must be positive")
        if not bgr_images:
            raise ValueError("ArcFace embedding batch cannot be empty")
        results: list[np.ndarray] = []
        for start in range(0, len(bgr_images), batch_size):
            images = bgr_images[start : start + batch_size]
            tensor = self._torch.cat(
                [facexlib_arcface_input(image, crop_border, self._torch) for image in images],
                dim=0,
            ).to(self._device)
            with self._torch.inference_mode():
                output = self._model(tensor).detach().cpu().numpy().astype(np.float64)
            output = output.reshape(len(images), -1)
            results.extend(
                normalized_embedding(row, source="facexlib ArcFace model") for row in output
            )
        return np.stack(results, axis=0)

    def similarity_from_embeddings(
        self, prediction: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        return cosine_similarity_from_embeddings(prediction, ground_truth)

    def __call__(self, prediction: np.ndarray, ground_truth: np.ndarray, crop_border: int) -> float:
        return self.similarity_from_embeddings(
            self.embedding(prediction, crop_border),
            self.embedding(ground_truth, crop_border),
        )


def create_arcface_metric(
    backend: str,
    model_path: Path,
    *,
    device: str = "cpu",
    confirm_gpu: bool = False,
) -> ArcFaceMetric | FacexlibArcFaceMetric:
    if backend == "onnx":
        return ArcFaceMetric(model_path, device=device, confirm_gpu=confirm_gpu)
    if backend == "facexlib-pth":
        return FacexlibArcFaceMetric(model_path, device=device, confirm_gpu=confirm_gpu)
    raise ValueError("ArcFace backend must be 'onnx' or 'facexlib-pth'")
