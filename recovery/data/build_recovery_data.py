#!/usr/bin/env python3
"""Build and verify deterministic face SR recovery datasets.

The generated image tree lives outside the repository by default. Ground-truth
images are always symlinked from the original, consistent FFHQ split; they are
never copied. Every generated LQ image has a deterministic JSONL record with
its recipe, seed, parameters, paths, dimensions, and byte/pixel hashes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


GENERATOR_VERSION = "1.0.0"
SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_FILENAME = "build_contract.json"
DEFAULT_SOURCE_ROOT = Path("/home/hermes/hat-face-training/data/ffhq_face_sr")
DEFAULT_OUTPUT_ROOT = Path("/home/hermes/hat-face-training/data/face_sr_recovery")
DEFAULT_SEED = 20260820
DEFAULT_PILOT_SEED = 20260821
DEFAULT_PILOT_SIZE = 512
TARGETS = ("clean", "mild-mixed", "benchmarks")
INTERPOLATIONS = ("area", "bilinear", "bicubic")


class RecoveryDataError(RuntimeError):
    """Raised for a user-actionable recovery data failure."""


@dataclass(frozen=True)
class SourceImage:
    image_id: str
    path: str
    relative_path: str
    split: str
    file_sha256: str
    pixel_sha256: str


@dataclass(frozen=True)
class GenerationJob:
    source: SourceImage
    gt_path: str
    lq_path: str
    bucket: str
    recipe: str
    sample_id: str
    seed: int
    scale: int
    repair: bool


@dataclass(frozen=True)
class BuildConfig:
    train_gt_root: Path
    val_gt_root: Path
    output_root: Path
    targets: tuple[str, ...] = TARGETS
    seed: int = DEFAULT_SEED
    pilot_seed: int = DEFAULT_PILOT_SEED
    pilot_size: int = DEFAULT_PILOT_SIZE
    workers: int = 1
    repair: bool = False
    dry_run: bool = False
    expected_train_count: int | None = 65000
    expected_val_count: int | None = 5000


@dataclass(frozen=True)
class DirectorySpec:
    """Exact expected membership and topology for one managed directory."""

    kind: str
    members: dict[str, str]
    symlink_target: str | None = None
    member_symlink_targets: dict[str, str] | None = None


@lru_cache(maxsize=1)
def _dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import basicsr
        import cv2
        import numpy as np
        import torch
        from basicsr.utils.matlab_functions import imresize
    except (ImportError, RuntimeError) as error:
        raise RecoveryDataError(
            "Image generation requires the project environment. Run with "
            "'/home/hermes/hat-face-training/hat-face/bin/python'. Original "
            f"import error: {error}"
        ) from error
    return np, cv2, imresize, basicsr, torch


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RecoveryDataError(f"Cannot hash '{path}': {error}") from error
    return digest.hexdigest()


def _hash_path(path: str) -> tuple[str, str]:
    item = Path(path)
    return path, _sha256_file(item)


def _audit_source_path(path: str) -> tuple[str, str, str]:
    image, content = _read_image(Path(path))
    np, _, _, _, _ = _dependencies()
    return (
        path,
        _sha256_bytes(content),
        _sha256_bytes(np.ascontiguousarray(image).tobytes()),
    )


def _stable_seed(global_seed: int, *parts: str) -> int:
    payload = "\0".join((str(global_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _manifest_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(record) for record in records)


def _atomic_write_if_changed(path: Path, content: bytes) -> bool:
    if path.exists():
        try:
            if path.read_bytes() == content:
                return False
        except OSError as error:
            raise RecoveryDataError(f"Cannot read existing file '{path}': {error}") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return True


def _write_artifact(path: Path, content: bytes, repair: bool) -> str:
    expected_hash = _sha256_bytes(content)
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise RecoveryDataError(f"Expected generated file but found symlink: '{path}'")
        actual_hash = _sha256_file(path)
        if actual_hash == expected_hash:
            return "reused"
        if not repair:
            raise RecoveryDataError(
                f"Existing artifact differs from deterministic output: '{path}'. "
                "Use --repair to replace mismatched generated artifacts."
            )
        action = "repaired"
    else:
        action = "created"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return action


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _ensure_symlink(link: Path, target: Path, repair: bool, directory: bool = False) -> str:
    target = target.resolve(strict=True)
    if link.is_symlink():
        try:
            if link.resolve(strict=True) == target:
                return "reused"
        except OSError:
            pass
        if not repair:
            raise RecoveryDataError(
                f"Symlink points to the wrong target: '{link}'. Use --repair to replace it."
            )
        link.unlink()
        action = "repaired"
    elif link.exists():
        kind = "directory" if link.is_dir() else "file"
        raise RecoveryDataError(
            f"Refusing to replace real {kind} at symlink location '{link}'. Move it manually first."
        )
    else:
        action = "created"
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary_link = link.with_name(f".{link.name}.tmp.{os.getpid()}")
    temporary_link.unlink(missing_ok=True)
    os.symlink(target, temporary_link, target_is_directory=directory)
    os.replace(temporary_link, link)
    return action


def _list_images(root: Path, split: str, expected_count: int | None) -> list[Path]:
    if not root.is_dir():
        raise RecoveryDataError(f"{split} GT root is not a readable directory: '{root}'")
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    invalid = [
        path.name
        for path in entries
        if path.is_symlink() or not path.is_file() or path.suffix != ".png"
    ]
    if invalid:
        raise RecoveryDataError(
            f"{split} GT root must contain only regular lowercase .png files; invalid: "
            f"{_preview_names(invalid)}"
        )
    files = entries
    if not files:
        raise RecoveryDataError(f"No supported images found in {split} GT root '{root}'")
    if expected_count is not None and len(files) != expected_count:
        raise RecoveryDataError(
            f"Expected {expected_count} {split} GT images in '{root}', found {len(files)}. "
            "Use the original 65,000/5,000 split or set the explicit expected count for a test dataset."
        )
    stems = [path.stem for path in files]
    if len(set(stems)) != len(stems):
        raise RecoveryDataError(f"Duplicate image stems found in {split} GT root '{root}'")
    return files


def _parallel_source_audit(paths: Sequence[Path], workers: int) -> dict[str, tuple[str, str]]:
    serialized = [str(path.resolve()) for path in paths]
    if workers == 1:
        results = map(_audit_source_path, serialized)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_audit_source_path, serialized, chunksize=32)
            return {path: (file_hash, pixel_hash) for path, file_hash, pixel_hash in results}
    return {path: (file_hash, pixel_hash) for path, file_hash, pixel_hash in results}


def audit_source_split(config: BuildConfig) -> tuple[list[SourceImage], list[SourceImage], dict[str, Any]]:
    """Validate the old split and reject exact cross-split content leakage."""
    train_root = config.train_gt_root.resolve()
    val_root = config.val_gt_root.resolve()
    output_root = _resolved(config.output_root)
    if train_root == val_root:
        raise RecoveryDataError("Train and validation GT roots must be different directories")
    for source_root in (train_root, val_root):
        if output_root == source_root or source_root in output_root.parents:
            raise RecoveryDataError(
                f"Output root '{output_root}' must not be inside source GT root '{source_root}'"
            )

    train_paths = _list_images(train_root, "train", config.expected_train_count)
    val_paths = _list_images(val_root, "val", config.expected_val_count)
    overlapping_names = sorted({path.name for path in train_paths} & {path.name for path in val_paths})
    if overlapping_names:
        preview = ", ".join(overlapping_names[:5])
        raise RecoveryDataError(f"Train/validation basenames overlap ({preview}); refusing a leaky split")

    all_hashes = _parallel_source_audit([*train_paths, *val_paths], config.workers)
    train_hash_to_paths: dict[str, list[str]] = {}
    train_pixel_hash_to_paths: dict[str, list[str]] = {}
    for path in train_paths:
        file_hash, pixel_hash = all_hashes[str(path.resolve())]
        train_hash_to_paths.setdefault(file_hash, []).append(str(path))
        train_pixel_hash_to_paths.setdefault(pixel_hash, []).append(str(path))
    file_collisions: list[tuple[str, str]] = []
    pixel_collisions: list[tuple[str, str]] = []
    for path in val_paths:
        file_hash, pixel_hash = all_hashes[str(path.resolve())]
        for train_path in train_hash_to_paths.get(file_hash, []):
            file_collisions.append((train_path, str(path)))
        for train_path in train_pixel_hash_to_paths.get(pixel_hash, []):
            pixel_collisions.append((train_path, str(path)))
    if file_collisions:
        train_path, val_path = file_collisions[0]
        raise RecoveryDataError(
            "Exact file content occurs in both source splits; refusing potential leakage: "
            f"train='{train_path}', val='{val_path}'"
        )
    if pixel_collisions:
        train_path, val_path = pixel_collisions[0]
        raise RecoveryDataError(
            "Exact decoded pixels occur in both source splits; refusing exact-pixel leakage: "
            f"train='{train_path}', val='{val_path}'"
        )

    def make_sources(paths: Sequence[Path], root: Path, split: str) -> list[SourceImage]:
        return [
            SourceImage(
                image_id=path.stem,
                path=str(path.resolve()),
                relative_path=path.relative_to(root).as_posix(),
                split=split,
                file_sha256=all_hashes[str(path.resolve())][0],
                pixel_sha256=all_hashes[str(path.resolve())][1],
            )
            for path in paths
        ]

    train_sources = make_sources(train_paths, train_root, "train")
    val_sources = make_sources(val_paths, val_root, "val")
    audit = {
        "hashes": ["sha256_file_bytes", "sha256_decoded_bgr_uint8_pixels"],
        "train_count": len(train_sources),
        "train_root": str(train_root),
        "val_count": len(val_sources),
        "val_root": str(val_root),
        "cross_split_basename_collisions": 0,
        "cross_split_file_hash_collisions": 0,
        "cross_split_pixel_hash_collisions": 0,
    }
    return train_sources, val_sources, audit


def _read_image(path: Path) -> tuple[Any, bytes]:
    np, cv2, _, _, _ = _dependencies()
    try:
        content = path.read_bytes()
    except OSError as error:
        raise RecoveryDataError(f"Cannot read source image '{path}': {error}") from error
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RecoveryDataError(f"OpenCV could not decode source image '{path}'")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise RecoveryDataError(
            f"Source image must decode as 8-bit 3-channel color, got shape={image.shape}, "
            f"dtype={image.dtype} for '{path}'"
        )
    return np.ascontiguousarray(image), content


def _quantize(image: Any) -> Any:
    np, _, _, _, _ = _dependencies()
    return np.ascontiguousarray(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8))


def matlab_bicubic_x4(image_u8: Any) -> Any:
    """Return uint8 x4 LQ using BasicSR's antialiased MATLAB-style bicubic."""
    np, _, imresize, _, _ = _dependencies()
    if image_u8.ndim != 3 or image_u8.shape[2] != 3 or image_u8.dtype != np.uint8:
        raise RecoveryDataError("matlab_bicubic_x4 expects an HWC uint8 3-channel image")
    height, width = image_u8.shape[:2]
    if height % 4 or width % 4:
        raise RecoveryDataError(f"GT dimensions must be divisible by 4, got {width}x{height}")
    resized = imresize(image_u8.astype(np.float32) / 255.0, 0.25, antialiasing=True)
    expected_shape = (height // 4, width // 4, 3)
    if resized.shape != expected_shape:
        raise RecoveryDataError(f"BasicSR imresize returned {resized.shape}; expected {expected_shape}")
    return _quantize(resized)


def _gaussian_blur(image: Any, sigma: float) -> tuple[Any, int]:
    _, cv2, _, _, _ = _dependencies()
    radius = max(1, int((3.0 * sigma) + 0.999999))
    kernel_size = 2 * radius + 1
    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return blurred, kernel_size


def _add_gaussian_noise(image: Any, sigma_255: float, gray: bool, seed: int) -> Any:
    np, _, _, _, _ = _dependencies()
    rng = np.random.Generator(np.random.PCG64(seed))
    noise_shape = (*image.shape[:2], 1 if gray else 3)
    noise = rng.normal(0.0, sigma_255 / 255.0, size=noise_shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def _jpeg_roundtrip(image: Any, quality: int) -> Any:
    np, cv2, _, _, _ = _dependencies()
    encoded_input = _quantize(image)
    ok, jpeg = cv2.imencode(".jpg", encoded_input, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RecoveryDataError(f"OpenCV JPEG encode failed at quality {quality}")
    decoded = cv2.imdecode(np.asarray(jpeg), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RecoveryDataError(f"OpenCV JPEG decode failed at quality {quality}")
    return decoded.astype(np.float32) / 255.0


def _resize_cv2(image: Any, width: int, height: int, method: str) -> Any:
    _, cv2, _, _, _ = _dependencies()
    flags = {
        "area": cv2.INTER_AREA,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    return cv2.resize(image, (width, height), interpolation=flags[method]).astype("float32")


def clean_parameters() -> dict[str, Any]:
    return {
        "antialiasing": True,
        "implementation": "basicsr.utils.matlab_functions.imresize",
        "input_range": "float32_[0,1]",
        "quantization": "numpy.rint_then_clip_uint8",
        "resize_scale": 0.25,
    }


def mild_parameters(seed: int) -> dict[str, Any]:
    np, _, _, _, _ = _dependencies()
    rng = np.random.Generator(np.random.PCG64(seed))
    sigma = round(float(rng.uniform(0.2, 1.2)), 6)
    return {
        "blur": {
            "border": "reflect_101",
            "kernel_size": 2 * max(1, int((3.0 * sigma) + 0.999999)) + 1,
            "sigma": sigma,
            "type": "isotropic_gaussian",
        },
        "downsample": clean_parameters(),
        "jpeg": {"backend": "opencv", "quality": int(rng.integers(70, 96))},
        "noise": {
            "distribution": "gaussian",
            "gray": bool(rng.random() < 0.2),
            "seed": _stable_seed(seed, "noise"),
            "sigma_255": round(float(rng.uniform(0.5, 8.0)), 6),
        },
        "order": ["blur", "downsample", "noise", "jpeg"],
    }


def hard_parameters(seed: int, gt_width: int, gt_height: int) -> dict[str, Any]:
    np, _, _, _, _ = _dependencies()
    rng = np.random.Generator(np.random.PCG64(seed))
    sigma1 = round(float(rng.uniform(0.8, 2.4)), 6)
    sigma2 = round(float(rng.uniform(0.2, 1.1)), 6)
    factor = float(rng.choice(np.asarray([1.25, 1.5, 1.75, 2.0])))
    lq_width, lq_height = gt_width // 4, gt_height // 4
    return {
        "stage1": {
            "blur": {
                "border": "reflect_101",
                "kernel_size": 2 * max(1, int((3.0 * sigma1) + 0.999999)) + 1,
                "sigma": sigma1,
                "type": "isotropic_gaussian",
            },
            "jpeg": {"backend": "opencv", "quality": int(rng.integers(35, 76))},
            "noise": {
                "distribution": "gaussian",
                "gray": bool(rng.random() < 0.35),
                "seed": _stable_seed(seed, "noise1"),
                "sigma_255": round(float(rng.uniform(4.0, 18.0)), 6),
            },
            "resize": {
                "height": int(round(lq_height * factor)),
                "implementation": "opencv",
                "method": INTERPOLATIONS[int(rng.integers(0, len(INTERPOLATIONS)))],
                "width": int(round(lq_width * factor)),
            },
        },
        "stage2": {
            "blur": {
                "border": "reflect_101",
                "kernel_size": 2 * max(1, int((3.0 * sigma2) + 0.999999)) + 1,
                "sigma": sigma2,
                "type": "isotropic_gaussian",
            },
            "jpeg": {"backend": "opencv", "quality": int(rng.integers(25, 66))},
            "noise": {
                "distribution": "gaussian",
                "gray": bool(rng.random() < 0.35),
                "seed": _stable_seed(seed, "noise2"),
                "sigma_255": round(float(rng.uniform(1.0, 10.0)), 6),
            },
            "resize": {
                "height": lq_height,
                "implementation": "opencv",
                "method": INTERPOLATIONS[int(rng.integers(0, len(INTERPOLATIONS)))],
                "width": lq_width,
            },
        },
        "order": [
            "stage1.blur",
            "stage1.resize",
            "stage1.noise",
            "stage1.jpeg",
            "stage2.blur",
            "stage2.resize",
            "stage2.noise",
            "stage2.jpeg",
        ],
    }


def _generate_lq(image_u8: Any, recipe: str, seed: int) -> tuple[Any, dict[str, Any]]:
    np, _, imresize, _, _ = _dependencies()
    height, width = image_u8.shape[:2]
    if height % 4 or width % 4:
        raise RecoveryDataError(f"GT dimensions must be divisible by 4, got {width}x{height}")
    if recipe == "clean":
        return matlab_bicubic_x4(image_u8), clean_parameters()

    image = image_u8.astype(np.float32) / 255.0
    if recipe == "mild":
        parameters = mild_parameters(seed)
        image, kernel_size = _gaussian_blur(image, parameters["blur"]["sigma"])
        if kernel_size != parameters["blur"]["kernel_size"]:
            raise RecoveryDataError("Internal mild blur parameter mismatch")
        image = imresize(image, 0.25, antialiasing=True)
        noise = parameters["noise"]
        image = _add_gaussian_noise(image, noise["sigma_255"], noise["gray"], noise["seed"])
        image = _jpeg_roundtrip(image, parameters["jpeg"]["quality"])
        return _quantize(image), parameters

    if recipe == "hard":
        parameters = hard_parameters(seed, width, height)
        stage1 = parameters["stage1"]
        image, _ = _gaussian_blur(image, stage1["blur"]["sigma"])
        resize1 = stage1["resize"]
        image = _resize_cv2(image, resize1["width"], resize1["height"], resize1["method"])
        noise1 = stage1["noise"]
        image = _add_gaussian_noise(image, noise1["sigma_255"], noise1["gray"], noise1["seed"])
        image = _jpeg_roundtrip(image, stage1["jpeg"]["quality"])
        stage2 = parameters["stage2"]
        image, _ = _gaussian_blur(image, stage2["blur"]["sigma"])
        resize2 = stage2["resize"]
        image = _resize_cv2(image, resize2["width"], resize2["height"], resize2["method"])
        noise2 = stage2["noise"]
        image = _add_gaussian_noise(image, noise2["sigma_255"], noise2["gray"], noise2["seed"])
        image = _jpeg_roundtrip(image, stage2["jpeg"]["quality"])
        return _quantize(image), parameters

    raise RecoveryDataError(f"Unknown degradation recipe '{recipe}'")


def _encode_png(image: Any) -> bytes:
    np, cv2, _, _, _ = _dependencies()
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RecoveryDataError("OpenCV PNG encode failed")
    return np.asarray(encoded).tobytes()


def _process_generation_job(job: GenerationJob) -> tuple[dict[str, Any], str]:
    np, _, _, _, _ = _dependencies()
    source_path = Path(job.source.path)
    image, source_bytes = _read_image(source_path)
    actual_source_hash = _sha256_bytes(source_bytes)
    if actual_source_hash != job.source.file_sha256:
        raise RecoveryDataError(
            f"Source changed after split audit: '{source_path}' "
            f"({job.source.file_sha256} -> {actual_source_hash})"
        )
    actual_pixel_hash = _sha256_bytes(np.ascontiguousarray(image).tobytes())
    if actual_pixel_hash != job.source.pixel_sha256:
        raise RecoveryDataError(
            f"Source decoded pixels changed after split audit: '{source_path}' "
            f"({job.source.pixel_sha256} -> {actual_pixel_hash})"
        )
    lq, parameters = _generate_lq(image, job.recipe, job.seed)
    encoded = _encode_png(lq)
    action = _write_artifact(Path(job.lq_path), encoded, job.repair)
    height, width = image.shape[:2]
    lq_height, lq_width = lq.shape[:2]
    record = {
        "bucket": job.bucket,
        "generator_version": GENERATOR_VERSION,
        "gt_path": job.gt_path,
        "gt_pixel_sha256": actual_pixel_hash,
        "gt_sha256": actual_source_hash,
        "gt_size": [width, height],
        "id": job.source.image_id,
        "lq_path": job.lq_path,
        "lq_pixel_sha256": _sha256_bytes(np.ascontiguousarray(lq).tobytes()),
        "lq_sha256": _sha256_bytes(encoded),
        "lq_size": [lq_width, lq_height],
        "parameters": parameters,
        "recipe": job.recipe,
        "sample_id": job.sample_id,
        "scale": job.scale,
        "schema_version": SCHEMA_VERSION,
        "seed": job.seed,
        "source_gt_path": job.source.path,
        "source_relative_path": job.source.relative_path,
        "split": job.source.split,
    }
    return record, action


def _run_jobs(jobs: Sequence[GenerationJob], workers: int, label: str) -> list[dict[str, Any]]:
    counts = {"created": 0, "repaired": 0, "reused": 0}
    records: list[dict[str, Any]] = []
    if workers == 1:
        results: Iterable[tuple[dict[str, Any], str]] = map(_process_generation_job, jobs)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_process_generation_job, jobs, chunksize=8)
    try:
        for index, (record, action) in enumerate(results, start=1):
            records.append(record)
            counts[action] += 1
            if index % 1000 == 0 or index == len(jobs):
                print(
                    f"{label}: {index}/{len(jobs)} "
                    f"created={counts['created']} reused={counts['reused']} repaired={counts['repaired']}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return records


def _environment_metadata() -> dict[str, str]:
    np, cv2, _, basicsr, torch = _dependencies()
    return {
        "basicsr": str(basicsr.__version__),
        "generator_version": GENERATOR_VERSION,
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }


def _write_manifest(
    path: Path,
    records: Sequence[dict[str, Any]],
    config: BuildConfig,
    audit: dict[str, Any],
) -> None:
    content = _manifest_bytes(records)
    metadata = {
        "environment": _environment_metadata(),
        "global_seed": config.seed,
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256_bytes(content),
        "pilot_seed": config.pilot_seed,
        "record_count": len(records),
        "schema_version": SCHEMA_VERSION,
        "source_split_audit": audit,
    }
    _atomic_write_if_changed(path, content)
    _atomic_write_if_changed(path.with_suffix(path.suffix + ".meta.json"), _json_bytes(metadata))


def _meta_info_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    lines: list[str] = []
    seen: set[str] = set()
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise RecoveryDataError(f"Duplicate sample_id while writing meta-info: '{sample_id}'")
        seen.add(sample_id)
        width, height = record["gt_size"]
        lines.append(f"{sample_id}.png ({height},{width},3)\n")
    return "".join(lines).encode("ascii")


def _write_meta_info(dataset_root: Path, records: Sequence[dict[str, Any]]) -> None:
    _atomic_write_if_changed(dataset_root / "meta_info.txt", _meta_info_bytes(records))


def _jobs_for_recipe(
    sources: Sequence[SourceImage],
    gt_root: Path,
    lq_root: Path,
    bucket: str,
    recipe: str,
    config: BuildConfig,
    suffix: str = "",
) -> list[GenerationJob]:
    jobs = []
    for source in sources:
        filename = f"{source.image_id}{suffix}.png"
        jobs.append(
            GenerationJob(
                source=source,
                gt_path=str((gt_root / filename).absolute()),
                lq_path=str((lq_root / filename).absolute()),
                bucket=bucket,
                recipe=recipe,
                sample_id=f"{source.image_id}{suffix}",
                seed=_stable_seed(config.seed, source.split, recipe, source.relative_path),
                scale=4,
                repair=config.repair,
            )
        )
    return jobs


def _build_clean_split(
    split: str,
    sources: Sequence[SourceImage],
    source_root: Path,
    config: BuildConfig,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset_root = config.output_root / "clean" / split
    _ensure_symlink(dataset_root / "gt", source_root, config.repair, directory=True)
    jobs = _jobs_for_recipe(sources, dataset_root / "gt", dataset_root / "lq", "clean", "clean", config)
    records = _run_jobs(jobs, config.workers, f"clean/{split}")
    _write_manifest(config.output_root / "manifests" / f"clean_{split}.jsonl", records, config, audit)
    _write_meta_info(dataset_root, records)
    return records


def _build_validation_recipe(
    recipe: str,
    sources: Sequence[SourceImage],
    source_root: Path,
    config: BuildConfig,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset_root = config.output_root / recipe / "val"
    _ensure_symlink(dataset_root / "gt", source_root, config.repair, directory=True)
    jobs = _jobs_for_recipe(sources, dataset_root / "gt", dataset_root / "lq", recipe, recipe, config)
    records = _run_jobs(jobs, config.workers, f"{recipe}/val")
    _write_manifest(config.output_root / "manifests" / f"{recipe}_val.jsonl", records, config, audit)
    _write_meta_info(dataset_root, records)
    return records


def _record_for_symlinked_sample(
    original: dict[str, Any],
    bucket: str,
    sample_id: str,
    gt_path: Path,
    lq_path: Path,
) -> dict[str, Any]:
    record = dict(original)
    record.update(
        {
            "bucket": bucket,
            "gt_path": str(gt_path.absolute()),
            "lq_path": str(lq_path.absolute()),
            "sample_id": sample_id,
        }
    )
    return record


def _build_mild_mixed_train(
    sources: Sequence[SourceImage],
    clean_records: Sequence[dict[str, Any]],
    config: BuildConfig,
    audit: dict[str, Any],
) -> None:
    dataset_root = config.output_root / "mild_mixed" / "train"
    gt_root, lq_root = dataset_root / "gt", dataset_root / "lq"
    gt_root.mkdir(parents=True, exist_ok=True)
    lq_root.mkdir(parents=True, exist_ok=True)
    clean_by_id = {record["id"]: record for record in clean_records}
    jobs = _jobs_for_recipe(
        sources,
        gt_root,
        lq_root,
        "mild_mixed",
        "mild",
        config,
        suffix="_mild",
    )
    mild_records = _run_jobs(jobs, config.workers, "mild_mixed/train mild half")
    mild_by_id = {record["id"]: record for record in mild_records}
    mixed_records: list[dict[str, Any]] = []
    for source in sources:
        clean_gt = gt_root / f"{source.image_id}_clean.png"
        mild_gt = gt_root / f"{source.image_id}_mild.png"
        clean_lq = lq_root / f"{source.image_id}_clean.png"
        _ensure_symlink(clean_gt, Path(source.path), config.repair)
        _ensure_symlink(mild_gt, Path(source.path), config.repair)
        _ensure_symlink(
            clean_lq,
            config.output_root / "clean" / "train" / "lq" / f"{source.image_id}.png",
            config.repair,
        )
        clean_record = _record_for_symlinked_sample(
            clean_by_id[source.image_id],
            "mild_mixed",
            f"{source.image_id}_clean",
            clean_gt,
            clean_lq,
        )
        mild_record = dict(mild_by_id[source.image_id])
        mild_record["gt_path"] = str(mild_gt.absolute())
        mixed_records.extend((clean_record, mild_record))
    _write_manifest(config.output_root / "manifests" / "mild_mixed_train.jsonl", mixed_records, config, audit)
    _write_meta_info(dataset_root, mixed_records)


def select_pilot(sources: Sequence[SourceImage], pilot_size: int, pilot_seed: int) -> list[SourceImage]:
    if pilot_size <= 0:
        raise RecoveryDataError("Pilot size must be positive")
    if pilot_size > len(sources):
        raise RecoveryDataError(f"Pilot size {pilot_size} exceeds validation count {len(sources)}")
    ranked = sorted(
        sources,
        key=lambda source: (
            hashlib.sha256(
                f"{pilot_seed}\0{source.relative_path}".encode("utf-8")
            ).digest(),
            source.relative_path,
        ),
    )
    return sorted(ranked[:pilot_size], key=lambda source: source.relative_path)


def _canonical_targets(targets: Sequence[str]) -> tuple[str, ...]:
    selected = set(targets)
    return tuple(target for target in TARGETS if target in selected)


def _expected_manifest_names(targets: Sequence[str]) -> tuple[str, ...]:
    selected = set(targets)
    names: list[str] = []
    if selected & {"clean", "mild-mixed"}:
        names.append("clean_train.jsonl")
    if selected & {"clean", "benchmarks"}:
        names.append("clean_val.jsonl")
    if "mild-mixed" in selected:
        names.append("mild_mixed_train.jsonl")
    if "benchmarks" in selected:
        names.extend(
            (
                "mild_val.jsonl",
                "hard_val.jsonl",
                "pilot_selection.jsonl",
                "clean_pilot.jsonl",
                "mild_pilot.jsonl",
                "hard_pilot.jsonl",
            )
        )
    return tuple(names)


def _inventory_fingerprint(sources: Sequence[SourceImage]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.file_sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(source.pixel_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_contract(
    config: BuildConfig,
    train_sources: Sequence[SourceImage],
    val_sources: Sequence[SourceImage],
) -> dict[str, Any]:
    targets = _canonical_targets(config.targets)
    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "environment": _environment_metadata(),
        "expected_manifests": list(_expected_manifest_names(targets)),
        "generator_version": GENERATOR_VERSION,
        "output_root": str(config.output_root.resolve()),
        "pilot": {
            "rank_method": "sha256(seed_NUL_source_relative_path)",
            "seed": config.pilot_seed,
            "size": config.pilot_size,
        },
        "recipes": {
            "clean": clean_parameters(),
            "hard": "deterministic_two_stage_v1",
            "meta_info": "basicsr_filename_space_(height,width,3)_v1",
            "mild": "deterministic_one_stage_v1",
            "scale": 4,
        },
        "seed": config.seed,
        "source_split": {
            "train": {
                "count": len(train_sources),
                "inventory_sha256": _inventory_fingerprint(train_sources),
                "root": str(config.train_gt_root.resolve()),
            },
            "val": {
                "count": len(val_sources),
                "inventory_sha256": _inventory_fingerprint(val_sources),
                "root": str(config.val_gt_root.resolve()),
            },
        },
        "targets": list(targets),
    }


def _source_names(sources: Sequence[SourceImage]) -> dict[str, str]:
    return {f"{source.image_id}.png": "regular_file" for source in sources}


def _expected_layout(
    config: BuildConfig,
    train_sources: Sequence[SourceImage],
    val_sources: Sequence[SourceImage],
    selected: Sequence[SourceImage],
) -> dict[Path, DirectorySpec]:
    """Return the complete managed tree. Missing entries are allowed only while building."""
    targets = set(config.targets)
    needs_clean_train = bool(targets & {"clean", "mild-mixed"})
    needs_clean_val = bool(targets & {"clean", "benchmarks"})
    train_names = _source_names(train_sources)
    val_names = _source_names(val_sources)
    root_members: dict[str, str] = {
        CONTRACT_FILENAME: "regular_file",
        "manifests": "real_dir",
    }
    specs: dict[Path, DirectorySpec] = {}

    def add(
        relative: str,
        kind: str,
        members: dict[str, str],
        target: Path | None = None,
        member_targets: dict[str, str] | None = None,
    ) -> None:
        specs[Path(relative)] = DirectorySpec(
            kind=kind,
            members=members,
            symlink_target=str(target.resolve()) if target is not None else None,
            member_symlink_targets=member_targets,
        )

    clean_splits: dict[str, str] = {}
    if needs_clean_train:
        clean_splits["train"] = "real_dir"
        add(
            "clean/train",
            "real_dir",
            {"gt": "symlink_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
        )
        add("clean/train/gt", "symlink_dir", train_names, config.train_gt_root)
        add("clean/train/lq", "real_dir", dict(train_names))
    if needs_clean_val:
        clean_splits["val"] = "real_dir"
        add(
            "clean/val",
            "real_dir",
            {"gt": "symlink_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
        )
        add("clean/val/gt", "symlink_dir", val_names, config.val_gt_root)
        add("clean/val/lq", "real_dir", dict(val_names))
    if clean_splits:
        root_members["clean"] = "real_dir"
        add("clean", "real_dir", clean_splits)

    if "mild-mixed" in targets:
        root_members["mild_mixed"] = "real_dir"
        mixed_gt: dict[str, str] = {}
        mixed_lq: dict[str, str] = {}
        mixed_gt_targets: dict[str, str] = {}
        mixed_lq_targets: dict[str, str] = {}
        for source in train_sources:
            mixed_gt[f"{source.image_id}_clean.png"] = "symlink_file"
            mixed_gt[f"{source.image_id}_mild.png"] = "symlink_file"
            mixed_lq[f"{source.image_id}_clean.png"] = "symlink_file"
            mixed_lq[f"{source.image_id}_mild.png"] = "regular_file"
            mixed_gt_targets[f"{source.image_id}_clean.png"] = source.path
            mixed_gt_targets[f"{source.image_id}_mild.png"] = source.path
            mixed_lq_targets[f"{source.image_id}_clean.png"] = str(
                (config.output_root / "clean" / "train" / "lq" / f"{source.image_id}.png").resolve(
                    strict=False
                )
            )
        add("mild_mixed", "real_dir", {"train": "real_dir"})
        add(
            "mild_mixed/train",
            "real_dir",
            {"gt": "real_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
        )
        add(
            "mild_mixed/train/gt",
            "real_dir",
            mixed_gt,
            member_targets=mixed_gt_targets,
        )
        add(
            "mild_mixed/train/lq",
            "real_dir",
            mixed_lq,
            member_targets=mixed_lq_targets,
        )

    if "benchmarks" in targets:
        for recipe in ("mild", "hard"):
            root_members[recipe] = "real_dir"
            add(recipe, "real_dir", {"val": "real_dir"})
            add(
                f"{recipe}/val",
                "real_dir",
                {"gt": "symlink_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
            )
            add(f"{recipe}/val/gt", "symlink_dir", val_names, config.val_gt_root)
            add(f"{recipe}/val/lq", "real_dir", dict(val_names))

        root_members["benchmarks"] = "real_dir"
        benchmark_members = {f"{recipe}_pilot": "real_dir" for recipe in ("clean", "mild", "hard")}
        add("benchmarks", "real_dir", benchmark_members)
        selected_names = _source_names(selected)
        selected_links = {name: "symlink_file" for name in selected_names}
        for recipe in ("clean", "mild", "hard"):
            benchmark = f"benchmarks/{recipe}_pilot"
            add(
                benchmark,
                "real_dir",
                {"gt": "real_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
            )
            gt_targets = {
                f"{source.image_id}.png": source.path for source in selected
            }
            lq_targets = {
                f"{source.image_id}.png": str(
                    (
                        config.output_root
                        / recipe
                        / "val"
                        / "lq"
                        / f"{source.image_id}.png"
                    ).resolve(strict=False)
                )
                for source in selected
            }
            add(
                f"{benchmark}/gt",
                "real_dir",
                dict(selected_links),
                member_targets=gt_targets,
            )
            add(
                f"{benchmark}/lq",
                "real_dir",
                dict(selected_links),
                member_targets=lq_targets,
            )

    manifest_members: dict[str, str] = {}
    for name in _expected_manifest_names(config.targets):
        manifest_members[name] = "regular_file"
        manifest_members[f"{name}.meta.json"] = "regular_file"
    add("manifests", "real_dir", manifest_members)
    add(".", "real_dir", root_members)
    return specs


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _matches_kind(path: Path, kind: str, symlink_target: str | None = None) -> bool:
    if kind == "regular_file":
        return path.is_file() and not path.is_symlink()
    if kind == "symlink_file":
        return path.is_symlink() and path.resolve(strict=False).is_file()
    if kind == "real_dir":
        return path.is_dir() and not path.is_symlink()
    if kind == "symlink_dir":
        if not path.is_symlink() or not path.resolve(strict=False).is_dir():
            return False
        return symlink_target is None or path.resolve(strict=False) == Path(symlink_target)
    raise RecoveryDataError(f"Internal error: unknown expected path kind '{kind}'")


def _preview_names(names: Iterable[str]) -> str:
    ordered = sorted(names)
    preview = ", ".join(ordered[:5])
    return preview + (f" (+{len(ordered) - 5} more)" if len(ordered) > 5 else "")


def _validate_layout(
    output_root: Path,
    specs: dict[Path, DirectorySpec],
    allow_missing: bool,
    missing_exceptions: set[Path] | None = None,
) -> None:
    """Reject extras and wrong topology without removing or replacing anything."""
    exceptions = missing_exceptions or set()
    root = output_root.resolve(strict=False)
    for relative, spec in sorted(specs.items(), key=lambda item: len(item[0].parts)):
        directory = root if relative == Path(".") else root / relative
        if not _present(directory):
            if allow_missing or relative in exceptions:
                continue
            raise RecoveryDataError(f"Managed directory is missing: '{directory}'")
        if not _matches_kind(directory, spec.kind, spec.symlink_target):
            expected = spec.kind.replace("_", " ")
            raise RecoveryDataError(f"Expected {expected} at managed path '{directory}'")
        try:
            actual_names = {entry.name for entry in directory.iterdir()}
        except OSError as error:
            raise RecoveryDataError(f"Cannot inspect managed directory '{directory}': {error}") from error
        expected_names = set(spec.members)
        extras = actual_names - expected_names
        if extras:
            raise RecoveryDataError(
                f"Unexpected member(s) in managed directory '{directory}': {_preview_names(extras)}. "
                "No files were removed; move unexpected data out manually."
            )
        missing = expected_names - actual_names
        if missing and not allow_missing:
            allowed_missing = {
                name for name in missing if (relative / name) in exceptions
            }
            missing -= allowed_missing
            if missing:
                raise RecoveryDataError(
                    f"Managed directory '{directory}' is incomplete; missing: {_preview_names(missing)}"
                )
        for name in actual_names & expected_names:
            member = directory / name
            if not _matches_kind(member, spec.members[name]):
                expected = spec.members[name].replace("_", " ")
                raise RecoveryDataError(
                    f"Expected {expected} for managed member '{member}', found incompatible topology"
                )
            if spec.member_symlink_targets and name in spec.member_symlink_targets:
                expected_target = Path(spec.member_symlink_targets[name]).resolve(strict=False)
                if member.resolve(strict=False) != expected_target:
                    raise RecoveryDataError(
                        f"Managed symlink has unexpected target: '{member}' -> "
                        f"'{member.resolve(strict=False)}'; expected '{expected_target}'"
                    )


def _contract_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                differences.append(f"{path}: unexpected={actual[key]!r}")
            elif key not in actual:
                differences.append(f"{path}: missing (expected {expected[key]!r})")
            else:
                differences.extend(_contract_differences(expected[key], actual[key], path))
        return differences
    if expected != actual:
        return [f"{prefix}: expected={expected!r}, existing={actual!r}"]
    return []


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryDataError(f"Cannot read {description} '{path}': {error}") from error
    if not isinstance(value, dict):
        raise RecoveryDataError(f"Expected a JSON object in {description} '{path}'")
    return value


def _validate_legacy_metadata(output_root: Path, expected: dict[str, Any]) -> None:
    """Safely adopt output produced by generator v1.0 before contracts existed."""
    metadata_paths = sorted((output_root / "manifests").glob("*.jsonl.meta.json"))
    if not metadata_paths:
        raise RecoveryDataError(
            f"Existing output root '{output_root}' has no {CONTRACT_FILENAME} and no compatible "
            "manifest metadata. Refusing to infer its invocation."
        )
    for path in metadata_paths:
        metadata = _read_json_object(path, "legacy manifest metadata")
        audit = metadata.get("source_split_audit", {})
        environment = metadata.get("environment", {})
        checks = {
            "global_seed": (expected["seed"], metadata.get("global_seed")),
            "pilot_seed": (expected["pilot"]["seed"], metadata.get("pilot_seed")),
            "schema_version": (SCHEMA_VERSION, metadata.get("schema_version")),
            "generator_version": (
                expected["generator_version"],
                environment.get("generator_version"),
            ),
            "train_count": (
                expected["source_split"]["train"]["count"],
                audit.get("train_count"),
            ),
            "train_root": (
                expected["source_split"]["train"]["root"],
                audit.get("train_root"),
            ),
            "val_count": (
                expected["source_split"]["val"]["count"],
                audit.get("val_count"),
            ),
            "val_root": (
                expected["source_split"]["val"]["root"],
                audit.get("val_root"),
            ),
        }
        mismatches = [
            f"{name}: expected={wanted!r}, existing={found!r}"
            for name, (wanted, found) in checks.items()
            if wanted != found
        ]
        if mismatches:
            raise RecoveryDataError(
                f"Legacy invocation metadata is incompatible in '{path}': " + "; ".join(mismatches)
            )
    selection_path = output_root / "manifests" / "pilot_selection.jsonl"
    if selection_path.exists():
        records = _load_manifest(selection_path)
        if len(records) != expected["pilot"]["size"]:
            raise RecoveryDataError(
                "Legacy pilot size is incompatible: "
                f"expected={expected['pilot']['size']}, existing={len(records)}"
            )


def _validate_existing_contract(output_root: Path, expected: dict[str, Any]) -> None:
    contract_path = output_root / CONTRACT_FILENAME
    if contract_path.is_symlink():
        raise RecoveryDataError(f"Build contract must be a regular file, not a symlink: '{contract_path}'")
    if contract_path.exists():
        actual = _read_json_object(contract_path, "build contract")
        differences = _contract_differences(expected, actual)
        if differences:
            raise RecoveryDataError(
                "Immutable recovery data build contract mismatch; use a new output root. "
                + "; ".join(differences[:8])
            )
        return
    if output_root.exists() and any(output_root.iterdir()):
        _validate_legacy_metadata(output_root, expected)


def _prepare_contract(
    config: BuildConfig,
    train_sources: Sequence[SourceImage],
    val_sources: Sequence[SourceImage],
    allow_missing: bool,
    write: bool,
) -> dict[str, Any]:
    selected = select_pilot(val_sources, config.pilot_size, config.pilot_seed)
    expected = _build_contract(config, train_sources, val_sources)
    _validate_existing_contract(config.output_root, expected)
    specs = _expected_layout(config, train_sources, val_sources, selected)
    contract_missing = not (config.output_root / CONTRACT_FILENAME).exists()
    missing_exceptions = {Path(CONTRACT_FILENAME)} if write and contract_missing else set()
    _validate_layout(
        config.output_root,
        specs,
        allow_missing=allow_missing,
        missing_exceptions=missing_exceptions,
    )
    if write:
        _atomic_write_if_changed(config.output_root / CONTRACT_FILENAME, _json_bytes(expected))
    return expected


def _build_pilot_benchmarks(
    selected: Sequence[SourceImage],
    records_by_recipe: dict[str, Sequence[dict[str, Any]]],
    config: BuildConfig,
    audit: dict[str, Any],
) -> None:
    selection_records = [
        {
            "id": source.image_id,
            "pilot_seed": config.pilot_seed,
            "rank_method": "sha256(seed_NUL_source_relative_path)",
            "schema_version": SCHEMA_VERSION,
            "source_gt_path": source.path,
            "source_relative_path": source.relative_path,
            "source_sha256": source.file_sha256,
            "source_pixel_sha256": source.pixel_sha256,
        }
        for source in selected
    ]
    _write_manifest(
        config.output_root / "manifests" / "pilot_selection.jsonl",
        selection_records,
        config,
        audit,
    )
    for recipe in ("clean", "mild", "hard"):
        benchmark_name = f"{recipe}_pilot"
        benchmark_root = config.output_root / "benchmarks" / benchmark_name
        gt_root, lq_root = benchmark_root / "gt", benchmark_root / "lq"
        gt_root.mkdir(parents=True, exist_ok=True)
        lq_root.mkdir(parents=True, exist_ok=True)
        full_by_id = {record["id"]: record for record in records_by_recipe[recipe]}
        benchmark_records = []
        for source in selected:
            gt_link = gt_root / f"{source.image_id}.png"
            lq_link = lq_root / f"{source.image_id}.png"
            _ensure_symlink(gt_link, Path(source.path), config.repair)
            _ensure_symlink(
                lq_link,
                config.output_root / recipe / "val" / "lq" / f"{source.image_id}.png",
                config.repair,
            )
            benchmark_records.append(
                _record_for_symlinked_sample(
                    full_by_id[source.image_id],
                    benchmark_name,
                    source.image_id,
                    gt_link,
                    lq_link,
                )
            )
        _write_manifest(
            config.output_root / "manifests" / f"{benchmark_name}.jsonl",
            benchmark_records,
            config,
            audit,
        )
        _write_meta_info(benchmark_root, benchmark_records)


def _validate_config(config: BuildConfig) -> None:
    unknown = sorted(set(config.targets) - set(TARGETS))
    if unknown:
        raise RecoveryDataError(f"Unknown build target(s): {', '.join(unknown)}")
    if config.workers < 1:
        raise RecoveryDataError("--workers must be at least 1")
    if config.seed < 0 or config.pilot_seed < 0:
        raise RecoveryDataError("Seeds must be non-negative integers")
    if len(config.targets) != len(set(config.targets)):
        raise RecoveryDataError("Build targets must not be repeated")


def _print_plan(config: BuildConfig, train_count: int, val_count: int) -> None:
    targets = set(config.targets)
    clean_train = train_count if targets & {"clean", "mild-mixed"} else 0
    clean_val = val_count if targets & {"clean", "benchmarks"} else 0
    mild_train = train_count if "mild-mixed" in targets else 0
    benchmark_lq = 2 * val_count if "benchmarks" in targets else 0
    print("Recovery data plan")
    print(f"  output root: {config.output_root.resolve()}")
    print(f"  targets: {', '.join(config.targets)}")
    print(f"  clean generated LQ: {clean_train + clean_val}")
    print(f"  mild mixed generated LQ: {mild_train}")
    print(f"  mild/hard validation generated LQ: {benchmark_lq}")
    if "mild-mixed" in targets:
        print(f"  Stage B pairs: {2 * train_count} (exactly 50% clean, 50% mild)")
    if "benchmarks" in targets:
        print(f"  pilot benchmark pairs: {3 * config.pilot_size} ({config.pilot_size} per bucket)")


def build_recovery_data(config: BuildConfig) -> None:
    _validate_config(config)
    train_sources, val_sources, audit = audit_source_split(config)
    _print_plan(config, len(train_sources), len(val_sources))
    _prepare_contract(
        config,
        train_sources,
        val_sources,
        allow_missing=True,
        write=not config.dry_run,
    )
    if config.dry_run:
        print("Dry run complete: source split and immutable contract passed; no files were written.")
        return

    targets = set(config.targets)
    clean_train_records: list[dict[str, Any]] = []
    clean_val_records: list[dict[str, Any]] = []
    if targets & {"clean", "mild-mixed"}:
        clean_train_records = _build_clean_split(
            "train", train_sources, config.train_gt_root, config, audit
        )
    if targets & {"clean", "benchmarks"}:
        clean_val_records = _build_clean_split("val", val_sources, config.val_gt_root, config, audit)
    if "mild-mixed" in targets:
        _build_mild_mixed_train(train_sources, clean_train_records, config, audit)
    if "benchmarks" in targets:
        mild_val_records = _build_validation_recipe(
            "mild", val_sources, config.val_gt_root, config, audit
        )
        hard_val_records = _build_validation_recipe(
            "hard", val_sources, config.val_gt_root, config, audit
        )
        selected = select_pilot(val_sources, config.pilot_size, config.pilot_seed)
        _build_pilot_benchmarks(
            selected,
            {
                "clean": clean_val_records,
                "mild": mild_val_records,
                "hard": hard_val_records,
            },
            config,
            audit,
        )
    print(f"Build complete: {config.output_root.resolve()}")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise RecoveryDataError(f"Cannot read manifest '{path}': {error}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RecoveryDataError(f"Blank line in manifest '{path}' at line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecoveryDataError(
                f"Invalid JSON in manifest '{path}' at line {line_number}: {error}"
            ) from error
        records.append(record)
    return records


def _verify_record(record: dict[str, Any], recompute: bool) -> None:
    np, cv2, _, _, _ = _dependencies()
    required = {
        "bucket",
        "gt_path",
        "gt_pixel_sha256",
        "gt_sha256",
        "gt_size",
        "id",
        "lq_path",
        "lq_pixel_sha256",
        "lq_sha256",
        "lq_size",
        "parameters",
        "recipe",
        "sample_id",
        "scale",
        "schema_version",
        "seed",
        "source_gt_path",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise RecoveryDataError(f"Manifest record '{record.get('sample_id', '?')}' lacks: {', '.join(missing)}")
    gt_path = Path(record["gt_path"])
    source_path = Path(record["source_gt_path"])
    lq_path = Path(record["lq_path"])
    if not gt_path.exists() or not source_path.exists() or not lq_path.exists():
        raise RecoveryDataError(
            f"Missing artifact for sample '{record['sample_id']}': "
            f"gt={gt_path.exists()} source={source_path.exists()} lq={lq_path.exists()}"
        )
    try:
        if not os.path.samefile(gt_path, source_path):
            raise RecoveryDataError(
                f"GT path is not the recorded source for sample '{record['sample_id']}': '{gt_path}'"
            )
    except OSError as error:
        raise RecoveryDataError(f"Cannot compare GT link for sample '{record['sample_id']}': {error}") from error
    if _sha256_file(source_path) != record["gt_sha256"]:
        raise RecoveryDataError(f"GT hash mismatch for sample '{record['sample_id']}'")
    gt, _ = _read_image(source_path)
    actual_gt_size = [int(gt.shape[1]), int(gt.shape[0])]
    if actual_gt_size != record["gt_size"]:
        raise RecoveryDataError(
            f"GT size mismatch for sample '{record['sample_id']}': "
            f"{actual_gt_size} != {record['gt_size']}"
        )
    if _sha256_bytes(np.ascontiguousarray(gt).tobytes()) != record["gt_pixel_sha256"]:
        raise RecoveryDataError(f"GT pixel hash mismatch for sample '{record['sample_id']}'")
    if _sha256_file(lq_path) != record["lq_sha256"]:
        raise RecoveryDataError(f"LQ file hash mismatch for sample '{record['sample_id']}'")
    lq_bytes = lq_path.read_bytes()
    lq = cv2.imdecode(np.frombuffer(lq_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if lq is None or lq.ndim != 3 or lq.shape[2] != 3 or lq.dtype != np.uint8:
        raise RecoveryDataError(f"Invalid LQ image for sample '{record['sample_id']}'")
    actual_size = [int(lq.shape[1]), int(lq.shape[0])]
    if actual_size != record["lq_size"]:
        raise RecoveryDataError(
            f"LQ size mismatch for sample '{record['sample_id']}': {actual_size} != {record['lq_size']}"
        )
    if _sha256_bytes(np.ascontiguousarray(lq).tobytes()) != record["lq_pixel_sha256"]:
        raise RecoveryDataError(f"LQ pixel hash mismatch for sample '{record['sample_id']}'")
    if recompute:
        expected_lq, expected_parameters = _generate_lq(gt, record["recipe"], int(record["seed"]))
        if expected_parameters != record["parameters"]:
            raise RecoveryDataError(f"Parameter mismatch for sample '{record['sample_id']}'")
        if not np.array_equal(lq, expected_lq):
            raise RecoveryDataError(f"Regenerated pixels differ for sample '{record['sample_id']}'")
        if _sha256_bytes(_encode_png(expected_lq)) != record["lq_sha256"]:
            raise RecoveryDataError(f"Regenerated PNG differs for sample '{record['sample_id']}'")


def _verify_record_job(payload: tuple[dict[str, Any], bool]) -> None:
    record, recompute = payload
    _verify_record(record, recompute)


def _verify_contract_structure(output_root: Path) -> dict[str, Any]:
    contract_path = output_root / CONTRACT_FILENAME
    if not contract_path.is_file() or contract_path.is_symlink():
        raise RecoveryDataError(
            f"Missing immutable regular-file build contract '{contract_path}'. "
            "For a completed legacy v1.0 build, rerun the same idempotent build command first."
        )
    contract = _read_json_object(contract_path, "build contract")
    if contract.get("contract_schema_version") != CONTRACT_SCHEMA_VERSION:
        raise RecoveryDataError(
            f"Unsupported build contract schema in '{contract_path}': "
            f"{contract.get('contract_schema_version')!r}"
        )
    if contract.get("output_root") != str(output_root.resolve()):
        raise RecoveryDataError(
            f"Build contract output root mismatch: expected '{output_root.resolve()}', "
            f"recorded {contract.get('output_root')!r}"
        )
    targets = contract.get("targets")
    if not isinstance(targets, list) or tuple(targets) != _canonical_targets(targets):
        raise RecoveryDataError(f"Invalid target list in build contract '{contract_path}'")
    expected_manifests = contract.get("expected_manifests")
    if expected_manifests != list(_expected_manifest_names(targets)):
        raise RecoveryDataError(f"Invalid expected manifest list in build contract '{contract_path}'")

    selected = set(targets)
    root_members: dict[str, str] = {
        CONTRACT_FILENAME: "regular_file",
        "manifests": "real_dir",
    }
    specs: dict[Path, DirectorySpec] = {}

    def add(relative: str, members: dict[str, str]) -> None:
        specs[Path(relative)] = DirectorySpec("real_dir", members)

    clean_splits: dict[str, str] = {}
    if selected & {"clean", "mild-mixed"}:
        clean_splits["train"] = "real_dir"
        add(
            "clean/train",
            {"gt": "symlink_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
        )
    if selected & {"clean", "benchmarks"}:
        clean_splits["val"] = "real_dir"
        add(
            "clean/val",
            {"gt": "symlink_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
        )
    if clean_splits:
        root_members["clean"] = "real_dir"
        add("clean", clean_splits)
    if "mild-mixed" in selected:
        root_members["mild_mixed"] = "real_dir"
        add("mild_mixed", {"train": "real_dir"})
        add(
            "mild_mixed/train",
            {"gt": "real_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
        )
    if "benchmarks" in selected:
        for recipe in ("mild", "hard"):
            root_members[recipe] = "real_dir"
            add(recipe, {"val": "real_dir"})
            add(
                f"{recipe}/val",
                {"gt": "symlink_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
            )
        root_members["benchmarks"] = "real_dir"
        add(
            "benchmarks",
            {f"{recipe}_pilot": "real_dir" for recipe in ("clean", "mild", "hard")},
        )
        for recipe in ("clean", "mild", "hard"):
            add(
                f"benchmarks/{recipe}_pilot",
                {"gt": "real_dir", "lq": "real_dir", "meta_info.txt": "regular_file"},
            )
    manifest_members: dict[str, str] = {}
    for name in expected_manifests:
        manifest_members[name] = "regular_file"
        manifest_members[f"{name}.meta.json"] = "regular_file"
    add("manifests", manifest_members)
    specs[Path(".")] = DirectorySpec("real_dir", root_members)
    _validate_layout(output_root, specs, allow_missing=False)
    return contract


def _manifest_data_paths(manifest_name: str) -> tuple[Path, Path, str]:
    mappings = {
        "clean_train.jsonl": (Path("clean/train/gt"), Path("clean/train/lq"), "full"),
        "clean_val.jsonl": (Path("clean/val/gt"), Path("clean/val/lq"), "full"),
        "mild_val.jsonl": (Path("mild/val/gt"), Path("mild/val/lq"), "full"),
        "hard_val.jsonl": (Path("hard/val/gt"), Path("hard/val/lq"), "full"),
        "mild_mixed_train.jsonl": (
            Path("mild_mixed/train/gt"),
            Path("mild_mixed/train/lq"),
            "mixed",
        ),
        "clean_pilot.jsonl": (
            Path("benchmarks/clean_pilot/gt"),
            Path("benchmarks/clean_pilot/lq"),
            "pilot",
        ),
        "mild_pilot.jsonl": (
            Path("benchmarks/mild_pilot/gt"),
            Path("benchmarks/mild_pilot/lq"),
            "pilot",
        ),
        "hard_pilot.jsonl": (
            Path("benchmarks/hard_pilot/gt"),
            Path("benchmarks/hard_pilot/lq"),
            "pilot",
        ),
    }
    try:
        return mappings[manifest_name]
    except KeyError as error:
        raise RecoveryDataError(f"Unsupported recovery data manifest '{manifest_name}'") from error


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _verify_manifest_membership(
    output_root: Path,
    manifest_path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    gt_relative, lq_relative, topology = _manifest_data_paths(manifest_path.name)
    gt_root = _lexical_absolute(output_root / gt_relative)
    lq_root = _lexical_absolute(output_root / lq_relative)
    gt_members: dict[str, str] = {}
    lq_members: dict[str, str] = {}
    source_parents: set[Path] = set()
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise RecoveryDataError(f"Invalid sample_id in '{manifest_path}'")
        expected_name = f"{sample_id}.png"
        gt_path = _lexical_absolute(Path(record.get("gt_path", "")))
        lq_path = _lexical_absolute(Path(record.get("lq_path", "")))
        if gt_path.parent != gt_root or gt_path.name != expected_name:
            raise RecoveryDataError(
                f"GT path escapes manifest directory for sample '{sample_id}': '{gt_path}'"
            )
        if lq_path.parent != lq_root or lq_path.name != expected_name:
            raise RecoveryDataError(
                f"LQ path escapes manifest directory for sample '{sample_id}': '{lq_path}'"
            )
        if expected_name in gt_members or expected_name in lq_members:
            raise RecoveryDataError(f"Duplicate sample path '{expected_name}' in '{manifest_path}'")
        source_parents.add(Path(record.get("source_gt_path", "")).resolve().parent)
        if topology == "full":
            gt_members[expected_name] = "regular_file"
            lq_members[expected_name] = "regular_file"
        elif topology == "mixed":
            gt_members[expected_name] = "symlink_file"
            lq_members[expected_name] = (
                "symlink_file" if sample_id.endswith("_clean") else "regular_file"
            )
        else:
            gt_members[expected_name] = "symlink_file"
            lq_members[expected_name] = "symlink_file"

    if topology == "full":
        if len(source_parents) != 1:
            raise RecoveryDataError(f"Full manifest '{manifest_path}' has multiple source GT roots")
        gt_spec = DirectorySpec("symlink_dir", gt_members, str(next(iter(source_parents))))
    else:
        gt_spec = DirectorySpec("real_dir", gt_members)
    specs = {
        gt_relative: gt_spec,
        lq_relative: DirectorySpec("real_dir", lq_members),
    }
    _validate_layout(output_root, specs, allow_missing=False)
    meta_info_path = output_root / gt_relative.parent / "meta_info.txt"
    try:
        actual_meta_info = meta_info_path.read_bytes()
    except OSError as error:
        raise RecoveryDataError(f"Cannot read BasicSR meta-info '{meta_info_path}': {error}") from error
    expected_meta_info = _meta_info_bytes(records)
    if actual_meta_info != expected_meta_info:
        raise RecoveryDataError(
            f"BasicSR meta-info does not exactly match manifest '{manifest_path.name}': "
            f"'{meta_info_path}'"
        )

    if topology == "mixed":
        for record in records:
            if not record["sample_id"].endswith("_clean"):
                continue
            expected_target = (
                output_root / "clean" / "train" / "lq" / f"{record['id']}.png"
            ).resolve(strict=True)
            if Path(record["lq_path"]).resolve(strict=True) != expected_target:
                raise RecoveryDataError(
                    f"Mixed clean LQ symlink has wrong target for '{record['sample_id']}'"
                )
    elif topology == "pilot":
        for record in records:
            expected_target = (
                output_root
                / record["recipe"]
                / "val"
                / "lq"
                / f"{record['id']}.png"
            ).resolve(strict=True)
            if Path(record["lq_path"]).resolve(strict=True) != expected_target:
                raise RecoveryDataError(
                    f"Pilot LQ symlink has wrong target for '{record['sample_id']}'"
                )


def verify_manifests(
    output_root: Path,
    manifests: Sequence[Path] | None = None,
    recompute: bool = False,
    workers: int = 1,
) -> None:
    if workers < 1:
        raise RecoveryDataError("--workers must be at least 1")
    output_root = output_root.resolve()
    contract = _verify_contract_structure(output_root)
    allowed_manifests = set(contract["expected_manifests"])
    if manifests:
        paths = [path.resolve() for path in manifests]
    else:
        manifest_root = output_root / "manifests"
        paths = sorted(manifest_root.glob("*.jsonl"))
    if not paths:
        raise RecoveryDataError(f"No JSONL manifests found under '{output_root / 'manifests'}'")
    for path in paths:
        if path.parent != output_root / "manifests" or path.name not in allowed_manifests:
            raise RecoveryDataError(f"Manifest is outside the immutable build contract: '{path}'")
    pilot_ids: dict[str, list[str]] = {}
    for path in paths:
        records = _load_manifest(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        try:
            metadata = json.loads(meta_path.read_text(encoding="ascii"))
        except (OSError, json.JSONDecodeError) as error:
            raise RecoveryDataError(f"Cannot read manifest metadata '{meta_path}': {error}") from error
        actual_manifest_hash = _sha256_file(path)
        if metadata.get("manifest_sha256") != actual_manifest_hash:
            raise RecoveryDataError(f"Manifest hash mismatch for '{path}'")
        if metadata.get("record_count") != len(records):
            raise RecoveryDataError(f"Manifest count mismatch for '{path}'")
        if path.name == "pilot_selection.jsonl":
            if len(records) != contract["pilot"]["size"]:
                raise RecoveryDataError(
                    f"Pilot selection count mismatch: {len(records)} != {contract['pilot']['size']}"
                )
            for record in records:
                source_path = Path(record["source_gt_path"])
                if _sha256_file(source_path) != record.get("source_sha256"):
                    raise RecoveryDataError(f"Pilot source hash mismatch for '{record['id']}'")
                source_image, _ = _read_image(source_path)
                np, _, _, _, _ = _dependencies()
                source_pixel_hash = _sha256_bytes(np.ascontiguousarray(source_image).tobytes())
                if source_pixel_hash != record.get("source_pixel_sha256"):
                    raise RecoveryDataError(
                        f"Pilot source pixel hash mismatch for '{record['id']}'"
                    )
            pilot_ids[path.name] = [record["id"] for record in records]
            print(f"verified {path.name}: {len(records)} selection records")
            continue
        _verify_manifest_membership(output_root, path, records)
        if path.name.endswith("_pilot.jsonl"):
            pilot_ids[path.name] = [record["id"] for record in records]
        payloads = ((record, recompute) for record in records)
        if workers == 1:
            for payload in payloads:
                _verify_record_job(payload)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for _ in executor.map(_verify_record_job, payloads, chunksize=8):
                    pass
        print(f"verified {path.name}: {len(records)} records (recompute={recompute})")
    if not manifests and "benchmarks" in contract["targets"]:
        selection = pilot_ids.get("pilot_selection.jsonl")
        for name in ("clean_pilot.jsonl", "mild_pilot.jsonl", "hard_pilot.jsonl"):
            if selection != pilot_ids.get(name):
                raise RecoveryDataError(f"Pilot IDs/order differ between selection and '{name}'")
    print(f"Verification complete: {len(paths)} manifest(s)")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify deterministic HAT face SR recovery data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="audit sources and build selected datasets")
    build.add_argument("--train-gt-root", type=Path, default=DEFAULT_SOURCE_ROOT / "train" / "gt")
    build.add_argument("--val-gt-root", type=Path, default=DEFAULT_SOURCE_ROOT / "val" / "gt")
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    build.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        dest="targets",
        help="build target; repeat for multiple targets (default: all)",
    )
    build.add_argument("--seed", type=_nonnegative_int, default=DEFAULT_SEED)
    build.add_argument("--pilot-seed", type=_nonnegative_int, default=DEFAULT_PILOT_SEED)
    build.add_argument("--pilot-size", type=_positive_int, default=DEFAULT_PILOT_SIZE)
    build.add_argument("--workers", type=_positive_int, default=max(1, min(8, os.cpu_count() or 1)))
    build.add_argument("--expected-train-count", type=_positive_int, default=65000)
    build.add_argument("--expected-val-count", type=_positive_int, default=5000)
    build.add_argument("--repair", action="store_true", help="replace mismatched generated files/symlinks")
    build.add_argument("--dry-run", action="store_true", help="audit and print the plan without writing files")

    verify = subparsers.add_parser("verify", help="verify manifests, links, hashes, and optionally recipes")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify.add_argument("--manifest", action="append", type=Path, dest="manifests")
    verify.add_argument("--recompute", action="store_true", help="regenerate every LQ in memory")
    verify.add_argument("--workers", type=_positive_int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            config = BuildConfig(
                train_gt_root=args.train_gt_root,
                val_gt_root=args.val_gt_root,
                output_root=args.output_root,
                targets=tuple(args.targets or TARGETS),
                seed=args.seed,
                pilot_seed=args.pilot_seed,
                pilot_size=args.pilot_size,
                workers=args.workers,
                repair=args.repair,
                dry_run=args.dry_run,
                expected_train_count=args.expected_train_count,
                expected_val_count=args.expected_val_count,
            )
            build_recovery_data(config)
        else:
            verify_manifests(
                output_root=args.output_root,
                manifests=args.manifests,
                recompute=args.recompute,
                workers=args.workers,
            )
    except RecoveryDataError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
