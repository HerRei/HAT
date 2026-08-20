#!/usr/bin/env python3
"""Prepare and explicitly run the immutable face-SR pilot inference matrix.

The ``static-check``, ``check``, ``prepare``, ``status``, and ``eval-command``
subcommands never invoke HAT. Only ``run --confirm-gpu-run`` starts ``hat/test.py``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError as error:  # pragma: no cover - exercised only outside the project environment
    raise SystemExit(
        "PyYAML is required. Run with /home/hermes/hat-face-training/hat-face/bin/python"
    ) from error


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "recovery" / "inference" / "pilot_matrix.json"
ARTIFACT_ROOT = REPO_ROOT / "results" / "recovery_pilot_matrix"
ORCHESTRATION_ROOT = ARTIFACT_ROOT / "orchestration"
OUTPUT_ROOT = ARTIFACT_ROOT / "outputs"
EVALUATION_ROOT = ARTIFACT_ROOT / "evaluations"
DEFAULT_PYTHON = Path("/home/hermes/hat-face-training/hat-face/bin/python")
IMAGE_SUFFIXES = {".png"}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDERS = {
    "__EXPERIMENT_NAME__",
    "__CHECKPOINT__",
    "__RESULTS_ROOT__",
    "__MODEL_ID__",
}
FORBIDDEN_OPTION_KEYS = {"auto_resume", "resume_state"}
TREE_DIGEST_ALGORITHM = "sha256(sorted(relative_posix_path_NUL_size_NUL_sha256_LF))"
GPU_PROBE_CODE = r"""
import json
import torch

devices = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    devices.append({
        "index": index,
        "name": props.name,
        "total_memory": int(props.total_memory),
        "major": int(props.major),
        "minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
        "warp_size": int(props.warp_size),
        "gcn_arch_name": getattr(props, "gcnArchName", None),
    })
print(json.dumps({
    "device_count": len(devices),
    "devices": devices,
    "torch_cuda_version": torch.version.cuda,
    "torch_hip_version": torch.version.hip,
}, sort_keys=True, separators=(",", ":")))
""".strip()


class PilotMatrixError(RuntimeError):
    """Raised when reproducibility or safety preflight fails."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PilotMatrixError(f"Cannot hash '{path}': {error}") from error
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotMatrixError(f"Cannot read JSON '{path}': {error}") from error


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="ascii"))
    except (OSError, yaml.YAMLError) as error:
        raise PilotMatrixError(f"Cannot read YAML '{path}': {error}") from error
    if not isinstance(value, dict):
        raise PilotMatrixError(f"Expected a YAML mapping in '{path}'")
    return value


def _capture_command(command: Sequence[str], label: str) -> bytes:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise PilotMatrixError(f"Cannot run {label}: {error}") from error
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise PilotMatrixError(
            f"{label} failed with exit code {completed.returncode}: {stderr}"
        )
    return completed.stdout


def _collect_provenance(final_directory: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Capture code, environment, package, and runtime-visible GPU identity."""
    git_head_bytes = _capture_command(("git", "rev-parse", "HEAD"), "git HEAD")
    git_head = git_head_bytes.decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise PilotMatrixError(f"Unexpected git HEAD value '{git_head}'")
    contents = {
        "git_status_porcelain_v1_z.bin": _capture_command(
            ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
            "git status",
        ),
        "git_diff_head_binary.patch": _capture_command(
            ("git", "diff", "--binary", "HEAD", "--"), "binary git diff"
        ),
        "pip_freeze.txt": _capture_command(
            (sys.executable, "-m", "pip", "freeze", "--all"), "pip freeze"
        ),
        "torch_gpu_visibility.json": _capture_command(
            (sys.executable, "-c", GPU_PROBE_CODE), "Torch GPU visibility probe"
        ),
    }
    try:
        gpu_probe = json.loads(contents["torch_gpu_visibility.json"])
    except json.JSONDecodeError as error:
        raise PilotMatrixError(f"Torch GPU visibility probe returned invalid JSON: {error}") from error
    if not isinstance(gpu_probe, dict) or int(gpu_probe.get("device_count", 0)) < 1:
        raise PilotMatrixError("Torch runtime does not expose any GPU during provenance capture")

    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is not None:
        contents["rocm_smi_identity.json"] = _capture_command(
            (
                rocm_smi,
                "--showproductname",
                "--showdriverversion",
                "--showuniqueid",
                "--showvbios",
                "--json",
            ),
            "rocm-smi identity probe",
        )

    hat_entries = []
    for path in sorted((REPO_ROOT / "hat").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        hat_entries.append(
            {"path": relative, "sha256": _sha256_file(path), "size": path.stat().st_size}
        )
    if not hat_entries:
        raise PilotMatrixError("No hat/**/*.py sources found for provenance capture")
    hat_content = _json_bytes(hat_entries)
    contents["hat_python_hashes.json"] = hat_content

    try:
        import basicsr
        import cv2
        import torch
    except (ImportError, RuntimeError) as error:
        raise PilotMatrixError(f"Cannot import runtime packages for provenance: {error}") from error

    evidence_files = {
        name: {
            "path": str(final_directory / name),
            "sha256": _sha256_bytes(content),
            "size": len(content),
        }
        for name, content in sorted(contents.items())
    }
    snapshot = {
        "schema_version": 1,
        "git": {
            "head": git_head,
            "status_sha256": evidence_files["git_status_porcelain_v1_z.bin"]["sha256"],
            "binary_diff_sha256": evidence_files["git_diff_head_binary.patch"]["sha256"],
        },
        "source": {
            "hat_python_file_count": len(hat_entries),
            "hat_python_tree_sha256": _sha256_bytes(hat_content),
            "hat_python_files": hat_entries,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "template": {
                "path": str(_repo_path("options/test/recovery/pilot_template.yml")),
                "sha256": _sha256_file(
                    _repo_path("options/test/recovery/pilot_template.yml")
                ),
            },
            "matrix": {"path": str(MATRIX_PATH), "sha256": _sha256_file(MATRIX_PATH)},
        },
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch": str(torch.__version__),
            "torch_hip": str(torch.version.hip),
            "torch_cuda": str(torch.version.cuda),
            "basicsr": str(basicsr.__version__),
            "opencv": str(cv2.__version__),
            "visibility_environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                )
            },
            "gpu_probe": gpu_probe,
            "rocm_smi_path": rocm_smi,
            "rocm_smi_binary_sha256": _sha256_file(Path(rocm_smi)) if rocm_smi else None,
        },
        "pip_freeze": {
            "path": evidence_files["pip_freeze.txt"]["path"],
            "sha256": evidence_files["pip_freeze.txt"]["sha256"],
        },
        "evidence_files": evidence_files,
    }
    return snapshot, contents


def _verify_stored_provenance(snapshot: Mapping[str, Any]) -> None:
    evidence = snapshot.get("evidence_files")
    if not isinstance(evidence, Mapping) or not evidence:
        raise PilotMatrixError("Prepared provenance lacks evidence files")
    for name, entry in evidence.items():
        if not isinstance(entry, Mapping):
            raise PilotMatrixError(f"Invalid provenance evidence entry '{name}'")
        path = Path(str(entry.get("path", "")))
        if not path.is_file():
            raise PilotMatrixError(f"Provenance evidence is missing: '{path}'")
        if path.name != name or _sha256_file(path) != entry.get("sha256"):
            raise PilotMatrixError(f"Provenance evidence changed: '{path}'")
        if path.stat().st_size != entry.get("size"):
            raise PilotMatrixError(f"Provenance evidence size changed: '{path}'")


def _safe_id(value: str, label: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise PilotMatrixError(
            f"Invalid {label} '{value}'; use 1-48 lowercase letters, digits, '_' or '-', starting alphanumeric"
        )
    return value


def _repo_path(relative: str) -> Path:
    candidate = (REPO_ROOT / relative).resolve(strict=False)
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as error:
        raise PilotMatrixError(f"Repository path escapes the checkout: '{relative}'") from error
    return candidate


def _local_path(value: str, label: str) -> Path:
    """Resolve an explicit local path without accepting URLs or implicit downloads."""
    if "://" in value:
        raise PilotMatrixError(f"{label} must be a local path, not a URL: '{value}'")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return _repo_path(value)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def load_matrix() -> dict[str, Any]:
    matrix = _load_json(MATRIX_PATH)
    if not isinstance(matrix, dict) or matrix.get("schema_version") != 1:
        raise PilotMatrixError("pilot_matrix.json must use schema_version 1")
    _safe_id(str(matrix.get("matrix_id", "")), "matrix id")
    if matrix.get("expected_count_per_dataset") != 512:
        raise PilotMatrixError("The v1 pilot matrix must contain exactly 512 images per dataset")
    if matrix.get("checkpoint_param_key") != "params_ema":
        raise PilotMatrixError("The recovery matrix must evaluate checkpoint key 'params_ema'")
    data_root = Path(str(matrix.get("data_root", "")))
    if not data_root.is_absolute():
        raise PilotMatrixError("data_root must be absolute")
    _repo_path(str(matrix.get("template", "")))

    models = matrix.get("models")
    if not isinstance(models, list) or not models:
        raise PilotMatrixError("The matrix must declare at least one model")
    model_ids: list[str] = []
    checkpoints: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            raise PilotMatrixError("Every model entry must be a mapping")
        model_id = _safe_id(str(model.get("id", "")), "model id")
        checkpoint = str(model.get("checkpoint", ""))
        _repo_path(checkpoint)
        expected_hash = str(model.get("sha256", ""))
        if not SHA256.fullmatch(expected_hash):
            raise PilotMatrixError(f"Model '{model_id}' has an invalid SHA-256")
        model_ids.append(model_id)
        checkpoints.append(checkpoint)
    if len(model_ids) != len(set(model_ids)):
        raise PilotMatrixError("Model IDs must be unique")
    if len(checkpoints) != len(set(checkpoints)):
        raise PilotMatrixError("Checkpoint paths must be unique")
    if model_ids[0] != "base":
        raise PilotMatrixError("The first matrix model must be the 'base' comparison baseline")

    datasets = matrix.get("datasets")
    if not isinstance(datasets, list) or [item.get("id") for item in datasets] != ["clean", "mild", "hard"]:
        raise PilotMatrixError("Datasets must be ordered clean, mild, hard")
    dataset_names: list[str] = []
    for dataset in datasets:
        _safe_id(str(dataset.get("id", "")), "dataset id")
        name = str(dataset.get("name", ""))
        root = Path(str(dataset.get("root", "")))
        manifest = Path(str(dataset.get("manifest", "")))
        if not name or root.is_absolute() or manifest.is_absolute() or ".." in root.parts or ".." in manifest.parts:
            raise PilotMatrixError(f"Invalid dataset entry: {dataset!r}")
        dataset_names.append(name)
    if len(dataset_names) != len(set(dataset_names)):
        raise PilotMatrixError("BasicSR dataset names must be unique")
    return matrix


def validate_template(template: Mapping[str, Any], matrix: Mapping[str, Any]) -> None:
    if template.get("name") != "__EXPERIMENT_NAME__":
        raise PilotMatrixError("Template name placeholder is missing")
    if template.get("model_type") != "HATModel" or template.get("scale") != 4:
        raise PilotMatrixError("Template must use HATModel at scale 4")
    if template.get("num_gpu") != 1 or template.get("manual_seed") != 0:
        raise PilotMatrixError("Template must explicitly request one GPU and manual_seed 0")
    forbidden = sorted(FORBIDDEN_OPTION_KEYS.intersection(_walk_keys(template)))
    if forbidden:
        raise PilotMatrixError(f"Resume behavior is forbidden in pilot options: {', '.join(forbidden)}")

    expected_architecture = {
        "type": "HAT",
        "upscale": 4,
        "in_chans": 3,
        "img_size": 64,
        "window_size": 16,
        "compress_ratio": 24,
        "squeeze_factor": 24,
        "conv_scale": 0.01,
        "overlap_ratio": 0.5,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 144,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
    }
    if template.get("network_g") != expected_architecture:
        raise PilotMatrixError("Template network_g no longer exactly matches HAT-S SRx4")

    path_options = template.get("path")
    if not isinstance(path_options, Mapping):
        raise PilotMatrixError("Template path section is missing")
    expected_path = {
        "pretrain_network_g": "__CHECKPOINT__",
        "strict_load_g": True,
        "param_key_g": "params_ema",
        "results_root": "__RESULTS_ROOT__",
    }
    if dict(path_options) != expected_path:
        raise PilotMatrixError("Template path options must use strict params_ema loading and placeholders")

    val = template.get("val")
    if not isinstance(val, Mapping) or val.get("save_img") is not True or val.get("suffix") != "__MODEL_ID__":
        raise PilotMatrixError("Template must save PNG outputs with the model-ID suffix")
    metrics = val.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"psnr", "ssim"}:
        raise PilotMatrixError("Template must report PSNR and SSIM")
    for metric in metrics.values():
        if metric.get("crop_border") != 4 or metric.get("test_y_channel") is not True:
            raise PilotMatrixError("Template metrics must use Y-channel evaluation with crop_border 4")

    template_datasets = template.get("datasets")
    if not isinstance(template_datasets, Mapping) or len(template_datasets) != 3:
        raise PilotMatrixError("Template must contain exactly three test datasets")
    actual = []
    for dataset in template_datasets.values():
        if dataset.get("type") != "PairedImageDataset" or dataset.get("io_backend") != {"type": "disk"}:
            raise PilotMatrixError("Every pilot dataset must be a disk PairedImageDataset")
        actual.append(
            (
                dataset.get("name"),
                Path(str(dataset.get("dataroot_gt"))),
                Path(str(dataset.get("dataroot_lq"))),
            )
        )
    data_root = Path(str(matrix["data_root"]))
    expected = [
        (
            dataset["name"],
            data_root / dataset["root"] / "gt",
            data_root / dataset["root"] / "lq",
        )
        for dataset in matrix["datasets"]
    ]
    if actual != expected:
        raise PilotMatrixError("Template dataset roots do not match pilot_matrix.json")


def load_canonical() -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = load_matrix()
    template_path = _repo_path(str(matrix["template"]))
    template = _load_yaml(template_path)
    validate_template(template, matrix)
    return matrix, template


def _load_manifest(path: Path, expected_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        raw_lines = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise PilotMatrixError(f"Cannot read data manifest '{path}': {error}") from error
    if len(raw_lines) != expected_count or any(not line.strip() for line in raw_lines):
        raise PilotMatrixError(
            f"Manifest '{path}' must have exactly {expected_count} nonblank records, found {len(raw_lines)}"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PilotMatrixError(f"Invalid JSON in '{path}' line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise PilotMatrixError(f"Manifest '{path}' line {line_number} is not a mapping")
        records.append(record)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = _load_json(meta_path)
    manifest_hash = _sha256_file(path)
    if metadata.get("manifest_sha256") != manifest_hash:
        raise PilotMatrixError(f"Manifest metadata hash mismatch for '{path}'")
    if metadata.get("record_count") != expected_count or metadata.get("schema_version") != 1:
        raise PilotMatrixError(f"Manifest metadata count/schema mismatch for '{path}'")
    snapshot = {
        "path": str(path.resolve()),
        "sha256": manifest_hash,
        "metadata_path": str(meta_path.resolve()),
        "metadata_sha256": _sha256_file(meta_path),
        "record_count": expected_count,
    }
    return records, snapshot


def _directory_pngs(path: Path, expected_count: int) -> dict[str, Path]:
    if not path.is_dir():
        raise PilotMatrixError(f"Pilot image directory is missing: '{path}'")
    entries = sorted((item for item in path.iterdir() if not item.name.startswith(".")), key=lambda item: item.name)
    invalid = [item.name for item in entries if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES]
    if invalid:
        raise PilotMatrixError(f"Unexpected entries in '{path}': {', '.join(invalid[:5])}")
    if len(entries) != expected_count:
        raise PilotMatrixError(f"Expected {expected_count} PNG files in '{path}', found {len(entries)}")
    by_id = {item.stem: item for item in entries}
    if len(by_id) != len(entries):
        raise PilotMatrixError(f"Duplicate image IDs in '{path}'")
    return by_id


def preflight_data(
    matrix: Mapping[str, Any], selected_ids: set[str] | None = None
) -> dict[str, Any]:
    """Verify manifests, links, hashes, scale, and identical membership."""
    data_root = Path(str(matrix["data_root"]))
    expected_count = int(matrix["expected_count_per_dataset"])
    hash_cache: dict[Path, str] = {}

    def checked_hash(path: Path) -> str:
        resolved = path.resolve(strict=True)
        if resolved not in hash_cache:
            hash_cache[resolved] = _sha256_file(resolved)
        return hash_cache[resolved]

    reports: dict[str, Any] = {}
    common_ids: list[str] | None = None
    common_gt_hashes: dict[str, str] | None = None
    datasets = [
        dataset
        for dataset in matrix["datasets"]
        if selected_ids is None or str(dataset["id"]) in selected_ids
    ]
    if not datasets:
        raise PilotMatrixError("No datasets selected for data preflight")
    for dataset in datasets:
        dataset_id = str(dataset["id"])
        dataset_root = data_root / str(dataset["root"])
        gt_by_id = _directory_pngs(dataset_root / "gt", expected_count)
        lq_by_id = _directory_pngs(dataset_root / "lq", expected_count)
        manifest_path = data_root / str(dataset["manifest"])
        records, snapshot = _load_manifest(manifest_path, expected_count)
        record_by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            image_id = str(record.get("id", ""))
            if not image_id or image_id in record_by_id:
                raise PilotMatrixError(f"Missing or duplicate image ID '{image_id}' in '{manifest_path}'")
            record_by_id[image_id] = record
        ids = sorted(record_by_id)
        if ids != sorted(gt_by_id) or ids != sorted(lq_by_id):
            raise PilotMatrixError(f"GT/LQ/manifest membership differs for '{dataset_id}'")
        gt_hashes: dict[str, str] = {}
        for image_id in ids:
            record = record_by_id[image_id]
            gt_path, lq_path = gt_by_id[image_id], lq_by_id[image_id]
            required = {
                "bucket",
                "gt_path",
                "gt_sha256",
                "gt_size",
                "id",
                "lq_path",
                "lq_sha256",
                "lq_size",
                "sample_id",
                "scale",
                "source_gt_path",
            }
            missing = sorted(required - set(record))
            if missing:
                raise PilotMatrixError(f"Record '{image_id}' in '{manifest_path}' lacks {', '.join(missing)}")
            if record["bucket"] != f"{dataset_id}_pilot" or record["sample_id"] != image_id:
                raise PilotMatrixError(f"Record identity/bucket mismatch for '{dataset_id}/{image_id}'")
            if int(record["scale"]) != 4:
                raise PilotMatrixError(f"Scale mismatch for '{dataset_id}/{image_id}'")
            gt_size, lq_size = record["gt_size"], record["lq_size"]
            if (
                not isinstance(gt_size, list)
                or not isinstance(lq_size, list)
                or len(gt_size) != 2
                or len(lq_size) != 2
                or gt_size != [4 * int(lq_size[0]), 4 * int(lq_size[1])]
            ):
                raise PilotMatrixError(f"Invalid x4 dimensions for '{dataset_id}/{image_id}'")
            try:
                same_gt = os.path.samefile(gt_path, Path(str(record["gt_path"])))
                same_lq = os.path.samefile(lq_path, Path(str(record["lq_path"])))
                same_source = os.path.samefile(gt_path, Path(str(record["source_gt_path"])))
            except OSError as error:
                raise PilotMatrixError(f"Cannot verify links for '{dataset_id}/{image_id}': {error}") from error
            if not same_gt or not same_lq or not same_source:
                raise PilotMatrixError(f"Manifest paths do not resolve to pilot files for '{dataset_id}/{image_id}'")
            actual_gt_hash = checked_hash(gt_path)
            actual_lq_hash = checked_hash(lq_path)
            if actual_gt_hash != record["gt_sha256"] or actual_lq_hash != record["lq_sha256"]:
                raise PilotMatrixError(f"File hash mismatch for '{dataset_id}/{image_id}'")
            gt_hashes[image_id] = actual_gt_hash
        if common_ids is None:
            common_ids = ids
            common_gt_hashes = gt_hashes
        elif ids != common_ids or gt_hashes != common_gt_hashes:
            raise PilotMatrixError("Clean, mild, and hard pilots must use identical IDs and GT bytes")
        snapshot["dataset_name"] = dataset["name"]
        snapshot["ids_sha256"] = _sha256_bytes(("\n".join(ids) + "\n").encode("ascii"))
        reports[dataset_id] = {"snapshot": snapshot, "ids": ids}

    selection_path = data_root / "manifests" / "pilot_selection.jsonl"
    selection, selection_snapshot = _load_manifest(selection_path, expected_count)
    selection_by_id = {str(record.get("id", "")): record for record in selection}
    if sorted(selection_by_id) != common_ids or len(selection_by_id) != expected_count:
        raise PilotMatrixError("pilot_selection.jsonl membership differs from benchmark manifests")
    assert common_gt_hashes is not None
    for image_id, record in selection_by_id.items():
        if record.get("source_sha256") != common_gt_hashes[image_id]:
            raise PilotMatrixError(f"Pilot selection source hash mismatch for '{image_id}'")
    reports["pilot_selection"] = {"snapshot": selection_snapshot, "ids": list(common_ids or [])}
    return reports


def _state_signature(path: Path, param_key: str) -> tuple[dict[str, tuple[list[int], str]], int]:
    try:
        import torch
    except (ImportError, RuntimeError) as error:
        raise PilotMatrixError(
            "Checkpoint inspection requires the preserved project PyTorch environment"
        ) from error
    try:
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except Exception as error:
        raise PilotMatrixError(f"Cannot load checkpoint '{path}' safely on CPU: {error}") from error
    if not isinstance(payload, Mapping) or param_key not in payload:
        raise PilotMatrixError(f"Checkpoint '{path}' lacks top-level key '{param_key}'")
    state = payload[param_key]
    if not isinstance(state, Mapping) or not state:
        raise PilotMatrixError(f"Checkpoint '{path}' key '{param_key}' is not a nonempty state dict")
    signature: dict[str, tuple[list[int], str]] = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise PilotMatrixError(f"Checkpoint '{path}' contains a non-tensor parameter '{key}'")
        if tensor.device.type != "cpu":
            raise PilotMatrixError(f"Checkpoint '{path}' was not mapped entirely to CPU")
        signature[key] = ([int(size) for size in tensor.shape], str(tensor.dtype))
    tensor_count = len(signature)
    del state
    del payload
    gc.collect()
    return signature, tensor_count


def _canonical_model_specs(
    matrix: Mapping[str, Any], selected_ids: Sequence[str]
) -> list[dict[str, str]]:
    selected = set(selected_ids)
    return [
        {
            "id": str(model["id"]),
            "checkpoint": str(_repo_path(str(model["checkpoint"]))),
            "sha256": str(model["sha256"]),
            "source": "canonical_matrix",
        }
        for model in matrix["models"]
        if model["id"] in selected
    ]


def _candidate_model_specs(
    matrix: Mapping[str, Any], candidate_id: str, checkpoint_value: str, expected_sha256: str
) -> list[dict[str, str]]:
    candidate_id = _safe_id(candidate_id, "candidate model id")
    canonical_ids = {str(model["id"]) for model in matrix["models"]}
    if candidate_id in canonical_ids:
        raise PilotMatrixError(
            f"Candidate model ID '{candidate_id}' collides with the canonical matrix"
        )
    if not SHA256.fullmatch(expected_sha256):
        raise PilotMatrixError("Candidate --sha256 must be an explicit lowercase 64-digit SHA-256")
    checkpoint = _local_path(checkpoint_value, "Candidate checkpoint")
    canonical_paths = {
        _repo_path(str(model["checkpoint"])) for model in matrix["models"]
    }
    if checkpoint in canonical_paths:
        raise PilotMatrixError("Candidate checkpoint must not alias a canonical matrix checkpoint")
    base = _canonical_model_specs(matrix, ["base"])[0]
    return [
        base,
        {
            "id": candidate_id,
            "checkpoint": str(checkpoint),
            "sha256": expected_sha256,
            "source": "explicit_candidate",
        },
    ]


def preflight_model_specs(
    matrix: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Verify pinned bytes and exact params_ema compatibility with canonical base."""
    if not model_specs:
        raise PilotMatrixError("No models selected for checkpoint preflight")
    selected_ids: list[str] = []
    normalized: list[dict[str, str]] = []
    for model in model_specs:
        model_id = _safe_id(str(model.get("id", "")), "model id")
        expected_hash = str(model.get("sha256", ""))
        source = str(model.get("source", ""))
        if not SHA256.fullmatch(expected_hash):
            raise PilotMatrixError(f"Model '{model_id}' has an invalid pinned SHA-256")
        if source not in {"canonical_matrix", "explicit_candidate"}:
            raise PilotMatrixError(f"Model '{model_id}' has invalid provenance source '{source}'")
        normalized.append(
            {
                "id": model_id,
                "checkpoint": str(_local_path(str(model.get("checkpoint", "")), "Checkpoint")),
                "sha256": expected_hash,
                "source": source,
            }
        )
        selected_ids.append(model_id)
    if len(selected_ids) != len(set(selected_ids)):
        raise PilotMatrixError("Selected model IDs must be unique")

    base = _canonical_model_specs(matrix, ["base"])[0]
    ordered = [base, *(model for model in normalized if model["id"] != "base")]
    reports: dict[str, Any] = {}
    reference_signature: dict[str, tuple[list[int], str]] | None = None
    for model in ordered:
        model_id = model["id"]
        checkpoint = _local_path(model["checkpoint"], "Checkpoint")
        if not checkpoint.is_file():
            raise PilotMatrixError(f"Checkpoint is missing: '{checkpoint}'")
        actual_hash = _sha256_file(checkpoint)
        if actual_hash != model["sha256"]:
            raise PilotMatrixError(
                f"Checkpoint hash mismatch for '{model_id}': {actual_hash} != {model['sha256']}"
            )
        signature, tensor_count = _state_signature(
            checkpoint, str(matrix["checkpoint_param_key"])
        )
        if model_id == "base":
            reference_signature = signature
        elif reference_signature is None or signature != reference_signature:
            raise PilotMatrixError(
                f"Checkpoint params_ema signature differs from canonical base for '{model_id}'"
            )
        reports[model_id] = {
            "path": str(checkpoint),
            "sha256": actual_hash,
            "param_key": matrix["checkpoint_param_key"],
            "tensor_count": tensor_count,
            "signature_sha256": _sha256_bytes(_json_bytes(signature)),
            "source": model["source"],
        }
    return {model_id: reports[model_id] for model_id in selected_ids}


def preflight_checkpoints(
    matrix: Mapping[str, Any], selected_ids: set[str] | None = None
) -> dict[str, Any]:
    """Verify selected canonical checkpoint entries."""
    model_ids = [
        str(model["id"])
        for model in matrix["models"]
        if selected_ids is None or model["id"] in selected_ids
    ]
    return preflight_model_specs(matrix, _canonical_model_specs(matrix, model_ids))


def cell_id(model_id: str, dataset_id: str) -> str:
    return f"{model_id}__{dataset_id}"


def experiment_name(run_id: str, model_id: str, dataset_id: str) -> str:
    return f"face_sr_pilot_{run_id}_{model_id}_{dataset_id}"


def result_dir(run_id: str, model_id: str, dataset_id: str) -> Path:
    return OUTPUT_ROOT / run_id / experiment_name(run_id, model_id, dataset_id)


def materialize_config(
    template: Mapping[str, Any],
    run_id: str,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = _safe_id(run_id, "run id")
    model_id = _safe_id(str(model["id"]), "model id")
    dataset_id = _safe_id(str(dataset["id"]), "dataset id")
    config = copy.deepcopy(dict(template))
    config["name"] = experiment_name(run_id, model_id, dataset_id)
    config["path"]["pretrain_network_g"] = str(
        _local_path(str(model["checkpoint"]), "Checkpoint")
    )
    config["path"]["results_root"] = str(OUTPUT_ROOT / run_id)
    config["val"]["suffix"] = model_id
    matching_datasets = {
        key: value
        for key, value in config["datasets"].items()
        if value.get("name") == dataset["name"]
    }
    if len(matching_datasets) != 1:
        raise PilotMatrixError(f"Template does not contain exactly one '{dataset_id}' dataset")
    config["datasets"] = matching_datasets
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    if any(placeholder in rendered for placeholder in PLACEHOLDERS):
        raise PilotMatrixError(f"Unresolved placeholder in generated config for '{model_id}'")
    if FORBIDDEN_OPTION_KEYS.intersection(_walk_keys(config)):
        raise PilotMatrixError(f"Resume option leaked into generated config for '{model_id}'")
    expected = (Path(config["path"]["results_root"]) / config["name"]).resolve()
    if expected != result_dir(run_id, model_id, dataset_id).resolve():
        raise PilotMatrixError(f"Unexpected BasicSR result directory for '{model_id}/{dataset_id}'")
    return config


def _render_config(config: Mapping[str, Any]) -> bytes:
    header = "# Generated by recovery.inference.pilot_matrix; do not edit.\n"
    return (header + yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=False)).encode("ascii")


def _snapshot_data(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value["snapshot"] for key, value in report.items()}


def _workspace(run_id: str) -> Path:
    return ORCHESTRATION_ROOT / run_id


def prepare(
    run_id: str,
    model_values: Sequence[str] = ("all",),
    dataset_values: Sequence[str] = ("all",),
) -> Path:
    matrix, template = load_canonical()
    model_ids = _selected_ids(
        [str(model["id"]) for model in matrix["models"]], model_values, "model"
    )
    dataset_ids = _selected_ids(
        [str(dataset["id"]) for dataset in matrix["datasets"]], dataset_values, "dataset"
    )
    model_specs = _canonical_model_specs(matrix, model_ids)
    return _prepare_specs(
        run_id, matrix, template, model_specs, dataset_ids, mode="canonical_matrix"
    )


def prepare_candidate(
    run_id: str,
    candidate_id: str,
    checkpoint_value: str,
    expected_sha256: str,
    dataset_values: Sequence[str],
) -> Path:
    """Prepare base plus one explicit, checksum-pinned candidate without matrix mutation."""
    matrix, template = load_canonical()
    dataset_ids = _selected_ids(
        [str(dataset["id"]) for dataset in matrix["datasets"]], dataset_values, "dataset"
    )
    model_specs = _candidate_model_specs(
        matrix, candidate_id, checkpoint_value, expected_sha256
    )
    return _prepare_specs(
        run_id, matrix, template, model_specs, dataset_ids, mode="explicit_candidate"
    )


def _prepare_specs(
    run_id: str,
    matrix: Mapping[str, Any],
    template: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    dataset_ids: Sequence[str],
    mode: str,
) -> Path:
    run_id = _safe_id(run_id, "run id")
    if mode not in {"canonical_matrix", "explicit_candidate"}:
        raise PilotMatrixError(f"Unknown preparation mode '{mode}'")
    model_ids = [str(model["id"]) for model in model_specs]
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise PilotMatrixError("Prepared model selection must be nonempty and unique")
    if mode == "explicit_candidate":
        if len(model_specs) != 2 or model_ids[0] != "base":
            raise PilotMatrixError("Candidate preparation must contain canonical base plus one candidate")
        if [str(model.get("source")) for model in model_specs] != [
            "canonical_matrix",
            "explicit_candidate",
        ]:
            raise PilotMatrixError("Candidate preparation has invalid model provenance")
    selected_datasets = [
        dataset for dataset in matrix["datasets"] if dataset["id"] in set(dataset_ids)
    ]
    if [str(dataset["id"]) for dataset in selected_datasets] != list(dataset_ids):
        raise PilotMatrixError("Prepared datasets must be a canonical ordered subset")
    workspace = _workspace(run_id)
    outputs = OUTPUT_ROOT / run_id
    if workspace.exists() or workspace.is_symlink():
        raise PilotMatrixError(f"Run workspace already exists: '{workspace}'. Choose a new run ID")
    if outputs.exists() or outputs.is_symlink():
        raise PilotMatrixError(f"Run output root already exists: '{outputs}'. Choose a new run ID")

    print("CPU preflight: verifying deterministic pilot data and manifests", flush=True)
    data_report = preflight_data(matrix, set(dataset_ids))
    print("CPU preflight: verifying checkpoint hashes and params_ema signatures", flush=True)
    checkpoint_report = preflight_model_specs(matrix, model_specs)
    print("CPU/runtime preflight: capturing code and environment provenance", flush=True)
    final_provenance_dir = workspace / "provenance"
    provenance, provenance_contents = _collect_provenance(final_provenance_dir)
    ORCHESTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=ORCHESTRATION_ROOT))
    final_config_dir = workspace / "configs"
    try:
        temp_config_dir = temporary / "configs"
        temp_config_dir.mkdir()
        temp_provenance_dir = temporary / "provenance"
        temp_provenance_dir.mkdir()
        for name, content in provenance_contents.items():
            (temp_provenance_dir / name).write_bytes(content)
        configs: dict[str, Any] = {}
        for model in model_specs:
            model_id = str(model["id"])
            for dataset in selected_datasets:
                dataset_id = str(dataset["id"])
                current_cell = cell_id(model_id, dataset_id)
                config_path = temp_config_dir / f"{current_cell}.yml"
                content = _render_config(materialize_config(template, run_id, model, dataset))
                config_path.write_bytes(content)
                configs[current_cell] = {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "path": str(final_config_dir / config_path.name),
                    "sha256": _sha256_bytes(content),
                    "experiment_name": experiment_name(run_id, model_id, dataset_id),
                    "result_dir": str(result_dir(run_id, model_id, dataset_id)),
                    "command": [str(DEFAULT_PYTHON), str(REPO_ROOT / "hat" / "test.py"), "-opt", str(final_config_dir / config_path.name)],
                    "auto_resume": False,
                }
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "mode": mode,
            "matrix_id": matrix["matrix_id"],
            "matrix_path": str(MATRIX_PATH),
            "matrix_sha256": _sha256_file(MATRIX_PATH),
            "template_path": str(_repo_path(str(matrix["template"]))),
            "template_sha256": _sha256_file(_repo_path(str(matrix["template"]))),
            "checkpoint_param_key": matrix["checkpoint_param_key"],
            "models": [dict(model) for model in model_specs],
            "selection": {
                "models": model_ids,
                "datasets": dataset_ids,
                "cells": [
                    cell_id(model_id, dataset_id)
                    for model_id in model_ids
                    for dataset_id in dataset_ids
                ],
            },
            "checkpoints": checkpoint_report,
            "data": _snapshot_data(data_report),
            "provenance": provenance,
            "configs": configs,
            "output_root": str(outputs),
            "auto_resume": False,
        }
        manifest_content = _json_bytes(manifest)
        manifest_path = temporary / "run_manifest.json"
        digest_path = temporary / "run_manifest.sha256"
        manifest_path.write_bytes(manifest_content)
        digest_path.write_text(
            f"{_sha256_bytes(manifest_content)}  run_manifest.json\n", encoding="ascii"
        )
        for path in (
            *temp_config_dir.iterdir(),
            *temp_provenance_dir.iterdir(),
            manifest_path,
            digest_path,
        ):
            path.chmod(0o444)
        if workspace.exists():
            raise PilotMatrixError(f"Run workspace appeared during preparation: '{workspace}'")
        os.replace(temporary, workspace)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Prepared immutable run: {workspace}")
    print("No HAT process or GPU workload was started.")
    return workspace


def _prepared_manifest(run_id: str) -> dict[str, Any]:
    run_id = _safe_id(run_id, "run id")
    manifest_path = _workspace(run_id) / "run_manifest.json"
    digest_path = _workspace(run_id) / "run_manifest.sha256"
    try:
        expected_digest_line = digest_path.read_text(encoding="ascii")
    except OSError as error:
        raise PilotMatrixError(f"Cannot read prepared manifest digest '{digest_path}': {error}") from error
    actual_manifest_hash = _sha256_file(manifest_path)
    if expected_digest_line != f"{actual_manifest_hash}  run_manifest.json\n":
        raise PilotMatrixError(f"Prepared manifest digest mismatch in '{digest_path}'")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 2 or manifest.get("run_id") != run_id:
        raise PilotMatrixError(f"Prepared manifest identity mismatch in '{manifest_path}'")
    if manifest.get("auto_resume") is not False:
        raise PilotMatrixError("Prepared manifest must explicitly record auto_resume=false")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise PilotMatrixError("Prepared manifest lacks environment provenance")
    _verify_stored_provenance(provenance)
    matrix, template = load_canonical()
    if manifest.get("matrix_sha256") != _sha256_file(MATRIX_PATH):
        raise PilotMatrixError("Canonical pilot_matrix.json changed after this run was prepared")
    template_path = _repo_path(str(matrix["template"]))
    if manifest.get("template_sha256") != _sha256_file(template_path):
        raise PilotMatrixError("Canonical pilot template changed after this run was prepared")
    prepared_model_list = manifest.get("models")
    if not isinstance(prepared_model_list, list):
        raise PilotMatrixError("Prepared manifest lacks frozen model definitions")
    models: dict[str, Mapping[str, Any]] = {}
    canonical_by_id = {str(model["id"]): model for model in matrix["models"]}
    candidate_count = 0
    for model in prepared_model_list:
        if not isinstance(model, dict):
            raise PilotMatrixError("Prepared model definition is not a mapping")
        model_id = _safe_id(str(model.get("id", "")), "prepared model id")
        if model_id in models:
            raise PilotMatrixError(f"Duplicate prepared model ID '{model_id}'")
        source = str(model.get("source", ""))
        checkpoint = str(_local_path(str(model.get("checkpoint", "")), "Checkpoint"))
        sha256 = str(model.get("sha256", ""))
        if not SHA256.fullmatch(sha256):
            raise PilotMatrixError(f"Prepared model '{model_id}' has an invalid SHA-256")
        normalized = {
            "id": model_id,
            "checkpoint": checkpoint,
            "sha256": sha256,
            "source": source,
        }
        if source == "canonical_matrix":
            if model_id not in canonical_by_id:
                raise PilotMatrixError(f"Unknown canonical prepared model '{model_id}'")
            if normalized != _canonical_model_specs(matrix, [model_id])[0]:
                raise PilotMatrixError(f"Canonical model definition changed for '{model_id}'")
        elif source == "explicit_candidate":
            candidate_count += 1
            if model_id in canonical_by_id:
                raise PilotMatrixError(f"Candidate model ID collides with canonical ID '{model_id}'")
            canonical_paths = {
                _repo_path(str(item["checkpoint"])) for item in matrix["models"]
            }
            if Path(checkpoint) in canonical_paths:
                raise PilotMatrixError(
                    f"Candidate checkpoint aliases a canonical checkpoint for '{model_id}'"
                )
        else:
            raise PilotMatrixError(f"Prepared model '{model_id}' has invalid source '{source}'")
        models[model_id] = normalized
    mode = manifest.get("mode")
    if mode == "canonical_matrix":
        if candidate_count:
            raise PilotMatrixError("Canonical run unexpectedly contains a candidate model")
    elif mode == "explicit_candidate":
        if candidate_count != 1 or list(models)[0] != "base" or len(models) != 2:
            raise PilotMatrixError("Candidate run must freeze canonical base plus one candidate")
    else:
        raise PilotMatrixError(f"Prepared manifest has invalid mode '{mode}'")
    datasets = {str(dataset["id"]): dataset for dataset in matrix["datasets"]}
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise PilotMatrixError("Prepared manifest lacks an explicit matrix selection")
    selected_models = selection.get("models")
    selected_datasets = selection.get("datasets")
    if not isinstance(selected_models, list) or not isinstance(selected_datasets, list):
        raise PilotMatrixError("Prepared manifest selection is malformed")
    canonical_dataset_ids = [str(dataset["id"]) for dataset in matrix["datasets"]]
    if (
        list(models) != selected_models
        or _selected_ids(canonical_dataset_ids, selected_datasets, "dataset") != selected_datasets
    ):
        raise PilotMatrixError("Prepared manifest selection is not in canonical order")
    expected_cells = {
        cell_id(model_id, dataset_id)
        for model_id in selected_models
        for dataset_id in selected_datasets
    }
    if selection.get("cells") != [
        cell_id(model_id, dataset_id)
        for model_id in selected_models
        for dataset_id in selected_datasets
    ]:
        raise PilotMatrixError("Prepared manifest cell order does not match its selection")
    if set(manifest.get("checkpoints", {})) != set(selected_models):
        raise PilotMatrixError("Prepared checkpoint provenance does not match its model selection")
    if set(manifest.get("data", {})) != {*selected_datasets, "pilot_selection"}:
        raise PilotMatrixError("Prepared data provenance does not match its dataset selection")
    if set(manifest.get("configs", {})) != expected_cells:
        raise PilotMatrixError("Prepared config set no longer matches the canonical matrix")
    for current_cell, entry in manifest["configs"].items():
        model_id = str(entry.get("model_id", ""))
        dataset_id = str(entry.get("dataset_id", ""))
        if current_cell != cell_id(model_id, dataset_id) or model_id not in models or dataset_id not in datasets:
            raise PilotMatrixError(f"Prepared config has invalid cell identity '{current_cell}'")
        config_path = Path(str(entry["path"]))
        expected_content = _render_config(
            materialize_config(template, run_id, models[model_id], datasets[dataset_id])
        )
        if not config_path.is_file() or config_path.read_bytes() != expected_content:
            raise PilotMatrixError(f"Prepared config was changed or removed: '{config_path}'")
        if entry.get("sha256") != _sha256_bytes(expected_content):
            raise PilotMatrixError(f"Prepared config hash metadata mismatch for '{current_cell}'")
        expected_command = [
            str(DEFAULT_PYTHON),
            str(REPO_ROOT / "hat" / "test.py"),
            "-opt",
            str(config_path),
        ]
        expected_fields = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "experiment_name": experiment_name(run_id, model_id, dataset_id),
            "result_dir": str(result_dir(run_id, model_id, dataset_id)),
            "command": expected_command,
            "auto_resume": False,
        }
        for field, expected in expected_fields.items():
            if entry.get(field) != expected:
                raise PilotMatrixError(
                    f"Prepared config metadata field '{field}' is invalid for '{current_cell}'"
                )
    return manifest


def _selected_ids(known: Sequence[str], values: Sequence[str], label: str) -> list[str]:
    if values == ["all"]:
        return list(known)
    if "all" in values:
        raise PilotMatrixError(f"Use --{label} all alone, not mixed with explicit IDs")
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise PilotMatrixError(f"Unknown {label} ID(s): {', '.join(unknown)}")
    selected = [item_id for item_id in known if item_id in set(values)]
    if not selected:
        raise PilotMatrixError(f"Select at least one {label} with --{label}")
    return selected


def _active_hat_processes() -> list[tuple[int, str]]:
    active = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (OSError, PermissionError):
            continue
        if "hat/train.py" in command or "hat/test.py" in command:
            active.append((int(entry.name), command.strip()))
    return active


def _acquire_gpu_lock(run_id: str, cells: Sequence[str]) -> tuple[int, Path]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = ARTIFACT_ROOT / "active_gpu.lock"
    content = _json_bytes({"pid": os.getpid(), "run_id": run_id, "cells": list(cells)})
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise PilotMatrixError(
            f"GPU inference lock already exists: '{lock_path}'. Inspect its owner; do not delete it blindly"
        ) from error
    os.write(descriptor, content)
    os.fsync(descriptor)
    return descriptor, lock_path


def _archived_siblings(target: Path) -> tuple[str, ...]:
    if not target.parent.is_dir():
        return ()
    prefix = f"{target.name}_archived_"
    return tuple(
        str(path)
        for path in sorted(target.parent.iterdir(), key=lambda item: item.name)
        if path.name.startswith(prefix)
    )


def _invoke_hat_cell(
    command: Sequence[str], target: Path, environment: Mapping[str, str]
) -> dict[str, list[str]]:
    """Recheck the cell path, invoke HAT, and expose BasicSR archive races."""
    archived_before = _archived_siblings(target)
    if target.exists() or target.is_symlink():
        raise PilotMatrixError(
            f"Result path appeared immediately before HAT launch: '{target}'. "
            "Nothing was deleted or renamed"
        )
    completed = subprocess.run(
        list(command), cwd=REPO_ROOT, env=dict(environment), check=False
    )
    archived_after = _archived_siblings(target)
    archived_created = sorted(set(archived_after) - set(archived_before))
    if archived_created:
        raise PilotMatrixError(
            "BasicSR created archived sibling(s) during the cell, proving a result-path race: "
            + ", ".join(archived_created)
            + f". HAT exit code was {completed.returncode}; all partial/archive evidence is preserved"
        )
    if completed.returncode != 0:
        raise PilotMatrixError(
            f"HAT inference failed with exit code {completed.returncode}. "
            "The partial result directory is preserved; do not reuse it"
        )
    return {
        "before": list(archived_before),
        "after": list(archived_after),
        "created_during_cell": [],
    }


def _validate_saved_outputs(
    matrix: Mapping[str, Any],
    run_id: str,
    model_id: str,
    dataset_id: str,
    data_report: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = next(item for item in matrix["datasets"] if item["id"] == dataset_id)
    root = result_dir(run_id, model_id, dataset_id) / "visualization"
    output_dir = root / str(dataset["name"])
    files = _directory_pngs(output_dir, int(matrix["expected_count_per_dataset"]))
    expected_names = {f"{image_id}_{model_id}" for image_id in data_report[dataset_id]["ids"]}
    if set(files) != expected_names:
        raise PilotMatrixError(f"Saved output membership/suffix mismatch in '{output_dir}'")
    tree_bytes = b"".join(
        path.name.encode("utf-8")
        + b"\0"
        + str(path.stat().st_size).encode("ascii")
        + b"\0"
        + _sha256_file(path).encode("ascii")
        + b"\n"
        for _, path in sorted(files.items(), key=lambda item: item[1].name)
    )
    tree_digest = _sha256_bytes(tree_bytes)
    return {
        "root": str(output_dir),
        "file_count": len(files),
        "prediction_suffix": f"_{model_id}",
        "files_manifest_sha256": tree_digest,
        "tree_sha256": tree_digest,
        "tree_sha256_algorithm": TREE_DIGEST_ALGORITHM,
    }


def _completion_record(
    *,
    run_id: str,
    current_cell: str,
    model_id: str,
    dataset_id: str,
    checkpoint: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    command: Sequence[str],
    config_path: Path,
    config_sha256: str,
    provenance: Mapping[str, Any],
    outputs: Mapping[str, Any],
    archive_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the completion schema consumed by the strict evaluator."""
    return {
        "schema_version": 3,
        "status": "complete",
        "run_id": run_id,
        "cell_id": current_cell,
        "model": {
            "name": model_id,
            "checkpoint": dict(checkpoint),
        },
        "dataset": {
            "id": dataset_id,
            "provenance": dict(data_provenance),
        },
        "command": list(command),
        "config": {
            "path": str(config_path),
            "sha256": config_sha256,
        },
        "provenance": dict(provenance),
        "outputs": dict(outputs),
        "basicsr_archived_siblings": dict(archive_report),
        "auto_resume": False,
    }


def _write_immutable_completion(path: Path, record: Mapping[str, Any]) -> tuple[str, Path]:
    """Exclusively write a read-only completion and its checksum sidecar."""
    digest_path = Path(f"{path}.sha256")
    if path.exists() or path.is_symlink() or digest_path.exists() or digest_path.is_symlink():
        raise PilotMatrixError(
            f"Completion evidence already exists; refusing to overwrite it: '{path}'"
        )
    content = _json_bytes(record)
    digest = _sha256_bytes(content)
    for output, payload in (
        (path, content),
        (digest_path, f"{digest}  {path.name}\n".encode("ascii")),
    ):
        try:
            descriptor = os.open(
                output,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o444,
            )
        except OSError as error:
            raise PilotMatrixError(
                f"Cannot exclusively create immutable completion evidence '{output}': {error}. "
                "Any partial evidence is preserved"
            ) from error
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            output.chmod(0o444)
        except BaseException:
            # Never erase a partially written record: its presence prevents unsafe reuse.
            raise
    return digest, digest_path


def _verify_immutable_completion(path: Path) -> None:
    digest_path = Path(f"{path}.sha256")
    try:
        digest_line = digest_path.read_text(encoding="ascii")
    except OSError as error:
        raise PilotMatrixError(
            f"Cannot read completion digest sidecar '{digest_path}': {error}"
        ) from error
    actual_digest = _sha256_file(path)
    if digest_line != f"{actual_digest}  {path.name}\n":
        raise PilotMatrixError(f"Completion digest mismatch in '{digest_path}'")
    for evidence_path in (path, digest_path):
        if evidence_path.stat().st_mode & 0o222:
            raise PilotMatrixError(
                f"Completion evidence is writable instead of immutable: '{evidence_path}'"
            )


def run_prepared(
    run_id: str,
    model_values: Sequence[str],
    dataset_values: Sequence[str],
    python: Path,
    confirmed: bool,
) -> None:
    if not confirmed:
        raise PilotMatrixError("GPU execution requires the explicit --confirm-gpu-run flag")
    manifest = _prepared_manifest(run_id)
    matrix, _ = load_canonical()
    model_ids = _selected_ids(manifest["selection"]["models"], model_values, "model")
    dataset_ids = _selected_ids(
        manifest["selection"]["datasets"], dataset_values, "dataset"
    )
    cells = [cell_id(model_id, dataset_id) for model_id in model_ids for dataset_id in dataset_ids]
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise PilotMatrixError(f"Python executable is missing or not executable: '{python}'")
    descriptor, lock_path = _acquire_gpu_lock(run_id, cells)
    os.close(descriptor)
    try:
        existing = [
            str(result_dir(run_id, model_id, dataset_id))
            for model_id in model_ids
            for dataset_id in dataset_ids
            if result_dir(run_id, model_id, dataset_id).exists()
            or result_dir(run_id, model_id, dataset_id).is_symlink()
        ]
        if existing:
            raise PilotMatrixError(
                "Refusing to mix with existing or partial result directories; choose a new run ID or select only pending cells: "
                + ", ".join(existing)
            )
        active = _active_hat_processes()
        if active:
            detail = "; ".join(f"PID {pid}: {command}" for pid, command in active)
            raise PilotMatrixError(
                f"Another HAT train/test process is active; refusing to compete for the GPU: {detail}"
            )

        print(
            f"Strict preflight before GPU launch: verifying {', '.join(dataset_ids)} pilot data",
            flush=True,
        )
        data_report = preflight_data(matrix, set(dataset_ids))
        print(
            f"Strict preflight before GPU launch: verifying {', '.join(model_ids)}",
            flush=True,
        )
        prepared_models = {
            str(model["id"]): model for model in manifest["models"]
        }
        checkpoint_report = preflight_model_specs(
            matrix, [prepared_models[model_id] for model_id in model_ids]
        )
        for model_id in model_ids:
            if checkpoint_report[model_id] != manifest["checkpoints"][model_id]:
                raise PilotMatrixError(
                    f"Checkpoint preflight differs from prepared manifest for '{model_id}'"
                )
        for dataset_id in dataset_ids:
            if data_report[dataset_id]["snapshot"] != manifest["data"][dataset_id]:
                raise PilotMatrixError(
                    f"Data preflight differs from prepared manifest for '{dataset_id}'"
                )
        print("Strict preflight before GPU launch: revalidating code/environment provenance", flush=True)
        current_provenance, _ = _collect_provenance(_workspace(run_id) / "provenance")
        if current_provenance != manifest["provenance"]:
            raise PilotMatrixError(
                "Code, environment, package, git, or GPU provenance drifted after prepare; "
                "refusing to launch HAT"
            )
        for model_id in model_ids:
            for dataset_id in dataset_ids:
                current_cell = cell_id(model_id, dataset_id)
                config_entry = manifest["configs"][current_cell]
                config_path = Path(str(config_entry["path"]))
                command = [str(python), str(REPO_ROOT / "hat" / "test.py"), "-opt", str(config_path)]
                if "--auto_resume" in command:
                    raise PilotMatrixError("Internal error: auto-resume is forbidden for pilot inference")
                print(f"Launching explicit GPU inference: {shlex.join(command)}", flush=True)
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                archive_report = _invoke_hat_cell(
                    command,
                    result_dir(run_id, model_id, dataset_id),
                    environment,
                )
                output_report = _validate_saved_outputs(
                    matrix, run_id, model_id, dataset_id, data_report
                )
                completion = _completion_record(
                    run_id=run_id,
                    current_cell=current_cell,
                    model_id=model_id,
                    dataset_id=dataset_id,
                    checkpoint=checkpoint_report[model_id],
                    data_provenance=data_report[dataset_id]["snapshot"],
                    command=command,
                    config_path=config_path,
                    config_sha256=str(config_entry["sha256"]),
                    provenance=manifest["provenance"],
                    outputs=output_report,
                    archive_report=archive_report,
                )
                completion_path = result_dir(run_id, model_id, dataset_id) / "completion.json"
                _write_immutable_completion(completion_path, completion)
                print(f"Validated complete output set: {completion_path}", flush=True)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def print_status(run_id: str) -> None:
    manifest = _prepared_manifest(run_id)
    print(f"run_id: {run_id}")
    for current_cell, entry in manifest["configs"].items():
        target = result_dir(run_id, entry["model_id"], entry["dataset_id"])
        completion = target / "completion.json"
        if completion.is_file():
            state = "complete"
        elif target.exists():
            state = "partial (never reuse)"
        else:
            state = "pending"
        print(f"  {current_cell}: {state} -> {target}")


def _validated_completion(
    matrix: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run_id: str,
    model_id: str,
    dataset_id: str,
    data_report: Mapping[str, Any],
) -> dict[str, Any]:
    current_cell = cell_id(model_id, dataset_id)
    completion_path = result_dir(run_id, model_id, dataset_id) / "completion.json"
    _verify_immutable_completion(completion_path)
    completion = _load_json(completion_path)
    expected_identity = {
        "schema_version": 3,
        "status": "complete",
        "run_id": run_id,
        "cell_id": current_cell,
        "model": {
            "name": model_id,
            "checkpoint": manifest["checkpoints"][model_id],
        },
        "dataset": {
            "id": dataset_id,
            "provenance": manifest["data"][dataset_id],
        },
        "config": {
            "path": manifest["configs"][current_cell]["path"],
            "sha256": manifest["configs"][current_cell]["sha256"],
        },
        "provenance": manifest["provenance"],
        "auto_resume": False,
    }
    for field, expected in expected_identity.items():
        if completion.get(field) != expected:
            raise PilotMatrixError(
                f"Completion field '{field}' is invalid in '{completion_path}'"
            )
    actual_outputs = _validate_saved_outputs(
        matrix, run_id, model_id, dataset_id, data_report
    )
    if completion.get("outputs") != actual_outputs:
        raise PilotMatrixError(f"Saved outputs changed after completion: '{completion_path}'")
    archive_report = completion.get("basicsr_archived_siblings")
    if (
        not isinstance(archive_report, dict)
        or archive_report.get("created_during_cell") != []
        or archive_report.get("before") != archive_report.get("after")
    ):
        raise PilotMatrixError(f"Invalid BasicSR archive report in '{completion_path}'")
    return completion


def _comparison_output(run_id: str, dataset_id: str, model_ids: Sequence[str]) -> Path:
    return EVALUATION_ROOT / run_id / dataset_id / "__".join(model_ids)


def evaluation_command(
    run_id: str,
    dataset_id: str,
    model_values: Sequence[str],
    python: Path,
    arcface_backend: str | None = None,
    arcface_model: Path | None = None,
    arcface_device: str = "cpu",
    arcface_batch_size: int = 16,
    confirm_arcface_gpu: bool = False,
    selection_count: int = 24,
    selection_metric: str = "psnr",
) -> list[str]:
    if selection_count < 1:
        raise PilotMatrixError("Selection count must be positive")
    if arcface_device not in {"cpu", "cuda"}:
        raise PilotMatrixError(f"Unsupported ArcFace device '{arcface_device}'")
    if arcface_batch_size < 1:
        raise PilotMatrixError("ArcFace batch size must be positive")
    if confirm_arcface_gpu and arcface_device != "cuda":
        raise PilotMatrixError("--confirm-arcface-gpu is valid only with --arcface-device cuda")
    if arcface_device == "cuda" and not confirm_arcface_gpu:
        raise PilotMatrixError("ArcFace CUDA/ROCm evaluation requires --confirm-arcface-gpu")
    manifest = _prepared_manifest(run_id)
    matrix, _ = load_canonical()
    model_ids = _selected_ids(manifest["selection"]["models"], model_values, "model")
    if "base" not in model_ids:
        raise PilotMatrixError("Evaluation selections must include --model base")
    dataset_by_id = {str(dataset["id"]): dataset for dataset in matrix["datasets"]}
    if dataset_id not in manifest["selection"]["datasets"]:
        raise PilotMatrixError(f"Dataset ID '{dataset_id}' was not prepared in this run")
    dataset = dataset_by_id[dataset_id]
    data_report = preflight_data(matrix, {dataset_id})
    if data_report[dataset_id]["snapshot"] != manifest["data"][dataset_id]:
        raise PilotMatrixError(f"Data preflight differs from prepared manifest for '{dataset_id}'")
    command = [
        str(python),
        "-m",
        "recovery.eval.evaluate_saved",
        "--gt",
        str(Path(str(matrix["data_root"])) / str(dataset["root"]) / "gt"),
    ]
    for model_id in model_ids:
        _validated_completion(
            matrix, manifest, run_id, model_id, dataset_id, data_report
        )
        prediction = result_dir(run_id, model_id, dataset_id) / "visualization" / str(dataset["name"])
        command.extend(("--prediction", f"{model_id}={prediction}"))
        command.extend(("--prediction-suffix", f"{model_id}=_{model_id}"))
        completion_path = result_dir(run_id, model_id, dataset_id) / "completion.json"
        command.extend(("--completion-record", f"{model_id}={completion_path}"))
    output = _comparison_output(run_id, dataset_id, model_ids)
    if output.exists() or output.is_symlink():
        raise PilotMatrixError(
            f"Evaluation output already exists: '{output}'. Preserve it and use a new run/model selection"
        )
    command.extend(
        (
            "--baseline",
            "base",
            "--pair-policy",
            "strict",
            "--color-space",
            "y",
            "--crop-border",
            "4",
            "--bootstrap-samples",
            "5000",
            "--out",
            str(output),
            "--selection-json",
            str(output / "contact_selection.json"),
            "--selection-count",
            str(selection_count),
            "--selection-metric",
            selection_metric,
        )
    )
    if (arcface_backend is None) != (arcface_model is None):
        raise PilotMatrixError(
            "ArcFace evaluation requires both --arcface-backend and --arcface-model"
        )
    if arcface_backend is not None and arcface_model is not None:
        if arcface_backend not in {"facexlib-pth", "onnx"}:
            raise PilotMatrixError(f"Unsupported ArcFace backend '{arcface_backend}'")
        resolved_arcface = _local_path(str(arcface_model), "ArcFace model")
        if not resolved_arcface.is_file():
            raise PilotMatrixError(f"ArcFace model is missing: '{resolved_arcface}'")
        command.extend(
            (
                "--arcface-backend",
                arcface_backend,
                "--arcface-model",
                str(resolved_arcface),
                "--arcface-device",
                arcface_device,
                "--arcface-batch-size",
                str(arcface_batch_size),
            )
        )
        if confirm_arcface_gpu:
            command.append("--confirm-arcface-gpu")
    return command


def contact_sheet_command(
    run_id: str,
    dataset_id: str,
    model_values: Sequence[str],
    python: Path,
    tile_size: int,
) -> list[str]:
    manifest = _prepared_manifest(run_id)
    model_ids = _selected_ids(manifest["selection"]["models"], model_values, "model")
    if dataset_id not in manifest["selection"]["datasets"]:
        raise PilotMatrixError(f"Dataset ID '{dataset_id}' was not prepared in this run")
    if tile_size < 32:
        raise PilotMatrixError("Contact-sheet tile size must be at least 32")
    output = _comparison_output(run_id, dataset_id, model_ids)
    selection = output / "contact_selection.json"
    contact_sheet = output / "contact_sheet.png"
    if not selection.is_file():
        raise PilotMatrixError(
            f"Evaluator selection is missing: '{selection}'. Run eval-command output first"
        )
    if contact_sheet.exists() or contact_sheet.is_symlink():
        raise PilotMatrixError(f"Contact sheet already exists: '{contact_sheet}'")
    return [
        str(python),
        "-m",
        "recovery.eval.build_contact_sheet",
        "--selection",
        str(selection),
        "--out",
        str(contact_sheet),
        "--tile-size",
        str(tile_size),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("static-check", help="validate canonical matrix/template only; no data, Torch, or GPU")
    subparsers.add_parser("check", help="CPU-verify all data and checkpoint bytes/state signatures")

    prepare_parser = subparsers.add_parser("prepare", help="strictly preflight and materialize an immutable run")
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument(
        "--model", action="append", dest="models", help="model ID, repeated, or 'all' (default: all)"
    )
    prepare_parser.add_argument(
        "--dataset", action="append", dest="datasets", help="clean/mild/hard, repeated, or 'all' (default: all)"
    )

    candidate_parser = subparsers.add_parser(
        "prepare-candidate",
        help="prepare canonical base plus one explicit SHA-pinned local checkpoint",
    )
    candidate_parser.add_argument("--run-id", required=True)
    candidate_parser.add_argument("--candidate-id", required=True)
    candidate_parser.add_argument("--checkpoint", required=True)
    candidate_parser.add_argument("--sha256", required=True)
    candidate_parser.add_argument(
        "--dataset", action="append", required=True, dest="datasets", help="clean/mild/hard, repeated, or 'all'"
    )

    status_parser = subparsers.add_parser("status", help="show pending, partial, and complete model outputs")
    status_parser.add_argument("--run-id", required=True)

    run_parser = subparsers.add_parser("run", help="explicitly launch HAT test inference after strict preflight")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--model", action="append", required=True, dest="models", help="model ID, repeated, or 'all'")
    run_parser.add_argument(
        "--dataset", action="append", required=True, dest="datasets", help="clean/mild/hard, repeated, or 'all'"
    )
    run_parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    run_parser.add_argument("--confirm-gpu-run", action="store_true")

    eval_parser = subparsers.add_parser("eval-command", help="print a strict saved-output evaluator command")
    eval_parser.add_argument("--run-id", required=True)
    eval_parser.add_argument("--dataset", required=True, choices=("clean", "mild", "hard"))
    eval_parser.add_argument("--model", action="append", required=True, dest="models")
    eval_parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    eval_parser.add_argument("--arcface-backend", choices=("facexlib-pth", "onnx"))
    eval_parser.add_argument("--arcface-model", type=Path)
    eval_parser.add_argument("--arcface-device", choices=("cpu", "cuda"), default="cpu")
    eval_parser.add_argument("--arcface-batch-size", type=int, default=16)
    eval_parser.add_argument("--confirm-arcface-gpu", action="store_true")
    eval_parser.add_argument("--selection-count", type=int, default=24)
    eval_parser.add_argument(
        "--selection-metric",
        choices=(
            "psnr",
            "ssim",
            "sharpness_absolute_error",
            "edge_correlation",
            "lpips",
            "arcface_identity_similarity",
        ),
        default="psnr",
    )

    contact_parser = subparsers.add_parser(
        "contact-command", help="print the contact-sheet command for a completed evaluation"
    )
    contact_parser.add_argument("--run-id", required=True)
    contact_parser.add_argument("--dataset", required=True, choices=("clean", "mild", "hard"))
    contact_parser.add_argument("--model", action="append", required=True, dest="models")
    contact_parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    contact_parser.add_argument("--tile-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "static-check":
            matrix, _ = load_canonical()
            print(
                f"Static matrix OK: {len(matrix['models'])} models x {len(matrix['datasets'])} datasets "
                f"x {matrix['expected_count_per_dataset']} images"
            )
            print("No checkpoint, dataset image, HAT process, or GPU was touched.")
        elif args.command == "check":
            matrix, _ = load_canonical()
            data_report = preflight_data(matrix)
            checkpoint_report = preflight_checkpoints(matrix)
            print(
                f"Strict CPU preflight OK: {len(checkpoint_report)} checkpoints, "
                f"{len(data_report) - 1} pilot datasets"
            )
            print("No HAT process or GPU workload was started.")
        elif args.command == "prepare":
            prepare(args.run_id, args.models or ["all"], args.datasets or ["all"])
        elif args.command == "prepare-candidate":
            prepare_candidate(
                args.run_id,
                args.candidate_id,
                args.checkpoint,
                args.sha256,
                args.datasets,
            )
        elif args.command == "status":
            print_status(args.run_id)
        elif args.command == "run":
            run_prepared(args.run_id, args.models, args.datasets, args.python, args.confirm_gpu_run)
        elif args.command == "eval-command":
            print(
                shlex.join(
                    evaluation_command(
                        args.run_id,
                        args.dataset,
                        args.models,
                        args.python,
                        args.arcface_backend,
                        args.arcface_model,
                        args.arcface_device,
                        args.arcface_batch_size,
                        args.confirm_arcface_gpu,
                        args.selection_count,
                        args.selection_metric,
                    )
                )
            )
        else:
            print(
                shlex.join(
                    contact_sheet_command(
                        args.run_id,
                        args.dataset,
                        args.models,
                        args.python,
                        args.tile_size,
                    )
                )
            )
    except PilotMatrixError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
