"""Evaluation orchestration for saved image directories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .metrics import LpipsMetric, create_arcface_metric, fidelity_metrics, read_bgr
from .provenance import (
    canonical_json_sha256,
    sha256_file,
    tree_digest,
    validate_completion_record,
)
from .statistics import paired_comparison, summarize

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_DIRECTIONS = {
    "psnr": True,
    "ssim": True,
    "sharpness_absolute_error": False,
    "edge_correlation": True,
    "lpips": False,
    "arcface_identity_similarity": True,
}


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    directory: Path
    filename_suffix: str = ""
    completion_record: Path | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    gt_directory: Path
    predictions: tuple[PredictionSpec, ...]
    output_directory: Path
    crop_border: int = 4
    color_space: str = "y"
    pair_policy: str = "strict"
    baseline: str | None = None
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    seed: int = 20260820
    tie_tolerance: float = 1e-12
    lpips: bool = False
    lpips_network: str = "alex"
    lpips_calibration_weights: Path | None = None
    lpips_allow_model_downloads: bool = False
    arcface_backend: str | None = None
    arcface_model: Path | None = None
    arcface_device: str = "cpu"
    arcface_confirm_gpu: bool = False
    arcface_batch_size: int = 16
    selection_json: Path | None = None
    selection_count: int = 24
    selection_metric: str = "psnr"


def _validate_name(name: str) -> None:
    valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not name or any(character not in valid_characters for character in name):
        raise ValueError(
            f"invalid model name {name!r}; use only ASCII letters, digits, '_' and '-'"
        )


def index_images(directory: Path, *, filename_suffix: str = "") -> dict[str, Path]:
    """Index recursively by basename after requiring/removing an explicit suffix."""
    if not directory.is_dir():
        raise NotADirectoryError(f"image directory does not exist: {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stem = path.stem
        if filename_suffix:
            if not stem.endswith(filename_suffix):
                continue
            stem = stem[: -len(filename_suffix)]
        if not stem:
            raise ValueError(f"suffix {filename_suffix!r} removes the whole stem: {path}")
        if stem in indexed:
            raise ValueError(
                f"duplicate image key {stem!r} after suffix removal: "
                f"{indexed[stem]} and {path}"
            )
        indexed[stem] = path
    if not indexed:
        suffix_description = f" ending in {filename_suffix!r}" if filename_suffix else ""
        raise ValueError(f"no supported images{suffix_description} found under {directory}")
    return indexed


def _pair_inputs(
    config: EvaluationConfig,
) -> tuple[list[str], dict[str, Path], dict[str, dict[str, Path]], dict[str, Any]]:
    gt_index = index_images(config.gt_directory)
    prediction_indexes: dict[str, dict[str, Path]] = {}
    seen_names: set[str] = set()
    pairing_models: dict[str, Any] = {}
    gt_keys = set(gt_index)
    for spec in config.predictions:
        _validate_name(spec.name)
        if spec.name in seen_names:
            raise ValueError(f"duplicate prediction name: {spec.name}")
        seen_names.add(spec.name)
        prediction_index = index_images(spec.directory, filename_suffix=spec.filename_suffix)
        prediction_indexes[spec.name] = prediction_index
        prediction_keys = set(prediction_index)
        missing = sorted(gt_keys - prediction_keys)
        extra = sorted(prediction_keys - gt_keys)
        pairing_models[spec.name] = {
            "output_count": len(prediction_index),
            "matched_to_gt_count": len(gt_keys & prediction_keys),
            "missing_gt_output_count": len(missing),
            "extra_output_count": len(extra),
            "missing_gt_output_examples": missing[:20],
            "extra_output_examples": extra[:20],
            "filename_suffix": spec.filename_suffix,
        }

    if config.pair_policy not in {"strict", "intersection"}:
        raise ValueError("pair policy must be 'strict' or 'intersection'")
    if config.pair_policy == "strict":
        invalid = [
            name
            for name, report in pairing_models.items()
            if report["missing_gt_output_count"] or report["extra_output_count"]
        ]
        if invalid:
            details = ", ".join(
                f"{name}: {pairing_models[name]['missing_gt_output_count']} missing, "
                f"{pairing_models[name]['extra_output_count']} extra"
                for name in invalid
            )
            raise ValueError(
                "strict pairing failed (use --pair-policy intersection only when a "
                f"partial common subset is intentional): {details}"
            )
        keys = sorted(gt_keys)
    else:
        common = set(gt_keys)
        for prediction_index in prediction_indexes.values():
            common &= set(prediction_index)
        keys = sorted(common)
        if not keys:
            raise ValueError("prediction directories and GT have no common image keys")

    pairing = {
        "policy": config.pair_policy,
        "key_definition": "case-sensitive basename after explicit filename suffix removal",
        "gt_count": len(gt_index),
        "evaluated_common_count": len(keys),
        "excluded_gt_count": len(gt_index) - len(keys),
        "models": pairing_models,
    }
    return keys, gt_index, prediction_indexes, pairing


def _dependency_versions() -> dict[str, str]:
    try:
        import basicsr

        basicsr_version = getattr(basicsr, "__version__", "unknown")
    except ImportError:
        basicsr_version = "unavailable"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "basicsr": basicsr_version,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _assert_empty_output_destination(config: EvaluationConfig) -> None:
    output_directory = config.output_directory
    if output_directory.is_symlink() and not output_directory.exists():
        raise FileExistsError(f"refusing broken-symlink output destination: {output_directory}")
    if output_directory.exists():
        if not output_directory.is_dir():
            raise FileExistsError(f"evaluation output exists and is not a directory: {output_directory}")
        if next(output_directory.iterdir(), None) is not None:
            raise FileExistsError(
                f"evaluation output directory is nonempty; preserve it and choose a new path: "
                f"{output_directory}"
            )
    if config.selection_json is not None:
        selection_path = config.selection_json
        if selection_path.exists() or selection_path.is_symlink():
            raise FileExistsError(
                f"selection output already exists; preserve it and choose a new path: {selection_path}"
            )


def _write_outputs(records: list[dict[str, Any]], aggregate: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    if next(output_directory.iterdir(), None) is not None:
        raise FileExistsError(
            f"evaluation output directory became nonempty before write: {output_directory}"
        )
    jsonl_path = output_directory / "per_image.jsonl"
    with jsonl_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")

    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_directory / "per_image.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(records))

    with (output_directory / "aggregate.json").open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(aggregate), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _stable_seed(base_seed: int, name: str, metric: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _make_selection(
    records: list[dict[str, Any]],
    *,
    baseline: str | None,
    count: int,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("selection count must be positive")
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_id.setdefault(record["id"], {})[record["model"]] = record
    all_ids = sorted(by_id)
    hash_ranked = sorted(
        all_ids,
        key=lambda image_id: hashlib.sha256(f"{seed}:{image_id}".encode()).digest(),
    )
    queues: list[tuple[str, list[str]]] = [("fixed_hash_sample", hash_ranked)]
    if baseline is not None and metric in METRIC_DIRECTIONS:
        candidates = sorted({record["model"] for record in records} - {baseline})
        for candidate in candidates:
            scored: list[tuple[float, str]] = []
            for image_id in all_ids:
                models = by_id[image_id]
                if baseline not in models or candidate not in models:
                    continue
                candidate_value = models[candidate].get(metric)
                baseline_value = models[baseline].get(metric)
                if not isinstance(candidate_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                    continue
                raw_difference = float(candidate_value) - float(baseline_value)
                improvement = (
                    raw_difference if METRIC_DIRECTIONS[metric] else -raw_difference
                )
                if math.isfinite(improvement):
                    scored.append((improvement, image_id))
            scored.sort(key=lambda item: (item[0], item[1]))
            queues.append((f"{candidate}_largest_loss", [image_id for _, image_id in scored]))
            queues.append((f"{candidate}_largest_gain", [image_id for _, image_id in reversed(scored)]))

    selected: list[str] = []
    reasons: dict[str, list[str]] = {}
    positions = [0] * len(queues)
    while len(selected) < min(count, len(all_ids)):
        progressed = False
        for queue_index, (reason, queue) in enumerate(queues):
            while positions[queue_index] < len(queue):
                image_id = queue[positions[queue_index]]
                positions[queue_index] += 1
                if image_id not in reasons:
                    selected.append(image_id)
                    reasons[image_id] = [reason]
                    progressed = True
                    break
                if reason not in reasons[image_id]:
                    reasons[image_id].append(reason)
            if len(selected) >= min(count, len(all_ids)):
                break
        if not progressed:
            break

    cases = []
    for image_id in selected:
        models = by_id[image_id]
        first = next(iter(models.values()))
        cases.append(
            {
                "id": image_id,
                "reasons": reasons[image_id],
                "gt_path": first["gt_path"],
                "predictions": {
                    model: {
                        "path": record["prediction_path"],
                        metric: record.get(metric),
                    }
                    for model, record in sorted(models.items())
                },
            }
        )
    return {
        "schema_version": 1,
        "selection_method": "round_robin_fixed_hash_and_candidate_extremes",
        "seed": seed,
        "metric": metric,
        "baseline": baseline,
        "requested_count": count,
        "selected_count": len(cases),
        "ordered_ids": selected,
        "cases": cases,
    }


def evaluate_saved_outputs(config: EvaluationConfig) -> dict[str, Any]:
    _assert_empty_output_destination(config)
    if not config.predictions:
        raise ValueError("at least one prediction is required")
    if config.crop_border < 0:
        raise ValueError("crop border cannot be negative")
    if config.color_space not in {"rgb", "y"}:
        raise ValueError("color space must be 'rgb' or 'y'")
    names = {spec.name for spec in config.predictions}
    if len(names) != len(config.predictions):
        raise ValueError("prediction names must be unique")
    if config.baseline is not None and config.baseline not in names:
        raise ValueError(f"baseline {config.baseline!r} is not a prediction name")
    if config.selection_metric not in METRIC_DIRECTIONS:
        raise ValueError(f"unsupported selection metric: {config.selection_metric}")
    if (config.arcface_backend is None) != (config.arcface_model is None):
        raise ValueError(
            "ArcFace requires both --arcface-backend and --arcface-model; the backend "
            "is never guessed from a filename"
        )
    if config.arcface_device not in {"cpu", "cuda"}:
        raise ValueError("ArcFace device must be 'cpu' or 'cuda'")
    if config.arcface_model is None and (
        config.arcface_device != "cpu" or config.arcface_confirm_gpu
    ):
        raise ValueError("ArcFace device/consent options require an explicit ArcFace model")
    if config.arcface_device == "cuda" and not config.arcface_confirm_gpu:
        raise RuntimeError(
            "ArcFace GPU execution requires both --arcface-device cuda and "
            "--confirm-arcface-gpu"
        )
    if config.arcface_device == "cpu" and config.arcface_confirm_gpu:
        raise ValueError("--confirm-arcface-gpu is only valid with --arcface-device cuda")
    if config.arcface_batch_size < 1:
        raise ValueError("ArcFace batch size must be positive")
    if config.lpips and not config.lpips_allow_model_downloads:
        raise RuntimeError(
            "LPIPS was requested without explicit model-download consent; set "
            "--lpips-allow-model-downloads after reviewing network/cache policy"
        )
    completion_flags = [spec.completion_record is not None for spec in config.predictions]
    if any(completion_flags) and not all(completion_flags):
        raise ValueError(
            "completion records are all-or-none: provide one explicit record per prediction"
        )

    keys, gt_index, prediction_indexes, pairing = _pair_inputs(config)
    gt_tree = tree_digest(config.gt_directory, gt_index.values())
    gt_provenance = {
        "root": str(config.gt_directory.resolve()),
        **gt_tree,
    }
    prediction_provenance: dict[str, Any] = {}
    for spec in config.predictions:
        paths = prediction_indexes[spec.name].values()
        if spec.completion_record is None:
            prediction_provenance[spec.name] = {
                "completion_record_status": "not_supplied",
                "outputs": {
                    "root": str(spec.directory.resolve()),
                    **tree_digest(spec.directory, paths),
                },
            }
        else:
            prediction_provenance[spec.name] = validate_completion_record(
                prediction_name=spec.name,
                prediction_directory=spec.directory,
                prediction_suffix=spec.filename_suffix,
                prediction_paths=paths,
                completion_record_path=spec.completion_record,
            )

    lpips_metric = None
    optional_metadata: dict[str, Any] = {
        "lpips": {"requested": config.lpips, "status": "not_requested"},
        "arcface": {"requested": config.arcface_model is not None, "status": "not_requested"},
    }
    if config.lpips:
        lpips_metric = LpipsMetric(
            network=config.lpips_network,
            calibration_weights=config.lpips_calibration_weights,
            allow_model_downloads=config.lpips_allow_model_downloads,
        )
        optional_metadata["lpips"] = {
            "requested": True,
            "status": "enabled",
            **lpips_metric.metadata,
        }
    arcface_metric = None
    if config.arcface_model is not None:
        assert config.arcface_backend is not None
        arcface_metric = create_arcface_metric(
            config.arcface_backend,
            config.arcface_model,
            device=config.arcface_device,
            confirm_gpu=config.arcface_confirm_gpu,
        )
        optional_metadata["arcface"] = {
            "requested": True,
            "status": "enabled",
            **arcface_metric.metadata,
        }
        preprocess_key = arcface_metric.preprocess_key(config.crop_border)
        cache_key = canonical_json_sha256(
            {
                "schema_version": 1,
                "model_key": arcface_metric.metadata["model_key"],
                "preprocess_key": preprocess_key,
                "ground_truth_tree_sha256": gt_tree["tree_sha256"],
                "evaluated_ids_sha256": hashlib.sha256(
                    ("\n".join(keys) + "\n").encode("utf-8")
                ).hexdigest(),
            }
        )
        optional_metadata["arcface"].update(
            {
                "preprocess_key": preprocess_key,
                "ground_truth_embedding_cache": {
                    "cache_key": cache_key,
                    "entry_key": "case-sensitive image ID",
                    "strategy": "per_evaluation_in_memory_one_embedding_per_gt",
                    "entries": len(keys),
                    "ground_truth_embedding_computations": len(keys),
                    "ground_truth_embedding_uses": len(keys) * len(config.predictions),
                    "ground_truth_embedding_reuses_avoided": len(keys)
                    * max(0, len(config.predictions) - 1),
                    "batch_size": config.arcface_batch_size,
                    "batching": (
                        f"native_bounded_{config.arcface_device}_batches"
                        if config.arcface_backend == "facexlib-pth"
                        else "conservative_sequential_calls_for_onnx_model_compatibility"
                    ),
                },
            }
        )

    dependency_versions = _dependency_versions()
    metric_protocol = {
        "psnr_ssim": "BasicSR 1.4.2-compatible NumPy definitions on uint8 decoded images",
        "color_space": config.color_space,
        "crop_border": config.crop_border,
        "sharpness": "variance of Laplacian on BasicSR Matlab-style Y; descriptive only",
        "edge_correlation": "Pearson correlation of Sobel gradient magnitudes on BasicSR Y",
        "optional_metric_crop": "same border crop, then LPIPS/ArcFace preprocessing",
    }
    source_directory = Path(__file__).resolve().parent
    protocol_descriptor = {
        "schema_version": 1,
        "metric_protocol": metric_protocol,
        "statistics": {
            "paired": True,
            "bootstrap_method": "deterministic_percentile",
            "bootstrap_samples": config.bootstrap_samples,
            "confidence": config.confidence,
            "seed": config.seed,
            "tie_tolerance": config.tie_tolerance,
            "pair_policy": config.pair_policy,
        },
        "optional_metrics": optional_metadata,
        "dependencies": dependency_versions,
        "evaluator_source_sha256": {
            name: sha256_file(source_directory / filename)
            for name, filename in {
                "core": "core.py",
                "metrics": "metrics.py",
                "provenance": "provenance.py",
                "statistics": "statistics.py",
            }.items()
        },
        "completion_record_policy": "optional_all_or_none_strict_schema_v3",
    }
    protocol_id = canonical_json_sha256(protocol_descriptor)
    evaluated_ids_sha256 = hashlib.sha256(
        ("\n".join(keys) + "\n").encode("utf-8")
    ).hexdigest()
    evaluation_descriptor = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "ground_truth": {
            "file_count": gt_tree["file_count"],
            "tree_sha256": gt_tree["tree_sha256"],
        },
        "evaluated_ids_sha256": evaluated_ids_sha256,
        "evaluated_common_count": len(keys),
        "pair_policy": config.pair_policy,
        "baseline": config.baseline,
        "predictions": [
            {
                "name": spec.name,
                "filename_suffix": spec.filename_suffix,
                "completion_record_sha256": prediction_provenance[spec.name].get(
                    "record_sha256"
                ),
                "checkpoint_sha256": prediction_provenance[spec.name]
                .get("checkpoint", {})
                .get("verified_sha256"),
                "config_sha256": prediction_provenance[spec.name]
                .get("config", {})
                .get("verified_sha256"),
                "file_count": prediction_provenance[spec.name]["outputs"]["file_count"],
                "tree_sha256": prediction_provenance[spec.name]["outputs"]["tree_sha256"],
            }
            for spec in config.predictions
        ],
    }
    evaluation_id = canonical_json_sha256(evaluation_descriptor)

    arcface_gt_embeddings: dict[str, np.ndarray] = {}
    arcface_prediction_embeddings: dict[str, dict[str, np.ndarray]] = {
        spec.name: {} for spec in config.predictions
    }
    if arcface_metric is not None:
        for start in range(0, len(keys), config.arcface_batch_size):
            batch_ids = keys[start : start + config.arcface_batch_size]
            gt_batch = arcface_metric.embeddings(
                [read_bgr(gt_index[image_id]) for image_id in batch_ids],
                config.crop_border,
                config.arcface_batch_size,
            )
            if gt_batch.shape[0] != len(batch_ids):
                raise RuntimeError("ArcFace backend returned the wrong GT batch length")
            arcface_gt_embeddings.update(zip(batch_ids, gt_batch, strict=True))
            for spec in config.predictions:
                prediction_batch = arcface_metric.embeddings(
                    [
                        read_bgr(prediction_indexes[spec.name][image_id])
                        for image_id in batch_ids
                    ],
                    config.crop_border,
                    config.arcface_batch_size,
                )
                if prediction_batch.shape[0] != len(batch_ids):
                    raise RuntimeError(
                        f"ArcFace backend returned the wrong batch length for {spec.name}"
                    )
                arcface_prediction_embeddings[spec.name].update(
                    zip(batch_ids, prediction_batch, strict=True)
                )
    records: list[dict[str, Any]] = []
    for image_id in keys:
        ground_truth = read_bgr(gt_index[image_id])
        for spec in config.predictions:
            prediction = read_bgr(prediction_indexes[spec.name][image_id])
            metrics = fidelity_metrics(
                prediction,
                ground_truth,
                crop_border=config.crop_border,
                color_space=config.color_space,
            )
            if lpips_metric is not None:
                metrics["lpips"] = lpips_metric(
                    prediction, ground_truth, config.crop_border
                )
            if arcface_metric is not None:
                metrics["arcface_identity_similarity"] = arcface_metric.similarity_from_embeddings(
                    arcface_prediction_embeddings[spec.name][image_id],
                    arcface_gt_embeddings[image_id],
                )
            records.append(
                {
                    "id": image_id,
                    "model": spec.name,
                    "prediction_path": str(prediction_indexes[spec.name][image_id].resolve()),
                    "gt_path": str(gt_index[image_id].resolve()),
                    "width": int(prediction.shape[1]),
                    "height": int(prediction.shape[0]),
                    **metrics,
                }
            )

    records_by_model = {
        name: {record["id"]: record for record in records if record["model"] == name}
        for name in names
    }
    metric_names = [
        name
        for name in METRIC_DIRECTIONS
        if any(name in record for record in records)
    ]
    if config.selection_json is not None and config.selection_metric not in metric_names:
        raise ValueError(
            f"selection metric {config.selection_metric!r} was not computed; enable its "
            "optional metric or choose an available metric"
        )
    if config.baseline is not None:
        baseline_records = records_by_model[config.baseline]
        for record in records:
            if record["model"] == config.baseline:
                continue
            baseline_record = baseline_records[record["id"]]
            record["baseline_model"] = config.baseline
            for metric_name in metric_names:
                record[f"{metric_name}_difference_vs_baseline"] = (
                    record[metric_name] - baseline_record[metric_name]
                )

    aggregate_models: dict[str, Any] = {}
    descriptive_names = metric_names + [
        "sharpness_laplacian_variance",
        "gt_sharpness_laplacian_variance",
    ]
    for name in sorted(names):
        model_records = records_by_model[name]
        aggregate_models[name] = {
            metric_name: summarize(
                [model_records[image_id][metric_name] for image_id in keys]
            )
            for metric_name in descriptive_names
            if metric_name in next(iter(model_records.values()))
        }

    comparisons: dict[str, Any] = {}
    if config.baseline is not None:
        for candidate_name in sorted(names - {config.baseline}):
            comparison_name = f"{candidate_name}_vs_{config.baseline}"
            comparisons[comparison_name] = {}
            for metric_name in metric_names:
                comparisons[comparison_name][metric_name] = paired_comparison(
                    [records_by_model[candidate_name][image_id][metric_name] for image_id in keys],
                    [records_by_model[config.baseline][image_id][metric_name] for image_id in keys],
                    higher_is_better=METRIC_DIRECTIONS[metric_name],
                    tie_tolerance=config.tie_tolerance,
                    bootstrap_samples=config.bootstrap_samples,
                    confidence=config.confidence,
                    seed=_stable_seed(config.seed, candidate_name, metric_name),
                )

    # Catch prediction/GT mutation while metrics were running before sealing results.
    if tree_digest(config.gt_directory, gt_index.values()) != gt_tree:
        raise RuntimeError("ground-truth file tree changed during evaluation")
    for spec in config.predictions:
        current_tree = tree_digest(spec.directory, prediction_indexes[spec.name].values())
        output_fields = prediction_provenance[spec.name]["outputs"]
        expected_tree = {
            key: output_fields[key]
            for key in (
                "algorithm",
                "file_count",
                "files_manifest_sha256",
                "tree_sha256",
            )
        }
        if current_tree != expected_tree:
            raise RuntimeError(f"prediction file tree changed during evaluation: {spec.name}")

    aggregate = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identifiers": {
            "protocol_id": protocol_id,
            "evaluation_id": evaluation_id,
        },
        "protocol_descriptor": protocol_descriptor,
        "metric_protocol": metric_protocol,
        "dependencies": dependency_versions,
        "optional_metrics": optional_metadata,
        "provenance": {
            "ground_truth": gt_provenance,
            "predictions": prediction_provenance,
            "evaluation_descriptor": evaluation_descriptor,
        },
        "configuration": {
            **asdict(config),
            "gt_directory": str(config.gt_directory.resolve()),
            "output_directory": str(config.output_directory.resolve()),
            "predictions": [
                {
                    "name": spec.name,
                    "directory": str(spec.directory.resolve()),
                    "filename_suffix": spec.filename_suffix,
                    "completion_record": (
                        str(spec.completion_record.resolve())
                        if spec.completion_record is not None
                        else None
                    ),
                }
                for spec in config.predictions
            ],
        },
        "pairing": pairing,
        "models": aggregate_models,
        "comparisons": comparisons,
    }
    _write_outputs(records, aggregate, config.output_directory)
    if config.selection_json is not None:
        selection = _make_selection(
            records,
            baseline=config.baseline,
            count=config.selection_count,
            metric=config.selection_metric,
            seed=config.seed,
        )
        selection["identifiers"] = {
            "protocol_id": protocol_id,
            "evaluation_id": evaluation_id,
        }
        config.selection_json.parent.mkdir(parents=True, exist_ok=True)
        with config.selection_json.open("x", encoding="utf-8") as handle:
            json.dump(_json_safe(selection), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    return aggregate
