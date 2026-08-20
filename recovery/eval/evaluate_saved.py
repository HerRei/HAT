"""Command-line entry point for saved-output evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import EvaluationConfig, PredictionSpec, evaluate_saved_outputs


def _name_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, parsed_value = value.split("=", 1)
    if not name or not parsed_value:
        raise argparse.ArgumentTypeError("NAME and VALUE must both be non-empty")
    return name, parsed_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate existing SR output directories without running inference."
    )
    parser.add_argument("--gt", required=True, type=Path, help="ground-truth image directory")
    parser.add_argument(
        "--prediction",
        required=True,
        action="append",
        type=_name_value,
        metavar="NAME=DIR",
        help="named prediction directory; repeat for each model",
    )
    parser.add_argument(
        "--prediction-suffix",
        action="append",
        default=[],
        type=_name_value,
        metavar="NAME=SUFFIX",
        help="required filename suffix to filter/remove for one prediction",
    )
    parser.add_argument(
        "--completion-record",
        action="append",
        default=[],
        type=_name_value,
        metavar="NAME=JSON",
        help=(
            "inference v3 completion record for one prediction; when supplied, repeat "
            "exactly once for every prediction"
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="new or empty result directory; nonempty directories are never overwritten",
    )
    parser.add_argument("--crop-border", type=int, default=4)
    parser.add_argument("--color-space", choices=("rgb", "y"), default="y")
    parser.add_argument("--pair-policy", choices=("strict", "intersection"), default="strict")
    parser.add_argument("--baseline", help="prediction name used for paired comparisons")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--lpips", action="store_true", help="request LPIPS; failure is fatal")
    parser.add_argument("--lpips-network", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--lpips-calibration-weights", type=Path)
    parser.add_argument(
        "--lpips-allow-model-downloads",
        action="store_true",
        help="explicitly allow upstream LPIPS/torchvision backbone downloads",
    )
    parser.add_argument(
        "--arcface-backend",
        choices=("facexlib-pth", "onnx"),
        help="required explicit backend when --arcface-model is supplied",
    )
    parser.add_argument(
        "--arcface-model",
        type=Path,
        help="explicit local ArcFace .pth or ONNX weight file; aligned faces only",
    )
    parser.add_argument(
        "--arcface-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="explicit identity-model device; never inferred (default: cpu)",
    )
    parser.add_argument(
        "--confirm-arcface-gpu",
        action="store_true",
        help="required consent in addition to --arcface-device cuda",
    )
    parser.add_argument(
        "--arcface-batch-size",
        type=int,
        default=16,
        help="bounded embedding batch size (facexlib backend; default: 16)",
    )
    parser.add_argument("--selection-json", type=Path, help="write deterministic contact-sheet case list")
    parser.add_argument("--selection-count", type=int, default=24)
    parser.add_argument(
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
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    suffixes = dict(args.prediction_suffix)
    completion_records = dict(args.completion_record)
    if len(completion_records) != len(args.completion_record):
        raise ValueError("completion record names must be unique")
    predictions = tuple(
        PredictionSpec(
            name,
            Path(directory),
            suffixes.get(name, ""),
            Path(completion_records[name]) if name in completion_records else None,
        )
        for name, directory in args.prediction
    )
    unknown_suffixes = set(suffixes) - {spec.name for spec in predictions}
    if unknown_suffixes:
        raise ValueError(
            "suffix supplied for unknown prediction(s): " + ", ".join(sorted(unknown_suffixes))
        )
    unknown_records = set(completion_records) - {spec.name for spec in predictions}
    if unknown_records:
        raise ValueError(
            "completion record supplied for unknown prediction(s): "
            + ", ".join(sorted(unknown_records))
        )
    return EvaluationConfig(
        gt_directory=args.gt,
        predictions=predictions,
        output_directory=args.out,
        crop_border=args.crop_border,
        color_space=args.color_space,
        pair_policy=args.pair_policy,
        baseline=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
        tie_tolerance=args.tie_tolerance,
        lpips=args.lpips,
        lpips_network=args.lpips_network,
        lpips_calibration_weights=args.lpips_calibration_weights,
        lpips_allow_model_downloads=args.lpips_allow_model_downloads,
        arcface_backend=args.arcface_backend,
        arcface_model=args.arcface_model,
        arcface_device=args.arcface_device,
        arcface_confirm_gpu=args.confirm_arcface_gpu,
        arcface_batch_size=args.arcface_batch_size,
        selection_json=args.selection_json,
        selection_count=args.selection_count,
        selection_metric=args.selection_metric,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        aggregate = evaluate_saved_outputs(config_from_args(args))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"evaluation error: {exc}\n")
    print(
        json.dumps(
            {
                "aggregate": str((args.out / "aggregate.json").resolve()),
                "evaluated_common_count": aggregate["pairing"]["evaluated_common_count"],
                "evaluation_id": aggregate["identifiers"]["evaluation_id"],
                "models": sorted(aggregate["models"]),
                "protocol_id": aggregate["identifiers"]["protocol_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
