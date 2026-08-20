#!/usr/bin/env python3
"""Create deterministic EMA weight interpolations between compatible checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


TensorMap = Mapping[str, torch.Tensor]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state(payload: Any, param_key: str, source: Path) -> TensorMap:
    if not isinstance(param_key, str) or not param_key:
        raise ValueError("param_key must be a non-empty string")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint {source} is not a mapping.")
    state = payload.get(param_key)
    if not isinstance(state, Mapping) or not state:
        available = ", ".join(sorted(str(key) for key in payload))
        raise KeyError(
            f"Checkpoint {source} has no non-empty mapping key {param_key!r}; "
            f"available: {available}"
        )
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Checkpoint {source} key {param_key!r} must map string names only "
                f"to tensors; invalid entry {key!r} ({type(value).__name__})"
            )
    return state


def interpolate_states(base: TensorMap, tuned: TensorMap, alpha: float) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if set(base) != set(tuned):
        missing = sorted(set(base) - set(tuned))
        extra = sorted(set(tuned) - set(base))
        raise ValueError(f"State keys differ; missing={missing[:5]}, extra={extra[:5]}")

    blended: dict[str, torch.Tensor] = {}
    for key in sorted(base):
        base_tensor = base[key].detach().cpu()
        tuned_tensor = tuned[key].detach().cpu()
        if base_tensor.shape != tuned_tensor.shape or base_tensor.dtype != tuned_tensor.dtype:
            raise ValueError(
                f"Tensor {key!r} differs: base={base_tensor.shape}/{base_tensor.dtype}, "
                f"tuned={tuned_tensor.shape}/{tuned_tensor.dtype}"
            )
        if base_tensor.is_floating_point() or base_tensor.is_complex():
            blended[key] = torch.lerp(base_tensor, tuned_tensor, alpha)
        else:
            if not torch.equal(base_tensor, tuned_tensor):
                raise ValueError(f"Non-floating tensor {key!r} differs between checkpoints.")
            blended[key] = base_tensor.clone()
    return blended


def alpha_label(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def write_interpolation(
    base_path: Path,
    tuned_path: Path,
    output_path: Path,
    alpha: float,
    param_key: str,
) -> dict[str, Any]:
    base_payload = torch.load(base_path, map_location="cpu", weights_only=True)
    tuned_payload = torch.load(tuned_path, map_location="cpu", weights_only=True)
    base_state = extract_state(base_payload, param_key, base_path)
    tuned_state = extract_state(tuned_payload, param_key, tuned_path)
    blended = interpolate_states(base_state, tuned_state, alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    payload = {
        "params_ema": blended,
        "recovery_meta": {
            "type": "linear_weight_interpolation",
            "alpha": alpha,
            "base_path": str(base_path.resolve()),
            "base_sha256": sha256_file(base_path),
            "tuned_path": str(tuned_path.resolve()),
            "tuned_sha256": sha256_file(tuned_path),
            "source_param_key": param_key,
        },
    }
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return {
        "alpha": alpha,
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "tensor_count": len(blended),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--prefix", default="base_95k_interp")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = args.base.resolve()
    tuned_path = args.tuned.resolve()
    output_dir = args.output_dir.resolve()
    results: list[dict[str, Any]] = []

    for alpha in args.alpha:
        output_path = output_dir / f"{args.prefix}_a{alpha_label(alpha)}.pth"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite explicitly.")
        results.append(write_interpolation(base_path, tuned_path, output_path, alpha, args.param_key))

    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary = {
        "base": str(base_path),
        "base_sha256": sha256_file(base_path),
        "tuned": str(tuned_path),
        "tuned_sha256": sha256_file(tuned_path),
        "source_param_key": args.param_key,
        "outputs": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
