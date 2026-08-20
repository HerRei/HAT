"""Deterministic, dependency-light summary statistics."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Any


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for sorted finite values."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Percentile bootstrap CI for a mean, using a deterministic local RNG."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "method": "paired_percentile_bootstrap",
            "confidence": confidence,
            "samples": samples,
            "finite_pair_count": 0,
            "low": None,
            "high": None,
        }
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    rng = random.Random(seed)
    count = len(finite)
    bootstrap_means = [
        sum(finite[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "paired_percentile_bootstrap",
        "confidence": confidence,
        "samples": samples,
        "finite_pair_count": count,
        "low": percentile(bootstrap_means, alpha),
        "high": percentile(bootstrap_means, 1.0 - alpha),
    }


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """Summarize finite values while reporting all excluded non-finite values."""
    converted = [float(value) for value in values]
    finite = [value for value in converted if math.isfinite(value)]
    result: dict[str, Any] = {
        "count": len(converted),
        "finite_count": len(finite),
        "nonfinite_count": len(converted) - len(finite),
        "mean": None,
        "std_population": None,
        "median": None,
        "min": None,
        "max": None,
    }
    if finite:
        result.update(
            mean=statistics.fmean(finite),
            std_population=statistics.pstdev(finite),
            median=statistics.median(finite),
            min=min(finite),
            max=max(finite),
        )
    return result


def paired_comparison(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    higher_is_better: bool,
    tie_tolerance: float,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Compare aligned per-image measurements without an independence assumption."""
    if len(candidate) != len(baseline):
        raise ValueError("paired comparison inputs must have equal lengths")
    pairs = [
        (float(candidate_value), float(baseline_value))
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
        if math.isfinite(float(candidate_value)) and math.isfinite(float(baseline_value))
    ]
    raw_differences = [candidate_value - baseline_value for candidate_value, baseline_value in pairs]
    improvements = raw_differences if higher_is_better else [-value for value in raw_differences]
    wins = sum(value > tie_tolerance for value in improvements)
    losses = sum(value < -tie_tolerance for value in improvements)
    ties = len(improvements) - wins - losses
    return {
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "total_pair_count": len(candidate),
        "finite_pair_count": len(pairs),
        "nonfinite_pair_count": len(candidate) - len(pairs),
        "mean_candidate_minus_baseline": statistics.fmean(raw_differences) if raw_differences else None,
        "mean_improvement": statistics.fmean(improvements) if improvements else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate_excluding_ties": wins / (wins + losses) if wins + losses else None,
        "win_rate_all_pairs": wins / len(improvements) if improvements else None,
        "tie_tolerance": tie_tolerance,
        "candidate_minus_baseline_ci": bootstrap_mean_ci(
            raw_differences,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        ),
    }
