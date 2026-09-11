"""Deterministic statistical primitives shared by experiment metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

Z_95 = 1.959963984540054


def rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def nearest_rank(values: Iterable[int | float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    estimate = successes / total
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / total
    centre = (estimate + z2 / (2 * total)) / denominator
    margin = (
        Z_95 * math.sqrt(estimate * (1 - estimate) / total + z2 / (4 * total * total)) / denominator
    )
    return {"lower": max(0.0, centre - margin), "upper": min(1.0, centre + margin)}


def proportion(successes: int, total: int) -> dict[str, Any]:
    return {
        "numerator": successes,
        "denominator": total,
        "estimate": rate(successes, total),
        "confidence_interval_95": wilson_interval(successes, total),
    }


def kaplan_meier(
    observations: list[dict[str, Any]],
    *,
    horizon: int | None = None,
    time_key: str = "time",
    event_key: str = "event_observed",
) -> dict[str, Any]:
    """Kaplan-Meier curve, Greenwood intervals, KM median, and RMST."""
    if not observations:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": "no observations",
            "sample_size": 0,
            "events": 0,
            "right_censored": 0,
            "curve": [],
            "fixed_window": None,
            "km_median": {"estimate": None, "status": "NOT_REACHED"},
            "rmst": None,
        }
    normalized = [
        {"time": max(0, int(row[time_key])), "event": bool(row[event_key])} for row in observations
    ]
    survival = 1.0
    greenwood_sum = 0.0
    curve: list[dict[str, Any]] = []
    for time in sorted({row["time"] for row in normalized if row["time"] > 0}):
        at_risk = sum(row["time"] >= time for row in normalized)
        events = sum(row["time"] == time and row["event"] for row in normalized)
        censored = sum(row["time"] == time and not row["event"] for row in normalized)
        if events:
            survival *= 1 - events / at_risk
            if at_risk > events:
                greenwood_sum += events / (at_risk * (at_risk - events))
        variance = survival * survival * greenwood_sum if survival else 0.0
        standard_error = math.sqrt(variance)
        curve.append(
            {
                "time": time,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival": survival,
                "greenwood_variance": variance,
                "confidence_interval_95": {
                    "lower": max(0.0, survival - Z_95 * standard_error),
                    "upper": min(1.0, survival + Z_95 * standard_error),
                },
            }
        )
    maximum_follow_up = max(row["time"] for row in normalized)
    requested_horizon = int(horizon) if horizon is not None else maximum_follow_up
    estimable = requested_horizon <= maximum_follow_up
    median_time = next((point["time"] for point in curve if point["survival"] <= 0.5), None)
    fixed_window = None
    rmst_result = None
    if estimable:
        point = {
            "survival": 1.0,
            "greenwood_variance": 0.0,
            "confidence_interval_95": {"lower": 1.0, "upper": 1.0},
        }
        for candidate in curve:
            if candidate["time"] > requested_horizon:
                break
            point = candidate
        interval = point["confidence_interval_95"]
        fixed_window = {
            "horizon": requested_horizon,
            "survival_probability": {
                "estimate": point["survival"],
                "greenwood_variance": point["greenwood_variance"],
                "confidence_interval_95": interval,
            },
            "cumulative_failure_probability": {
                "estimate": 1 - point["survival"],
                "greenwood_variance": point["greenwood_variance"],
                "confidence_interval_95": {
                    "lower": 1 - interval["upper"],
                    "upper": 1 - interval["lower"],
                },
            },
        }
        rmst_result = _rmst(curve, requested_horizon)
    return {
        "status": "COMPLETED" if estimable else "HORIZON_NOT_ESTIMABLE",
        "reason": None if estimable else "requested horizon exceeds maximum follow-up",
        "sample_size": len(normalized),
        "events": sum(row["event"] for row in normalized),
        "right_censored": sum(not row["event"] for row in normalized),
        "requested_horizon": requested_horizon,
        "maximum_observed_follow_up": maximum_follow_up,
        "curve": curve,
        "fixed_window": fixed_window,
        "km_median": {
            "estimate": median_time,
            "status": "ESTIMATED" if median_time is not None else "NOT_REACHED",
        },
        "rmst": rmst_result,
    }


def _rmst(curve: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    area = 0.0
    previous_time = 0
    previous_survival = 1.0
    for point in curve:
        bounded = min(int(point["time"]), horizon)
        area += previous_survival * max(0, bounded - previous_time)
        previous_time = bounded
        previous_survival = float(point["survival"])
        if point["time"] >= horizon:
            break
    area += previous_survival * max(0, horizon - previous_time)

    variance = 0.0
    for event_point in (point for point in curve if point["events"] and point["time"] <= horizon):
        event_time = int(event_point["time"])
        tail = 0.0
        previous_time = event_time
        previous_survival = float(event_point["survival"])
        for point in curve:
            if point["time"] <= event_time:
                continue
            bounded = min(int(point["time"]), horizon)
            tail += previous_survival * max(0, bounded - previous_time)
            previous_time = bounded
            previous_survival = float(point["survival"])
            if point["time"] >= horizon:
                break
        tail += previous_survival * max(0, horizon - previous_time)
        at_risk = int(event_point["at_risk"])
        events = int(event_point["events"])
        if at_risk > events:
            variance += tail * tail * events / (at_risk * (at_risk - events))
    standard_error = math.sqrt(variance)
    return {
        "horizon": horizon,
        "estimate": area,
        "greenwood_variance": variance,
        "confidence_interval_95": {
            "lower": max(0.0, area - Z_95 * standard_error),
            "upper": min(float(horizon), area + Z_95 * standard_error),
        },
    }


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    left, right = average_ranks(x), average_ranks(y)
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    cluster_key: str,
    statistic: Callable[[list[dict[str, Any]]], float | None],
    seed: int = 0,
    replicates: int = 1000,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[cluster_key])].append(row)
    keys = sorted(groups)
    if not keys:
        return {
            "replicates_requested": replicates,
            "replicates_valid": 0,
            "confidence_interval_95": None,
        }
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled: list[dict[str, Any]] = []
        for key in (rng.choice(keys) for _ in keys):
            sampled.extend(groups[key])
        value = statistic(sampled)
        if value is not None and math.isfinite(value):
            estimates.append(float(value))
    return {
        "replicates_requested": replicates,
        "replicates_valid": len(estimates),
        "confidence_interval_95": (
            {"lower": percentile(estimates, 0.025), "upper": percentile(estimates, 0.975)}
            if estimates
            else None
        ),
    }


def binary_probability_metrics(
    rows: list[dict[str, Any]],
    *,
    truth_key: str,
    probability_key: str,
    weight_key: str | None = None,
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get(truth_key) in (0, 1)
        and not isinstance(row.get(truth_key), bool)
        and isinstance(row.get(probability_key), (int, float))
        and not isinstance(row.get(probability_key), bool)
        and row.get("schema_valid", True)
        and math.isfinite(float(row[probability_key]))
        and 0 <= float(row[probability_key]) <= 1
    ]
    weights = [float(row.get(weight_key, 1.0)) if weight_key else 1.0 for row in valid]
    total_weight = sum(weights)
    positives = sum(
        weight * int(row[truth_key]) for row, weight in zip(valid, weights, strict=True)
    )
    prevalence = positives / total_weight if total_weight else None
    brier = (
        sum(
            weight * (float(row[probability_key]) - int(row[truth_key])) ** 2
            for row, weight in zip(valid, weights, strict=True)
        )
        / total_weight
        if total_weight
        else None
    )
    bins = []
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        members = [
            (row, weight)
            for row, weight in zip(valid, weights, strict=True)
            if low <= float(row[probability_key]) < high
            or (index == 9 and float(row[probability_key]) == 1.0)
        ]
        weight = sum(item[1] for item in members)
        mean_probability = (
            sum(item_weight * float(row[probability_key]) for row, item_weight in members) / weight
            if weight
            else None
        )
        observed_rate = (
            sum(item_weight * int(row[truth_key]) for row, item_weight in members) / weight
            if weight
            else None
        )
        if weight:
            ece += weight / total_weight * abs(mean_probability - observed_rate)
        bins.append(
            {
                "lower": low,
                "upper": high,
                "samples": len(members),
                "weight": weight,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    labels = {int(row[truth_key]) for row in valid}
    return {
        "status": "COMPLETED" if len(labels) == 2 else "INSUFFICIENT_CLASS_SUPPORT",
        "samples": len(rows),
        "valid_samples": len(valid),
        "invalid_samples": len(rows) - len(valid),
        "prevalence": prevalence,
        "auprc": _average_precision(valid, weights, truth_key, probability_key)
        if len(labels) == 2
        else None,
        "auroc": _auroc(valid, weights, truth_key, probability_key) if len(labels) == 2 else None,
        "brier_score": brier,
        "ece_10_equal_width": ece if total_weight else None,
        "nonempty_calibration_bins": sum(bool(item["samples"]) for item in bins),
        "calibration_bins": bins,
        "parse_failure_rate": rate(len(rows) - len(valid), len(rows)),
    }


def _average_precision(
    rows: list[dict[str, Any]],
    weights: list[float],
    truth_key: str,
    probability_key: str,
) -> float:
    positives = sum(weight * int(row[truth_key]) for row, weight in zip(rows, weights, strict=True))
    grouped: dict[float, list[tuple[int, float]]] = defaultdict(list)
    for row, weight in zip(rows, weights, strict=True):
        grouped[float(row[probability_key])].append((int(row[truth_key]), weight))
    true_positive = false_positive = 0.0
    previous_recall = area = 0.0
    for score in sorted(grouped, reverse=True):
        true_positive += sum(weight * label for label, weight in grouped[score])
        false_positive += sum(weight * (1 - label) for label, weight in grouped[score])
        recall = true_positive / positives
        precision_value = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision_value
        previous_recall = recall
    return area


def _auroc(
    rows: list[dict[str, Any]],
    weights: list[float],
    truth_key: str,
    probability_key: str,
) -> float:
    positives = [
        (float(row[probability_key]), weight)
        for row, weight in zip(rows, weights, strict=True)
        if int(row[truth_key]) == 1
    ]
    negatives = [
        (float(row[probability_key]), weight)
        for row, weight in zip(rows, weights, strict=True)
        if int(row[truth_key]) == 0
    ]
    denominator = sum(weight for _, weight in positives) * sum(weight for _, weight in negatives)
    concordance = 0.0
    for positive_score, positive_weight in positives:
        for negative_score, negative_weight in negatives:
            comparison = (
                1.0
                if positive_score > negative_score
                else 0.5
                if positive_score == negative_score
                else 0.0
            )
            concordance += positive_weight * negative_weight * comparison
    return concordance / denominator
