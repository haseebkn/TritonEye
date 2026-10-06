"""Conditional uncertainty summaries for NL research evaluation.

Binomial and Poisson intervals assume independent events; block resampling
keeps shared acquisition/geographic groups together. Neither establishes
province-wide or operational performance from a purposive pilot.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import beta, chi2


def _alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def binomial_interval(
    successes: int, total: int, alpha: float = 0.05
) -> list[float] | None:
    """Two-sided Clopper-Pearson interval; an empty denominator is unmeasured."""
    _alpha(alpha)
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("Counts must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("Counts must be integers")
    if not 0 <= successes <= total:
        raise ValueError("Counts must satisfy 0 <= successes <= total")
    if not total:
        return None
    lower = (
        0.0
        if not successes
        else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    )
    return [lower, upper]


def poisson_interval(
    events: int, exposure: float, alpha: float = 0.05
) -> list[float] | None:
    """Exact Poisson interval; zero events still has a positive upper bound."""
    _alpha(alpha)
    if isinstance(events, bool) or not isinstance(events, int) or events < 0:
        raise ValueError("Nonnegative integer events required")
    if not math.isfinite(exposure) or exposure < 0:
        raise ValueError("Finite nonnegative exposure required")
    if not exposure:
        if events:
            raise ValueError("Events without exposure")
        return None
    lower = (
        0.0 if not events else float(chi2.ppf(alpha / 2, 2 * events) / (2 * exposure))
    )
    upper = float(chi2.ppf(1 - alpha / 2, 2 * (events + 1)) / (2 * exposure))
    return [lower, upper]


def dependence_blocks(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Connected components: sharing any acquisition or geography joins whole ROIs."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(rows):
        for j, other in enumerate(rows[:i]):
            if (
                row["acquisition_group"] == other["acquisition_group"]
                or row["geographic_group"] == other["geographic_group"]
            ):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def block_interval(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    alpha: float = 0.05,
    seed: int = 20261005,
    resamples: int = 2000,
    min_blocks: int = 5,
) -> dict[str, Any]:
    """Paired percentile resampling of whole connected blocks; no IID-pixel fiction."""
    _alpha(alpha)
    if min_blocks < 2 or resamples < 100:
        raise ValueError("At least two blocks and 100 resamples required")
    blocks = dependence_blocks(rows)
    result: dict[str, Any] = {
        "method": "paired connected acquisition/geography block percentile bootstrap",
        "blocks": len(blocks),
        "seed": seed,
        "resamples": resamples,
        "interval": None,
        "reason": None,
    }
    if len(blocks) < min_blocks:
        result["reason"] = "insufficient independent blocks"
        return result
    totals = np.array(
        [
            [
                sum(float(rows[i][numerator]) for i in block),
                sum(float(rows[i][denominator]) for i in block),
            ]
            for block in blocks
        ]
    )
    if not np.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("Finite nonnegative block counts/exposures required")
    rng = np.random.default_rng(seed)
    sums = totals[rng.integers(0, len(blocks), (resamples, len(blocks)))].sum(axis=1)
    valid = sums[:, 1] > 0
    result["valid_resamples"] = int(valid.sum())
    if not valid.all():
        result["reason"] = "some resamples have an undefined denominator"
        return result
    values = sums[:, 0] / sums[:, 1]
    if np.ptp(values) == 0:
        result["reason"] = "degenerate empirical distribution; no uncertainty claim"
        return result
    result["interval"] = [
        float(x) for x in np.quantile(values, [alpha / 2, 1 - alpha / 2])
    ]
    return result
