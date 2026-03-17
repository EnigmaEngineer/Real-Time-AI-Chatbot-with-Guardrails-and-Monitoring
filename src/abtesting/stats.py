"""Statistical significance testing for A/B experiments."""

import math
from dataclasses import dataclass

from src.abtesting.router import ExperimentRecord


@dataclass
class ABTestResult:
    experiment: str
    variant_a: str
    variant_b: str
    metric: str
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int
    t_statistic: float
    p_value: float
    significant: bool
    recommended_winner: str | None


def welch_t_test(values_a: list[float], values_b: list[float]) -> tuple[float, float]:
    """Two-sample Welch's t-test (unequal variances). Returns (t_statistic, p_value)."""
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = sum(values_a) / n_a
    mean_b = sum(values_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in values_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in values_b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-10
    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    p_value = _t_distribution_p_value(abs(t_stat), df)
    return t_stat, p_value


def _t_distribution_p_value(t: float, df: float) -> float:
    """Approximate two-tailed p-value from t-distribution using normal approximation for large df."""
    # For df > 30, normal approximation is reasonable
    if df > 30:
        return 2.0 * _normal_cdf(-abs(t))
    # Simple approximation for smaller df
    x = df / (df + t * t)
    p = _incomplete_beta(df / 2.0, 0.5, x)
    return p


def _normal_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function (approximation via continued fraction)."""
    if x < 0 or x > 1:
        return 0.0
    if x == 0 or x == 1:
        return x

    # Use series expansion for small x
    result = 0.0
    term = 1.0
    for n in range(200):
        if n > 0:
            term *= (n - b) * x / n if n > 0 else x
        coeff = 1.0
        for k in range(n):
            coeff *= (a + k) / (a + b + k)
        contribution = coeff * term / (a + n)
        result += contribution
        if abs(contribution) < 1e-12:
            break

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    prefix = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) if x > 0 and (1 - x) > 0 else 0
    return min(max(result * prefix, 0.0), 1.0)


def analyze_experiment(
    records: list[ExperimentRecord],
    metric: str = "feedback",
    significance_level: float = 0.05,
) -> list[ABTestResult]:
    """Analyze an A/B experiment across all variant pairs."""
    by_variant: dict[str, list[ExperimentRecord]] = {}
    for r in records:
        by_variant.setdefault(r.variant, []).append(r)

    variant_names = sorted(by_variant.keys())
    results = []

    for i, va in enumerate(variant_names):
        for vb in variant_names[i + 1 :]:
            values_a = _extract_metric(by_variant[va], metric)
            values_b = _extract_metric(by_variant[vb], metric)

            if not values_a or not values_b:
                continue

            t_stat, p_val = welch_t_test(values_a, values_b)
            mean_a = sum(values_a) / len(values_a)
            mean_b = sum(values_b) / len(values_b)
            significant = p_val < significance_level

            winner = None
            if significant:
                # For feedback: higher is better; for latency: lower is better
                if metric == "latency_ms":
                    winner = va if mean_a < mean_b else vb
                else:
                    winner = va if mean_a > mean_b else vb

            experiment_name = records[0].experiment if records else "unknown"
            results.append(
                ABTestResult(
                    experiment=experiment_name,
                    variant_a=va,
                    variant_b=vb,
                    metric=metric,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    n_a=len(values_a),
                    n_b=len(values_b),
                    t_statistic=t_stat,
                    p_value=p_val,
                    significant=significant,
                    recommended_winner=winner,
                )
            )

    return results


def _extract_metric(records: list[ExperimentRecord], metric: str) -> list[float]:
    if metric == "feedback":
        return [float(r.feedback) for r in records if r.feedback is not None]
    if metric == "latency_ms":
        return [r.latency_ms for r in records]
    if metric == "token_count":
        return [float(r.token_count) for r in records]
    return []
