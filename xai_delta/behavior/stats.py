"""Exact McNemar tests, Holm correction and paired bootstrap confidence intervals
for the binary behavioral metric B(M, x)."""
import numpy as np
from scipy.stats import beta, binomtest, bootstrap


def clopper_pearson_ci(k, n, confidence_level=0.95):
    """Exact binomial confidence interval for a single proportion k/n."""
    alpha = 1 - confidence_level
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def contingency_2x2(y_a, y_b):
    """Paired 2x2 table of binary outcomes; n_ij counts y_a = i and y_b = j."""
    y_a, y_b = np.asarray(y_a), np.asarray(y_b)
    return {
        "n00": int(np.sum((y_a == 0) & (y_b == 0))),
        "n01": int(np.sum((y_a == 0) & (y_b == 1))),
        "n10": int(np.sum((y_a == 1) & (y_b == 0))),
        "n11": int(np.sum((y_a == 1) & (y_b == 1))),
    }


def exact_mcnemar(y_a, y_b, alternative="two-sided"):
    """Exact McNemar test: binomial test on the discordant pairs.
    alternative="greater" tests whether (y_a=1, y_b=0) pairs dominate."""
    table = contingency_2x2(y_a, y_b)
    n10, n01 = table["n10"], table["n01"]
    n_discordant = n10 + n01
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(n10, n_discordant, 0.5, alternative=alternative).pvalue)
    return {**table, "n_discordant": n_discordant, "statistic": n10, "p_value": p_value,
            "alternative": alternative}


def holm_correction(p_values):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p_values)):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def paired_bootstrap_ci(arrays, statistic_fn, n_resamples=10_000, seed=42,
                        method="BCa", confidence_level=0.95):
    """Paired bootstrap CI, resampling prompts (rows) jointly across all arrays.
    Falls back to the percentile method when BCa is degenerate."""
    rng = np.random.default_rng(seed)
    kwargs = dict(n_resamples=n_resamples, paired=True, confidence_level=confidence_level, rng=rng)
    try:
        res = bootstrap(arrays, statistic_fn, method=method, **kwargs)
        used = method
    except Exception:
        res = bootstrap(arrays, statistic_fn, method="percentile", **kwargs)
        used = "percentile"
    return {
        "point_estimate": float(statistic_fn(*arrays)),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "method": used,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def mean_diff(y_a, y_b, axis=-1):
    """mean(y_a) - mean(y_b); vectorized for paired_bootstrap_ci."""
    return np.mean(y_a, axis=axis) - np.mean(y_b, axis=axis)
