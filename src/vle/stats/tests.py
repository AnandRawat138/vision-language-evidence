"""Statistical analysis utilities (bootstrap CIs, paired tests, effect sizes, Holm correction)."""

from __future__ import annotations

import numpy as np
from scipy import stats

DEFAULT_B = 10_000
SEED = 2026


def describe(x: np.ndarray | list[float]) -> dict:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(x)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "ci95_lo": lo, "ci95_hi": hi}


def bootstrap_ci(x: np.ndarray, stat=np.mean, B: int = DEFAULT_B, alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        v = float(stat(x)) if len(x) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = stat(x[idx], axis=1) if stat in (np.mean, np.median) else np.array([stat(x[i]) for i in idx])
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, B: int = DEFAULT_B, seed: int = SEED) -> tuple[float, float, float]:
    """CI for a ratio of sums (e.g. micro-F1 components, CHAIR_i) by resampling images."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / max(den.sum(), 1e-12)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, B: int = DEFAULT_B, seed: int = SEED) -> dict:
    """Mean difference a-b with bootstrap CI and two-sided bootstrap p-value (shifted-null)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(1)
    obs = d.mean()
    null = bs - obs
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"mean_diff": float(obs), "ci95_lo": float(np.percentile(bs, 2.5)), "ci95_hi": float(np.percentile(bs, 97.5)),
            "p_bootstrap": max(p, 1.0 / B)}


def wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"W": 0.0, "p": 1.0}
    r = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"W": float(r.statistic), "p": float(r.pvalue)}


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) * len(b) > 4_000_000:
        rng = np.random.default_rng(SEED)
        a = rng.choice(a, 2000, replace=False) if len(a) > 2000 else a
        b = rng.choice(b, 2000, replace=False) if len(b) > 2000 else b
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    p = float(stats.binomtest(n01, n, 0.5).pvalue) if n else 1.0
    return {"a_only": n01, "b_only": n10, "p_exact": p}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    r = stats.ttest_ind(a, b, equal_var=False)
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"t": float(r.statistic), "p": float(r.pvalue), "hedges_g": float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0}
