"""Module G: confidence scoring, calibration metrics and selective reporting.

Definitions (also stated in the manuscript):
* An *observation* is one canonical label emitted for one image.
* Its *correctness* y in {0,1} is 1 iff the label is in the reference annotation
  (only evaluated where the reference is exhaustive, e.g. COCO objects).
* Confidence signals:
    verbal  — model's self-reported high/medium/low mapped to 0.9/0.6/0.3 (ordinal, uncalibrated)
    token   — geometric-mean probability of the generated tokens of the label (uncalibrated)
    ground  — CLIP image-text cosine similarity of "a photo of a {label}" (uncalibrated, unbounded)
    combined— logistic regression over the signals + quality flags, fit on the DEV split only;
              this is the only score reported as a calibrated probability.
"""

from __future__ import annotations

import numpy as np

VERBAL_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3, None: 0.6}


def ece(conf: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, y = np.asarray(conf, float), np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - y[m].mean())
    return float(e)


def reliability(conf: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({"bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]), "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else None, "acc": float(y[m].mean()) if m.any() else None})
    return out


def brier(conf: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(conf, float) - np.asarray(y, float)) ** 2))


def risk_coverage(conf: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Sort by confidence desc; coverage k/n and selective risk (error rate) of the top-k. Returns AURC."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    err = 1 - np.asarray(y, float)[order]
    k = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / k
    cov = k / len(err)
    aurc = float(np.trapezoid(risk, cov)) if len(err) > 1 else float(risk[0])
    return cov, risk, aurc


def selective_risk_at(conf: np.ndarray, y: np.ndarray, coverage: float) -> float:
    cov, risk, _ = risk_coverage(conf, y)
    i = int(np.searchsorted(cov, coverage))
    return float(risk[min(i, len(risk) - 1)])


def auroc(conf: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(conf)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Combiner:
    """Logistic-regression calibrator over confidence features (fit on dev, applied to test)."""

    FEATURES = ["token", "verbal", "ground", "n_quality_flags", "is_object", "is_clothing", "support_frac"]

    def __init__(self, features: list[str] | None = None, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.features = features or [f for f in self.FEATURES if f != "support_frac"]
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=C, max_iter=1000)

    def _X(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[float(r.get(f) if r.get(f) is not None else np.nan) for f in self.features] for r in rows])
        # impute missing with column means from fit (or 0.5 before fit)
        if hasattr(self, "_means"):
            m = np.where(np.isnan(X), self._means, X)
        else:
            m = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        return np.nan_to_num(m, nan=0.5)

    def fit(self, rows: list[dict], y: np.ndarray) -> "Combiner":
        X = np.array([[float(r.get(f) if r.get(f) is not None else np.nan) for f in self.features] for r in rows])
        self._means = np.nan_to_num(np.nanmean(X, axis=0), nan=0.5)
        Xi = self._X(rows)
        self.lr.fit(self.scaler.fit_transform(Xi), y)
        return self

    def predict(self, rows: list[dict]) -> np.ndarray:
        return self.lr.predict_proba(self.scaler.transform(self._X(rows)))[:, 1]

    def coefficients(self) -> dict:
        return {f: float(c) for f, c in zip(self.features, self.lr.coef_[0])}


def threshold_for_precision(conf: np.ndarray, y: np.ndarray, target_precision: float) -> float:
    """Smallest threshold whose retained set has precision >= target (computed on DEV)."""
    order = np.argsort(-conf)
    c, yy = conf[order], y[order]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    ok = np.where(prec >= target_precision)[0]
    if len(ok) == 0:
        return float(c[0]) + 1e-9
    return float(c[ok[-1]])
