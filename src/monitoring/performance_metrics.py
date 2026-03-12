import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# ── thresholds 

AUC_THRESHOLDS   = {"good": 0.75, "watch": 0.70}   # below watch → retrain
KS_THRESHOLDS    = {"good": 0.40, "watch": 0.30}
BRIER_THRESHOLDS = {"good": 0.16, "watch": 0.22}   # above watch → investigate


def _metric_status(value: float, thresholds: dict, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= thresholds["good"]:  return "good"
        if value >= thresholds["watch"]: return "watch"
        return "action"
    else:
        if value <= thresholds["good"]:  return "good"
        if value <= thresholds["watch"]: return "watch"
        return "action"


# ── KS statistic 

def calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:

    df = pd.DataFrame({"target": y_true, "prob": y_prob}).sort_values("prob", ascending=False)

    n_default    = df["target"].sum()
    n_non_default = len(df) - n_default

    if n_default == 0 or n_non_default == 0:
        return 0.0

    df["cum_default"]     = df["target"].cumsum()     / n_default
    df["cum_non_default"] = (1 - df["target"]).cumsum() / n_non_default

    return float((df["cum_default"] - df["cum_non_default"]).abs().max())


# ── result container 

@dataclass
class PerformanceResult:
    month_label:    str
    n_samples:      int
    n_defaults:     int
    auc:            float
    ks:             float
    brier:          float
    pr_auc:         float
    auc_status:     str
    ks_status:      str
    brier_status:   str

    @property
    def default_rate(self) -> float:
        return self.n_defaults / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def overall_status(self) -> str:
        statuses = [self.auc_status, self.ks_status, self.brier_status]
        if "action" in statuses: return "action"
        if "watch"  in statuses: return "watch"
        return "good"

    @property
    def needs_retraining(self) -> bool:
        return self.auc < AUC_THRESHOLDS["watch"] or self.ks < KS_THRESHOLDS["watch"]

    def to_dict(self) -> dict:
        return {
            "month":        self.month_label,
            "n_samples":    self.n_samples,
            "n_defaults":   self.n_defaults,
            "default_rate": round(self.default_rate, 4),
            "auc":          round(self.auc, 4),
            "ks":           round(self.ks, 4),
            "brier":        round(self.brier, 4),
            "pr_auc":       round(self.pr_auc, 4),
            "auc_status":   self.auc_status,
            "ks_status":    self.ks_status,
            "brier_status": self.brier_status,
            "overall":      self.overall_status,
        }

    def __repr__(self) -> str:
        return (f"PerformanceResult(month='{self.month_label}' | "
                f"AUC={self.auc:.4f} [{self.auc_status}] | "
                f"KS={self.ks:.4f} [{self.ks_status}] | "
                f"Brier={self.brier:.4f} [{self.brier_status}])")


# ── main function 

def calculate_performance(
    y_true:      np.ndarray,
    y_prob:      np.ndarray,
    month_label: str = "",
) -> PerformanceResult:

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_samples  = len(y_true)
    n_defaults = int(y_true.sum())

    # Handle edge cases (no defaults or all defaults → metrics undefined)
    if n_defaults == 0 or n_defaults == n_samples:
        return PerformanceResult(
            month_label  = month_label,
            n_samples    = n_samples,
            n_defaults   = n_defaults,
            auc          = 0.5,
            ks           = 0.0,
            brier        = 0.25,
            pr_auc       = 0.0,
            auc_status   = "action",
            ks_status    = "action",
            brier_status = "action",
        )

    auc   = roc_auc_score(y_true, y_prob)
    ks    = calculate_ks(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return PerformanceResult(
        month_label  = month_label,
        n_samples    = n_samples,
        n_defaults   = n_defaults,
        auc          = auc,
        ks           = ks,
        brier        = brier,
        pr_auc       = pr_auc,
        auc_status   = _metric_status(auc,   AUC_THRESHOLDS,   higher_is_better=True),
        ks_status    = _metric_status(ks,    KS_THRESHOLDS,    higher_is_better=True),
        brier_status = _metric_status(brier, BRIER_THRESHOLDS, higher_is_better=False),
    )


def performance_timeseries(results: list[PerformanceResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y   = (rng.uniform(0, 1, 5000) < 0.22).astype(int)
    p   = np.clip(y * 0.5 + rng.normal(0.2, 0.15, 5000), 0.01, 0.99)
    res = calculate_performance(y, p, month_label="2018-01")
    print(res)
