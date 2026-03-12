"""
prediction_drift.py
───────────────────
Monitors distributional shift in the model's output scores (predicted PD).

Even without ground-truth labels, we can detect early when the model is
producing systematically different predictions — a leading indicator of
portfolio composition change or model failure.

Metrics:
  - PSI on the PD score distribution (score shift)
  - Mean PD shift (predicted default rate drift)
  - High-risk segment share (% of loans with PD > threshold)
  - KS test between baseline and production PD distributions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))   # adds monitoring/ to path

from psi import calculate_psi_numeric, PSIResult


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class PredictionDriftResult:
    month_label:          str
    psi_score:            float         # PSI on PD score distribution
    psi_status:           str
    mean_pd_baseline:     float         # mean PD in training
    mean_pd_production:   float         # mean PD in production
    mean_pd_shift:        float         # absolute shift
    high_risk_share_base: float         # % with PD > threshold in baseline
    high_risk_share_prod: float         # % with PD > threshold in production
    ks_statistic:         float         # KS test on score distributions
    ks_pvalue:            float

    HIGH_RISK_THRESHOLD: float = 0.30   # class attribute

    @property
    def is_alert(self) -> bool:
        return (self.psi_score > 0.10
                or abs(self.mean_pd_shift) > 0.03
                or self.ks_pvalue < 0.01)

    @property
    def is_critical(self) -> bool:
        return self.psi_score > 0.25 or abs(self.mean_pd_shift) > 0.07

    def to_dict(self) -> dict:
        return {
            "month":                self.month_label,
            "psi_score":            round(self.psi_score, 4),
            "psi_status":           self.psi_status,
            "mean_pd_baseline":     round(self.mean_pd_baseline, 4),
            "mean_pd_production":   round(self.mean_pd_production, 4),
            "mean_pd_shift":        round(self.mean_pd_shift, 4),
            "high_risk_share_base": round(self.high_risk_share_base, 4),
            "high_risk_share_prod": round(self.high_risk_share_prod, 4),
            "ks_statistic":         round(self.ks_statistic, 4),
            "ks_pvalue":            round(self.ks_pvalue, 4),
        }

    def __repr__(self) -> str:
        return (f"PredictionDriftResult(month='{self.month_label}' | "
                f"PSI={self.psi_score:.4f} [{self.psi_status}] | "
                f"MeanPD {self.mean_pd_baseline:.4f} → {self.mean_pd_production:.4f} "
                f"(shift={self.mean_pd_shift:+.4f}))")


# ── main function ─────────────────────────────────────────────────────────────

def calculate_prediction_drift(
    baseline_scores:   np.ndarray,
    production_scores: np.ndarray,
    month_label:       str = "",
    high_risk_threshold: float = 0.30,
    n_bins:            int = 20,
) -> PredictionDriftResult:
    """
    Detect distributional drift in predicted PD scores.

    Parameters
    ----------
    baseline_scores   : PD predictions on the training / reference set.
    production_scores : PD predictions for the current production month.
    month_label       : e.g. "2018-05".
    high_risk_threshold: PD cutoff defining the 'high risk' segment.
    n_bins            : Bins for PSI computation.
    """
    base = np.clip(np.asarray(baseline_scores, dtype=float), 1e-7, 1 - 1e-7)
    prod = np.clip(np.asarray(production_scores, dtype=float), 1e-7, 1 - 1e-7)

    # PSI on score distribution
    psi_result: PSIResult = calculate_psi_numeric(
        pd.Series(base, name="pd_score"),
        pd.Series(prod, name="pd_score"),
        n_bins=n_bins,
    )

    # Mean PD shift
    mean_base = float(base.mean())
    mean_prod = float(prod.mean())
    mean_shift = mean_prod - mean_base

    # High-risk segment share
    hr_base = float((base > high_risk_threshold).mean())
    hr_prod = float((prod > high_risk_threshold).mean())

    # KS test: tests if two distributions are the same
    ks_stat, ks_p = stats.ks_2samp(base, prod)

    return PredictionDriftResult(
        month_label          = month_label,
        psi_score            = psi_result.psi,
        psi_status           = psi_result.status,
        mean_pd_baseline     = mean_base,
        mean_pd_production   = mean_prod,
        mean_pd_shift        = mean_shift,
        high_risk_share_base = hr_base,
        high_risk_share_prod = hr_prod,
        ks_statistic         = float(ks_stat),
        ks_pvalue            = float(ks_p),
        HIGH_RISK_THRESHOLD  = high_risk_threshold,
    )


def prediction_drift_timeseries(results: list[PredictionDriftResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


if __name__ == "__main__":
    rng  = np.random.default_rng(42)
    base = np.clip(rng.beta(2, 8, 10_000), 0.01, 0.99)
    prod = np.clip(rng.beta(2.8, 7, 5_000), 0.01, 0.99)  # slightly riskier
    res  = calculate_prediction_drift(base, prod, month_label="2018-08")
    print(res)
