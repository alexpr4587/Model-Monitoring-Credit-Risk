"""
calibration.py
──────────────
Monitors whether the model's predicted PD probabilities remain accurate
over time — i.e., "if the model says 10% default probability, do ~10% actually default?"

Calibration can degrade even when AUC stays high.
A well-discriminating model that is miscalibrated will under/over-estimate EL.

Methods:
  - Mean calibration error (predicted PD vs observed default rate)
  - Reliability diagram (calibration curve by decile)
  - Expected Calibration Error (ECE)
  - Hosmer-Lemeshow statistic
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class CalibrationResult:
    month_label:          str
    mean_predicted_pd:    float      # E[PD_hat]
    observed_default_rate: float     # actual default rate
    calibration_error:    float      # |mean_pred - obs_default|
    expected_calib_error: float      # ECE (reliability-diagram based)
    hl_statistic:         float      # Hosmer-Lemeshow chi-squared
    hl_pvalue:            float      # H-L p-value (< 0.05 = miscalibrated)
    decile_df:            pd.DataFrame = field(repr=False)  # for plotting

    @property
    def status(self) -> str:
        if self.calibration_error > 0.05 or self.hl_pvalue < 0.05:
            return "miscalibrated"
        if self.calibration_error > 0.02:
            return "watch"
        return "calibrated"

    @property
    def is_alert(self) -> bool:
        return self.status != "calibrated"

    def to_dict(self) -> dict:
        return {
            "month":                self.month_label,
            "mean_predicted_pd":    round(self.mean_predicted_pd, 4),
            "observed_default_rate":round(self.observed_default_rate, 4),
            "calibration_error":    round(self.calibration_error, 4),
            "ece":                  round(self.expected_calib_error, 4),
            "hl_statistic":         round(self.hl_statistic, 4),
            "hl_pvalue":            round(self.hl_pvalue, 4),
            "status":               self.status,
        }

    def __repr__(self) -> str:
        return (f"CalibrationResult(month='{self.month_label}' | "
                f"E[PD]={self.mean_predicted_pd:.4f} | "
                f"Obs DR={self.observed_default_rate:.4f} | "
                f"Error={self.calibration_error:.4f} | "
                f"Status='{self.status}')")


def calculate_calibration(
    y_true:      np.ndarray,
    y_prob:      np.ndarray,
    n_bins:      int = 10,
    month_label: str = "",
) -> CalibrationResult:
    """
    Full calibration assessment for one production month.

    Parameters
    ----------
    y_true      : Observed default labels (0/1).
    y_prob      : Predicted PD (probability of default).
    n_bins      : Number of bins for reliability diagram (default = deciles).
    month_label : e.g. "2018-04".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)

    mean_pred  = float(y_prob.mean())
    obs_rate   = float(y_true.mean())
    calib_err  = abs(mean_pred - obs_rate)

    # ── Reliability diagram by decile ────────────────────────────────────────
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)

    decile_df = (
        df.groupby("bin")
        .agg(
            mean_pred    = ("p", "mean"),
            obs_rate     = ("y", "mean"),
            n            = ("y", "count"),
        )
        .reset_index(drop=True)
    )
    decile_df["abs_error"] = (decile_df["mean_pred"] - decile_df["obs_rate"]).abs()

    # Expected Calibration Error (ECE) — weighted by bin size
    total_n = decile_df["n"].sum()
    ece     = float((decile_df["abs_error"] * decile_df["n"] / total_n).sum())

    # ── Hosmer-Lemeshow test ──────────────────────────────────────────────────
    # H0: model is well-calibrated. Reject if p < 0.05.
    hl_stat = 0.0
    for _, row in decile_df.iterrows():
        n     = row["n"]
        p_hat = row["mean_pred"]
        o1    = row["obs_rate"] * n          # observed defaults
        e1    = p_hat * n                    # expected defaults
        e0    = (1 - p_hat) * n              # expected non-defaults
        o0    = n - o1                       # observed non-defaults
        if e1 > 0:  hl_stat += (o1 - e1) ** 2 / e1
        if e0 > 0:  hl_stat += (o0 - e0) ** 2 / e0

    df_hl  = max(len(decile_df) - 2, 1)
    hl_p   = float(1 - stats.chi2.cdf(hl_stat, df=df_hl))

    return CalibrationResult(
        month_label           = month_label,
        mean_predicted_pd     = mean_pred,
        observed_default_rate = obs_rate,
        calibration_error     = calib_err,
        expected_calib_error  = ece,
        hl_statistic          = hl_stat,
        hl_pvalue             = hl_p,
        decile_df             = decile_df,
    )


def calibration_timeseries(results: list[CalibrationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y   = (rng.uniform(0, 1, 5000) < 0.22).astype(int)
    p   = np.clip(y * 0.4 + rng.normal(0.18, 0.1, 5000), 0.01, 0.99)
    res = calculate_calibration(y, p, month_label="2018-01")
    print(res)
    print(res.decile_df.round(4).to_string(index=False))
