"""
psi.py
──────
Population Stability Index (PSI) — the industry-standard metric for
detecting distributional shift between training and production data.

PSI = Σ (Actual_i − Expected_i) × ln(Actual_i / Expected_i)

Interpretation (standard banking thresholds):
  PSI < 0.10  → Stable, no action needed
  0.10 – 0.25 → Moderate shift, monitor closely
  PSI > 0.25  → Severe shift, investigate and consider retraining
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── result containers ──────────────────────────────────────────────────────────

@dataclass
class PSIResult:
    feature:     str
    psi:         float
    status:      str                      # "stable" | "moderate" | "severe"
    n_bins:      int
    expected_pct: np.ndarray             # baseline bin proportions
    actual_pct:   np.ndarray             # production bin proportions
    bin_edges:    Optional[np.ndarray]   # numeric features only
    bin_labels:   Optional[list]         # categorical features only

    @property
    def is_alert(self) -> bool:
        return self.psi > 0.10

    @property
    def is_critical(self) -> bool:
        return self.psi > 0.25

    def __repr__(self) -> str:
        return f"PSIResult(feature='{self.feature}', psi={self.psi:.4f}, status='{self.status}')"


@dataclass
class PSIReport:
    results:      list[PSIResult]
    month_label:  str = ""

    @property
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "feature": r.feature,
            "psi":     round(r.psi, 4),
            "status":  r.status,
        } for r in self.results]).sort_values("psi", ascending=False)

    @property
    def critical_features(self) -> list[str]:
        return [r.feature for r in self.results if r.is_critical]

    @property
    def alert_features(self) -> list[str]:
        return [r.feature for r in self.results if r.is_alert]

    @property
    def max_psi(self) -> float:
        return max(r.psi for r in self.results)

    def __repr__(self) -> str:
        return (f"PSIReport(month='{self.month_label}', "
                f"features={len(self.results)}, "
                f"critical={len(self.critical_features)}, "
                f"max_psi={self.max_psi:.4f})")


# ── core functions ─────────────────────────────────────────────────────────────

def _psi_status(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    elif psi < 0.25:
        return "moderate"
    return "severe"


def _psi_from_proportions(
    expected_pct: np.ndarray,
    actual_pct:   np.ndarray,
    epsilon:      float = 1e-6,
) -> float:
    """
    Core PSI formula applied to pre-computed proportions.
    epsilon prevents log(0) for empty bins.
    """
    e = np.clip(expected_pct, epsilon, None)
    a = np.clip(actual_pct,   epsilon, None)
    # Normalize so they sum to 1 (in case of floating-point errors)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def calculate_psi_numeric(
    baseline:     pd.Series,
    production:   pd.Series,
    n_bins:       int = 10,
    bin_edges:    Optional[np.ndarray] = None,
) -> PSIResult:
    """
    Calculate PSI for a numeric feature using equal-frequency bins
    fitted on the baseline distribution.

    Parameters
    ----------
    baseline   : Training / reference distribution.
    production : Current production distribution.
    n_bins     : Number of quantile bins (default 10 = deciles).
    bin_edges  : Pre-computed edges (use when comparing multiple months
                 against the same baseline — avoids edge drift).
    """
    feature = baseline.name or "unknown"

    # Drop nulls before binning
    base_clean = baseline.dropna()
    prod_clean = production.dropna()

    if len(base_clean) == 0 or len(prod_clean) == 0:
        return PSIResult(feature, 0.0, "stable", n_bins,
                         np.array([]), np.array([]), None, None)

    # Fit bin edges on baseline (quantile-based = equal expected frequency)
    if bin_edges is None:
        quantiles  = np.linspace(0, 100, n_bins + 1)
        bin_edges  = np.unique(np.percentile(base_clean, quantiles))

    # Extend edges to cover production extremes
    bin_edges[0]  = min(bin_edges[0],  prod_clean.min()) - 1e-9
    bin_edges[-1] = max(bin_edges[-1], prod_clean.max()) + 1e-9

    expected_counts = np.histogram(base_clean, bins=bin_edges)[0]
    actual_counts   = np.histogram(prod_clean, bins=bin_edges)[0]

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct   = actual_counts   / actual_counts.sum()

    psi = _psi_from_proportions(expected_pct, actual_pct)

    return PSIResult(
        feature      = feature,
        psi          = psi,
        status       = _psi_status(psi),
        n_bins       = len(bin_edges) - 1,
        expected_pct = expected_pct,
        actual_pct   = actual_pct,
        bin_edges    = bin_edges,
        bin_labels   = None,
    )


def calculate_psi_categorical(
    baseline:   pd.Series,
    production: pd.Series,
) -> PSIResult:
    """
    Calculate PSI for a categorical feature (home_ownership, purpose).
    Uses observed categories from the baseline as reference bins.
    New production categories are lumped into an 'other' bin.
    """
    feature = baseline.name or "unknown"

    # Proportions over known baseline categories
    categories     = baseline.dropna().unique().tolist()
    expected_counts = baseline.value_counts().reindex(categories, fill_value=0)
    actual_counts   = production.value_counts().reindex(categories, fill_value=0)

    expected_pct = expected_counts.values / expected_counts.values.sum()
    actual_pct   = actual_counts.values   / max(actual_counts.values.sum(), 1)

    psi = _psi_from_proportions(expected_pct, actual_pct)

    return PSIResult(
        feature      = feature,
        psi          = psi,
        status       = _psi_status(psi),
        n_bins       = len(categories),
        expected_pct = expected_pct,
        actual_pct   = actual_pct,
        bin_edges    = None,
        bin_labels   = categories,
    )


def calculate_psi_report(
    baseline:    pd.DataFrame,
    production:  pd.DataFrame,
    features:    Optional[list[str]] = None,
    month_label: str = "",
    n_bins:      int = 10,
) -> PSIReport:
    """
    Calculate PSI for all features and return a PSIReport.

    Parameters
    ----------
    baseline   : Baseline (training) DataFrame.
    production : Current production DataFrame.
    features   : List of feature columns to monitor.
                 Defaults to all shared numeric columns.
    month_label: e.g. "2018-03" for labelling.
    n_bins     : Bins for numeric features.
    """
    NUMERIC_FEATURES     = [
        "funded_amnt", "annual_inc", "dti", "int_rate",
        "emp_length", "term", "sub_grade_num",
        "loan_to_income", "payment_burden", "log_annual_inc",
        "dti_x_term", "int_rate_residual", "inc_stability",
    ]
    CATEGORICAL_FEATURES = ["home_ownership", "purpose"]

    if features is None:
        features = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                    if f in baseline.columns and f in production.columns]

    results = []
    for feat in features:
        if feat in CATEGORICAL_FEATURES:
            result = calculate_psi_categorical(
                baseline[feat].rename(feat),
                production[feat].rename(feat),
            )
        else:
            result = calculate_psi_numeric(
                baseline[feat].rename(feat),
                production[feat].rename(feat),
                n_bins=n_bins,
            )
        results.append(result)

    return PSIReport(results=results, month_label=month_label)


# ── convenience function ───────────────────────────────────────────────────────

def psi_timeseries(
    baseline:        pd.DataFrame,
    monthly_files:   list[tuple[str, pd.DataFrame]],
    features:        Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute PSI across multiple production months.

    Returns a DataFrame with columns [month, feature, psi, status].
    """
    rows = []
    for label, prod_df in monthly_files:
        report = calculate_psi_report(baseline, prod_df, features=features, month_label=label)
        for r in report.results:
            rows.append({"month": label, "feature": r.feature, "psi": r.psi, "status": r.status})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng = np.random.default_rng(42)
    n   = 5_000

    base = pd.DataFrame({
        "sub_grade_num": rng.choice(np.arange(1, 36), n),
        "annual_inc":    rng.lognormal(11.0, 0.6, n),
        "dti":           rng.lognormal(2.8, 0.5, n),
        "home_ownership": rng.choice(["rent", "mortgage", "own"], n, p=[0.47, 0.43, 0.10]),
    })
    # Introduce drift in production
    prod = base.copy()
    prod["sub_grade_num"] = rng.choice(np.arange(1, 36), n,
                                        p=np.ones(35) / 35)  # uniform → high drift
    prod["annual_inc"] *= 0.80   # income compression

    report = calculate_psi_report(base, prod, month_label="2018-06")
    print(report)
    print(report.summary.to_string(index=False))
