"""
business_metrics.py
───────────────────
Portfolio-level business metrics — the numbers that matter to credit risk
managers and the business, beyond pure model accuracy.

Tracks per month:
  - Approval rate           (based on ApprovalModel EP hurdle)
  - Observed default rate   (realized credit losses)
  - Expected Loss (EL)      PD × LGD × EAD — in dollars
  - Expected Profit (EP)    (1-PD) × income − EL — in dollars
  - EP ratio                EP / funded_amnt
  - EL ratio                EL / funded_amnt
  - Portfolio exposure      total funded_amnt approved
  - EP vs EL decomposition  profit/loss breakdown
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))   # adds src/ to path

from credit_risk_pipeline import (
    calculate_el,
    calculate_el_ratio,
    calculate_ep,
    calculate_ep_ratio,
)


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class PortfolioResult:
    month_label:        str
    n_applications:     int
    n_approved:         int
    approval_rate:      float
    total_exposure:     float         # sum of funded_amnt (approved loans)
    observed_default_rate: float      # realized, requires labels
    total_el:           float         # expected loss $
    total_ep:           float         # expected profit $
    mean_el_ratio:      float         # EL / funded_amnt
    mean_ep_ratio:      float         # EP / funded_amnt
    mean_pd:            float
    mean_lgd:           float
    mean_ead_ratio:     float

    @property
    def el_m(self) -> float:
        """Expected Loss in millions."""
        return self.total_el / 1e6

    @property
    def ep_m(self) -> float:
        """Expected Profit in millions."""
        return self.total_ep / 1e6

    @property
    def is_profitable(self) -> bool:
        return self.total_ep > 0

    def to_dict(self) -> dict:
        return {
            "month":                self.month_label,
            "n_applications":       self.n_applications,
            "n_approved":           self.n_approved,
            "approval_rate":        round(self.approval_rate, 4),
            "total_exposure":       round(self.total_exposure, 0),
            "observed_default_rate":round(self.observed_default_rate, 4),
            "total_el":             round(self.total_el, 0),
            "total_ep":             round(self.total_ep, 0),
            "mean_el_ratio":        round(self.mean_el_ratio, 4),
            "mean_ep_ratio":        round(self.mean_ep_ratio, 4),
            "mean_pd":              round(self.mean_pd, 4),
            "mean_lgd":             round(self.mean_lgd, 4),
            "mean_ead_ratio":       round(self.mean_ead_ratio, 4),
        }

    def __repr__(self) -> str:
        return (
            f"PortfolioResult(month='{self.month_label}' | "
            f"approved={self.n_approved:,}/{self.n_applications:,} "
            f"({self.approval_rate:.1%}) | "
            f"DR={self.observed_default_rate:.1%} | "
            f"EL=${self.el_m:.1f}M | "
            f"EP=${self.ep_m:.1f}M)"
        )


# ── main function ─────────────────────────────────────────────────────────────

def calculate_portfolio_metrics(
    df:             pd.DataFrame,
    pd_scores:      np.ndarray,
    lgd_scores:     np.ndarray,
    ead_scores:     np.ndarray,
    approved_mask:  np.ndarray,
    month_label:    str = "",
) -> PortfolioResult:
    """
    Calculate full portfolio metrics for one production month.

    Parameters
    ----------
    df            : Production DataFrame (must have funded_amnt, int_rate,
                    term, and optionally is_default).
    pd_scores     : Array of predicted PD probabilities.
    lgd_scores    : Array of predicted LGD values.
    ead_scores    : Array of predicted EAD values (dollar amount).
    approved_mask : Boolean array — True for approved loans.
    month_label   : e.g. "2018-07".
    """
    n_total    = len(df)
    n_approved = int(approved_mask.sum())

    # Work on approved book only
    df_app    = df[approved_mask].copy()
    pd_app    = pd_scores[approved_mask]
    lgd_app   = lgd_scores[approved_mask]
    ead_app   = ead_scores[approved_mask]

    funded    = df_app["funded_amnt"].values
    int_rate  = df_app["int_rate"].values
    term      = df_app["term"].values

    # EL and EP per loan
    el_arr    = calculate_el(pd_app, lgd_app, ead_app)
    ep_arr    = calculate_ep(pd_app, lgd_app, ead_app, funded, int_rate, term)
    el_ratio  = calculate_el_ratio(el_arr, funded)
    ep_ratio  = calculate_ep_ratio(ep_arr, funded)

    # Observed default rate (if labels exist)
    if "is_default" in df_app.columns:
        obs_dr = float(df_app["is_default"].mean())
    else:
        obs_dr = float("nan")

    return PortfolioResult(
        month_label           = month_label,
        n_applications        = n_total,
        n_approved            = n_approved,
        approval_rate         = n_approved / max(n_total, 1),
        total_exposure        = float(funded.sum()),
        observed_default_rate = obs_dr,
        total_el              = float(el_arr.sum()),
        total_ep              = float(ep_arr.sum()),
        mean_el_ratio         = float(el_ratio.mean()),
        mean_ep_ratio         = float(ep_ratio.mean()),
        mean_pd               = float(pd_app.mean()),
        mean_lgd              = float(lgd_app.mean()),
        mean_ead_ratio        = float((ead_app / (funded + 1e-9)).mean()),
    )


def portfolio_timeseries(results: list[PortfolioResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


def backtest_comparison(
    portfolio_ts: pd.DataFrame,
    backtest_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge live monitoring metrics with historical backtest results for
    side-by-side comparison (useful for the dashboard).
    """
    if backtest_path is None:
        backtest_path = str(
            Path(__file__).parents[2] / "data" / "baseline" / "backtest_results.csv"
        )

    backtest = pd.read_csv(backtest_path)
    backtest_summary = backtest.agg({
        "approval_rate":      "mean",
        "obs_default_rate":   "mean",
        "pred_ep_ratio":      "mean",
    }).rename({
        "approval_rate":    "bt_mean_approval_rate",
        "obs_default_rate": "bt_mean_default_rate",
        "pred_ep_ratio":    "bt_mean_ep_ratio",
    })

    df_out = portfolio_ts.copy()
    for col, val in backtest_summary.items():
        df_out[col] = round(val, 4)

    return df_out
