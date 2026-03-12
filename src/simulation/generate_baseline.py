"""
generate_baseline.py
────────────────────
Creates a synthetic baseline dataset that mirrors the LendingClub feature
distributions observed during model training (≤ 2015 cohort).

Run once to create  data/baseline/train_snapshot.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── output path ───────────────────────────────────────────────────────────────
BASELINE_PATH = Path(__file__).parents[2] / "data" / "baseline" / "train_snapshot.csv"

# ── feature distributions (calibrated from LendingClub 2007-2015 cohort) ──────
N_SAMPLES = 50_000          # representative baseline snapshot

SUB_GRADE_PROBS = [          # P(sub_grade_num = 1..35) — A/B heavy, G rare
    0.040, 0.040, 0.040, 0.035, 0.030,   # A1-A5
    0.055, 0.055, 0.050, 0.045, 0.040,   # B1-B5
    0.050, 0.048, 0.045, 0.040, 0.038,   # C1-C5
    0.035, 0.030, 0.028, 0.025, 0.022,   # D1-D5
    0.018, 0.016, 0.014, 0.012, 0.010,   # E1-E5
    0.008, 0.007, 0.006, 0.005, 0.004,   # F1-F5
    0.003, 0.002, 0.002, 0.001, 0.001,   # G1-G5
]
SUB_GRADE_PROBS = np.array(SUB_GRADE_PROBS) / sum(SUB_GRADE_PROBS)

HOME_OWNERSHIP_CATS  = ["rent", "mortgage", "own", "other"]
HOME_OWNERSHIP_PROBS = [0.47, 0.43, 0.08, 0.02]

PURPOSE_CATS  = [
    "debt_consolidation", "credit_card", "home_improvement",
    "other", "major_purchase", "medical", "small_business",
    "car", "vacation", "moving", "house", "wedding", "educational",
]
PURPOSE_PROBS = [0.44, 0.23, 0.08, 0.07, 0.04, 0.03, 0.03,
                 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]


def _sub_grade_to_int_rate(sub_grade_num: np.ndarray) -> np.ndarray:
    """Approximate int_rate from sub_grade (follows LendingClub grade structure)."""
    base = 5.0 + sub_grade_num * 0.82           # 5% for A1, ~34% for G5
    noise = RNG.normal(0, 0.6, size=len(sub_grade_num))
    return np.clip(base + noise, 5.0, 36.0)


def generate_baseline(n: int = N_SAMPLES, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sub_grade_num = rng.choice(np.arange(1, 36), size=n, p=SUB_GRADE_PROBS)
    int_rate      = _sub_grade_to_int_rate(sub_grade_num)
    funded_amnt   = np.clip(rng.lognormal(mean=9.6, sigma=0.6, size=n), 1_000, 40_000).round(-2)
    annual_inc    = np.clip(rng.lognormal(mean=11.0, sigma=0.6, size=n), 15_000, 500_000).round(-2)
    dti           = np.clip(rng.lognormal(mean=2.8, sigma=0.5, size=n), 0, 40)
    emp_length    = rng.choice(np.arange(0, 11), size=n,
                               p=[0.07, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.28])
    term          = rng.choice([36, 60], size=n, p=[0.72, 0.28])
    installment   = (funded_amnt * (int_rate / 100 / 12) /
                     (1 - (1 + int_rate / 100 / 12) ** (-term))).round(2)
    home_ownership = rng.choice(HOME_OWNERSHIP_CATS, size=n, p=HOME_OWNERSHIP_PROBS)
    purpose        = rng.choice(PURPOSE_CATS, size=n, p=PURPOSE_PROBS)

    # Engineered features (same logic as LoanFeatureEngineer)
    loan_to_income  = funded_amnt / (annual_inc + 1)
    payment_burden  = (installment * 12) / (annual_inc + 1)
    log_annual_inc  = np.log1p(annual_inc)
    dti_x_term      = dti * term
    # int_rate_residual: deviation from mean rate for that sub_grade
    mean_by_grade   = pd.Series(int_rate).groupby(sub_grade_num).transform("mean").values
    int_rate_residual = int_rate - mean_by_grade
    inc_stability   = annual_inc / (emp_length + 1)

    df = pd.DataFrame({
        "funded_amnt":       funded_amnt,
        "annual_inc":        annual_inc,
        "dti":               dti,
        "int_rate":          int_rate,
        "emp_length":        emp_length,
        "term":              term,
        "sub_grade_num":     sub_grade_num,
        "installment":       installment,
        "home_ownership":    home_ownership,
        "purpose":           purpose,
        "loan_to_income":    loan_to_income,
        "payment_burden":    payment_burden,
        "log_annual_inc":    log_annual_inc,
        "dti_x_term":        dti_x_term,
        "int_rate_residual": int_rate_residual,
        "inc_stability":     inc_stability,
    })

    return df


if __name__ == "__main__":
    df = generate_baseline()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASELINE_PATH, index=False)
    print(f"Baseline snapshot saved → {BASELINE_PATH}  ({len(df):,} rows)")
    print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]].round(2))
