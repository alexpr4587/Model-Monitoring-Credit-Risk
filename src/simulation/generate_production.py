"""
generate_production.py
──────────────────────
Generates 12 months of synthetic production data using a convex combination
between a "healthy" baseline population and a pre-defined "stressed" population.

The mixing parameter α increases linearly from 0 → 1 over the 12 months:

    X_month = (1 - α) * X_healthy  +  α * X_stressed

    Month  1 : α = 0.00  → 100% healthy population
    Month  6 : α = 0.45  → 55% healthy / 45% stressed
    Month 12 : α = 1.00  → 100% stressed population

This guarantees:
  - PSI alerts fire gradually (yellow first, then red)
  - AUC/KS degrade as a smooth negative slope, not a shock
  - Default rate increases continuously month-over-month
  - No discrete jumps caused by feature-level if/else onset logic

Stressed population calibrated to LendingClub post-2017 behavior:
  - Grade composition shifted toward D-G (riskier borrowers)
  - Income compressed ~25% (wage stagnation)
  - DTI elevated ~8pp (higher debt burden)
  - Interest rates ~4pp higher (risk repricing)

Outputs: data/production/YYYY_MM.csv  (one file per month)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from generate_baseline import (
    generate_baseline,
    HOME_OWNERSHIP_CATS, HOME_OWNERSHIP_PROBS,
    PURPOSE_CATS, PURPOSE_PROBS,
    RNG as _BASE_RNG,
)

PRODUCTION_PATH = Path(__file__).parents[2] / "data" / "production"

# ── grade distributions ───────────────────────────────────────────────────────

# Healthy population: A/B heavy — mirrors 2007-2015 training cohort
HEALTHY_GRADE_PROBS = np.array([
    0.040, 0.040, 0.040, 0.035, 0.030,   # A
    0.055, 0.055, 0.050, 0.045, 0.040,   # B
    0.050, 0.048, 0.045, 0.040, 0.038,   # C
    0.035, 0.030, 0.028, 0.025, 0.022,   # D
    0.018, 0.016, 0.014, 0.012, 0.010,   # E
    0.008, 0.007, 0.006, 0.005, 0.004,   # F
    0.003, 0.002, 0.002, 0.001, 0.001,   # G
])
HEALTHY_GRADE_PROBS /= HEALTHY_GRADE_PROBS.sum()

# Stressed population: D-F heavy — mirrors post-2017 credit loosening
STRESSED_GRADE_PROBS = np.array([
    0.015, 0.015, 0.014, 0.012, 0.010,   # A (reduced)
    0.025, 0.025, 0.023, 0.020, 0.018,   # B (reduced)
    0.055, 0.053, 0.050, 0.046, 0.042,   # C (increased)
    0.060, 0.057, 0.053, 0.048, 0.043,   # D (significantly increased)
    0.042, 0.038, 0.033, 0.028, 0.022,   # E (significantly increased)
    0.018, 0.015, 0.013, 0.010, 0.008,   # F
    0.008, 0.006, 0.005, 0.004, 0.003,   # G
])
STRESSED_GRADE_PROBS /= STRESSED_GRADE_PROBS.sum()

# ── scalar feature parameters ─────────────────────────────────────────────────
# Each entry: (healthy_mean, healthy_std, stressed_mean, stressed_std)
# used to draw from lognormal/normal distributions per-population

FEATURE_PARAMS = {
    #                  healthy                stressed
    "annual_inc":  dict(h_mu=11.00, h_sig=0.60, s_mu=10.75, s_sig=0.65),  # ~25% income drop
    "dti":         dict(h_mu=2.80,  h_sig=0.50, s_mu=3.10,  s_sig=0.55),  # higher debt burden
    "funded_amnt": dict(h_mu=9.60,  h_sig=0.60, s_mu=9.75,  s_sig=0.58),  # slightly larger loans
}

# ── alpha schedule ─────────────────────────────────────────────────────────────

def alpha(month: int, n_months: int = 12) -> float:
    """
    Convex mixing weight. Increases linearly from 0 to 1 over n_months.
    Month 1 → α=0 (pure healthy), Month n_months → α=1 (pure stressed).
    """
    return (month - 1) / max(n_months - 1, 1)


# ── population generators ──────────────────────────────────────────────────────

def _sample_grades(n: int, a: float, rng: np.random.Generator) -> np.ndarray:
    """Sample sub_grade_num using convex mix of healthy/stressed grade distributions."""
    probs = (1 - a) * HEALTHY_GRADE_PROBS + a * STRESSED_GRADE_PROBS
    probs = probs / probs.sum()
    return rng.choice(np.arange(1, 36), size=n, p=probs)


def _sample_lognormal(n: int, a: float, key: str, rng: np.random.Generator) -> np.ndarray:
    """Draw from convex mix of two lognormal distributions."""
    p = FEATURE_PARAMS[key]
    # Mix by sampling from each population proportionally then shuffling
    n_stressed = int(round(n * a))
    n_healthy  = n - n_stressed

    healthy  = rng.lognormal(p["h_mu"], p["h_sig"], size=n_healthy) if n_healthy  > 0 else np.array([])
    stressed = rng.lognormal(p["s_mu"], p["s_sig"], size=n_stressed) if n_stressed > 0 else np.array([])
    combined = np.concatenate([healthy, stressed])
    rng.shuffle(combined)
    return combined


def _apply_default_label(df: pd.DataFrame, a: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Synthetic default label. PD increases continuously with alpha.
    Calibrated so:
      α=0  → base default rate ~15% (training distribution)
      α=1  → base default rate ~38% (stressed portfolio)
    """
    grade_risk = 0.04 + (df["sub_grade_num"].values - 1) * 0.014
    dti_risk   = df["dti"].values * 0.004
    time_risk  = a * 0.23   # additional systematic risk from population shift
    pd_vector  = np.clip(grade_risk + dti_risk + time_risk, 0.02, 0.85)
    df["is_default"] = (rng.uniform(0, 1, size=len(df)) < pd_vector).astype(int)
    return df


# ── single-month generator ────────────────────────────────────────────────────

def generate_month(
    month:          int,
    n_applications: int = 10_000,
    n_months:       int = 12,
    seed:           int = None,
) -> pd.DataFrame:
    """
    Generate one month of production data.

    Parameters
    ----------
    month          : 1-based month index (1 = first production month).
    n_applications : Number of loan applications.
    n_months       : Total months in the simulation window (for α schedule).
    seed           : Random seed.
    """
    a   = alpha(month, n_months)
    rng = np.random.default_rng(seed if seed is not None else month * 100)

    n = n_applications

    # ── Grade + int_rate ──────────────────────────────────────────────────────
    sub_grade_num = _sample_grades(n, a, rng)
    base_rate     = 5.0 + sub_grade_num * 0.82
    int_rate      = np.clip(base_rate + rng.normal(0, 0.6, size=n), 5.0, 36.0)

    # ── Income ────────────────────────────────────────────────────────────────
    annual_inc = np.clip(
        _sample_lognormal(n, a, "annual_inc", rng),
        15_000, 500_000,
    ).round(-2)

    # ── DTI ───────────────────────────────────────────────────────────────────
    dti = np.clip(_sample_lognormal(n, a, "dti", rng), 0, 45)

    # ── Funded amount ─────────────────────────────────────────────────────────
    funded_amnt = np.clip(
        _sample_lognormal(n, a, "funded_amnt", rng),
        1_000, 40_000,
    ).round(-2)

    # ── Other features (stable across populations) ────────────────────────────
    emp_length = rng.choice(
        np.arange(0, 11), size=n,
        p=[0.07, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.28],
    )
    term           = rng.choice([36, 60], size=n, p=[0.72, 0.28])
    home_ownership = rng.choice(HOME_OWNERSHIP_CATS, size=n, p=HOME_OWNERSHIP_PROBS)
    purpose        = rng.choice(PURPOSE_CATS, size=n, p=PURPOSE_PROBS)

    # ── Installment ───────────────────────────────────────────────────────────
    installment = (
        funded_amnt * (int_rate / 100 / 12)
        / (1 - (1 + int_rate / 100 / 12) ** (-term))
    ).round(2)

    # ── Engineered features ───────────────────────────────────────────────────
    loan_to_income    = funded_amnt / (annual_inc + 1)
    payment_burden    = (installment * 12) / (annual_inc + 1)
    log_annual_inc    = np.log1p(annual_inc)
    dti_x_term        = dti * term
    mean_by_grade     = pd.Series(int_rate).groupby(sub_grade_num).transform("mean").values
    int_rate_residual = int_rate - mean_by_grade
    inc_stability     = annual_inc / (emp_length + 1)

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

    df = _apply_default_label(df, a, rng)
    return df


# ── full run ──────────────────────────────────────────────────────────────────

def generate_all_months(
    start_year:            int = 2018,
    start_month:           int = 1,
    n_months:              int = 12,
    applications_per_month:int = 10_000,
    overwrite:             bool = True,
) -> None:
    """Generate and save all production months."""
    PRODUCTION_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_months} months of production data (convex mixing α: 0.00 → 1.00)\n")
    print(f"{'Month':<8} {'File':<16} {'Apps':>8} {'α':>6} {'DR':>8} {'Mean Grade':>11}")
    print("-" * 65)

    for i in range(n_months):
        month_idx = i + 1
        year      = start_year + (start_month + i - 1) // 12
        month_num = (start_month + i - 1) % 12 + 1
        filename  = PRODUCTION_PATH / f"{year}_{month_num:02d}.csv"

        if filename.exists() and not overwrite:
            print(f"  Skipping {filename.name} (already exists)")
            continue

        df = generate_month(
            month          = month_idx,
            n_applications = applications_per_month,
            n_months       = n_months,
            seed           = 42 + i,
        )
        df.to_csv(filename, index=False)

        a   = alpha(month_idx, n_months)
        dr  = df["is_default"].mean()
        mg  = df["sub_grade_num"].mean()
        print(f"  {month_idx:<6} {filename.name:<16} {len(df):>8,} {a:>6.2f} {dr:>8.2%} {mg:>11.1f}")

    print(f"\nProduction data saved → {PRODUCTION_PATH}")


# ── single-month entry point (for cron jobs) ──────────────────────────────────

def generate_next_month(
    year:       int,
    month:      int,
    month_idx:  int,
    n_months:   int = 12,
    n_apps:     int = 10_000,
) -> Path:
    """
    Generate exactly one month. Called by the scheduler (cron / orchestrator).

    Parameters
    ----------
    year, month : Calendar year and month (e.g. 2018, 3).
    month_idx   : Position in the simulation window (1-based).
    """
    PRODUCTION_PATH.mkdir(parents=True, exist_ok=True)
    filename = PRODUCTION_PATH / f"{year}_{month:02d}.csv"
    df = generate_month(month=month_idx, n_applications=n_apps, n_months=n_months, seed=year * 100 + month)
    df.to_csv(filename, index=False)
    print(f"Generated {filename.name}  (α={alpha(month_idx, n_months):.2f}, DR={df['is_default'].mean():.2%})")
    return filename


if __name__ == "__main__":
    generate_all_months()
