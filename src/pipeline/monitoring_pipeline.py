import io
import sys
import json
import pickle
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))       # so  `import src.X`  works
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # so bare `import monitoring.X` works too

from monitoring.psi              import calculate_psi_report, PSIReport
from monitoring.performance_metrics import calculate_performance, performance_timeseries
from monitoring.calibration      import calculate_calibration, calibration_timeseries
from monitoring.prediction_drift import calculate_prediction_drift
from portfolio.business_metrics  import calculate_portfolio_metrics, portfolio_timeseries
from alerts.alert_rules          import evaluate_alerts, AlertReport
from credit_risk_pipeline import (
    ApprovalModel,
    PDPreprocessor,
    PDModel,
    ColumnNameCleaner,
    CategoricalCleaner,
    EmpLengthTransformer,
    TermTransformer,
    LoanStatusCleaner,
    GradeTransformer,
    IssueDateTransformer,
    Winsorizer,
    WoEEncoder,
    FeatureDropper,
    LoanFeatureEngineer,
)


import credit_risk_pipeline as _crp

_CLASS_REMAP = {
    name: getattr(_crp, name)
    for name in dir(_crp)
    if not name.startswith("_")
}

class _CreditRiskUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("__main__", "credit_risk_pipeline") and name in _CLASS_REMAP:
            return _CLASS_REMAP[name]
        return super().find_class(module, name)

def _safe_load(path: Path):
    try:
        # Try joblib first (handles numpy arrays efficiently)
        return joblib.load(path)
    except AttributeError:
        # Fall back to custom unpickler when __main__ remapping is needed
        with open(path, "rb") as f:
            return _CreditRiskUnpickler(f).load()


logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR    = PROJECT_ROOT / "models"
DATA_DIR      = PROJECT_ROOT / "data"
BASELINE_PATH = DATA_DIR / "baseline" / "train_snapshot.csv"
PROD_DIR      = DATA_DIR / "production"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "monitoring_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "funded_amnt", "annual_inc", "dti", "int_rate",
    "emp_length", "term", "sub_grade_num",
    "loan_to_income", "payment_burden", "log_annual_inc",
    "dti_x_term", "int_rate_residual", "inc_stability",
]
CATEGORICAL_FEATURES = ["home_ownership", "purpose"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HURDLE_RATE = 0.0   # EP-positive approval (calibrated in AprovalModel.ipynb)



class ModelRegistry:

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir   = models_dir
        self._models      = {}
        self._loaded      = False

    def load(self) -> "ModelRegistry":
        artifacts = [
            "pd_preprocessor", "pd_model",
            "lgd_preprocessor", "lgd_model",
            "ead_preprocessor", "ead_model",
        ]
        for name in artifacts:
            path = self.models_dir / f"{name}.pkl"
            if path.exists():
                self._models[name] = _safe_load(path)
                log.info(f"  Loaded {name}.pkl")
            else:
                log.warning(f"  Model artifact not found: {path} — using dummy predictor")
                self._models[name] = None

        self.approval_model = ApprovalModel(hurdle_rate=HURDLE_RATE)
        self._loaded = True
        return self

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted PD probabilities."""
        if self._models.get("pd_preprocessor") and self._models.get("pd_model"):
            X_t = self._models["pd_preprocessor"].transform(X)
            return self._models["pd_model"].predict_proba(X_t)
        # Fallback: deterministic synthetic PD from features
        log.debug("Using synthetic PD predictor (no model loaded)")
        return _synthetic_pd(X)

    def predict_lgd(self, X: pd.DataFrame) -> np.ndarray:
        if self._models.get("lgd_preprocessor") and self._models.get("lgd_model"):
            X_t = self._models["lgd_preprocessor"].transform(X)
            return np.clip(self._models["lgd_model"].predict(X_t), 0, 1)
        return _synthetic_lgd(X)

    def predict_ead(self, X: pd.DataFrame) -> np.ndarray:
        """Returns EAD in dollars (funded_amnt × EAD_ratio)."""
        if self._models.get("ead_preprocessor") and self._models.get("ead_model"):
            X_t = self._models["ead_preprocessor"].transform(X)
            ead_ratio = np.clip(self._models["ead_model"].predict(X_t), 0, 1)
            return ead_ratio * X["funded_amnt"].values
        return _synthetic_ead(X)



def _synthetic_pd(df: pd.DataFrame) -> np.ndarray:
    """Calibrated synthetic PD for demo runs without .pkl files."""
    rng        = np.random.default_rng(42)
    grade_term = (df["sub_grade_num"].values - 1) / 34   # 0→1
    dti_norm   = np.clip(df["dti"].values / 40, 0, 1)
    base       = 0.05 + grade_term * 0.55 + dti_norm * 0.15
    noise      = rng.normal(0, 0.04, size=len(df))
    return np.clip(base + noise, 0.01, 0.99)

def _synthetic_lgd(df: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(43)
    base = np.full(len(df), 0.40)
    return np.clip(base + rng.normal(0, 0.08, size=len(df)), 0.05, 0.95)

def _synthetic_ead(df: pd.DataFrame) -> np.ndarray:
    rng      = np.random.default_rng(44)
    ead_ratio = np.clip(0.75 + rng.normal(0, 0.10, size=len(df)), 0.3, 1.0)
    return ead_ratio * df["funded_amnt"].values



def run_monthly(
    prod_df:     pd.DataFrame,
    baseline_df: pd.DataFrame,
    registry:    ModelRegistry,
    month_label: str,
    baseline_pd_scores: Optional[np.ndarray] = None,
) -> dict:
    log.info(f"  Running month: {month_label}  ({len(prod_df):,} applications)")

    pd_scores  = registry.predict_pd(prod_df)
    lgd_scores = registry.predict_lgd(prod_df)
    ead_scores = registry.predict_ead(prod_df)

    approved_mask = registry.approval_model.approve(
        pd_score    = pd_scores,
        lgd_hat     = lgd_scores,
        ead_hat     = ead_scores,
        funded_amnt = prod_df["funded_amnt"].values,
        int_rate    = prod_df["int_rate"].values,
        term        = prod_df["term"].values,
    ).astype(bool)

    psi_report: PSIReport = calculate_psi_report(
        baseline   = baseline_df,
        production = prod_df,
        features   = ALL_FEATURES,
        month_label= month_label,
    )
    feature_psi_dict = {r.feature: r.psi for r in psi_report.results}

    if baseline_pd_scores is None:
        baseline_pd_scores = registry.predict_pd(baseline_df)

    pred_drift = calculate_prediction_drift(
        baseline_scores   = baseline_pd_scores,
        production_scores = pd_scores,
        month_label       = month_label,
    )

    if "is_default" in prod_df.columns:
        y_true = prod_df["is_default"].values
        perf   = calculate_performance(y_true, pd_scores, month_label=month_label)
        calib  = calculate_calibration(y_true, pd_scores, month_label=month_label)
    else:
        log.warning(f"  No labels for {month_label} — skipping performance metrics")
        perf  = None
        calib = None

    portfolio = calculate_portfolio_metrics(
        df            = prod_df,
        pd_scores     = pd_scores,
        lgd_scores    = lgd_scores,
        ead_scores    = ead_scores,
        approved_mask = approved_mask,
        month_label   = month_label,
    )

    metrics_for_alerts = {
        "auc":                   perf.auc   if perf  else None,
        "ks":                    perf.ks    if perf  else None,
        "brier":                 perf.brier if perf  else None,
        "calibration_error":     calib.calibration_error if calib else None,
        "observed_default_rate": portfolio.observed_default_rate,
        "el_m":                  portfolio.el_m,
        "mean_ep_ratio":         portfolio.mean_ep_ratio,
        "psi_score":             pred_drift.psi_score,
        "mean_pd_shift":         abs(pred_drift.mean_pd_shift),
    }
    alert_report = evaluate_alerts(
        metrics     = {k: v for k, v in metrics_for_alerts.items() if v is not None},
        month_label = month_label,
        feature_psi = feature_psi_dict,
    )

    log.info(f"    → alerts: {alert_report.n_critical} critical, "
             f"{alert_report.n_warnings} warnings | "
             f"retrain={alert_report.retrain_triggered}")

    return {
        "psi_report":    psi_report,
        "pred_drift":    pred_drift,
        "perf":          perf,
        "calib":         calib,
        "portfolio":     portfolio,
        "alert_report":  alert_report,
    }



def run_pipeline(
    production_dir: Path = PROD_DIR,
    output_dir:     Path = OUTPUT_DIR,
) -> dict:
    log.info("=" * 60)
    log.info("CREDIT RISK MONITORING PIPELINE")
    log.info("=" * 60)

    log.info("\n[1/5] Loading model artifacts...")
    registry = ModelRegistry().load()

    log.info("\n[2/5] Loading baseline data...")
    if not BASELINE_PATH.exists():
        log.info("  Baseline not found — generating now...")
        from simulation.generate_baseline import generate_baseline
        baseline_df = generate_baseline()
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        baseline_df.to_csv(BASELINE_PATH, index=False)
    else:
        baseline_df = pd.read_csv(BASELINE_PATH)
    log.info(f"  Baseline loaded: {len(baseline_df):,} rows")

    # Pre-compute baseline PD scores once
    baseline_pd_scores = registry.predict_pd(baseline_df)

    log.info("\n[3/5] Loading production data...")
    prod_files = sorted(production_dir.glob("*.csv"))
    if not prod_files:
        log.info("  No production files found — generating now...")
        from simulation.generate_production import generate_all_months
        generate_all_months()
        prod_files = sorted(production_dir.glob("*.csv"))

    log.info(f"  Found {len(prod_files)} production files")

    log.info("\n[4/5] Running monthly monitoring checks...")

    all_perf      = []
    all_calib     = []
    all_portfolio = []
    all_pred_drift= []
    all_alerts    = []
    all_psi_rows  = []

    for f in prod_files:
        month_label = f.stem.replace("_", "-")   # "2018_03" → "2018-03"
        prod_df     = pd.read_csv(f)

        results = run_monthly(
            prod_df            = prod_df,
            baseline_df        = baseline_df,
            registry           = registry,
            month_label        = month_label,
            baseline_pd_scores = baseline_pd_scores,
        )

        # Collect PSI results
        for psi_r in results["psi_report"].results:
            all_psi_rows.append({
                "month":   month_label,
                "feature": psi_r.feature,
                "psi":     round(psi_r.psi, 4),
                "status":  psi_r.status,
            })

        if results["perf"]:
            all_perf.append(results["perf"])
        if results["calib"]:
            all_calib.append(results["calib"])

        all_portfolio.append(results["portfolio"])
        all_pred_drift.append(results["pred_drift"])
        all_alerts.append(results["alert_report"])

    log.info("\n[5/5] Saving results...")

    df_perf      = performance_timeseries(all_perf)      if all_perf      else pd.DataFrame()
    df_calib     = calibration_timeseries(all_calib)     if all_calib     else pd.DataFrame()
    df_portfolio = portfolio_timeseries(all_portfolio)
    df_psi       = pd.DataFrame(all_psi_rows)
    df_pred_drift = pd.DataFrame([r.to_dict() for r in all_pred_drift])
    df_alerts    = pd.concat([r.summary_df() for r in all_alerts], ignore_index=True) \
                   if any(r.has_alerts for r in all_alerts) else pd.DataFrame()

    # Alert summary per month
    df_alert_summary = pd.DataFrame([{
        "month":           r.month_label,
        "n_critical":      r.n_critical,
        "n_warnings":      r.n_warnings,
        "retrain":         r.retrain_triggered,
        "overall_status":  r.overall_status,
    } for r in all_alerts])

    # Save to CSV
    output_files = {
        "performance.csv":      df_perf,
        "calibration.csv":      df_calib,
        "portfolio.csv":        df_portfolio,
        "psi_timeseries.csv":   df_psi,
        "prediction_drift.csv": df_pred_drift,
        "alerts.csv":           df_alerts,
        "alert_summary.csv":    df_alert_summary,
    }
    for fname, df in output_files.items():
        if not df.empty:
            path = output_dir / fname
            df.to_csv(path, index=False)
            log.info(f"  Saved {fname}  ({len(df)} rows)")

    # Save latest alert report as JSON
    latest_alert = all_alerts[-1] if all_alerts else None
    if latest_alert:
        json_path = output_dir / "latest_alert.json"
        json_path.write_text(latest_alert.to_json())
        log.info(f"  Saved latest_alert.json")

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  Months processed : {len(prod_files)}")
    log.info(f"  Total alerts     : {len(df_alerts) if not df_alerts.empty else 0}")
    if df_alert_summary["retrain"].any():
        first_retrain = df_alert_summary[df_alert_summary["retrain"]].iloc[0]["month"]
        log.info(f"  First retrain trigger: {first_retrain}")
    log.info("=" * 60)

    return {
        "performance":    df_perf,
        "calibration":    df_calib,
        "portfolio":      df_portfolio,
        "psi":            df_psi,
        "prediction_drift": df_pred_drift,
        "alerts":         df_alerts,
        "alert_summary":  df_alert_summary,
    }


if __name__ == "__main__":
    results = run_pipeline()
