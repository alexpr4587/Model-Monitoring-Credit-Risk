from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


# ── alert severity levels 

CRITICAL = "critical"
WARNING  = "warning"
INFO     = "info"


# ── individual alert 

@dataclass
class Alert:
    rule_id:     str
    severity:    str          # "critical" | "warning" | "info"
    category:    str          # "drift" | "performance" | "calibration" | "portfolio"
    metric:      str          # metric name
    value:       float        # observed value
    threshold:   float        # threshold that was breached
    message:     str          # human-readable description
    month_label: str = ""

    @property
    def icon(self) -> str:
        return {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(self.severity, "⚪")

    def to_dict(self) -> dict:
        return {
            "rule_id":     self.rule_id,
            "severity":    self.severity,
            "category":    self.category,
            "metric":      self.metric,
            "value":       round(self.value, 4),
            "threshold":   self.threshold,
            "message":     self.message,
            "month":       self.month_label,
        }

    def __repr__(self) -> str:
        return f"{self.icon} [{self.severity.upper()}] {self.message} ({self.metric}={self.value:.4f})"


# ── alert report 

@dataclass
class AlertReport:
    month_label:        str
    alerts:             list[Alert] = field(default_factory=list)
    retrain_triggered:  bool = False
    generated_at:       str  = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def critical(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity == CRITICAL]

    @property
    def warnings(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity == WARNING]

    @property
    def n_critical(self) -> int:
        return len(self.critical)

    @property
    def n_warnings(self) -> int:
        return len(self.warnings)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    @property
    def overall_status(self) -> str:
        if self.n_critical > 0: return CRITICAL
        if self.n_warnings > 0: return WARNING
        return "healthy"

    def summary_df(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame(columns=["rule_id", "severity", "category", "metric", "value", "threshold", "message"])
        return pd.DataFrame([a.to_dict() for a in self.alerts])

    def to_json(self) -> str:
        return json.dumps({
            "month":           self.month_label,
            "overall_status":  self.overall_status,
            "retrain":         self.retrain_triggered,
            "n_critical":      self.n_critical,
            "n_warnings":      self.n_warnings,
            "alerts":          [a.to_dict() for a in self.alerts],
            "generated_at":    self.generated_at,
        }, indent=2)

    def __repr__(self) -> str:
        return (f"AlertReport(month='{self.month_label}' | "
                f"status={self.overall_status} | "
                f"critical={self.n_critical}, warnings={self.n_warnings} | "
                f"retrain={self.retrain_triggered})")


# ── rule definitions 

# Each rule: (rule_id, severity, category, metric_key, threshold, direction, message_template)
# direction: "above" → alert if value > threshold | "below" → alert if value < threshold

RULES = [
    # ── Data Drift Rules ──
    ("PSI_CRITICAL",    CRITICAL, "drift",       "psi",               0.25, "above",
     "Severe feature distribution shift detected — PSI = {value:.3f} (threshold: {threshold})"),
    ("PSI_WARNING",     WARNING,  "drift",       "psi",               0.10, "above",
     "Moderate feature drift — PSI = {value:.3f}. Monitor trend."),

    # ── Score Drift Rules ──
    ("SCORE_DRIFT_PSI", WARNING,  "drift",       "psi_score",         0.15, "above",
     "PD score distribution shifted — PSI_score = {value:.3f}"),
    ("MEAN_PD_SHIFT",   WARNING,  "drift",       "mean_pd_shift",     0.05, "above",
     "Mean predicted PD shifted by {value:.3f} — possible covariate shift"),

    # ── Model Performance Rules ──
    ("AUC_CRITICAL",    CRITICAL, "performance", "auc",               0.70, "below",
     "AUC below minimum threshold — AUC = {value:.4f} (min: {threshold})"),
    ("AUC_WARNING",     WARNING,  "performance", "auc",               0.75, "below",
     "AUC declining — AUC = {value:.4f}. Below good threshold of {threshold}."),
    ("KS_CRITICAL",     CRITICAL, "performance", "ks",                0.30, "below",
     "KS statistic below minimum — KS = {value:.4f} (min: {threshold})"),
    ("KS_WARNING",      WARNING,  "performance", "ks",                0.40, "below",
     "KS declining — KS = {value:.4f}. Below good threshold of {threshold}."),
    ("BRIER_WARNING",   WARNING,  "performance", "brier",             0.20, "above",
     "Brier score elevated — {value:.4f}. Model calibration degrading."),

    # ── Calibration Rules ──
    ("CALIB_CRITICAL",  CRITICAL, "calibration", "calibration_error", 0.05, "above",
     "Model miscalibration — predicted PD vs observed DR gap = {value:.4f}"),
    ("CALIB_WARNING",   WARNING,  "calibration", "calibration_error", 0.02, "above",
     "Calibration drift — predicted PD vs observed DR gap = {value:.4f}"),

    # ── Portfolio Rules ──
    ("DR_CRITICAL",     CRITICAL, "portfolio",   "observed_default_rate", 0.30, "above",
     "Default rate spike — {value:.1%} (threshold: {threshold:.0%})"),
    ("DR_WARNING",      WARNING,  "portfolio",   "observed_default_rate", 0.25, "above",
     "Default rate elevated — {value:.1%}. Monitor portfolio quality."),
    ("EL_WARNING",      WARNING,  "portfolio",   "el_m",              100.0, "above",
     "Expected Loss above $100M — ${value:.1f}M"),
    ("EP_CRITICAL",     CRITICAL, "portfolio",   "mean_ep_ratio",     0.00, "below",
     "Portfolio expected profit is negative — EP ratio = {value:.4f}"),
    ("EP_WARNING",      WARNING,  "portfolio",   "mean_ep_ratio",     0.03, "below",
     "EP ratio declining — {value:.4f}. Below healthy threshold of {threshold}."),
]


# ── retraining trigger logic 

def _should_retrain(alerts: list[Alert], metrics: dict) -> bool:

    critical_count = sum(1 for a in alerts if a.severity == CRITICAL)
    psi_critical   = any(a.rule_id == "PSI_CRITICAL" for a in alerts)
    auc_low        = metrics.get("auc", 1.0) < 0.72
    ks_low         = metrics.get("ks", 1.0) < 0.28

    return psi_critical or auc_low or ks_low or critical_count >= 3


# ── main evaluation function 

def evaluate_alerts(
    metrics:      dict,
    month_label:  str = "",
    feature_psi:  Optional[dict] = None,   # {"sub_grade_num": 0.32, "dti": 0.14, ...}
) -> AlertReport:

    fired_alerts: list[Alert] = []
    already_fired: set[str] = set()   # prevent duplicate rule fires per feature

    # ── Evaluate generic scalar rules 
    for rule_id, severity, category, metric_key, threshold, direction, msg_template in RULES:
        if metric_key not in metrics:
            continue
        value = metrics[metric_key]
        if value is None:
            continue

        breached = (direction == "above" and value > threshold) or \
                   (direction == "below" and value < threshold)

        if breached:
            # Avoid duplicate warnings when both WARNING + CRITICAL fire for same metric
            # (only fire the most severe one)
            base_key = f"{category}:{metric_key}"
            if base_key in already_fired and severity == WARNING:
                continue

            msg = msg_template.format(value=value, threshold=threshold)
            fired_alerts.append(Alert(
                rule_id    = rule_id,
                severity   = severity,
                category   = category,
                metric     = metric_key,
                value      = float(value),
                threshold  = threshold,
                message    = msg,
                month_label= month_label,
            ))
            if severity == CRITICAL:
                already_fired.add(base_key)

    # ── Evaluate per-feature PSI 
    if feature_psi:
        for feature, psi_val in feature_psi.items():
            if psi_val > 0.25:
                fired_alerts.append(Alert(
                    rule_id    = f"PSI_CRITICAL_{feature.upper()}",
                    severity   = CRITICAL,
                    category   = "drift",
                    metric     = "psi",
                    value      = psi_val,
                    threshold  = 0.25,
                    message    = f"Severe drift in '{feature}' — PSI = {psi_val:.3f}",
                    month_label= month_label,
                ))
            elif psi_val > 0.10:
                fired_alerts.append(Alert(
                    rule_id    = f"PSI_WARNING_{feature.upper()}",
                    severity   = WARNING,
                    category   = "drift",
                    metric     = "psi",
                    value      = psi_val,
                    threshold  = 0.10,
                    message    = f"Moderate drift in '{feature}' — PSI = {psi_val:.3f}",
                    month_label= month_label,
                ))

    # Sort: critical first, then warning
    fired_alerts.sort(key=lambda a: (0 if a.severity == CRITICAL else 1, -a.value))

    retrain = _should_retrain(fired_alerts, metrics)

    return AlertReport(
        month_label       = month_label,
        alerts            = fired_alerts,
        retrain_triggered = retrain,
    )


if __name__ == "__main__":
    # Example: month 10 metrics (degraded model)
    sample_metrics = {
        "auc":                   0.695,
        "ks":                    0.285,
        "brier":                 0.225,
        "calibration_error":     0.062,
        "observed_default_rate": 0.312,
        "el_m":                  118.0,
        "mean_ep_ratio":         0.018,
        "psi_score":             0.19,
        "mean_pd_shift":         0.06,
    }
    sample_psi = {
        "sub_grade_num": 0.38,
        "annual_inc":    0.14,
        "dti":           0.12,
        "int_rate":      0.08,
        "funded_amnt":   0.05,
        "emp_length":    0.03,
    }

    report = evaluate_alerts(sample_metrics, month_label="2018-10", feature_psi=sample_psi)
    print(report)
    print()
    for a in report.alerts:
        print(a)
    print(f"\nRetrain triggered: {report.retrain_triggered}")
    print(report.to_json())
