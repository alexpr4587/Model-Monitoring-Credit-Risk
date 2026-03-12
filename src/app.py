import sys
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "data" / "monitoring_results"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

st.set_page_config(page_title="Credit Risk Monitor", page_icon="📊", layout="wide")

# COLOURS
C_BG      = "#0B0F19"
C_CARD    = "#111726"
C_BORDER  = "rgba(255, 255, 255, 0.08)"
C_ACCENT  = "#4F46E5"
C_GREEN   = "#10B981"
C_AMBER   = "#F59E0B"
C_RED     = "#EF4444"
C_BLUE    = "#3B82F6"
C_PURPLE  = "#8B5CF6"
C_TEXT    = "#F3F4F6"
C_MUTED   = "#9CA3AF"
MONO      = "JetBrains Mono"

# css styles
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: {C_BG};
  }}

  .metric-card {{
      background: {C_CARD};
      border: 1px solid {C_BORDER};
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 12px;
      min-height: 160px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: transform 0.2s ease, border 0.2s ease;
  }}
  .metric-card:hover {{
      border: 1px solid rgba(79, 70, 229, 0.4);
      transform: translateY(-2px);
  }}
  .metric-label {{
      font-size: 11px;
      font-weight: 600;
      color: {C_MUTED};
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }}
  .metric-value {{
      font-size: 32px;
      font-weight: 700;
      color: {C_TEXT};
      line-height: 1.1;
  }}
  .metric-delta {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      margin-top: 8px;
      display: flex;
      align-items: center;
      gap: 4px;
  }}
  .alert-critical {{ background: rgba(239,68,68,0.04); border: 1px solid rgba(239,68,68,0.2); border-radius: 12px; padding: 16px; margin-bottom: 10px; }}
  .alert-warning  {{ background: rgba(245,158,11,0.04); border: 1px solid rgba(245,158,11,0.2); border-radius: 12px; padding: 16px; margin-bottom: 10px; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
  .stTabs [data-baseweb="tab"] {{
      height: 45px;
      background-color: transparent !important;
      color: {C_MUTED} !important;
      font-weight: 600;
  }}
  .stTabs [aria-selected="true"] {{
      color: {C_ACCENT} !important;
      border-bottom: 2px solid {C_ACCENT} !important;
  }}
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=C_MUTED, size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, linecolor=C_BORDER),
    yaxis=dict(showgrid=True, gridcolor=C_BORDER, zeroline=False),
)

# ── Required columns for production ingest
REQUIRED_COLS = [
    "funded_amnt", "annual_inc", "dti", "int_rate", "emp_length",
    "term", "sub_grade_num", "installment", "home_ownership", "purpose",
    "loan_to_income", "payment_burden", "log_annual_inc", "dti_x_term",
    "int_rate_residual", "inc_stability",
]


# ── Helpers

def kpi(label, value, sub=None, delta=None, delta_up=True, color=C_TEXT, inverted=False):
    if inverted:
        d_color = C_RED if delta_up else C_GREEN
    else:
        d_color = C_GREEN if delta_up else C_RED
    arrow = "↑" if delta_up else "↓"
    delta_html = f'<div class="metric-delta" style="color:{d_color};">{arrow} {delta}</div>' if delta else ""
    sub_html   = f'<div style="color:{C_MUTED}; font-size:11px; margin-top:4px; min-height:14px;">{sub if sub else ""}</div>'
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div>'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'{sub_html}'
        f'{delta_html}'
        f'</div></div>'
    )


def linechart(df, x, ycols, colors, title="", refs=None):
    fig = go.Figure()
    for col, c in zip(ycols, colors):
        if col not in df.columns:
            continue
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], mode="lines", name=col,
            line=dict(color=c, width=3),
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.03)",
        ))
    if refs:
        for r in refs:
            fig.add_hline(y=r["y"], line_color=r.get("color", C_RED), line_dash="dot", opacity=0.4)
    fig.update_layout(**LAYOUT, title=dict(text=title.upper(), font=dict(size=12, color=C_TEXT)))
    return fig


@st.cache_data(ttl=300)
def load_data():
    def rd(name):
        p = RESULTS_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    d = {k: rd(v) for k, v in {
        "perf":         "performance.csv",
        "calib":        "calibration.csv",
        "portfolio":    "portfolio.csv",
        "psi":          "psi_timeseries.csv",
        "pred_drift":   "prediction_drift.csv",
        "alerts":       "alerts.csv",
        "alert_summary":"alert_summary.csv",
    }.items()}
    jp = RESULTS_DIR / "latest_alert.json"
    d["aj"] = json.loads(jp.read_text()) if jp.exists() else {}
    return d


# ── Header

def header(d):
    aj      = d.get("aj", {})
    status  = aj.get("overall_status", "healthy").upper()
    month   = aj.get("month", "—")
    sc      = C_GREEN if status == "HEALTHY" else C_RED
    # pre-compute to avoid backslash in f-string
    sc_dot  = f'<span style="color:{sc};">●</span>'
    st.markdown(
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem;">'
        f'<div>'
        f'<h1 style="font-size:22px; margin:0; color:{C_TEXT};">Risk Engine Monitor</h1>'
        f'<p style="color:{C_MUTED}; font-size:11px; font-family:{MONO};">LendingClub Production Pipeline</p>'
        f'</div>'
        f'<div style="display:flex; gap:24px;">'
        f'<div style="text-align:right;">'
        f'<div style="font-size:10px; color:{C_MUTED}; letter-spacing:1px;">STATUS</div>'
        f'<div style="color:{sc}; font-weight:700; font-size:13px;">{sc_dot} {status}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:10px; color:{C_MUTED}; letter-spacing:1px;">PERIOD</div>'
        f'<div style="color:{C_TEXT}; font-weight:700; font-size:13px;">{month}</div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ── Tab: Performance

def tab_performance(d):
    df = d["perf"]
    if df.empty:
        return
    lt, pv = df.iloc[-1], (df.iloc[-2] if len(df) > 1 else df.iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("ROC-AUC", f"{lt['auc']:.3f}",
                        delta=f"{abs(lt['auc']-pv['auc']):.3f}", delta_up=lt["auc"] >= pv["auc"]),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("KS STAT", f"{lt['ks']:.3f}",
                        delta=f"{abs(lt['ks']-pv['ks']):.3f}", delta_up=lt["ks"] >= pv["ks"]),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("BRIER SCORE", f"{lt['brier']:.3f}", sub="Calibration",
                        delta=f"{abs(lt['brier']-pv['brier']):.3f}",
                        delta_up=lt["brier"] > pv["brier"], inverted=True),
                    unsafe_allow_html=True)
    with c4:
        dr_color = C_RED if lt["default_rate"] > 0.3 else C_TEXT
        st.markdown(kpi("DEFAULT RATE", f"{lt['default_rate']:.1%}",
                        sub=f"N={int(lt['n_samples']):,}",
                        delta=f"{abs(lt['default_rate']-pv['default_rate']):.1%}",
                        delta_up=lt["default_rate"] > pv["default_rate"],
                        inverted=True, color=dr_color),
                    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(linechart(df, "month", ["auc", "ks"], [C_ACCENT, C_GREEN],
                                  "Model Discrimination"), use_container_width=True)
    with cr:
        st.plotly_chart(linechart(df, "month", ["brier"], [C_AMBER],
                                  "Calibration Quality"), use_container_width=True)


# ── Tab: Drift

def tab_drift(d):
    df = d["psi"]
    if df.empty:
        return
    st.markdown(
        f'<div style="margin-bottom:1rem;">'
        f'<h3 style="font-size:13px; color:{C_TEXT}; letter-spacing:1px;">POPULATION STABILITY (PSI)</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )
    piv = df.pivot(index="feature", columns="month", values="psi")
    fig = px.imshow(
        piv,
        color_continuous_scale=[[0, C_CARD], [0.3, C_AMBER], [1, C_RED]],
        zmin=0, zmax=0.5, aspect="auto",
    )
    fig.update_layout(**LAYOUT, height=350)
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        fig2 = go.Figure()
        feat_colors = [C_BLUE, C_PURPLE, C_AMBER, C_RED, C_ACCENT]
        for i, feat in enumerate(df["feature"].unique()[:5]):
            dff = df[df["feature"] == feat]
            fig2.add_trace(go.Scatter(
                x=dff["month"], y=dff["psi"],
                name=feat.replace("_num", "").upper(),
                line=dict(width=2.5, color=feat_colors[i]),
            ))
        fig2.update_layout(**LAYOUT, title_text="STABILITY TRENDS")
        st.plotly_chart(fig2, use_container_width=True)
    with cr:
        dpd = d["pred_drift"]
        if not dpd.empty:
            st.plotly_chart(linechart(dpd, "month",
                                      ["mean_pd_baseline", "mean_pd_production"],
                                      [C_MUTED, C_RED], "Shift in PD Predictions"),
                            use_container_width=True)


# ── Tab: Portfolio

def tab_portfolio(d):
    df = d["portfolio"]
    if df.empty:
        return
    lt = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("APPROVALS", f"{lt['approval_rate']:.1%}", sub="Selection Rate"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("EXP. LOSS", f"${lt['total_el']/1e6:.1f}M", sub="Portfolio EL"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("EP RATIO", f"{lt['mean_ep_ratio']:.2%}", sub="Profitability"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("MEAN LGD", f"{lt['mean_lgd']:.1%}", sub="Severity"),
                    unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(linechart(df, "month",
                                  ["approval_rate", "observed_default_rate"],
                                  [C_GREEN, C_RED], "Volume vs Performance"),
                        use_container_width=True)
    with cr:
        st.plotly_chart(linechart(df, "month", ["mean_ep_ratio"], [C_ACCENT],
                                  "Expected Profit Trend"),
                        use_container_width=True)


# ── Tab: Alerts

def tab_alerts(d):
    aj   = d.get("aj", {})
    ds   = d["alert_summary"]
    acts = aj.get("alerts", [])

    col_main, col_side = st.columns([2, 1.1])

    with col_main:
        st.markdown(
            f'<h3 style="font-size:14px; color:{C_TEXT}; margin-bottom:1.2rem;">ACTIVE ANOMALIES</h3>',
            unsafe_allow_html=True,
        )
        if not acts:
            st.success("System operational. No active alerts.")
        else:
            for a in acts:
                css     = "alert-critical" if a.get("severity") == "critical" else "alert-warning"
                msg     = a.get("message", "")
                metric  = a.get("metric", "").upper()
                val     = a.get("value", 0)
                thresh  = a.get("threshold", 0)
                mo      = a.get("month", "")
                st.markdown(
                    f'<div class="{css}">'
                    f'<div style="font-weight:700; color:{C_TEXT}; font-size:13px;">{msg}</div>'
                    f'<div style="color:{C_MUTED}; font-size:11px; font-family:{MONO}; margin-top:5px;">'
                    f'{metric} : {val:.4f} (Threshold: {thresh}) · {mo}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        if not ds.empty:
            st.markdown(
                f'<br><h3 style="font-size:14px; color:{C_TEXT}; margin-bottom:1rem;">ALERT HISTORY</h3>',
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ds["month"], y=ds["n_critical"], name="Critical",
                                 marker=dict(color=C_RED, line=dict(width=0), cornerradius=4)))
            fig.add_trace(go.Bar(x=ds["month"], y=ds["n_warnings"], name="Warnings",
                                 marker=dict(color=C_AMBER, line=dict(width=0), cornerradius=4)))
            fig.update_layout(**LAYOUT, barmode="stack", height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown(
            f'<h3 style="font-size:14px; color:{C_TEXT}; margin-bottom:1.2rem;">MONITORING RULES</h3>',
            unsafe_allow_html=True,
        )
        RULES = [
            ("PSI > 0.25",        "Severe feature drift",    C_RED),
            ("PSI > 0.10",        "Moderate drift",          C_AMBER),
            ("AUC < 0.70",        "Performance critical",    C_RED),
            ("AUC < 0.75",        "AUC declining",           C_AMBER),
            ("KS < 0.30",         "Discrimination loss",     C_RED),
            ("Brier > 0.20",      "Calibration degrading",   C_AMBER),
            ("Default rate > 30%","Portfolio stress",        C_RED),
            ("EL > $100M",        "Expected loss rising",    C_AMBER),
            ("EP ratio < 0%",     "Portfolio unprofitable",  C_RED),
        ]
        for rule, desc, color in RULES:
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; align-items:center;'
                f' padding:10px 0; border-bottom:1px solid {C_BORDER};">'
                f'<div style="font-family:{MONO}; font-size:11px; color:{color};">{rule}</div>'
                f'<div style="color:{C_MUTED}; font-size:11px;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<br><h3 style="font-size:14px; color:{C_TEXT}; margin-bottom:1rem;">STATUS ENGINE</h3>',
            unsafe_allow_html=True,
        )
        rt = aj.get("retrain", False)
        st.info("Retraining: " + ("TRIGGERED" if rt else "STANDBY"))


# ── Production Mode helpers

def _append_results(new_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_map = {
        "perf":          "performance.csv",
        "calib":         "calibration.csv",
        "portfolio":     "portfolio.csv",
        "psi":           "psi_timeseries.csv",
        "pred_drift":    "prediction_drift.csv",
        "alerts":        "alerts.csv",
        "alert_summary": "alert_summary.csv",
    }
    for key, fname in file_map.items():
        new_df = new_results.get(key)
        if new_df is None or new_df.empty:
            continue
        path = RESULTS_DIR / fname
        if path.exists():
            existing = pd.read_csv(path)
            month_label = new_df["month"].iloc[0] if "month" in new_df.columns else None
            if month_label and "month" in existing.columns:
                existing = existing[existing["month"] != month_label]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(path, index=False)

    if new_results.get("alert_json"):
        (RESULTS_DIR / "latest_alert.json").write_text(
            json.dumps(new_results["alert_json"], indent=2)
        )


def _run_single_month(prod_df, month_label):
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from monitoring.psi                 import calculate_psi_report
    from monitoring.performance_metrics import calculate_performance
    from monitoring.calibration         import calculate_calibration
    from monitoring.prediction_drift    import calculate_prediction_drift
    from portfolio.business_metrics     import calculate_portfolio_metrics
    from alerts.alert_rules             import evaluate_alerts
    from pipeline.monitoring_pipeline   import ModelRegistry

    baseline_path = PROJECT_ROOT / "data" / "baseline" / "train_snapshot.csv"
    if not baseline_path.exists():
        st.error("Baseline snapshot not found. Run generate_baseline.py first.")
        return {}

    baseline_df = pd.read_csv(baseline_path)
    registry    = ModelRegistry().load()
    base_scores = registry.predict_pd(baseline_df)

    pd_scores  = registry.predict_pd(prod_df)
    lgd_scores = registry.predict_lgd(prod_df)
    ead_scores = registry.predict_ead(prod_df)

    from credit_risk_pipeline import ApprovalModel
    approved = ApprovalModel(hurdle_rate=0.0).approve(
        pd_score    = pd_scores,
        lgd_hat     = lgd_scores,
        ead_hat     = ead_scores,
        funded_amnt = prod_df["funded_amnt"].values,
        int_rate    = prod_df["int_rate"].values,
        term        = prod_df["term"].values,
    ).astype(bool)

    all_features = [c for c in REQUIRED_COLS if c in baseline_df.columns and c in prod_df.columns]

    psi_report  = calculate_psi_report(baseline_df, prod_df, features=all_features, month_label=month_label)
    pred_drift  = calculate_prediction_drift(base_scores, pd_scores, month_label=month_label)
    portfolio   = calculate_portfolio_metrics(prod_df, pd_scores, lgd_scores, ead_scores, approved, month_label=month_label)
    feature_psi = {r.feature: r.psi for r in psi_report.results}

    metrics = {
        "psi_score":             pred_drift.psi_score,
        "mean_pd_shift":         abs(pred_drift.mean_pd_shift),
        "observed_default_rate": portfolio.observed_default_rate,
        "el_m":                  portfolio.el_m,
        "mean_ep_ratio":         portfolio.mean_ep_ratio,
    }

    perf = calib = None
    if "is_default" in prod_df.columns:
        y     = prod_df["is_default"].values
        perf  = calculate_performance(y, pd_scores, month_label=month_label)
        calib = calculate_calibration(y, pd_scores, month_label=month_label)
        metrics.update({
            "auc":               perf.auc,
            "ks":                perf.ks,
            "brier":             perf.brier,
            "calibration_error": calib.calibration_error,
        })

    alert_report = evaluate_alerts(metrics, month_label=month_label, feature_psi=feature_psi)

    psi_rows = [
        {"month": month_label, "feature": r.feature, "psi": round(r.psi, 4), "status": r.status}
        for r in psi_report.results
    ]

    return {
        "perf":          pd.DataFrame([perf.to_dict()])  if perf  else pd.DataFrame(),
        "calib":         pd.DataFrame([calib.to_dict()]) if calib else pd.DataFrame(),
        "portfolio":     pd.DataFrame([portfolio.to_dict()]),
        "psi":           pd.DataFrame(psi_rows),
        "pred_drift":    pd.DataFrame([pred_drift.to_dict()]),
        "alerts":        alert_report.summary_df(),
        "alert_summary": pd.DataFrame([{
            "month":          month_label,
            "n_critical":     alert_report.n_critical,
            "n_warnings":     alert_report.n_warnings,
            "retrain":        alert_report.retrain_triggered,
            "overall_status": alert_report.overall_status,
        }]),
        "alert_json": json.loads(alert_report.to_json()),
    }


# ── Tab: Ingest

def tab_ingest(d):
    st.markdown(
        f'<div style="margin-bottom:1.5rem;">'
        f'<h3 style="font-size:14px; color:{C_TEXT}; letter-spacing:1px; margin-bottom:6px;">PRODUCTION INGEST</h3>'
        f'<p style="color:{C_MUTED}; font-size:12px; margin:0;">'
        f'Upload a monthly portfolio CSV to run monitoring and append results to history.'
        f'</p></div>',
        unsafe_allow_html=True,
    )

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        month_label = st.text_input(
            "Month label (YYYY-MM)",
            value=datetime.date.today().strftime("%Y-%m"),
            help="Used as the x-axis label in all charts.",
        )

        uploaded = st.file_uploader(
            "Upload portfolio CSV",
            type=["csv"],
            help="Same schema as data/production/ files.",
        )

        if uploaded is not None:
            prod_df = pd.read_csv(uploaded)

            missing = [c for c in REQUIRED_COLS if c not in prod_df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                return

            # pre-compute label badge to avoid backslash in f-string
            has_labels = "is_default" in prod_df.columns
            label_badge = (
                f'<span style="color:{C_GREEN};">labels present</span>'
                if has_labels else
                f'<span style="color:{C_AMBER};">no labels — perf metrics skipped</span>'
            )

            st.markdown(
                f'<div style="background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:12px; padding:16px; margin:12px 0;">'
                f'<div style="color:{C_MUTED}; font-size:11px; letter-spacing:1px; margin-bottom:8px;">FILE PREVIEW</div>'
                f'<div style="font-family:{MONO}; font-size:12px; color:{C_TEXT};">'
                f'{len(prod_df):,} rows &nbsp;·&nbsp; {len(prod_df.columns)} columns &nbsp;·&nbsp; '
                f'Month: <span style="color:{C_ACCENT};">{month_label}</span>'
                f' &nbsp;·&nbsp; {label_badge}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            existing_perf = d.get("perf", pd.DataFrame())
            if (not existing_perf.empty
                    and "month" in existing_perf.columns
                    and month_label in existing_perf["month"].values):
                st.warning(f"Month `{month_label}` already exists. Running will overwrite it.")

            if st.button("▶ Run Monitoring", type="primary", use_container_width=True):
                with st.spinner(f"Running pipeline for {month_label}..."):
                    try:
                        results = _run_single_month(prod_df, month_label)
                        if results:
                            _append_results(results)
                            st.cache_data.clear()
                            st.success(f"Month {month_label} ingested successfully.")

                            cols = st.columns(4)
                            port = results["portfolio"]
                            perf = results["perf"]
                            alert_s = results["alert_summary"]
                            summaries = []
                            if not port.empty:
                                summaries += [
                                    ("DEFAULT RATE", f"{port['observed_default_rate'].iloc[0]:.1%}"),
                                    ("EXPECTED LOSS", f"${port['total_el'].iloc[0]/1e6:.1f}M"),
                                ]
                            if not perf.empty:
                                summaries += [
                                    ("AUC", f"{perf['auc'].iloc[0]:.4f}"),
                                    ("KS",  f"{perf['ks'].iloc[0]:.4f}"),
                                ]
                            for i, (lbl, val) in enumerate(summaries[:4]):
                                with cols[i]:
                                    st.metric(lbl, val)

                            if not alert_s.empty and alert_s["retrain"].iloc[0]:
                                st.error("🔁 Retraining triggered for this month.")

                            st.info("Reload the page to see updated charts.")
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        raise

    with col_info:
        schema_rows = [
            ("funded_amnt",    "Loan amount ($)"),
            ("annual_inc",     "Borrower income"),
            ("dti",            "Debt-to-income ratio"),
            ("int_rate",       "Interest rate (%)"),
            ("sub_grade_num",  "Grade 1–35"),
            ("term",           "36 or 60 months"),
            ("home_ownership", "rent/mortgage/own"),
            ("purpose",        "Loan purpose"),
            ("loan_to_income", "Engineered feature"),
            ("is_default",     "Optional — 0/1 label"),
        ]
        st.markdown(
            f'<div style="background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:12px; padding:20px;">'
            f'<div style="color:{C_MUTED}; font-size:11px; letter-spacing:1px; margin-bottom:12px;">EXPECTED SCHEMA</div>',
            unsafe_allow_html=True,
        )
        for col_name, desc in schema_rows:
            col_color = C_ACCENT if col_name == "is_default" else C_TEXT
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid {C_BORDER};">'
                f'<div style="font-family:{MONO}; font-size:10px; color:{col_color};">{col_name}</div>'
                f'<div style="color:{C_MUTED}; font-size:10px;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        perf_path = RESULTS_DIR / "performance.csv"
        if perf_path.exists():
            existing_months = pd.read_csv(perf_path)["month"].tolist()
            if existing_months:
                st.markdown(
                    f'<div style="background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:12px; padding:20px; margin-top:12px;">'
                    f'<div style="color:{C_MUTED}; font-size:11px; letter-spacing:1px; margin-bottom:10px;">INGESTED MONTHS ({len(existing_months)})</div>',
                    unsafe_allow_html=True,
                )
                for m in existing_months:
                    st.markdown(
                        f'<div style="font-family:{MONO}; font-size:11px; color:{C_TEXT}; padding:4px 0;">● {m}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)


# ── Main

def main():
    data = load_data()
    tabs = ["Performance", "Stability", "Portfolio", "Signals", "⬆ Ingest"]
    if data["portfolio"].empty:
        st.warning("Monitoring data not found. Ensure pipeline has run.")
        t = st.tabs(tabs)
        with t[4]:
            tab_ingest(data)
        return
    header(data)
    t = st.tabs(tabs)
    with t[0]: tab_performance(data)
    with t[1]: tab_drift(data)
    with t[2]: tab_portfolio(data)
    with t[3]: tab_alerts(data)
    with t[4]: tab_ingest(data)


if __name__ == "__main__":
    main()
