"""
app/copilot.py — Demand Forecast Explainer Dashboard
Run: streamlit run app/copilot.py
"""
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from features import build_features, get_feature_cols
from inference import Forecaster

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecast Explainer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b22; }
.metric-card {
    background: #1c2128; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 12px;
    border-left: 4px solid #58a6ff;
}
.driver-card {
    background: #1c2128; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
}
.driver-up   { border-left: 4px solid #3fb950; }
.driver-down { border-left: 4px solid #f85149; }
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 12px; font-weight: 600;
}
.badge-up   { background: #1a3a2a; color: #3fb950; }
.badge-down { background: #3a1a1a; color: #f85149; }
h1,h2,h3,h4,p,label,div { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data & model (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_forecaster():
    return Forecaster()

@st.cache_data
def load_processed():
    df = pd.read_csv(ROOT / "data" / "processed" / "train_ready.csv", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_metrics():
    with open(ROOT / "artifacts" / "metrics.json") as f:
        return json.load(f)

forecaster = load_forecaster()
df_all     = load_processed()
metrics    = load_metrics()
stores     = sorted(df_all["Store"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Demand Forecast\nExplainer")
    st.markdown("---")
    page = st.radio("Navigate", ["🏪 Store Forecast", "📊 Model Performance", "📂 Batch Upload"])
    st.markdown("---")
    st.markdown("**Model metrics**")
    st.metric("Test MAPE", f"{metrics['test_mape']*100:.1f}%")
    st.metric("Naive baseline MAPE", f"{metrics['naive_mape']*100:.1f}%")
    st.metric("Test RMSE", f"€{metrics['test_rmse']:,.0f}")
    st.metric("Features", metrics["n_features"])
    st.markdown("---")
    st.caption("Rossmann Store Sales · LightGBM + SHAP")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Store Forecast
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏪 Store Forecast":
    st.markdown("# 🏪 Store Sales Forecast")
    st.markdown("Select a store to see its historical sales, forecast, and SHAP-driven explanations.")

    col_sel, col_what = st.columns([2, 2])
    with col_sel:
        selected_store = st.selectbox("Store", stores, index=0)
    with col_what:
        whatif_promo = st.toggle("🏷️ Simulate promotion ON", value=False)

    store_df = df_all[df_all["Store"] == selected_store].copy()
    store_feat = build_features(store_df, is_train=True)
    feature_cols = forecaster.feature_cols

    # Fill any missing feature cols (edge cases)
    for c in feature_cols:
        if c not in store_feat.columns:
            store_feat[c] = 0

    # What-if: flip promo
    if whatif_promo:
        store_feat["Promo"] = 1

    preds = forecaster.predict(store_feat)
    store_feat["pred_median"] = preds["median"]
    store_feat["pred_q10"]    = preds["q10"]
    store_feat["pred_q90"]    = preds["q90"]

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    recent = store_feat.tail(30)
    k1.metric("Avg Daily Sales (last 30d)", f"€{recent['Sales'].mean():,.0f}")
    k2.metric("Avg Forecast (last 30d)",    f"€{recent['pred_median'].mean():,.0f}")
    k3.metric("Forecast Range (last 30d)",
              f"€{recent['pred_q10'].mean():,.0f} – €{recent['pred_q90'].mean():,.0f}")
    promo_lift = (store_feat[store_feat["Promo"]==1]["pred_median"].mean() /
                  (store_feat[store_feat["Promo"]==0]["pred_median"].mean() + 1e-9) - 1) * 100
    k4.metric("Promo lift (est.)", f"+{promo_lift:.1f}%")

    # ── Time-series chart ─────────────────────────────────────────────────────
    st.markdown("### Sales vs Forecast")
    plot_df = store_feat.tail(120)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["pred_q10"],
        fill=None, mode="lines", line=dict(width=0),
        showlegend=False, name="Q10"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["pred_q90"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(88,166,255,0.15)", name="80% forecast band"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["Sales"],
        mode="lines", name="Actual Sales",
        line=dict(color="#e6edf3", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["pred_median"],
        mode="lines", name="Forecast (median)",
        line=dict(color="#58a6ff", width=2, dash="dot")
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        height=360, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP Driver Cards ─────────────────────────────────────────────────────
    st.markdown("### Why is this forecast high or low?")
    st.caption("SHAP values show how each feature pushes the forecast up or down from the average.")

    latest_row = store_feat.tail(1)
    drivers    = forecaster.explain(latest_row)[0]

    left_col, right_col = st.columns(2)
    pos_drivers = [d for d in drivers if d["shap_value"] > 0]
    neg_drivers = [d for d in drivers if d["shap_value"] <= 0]

    with left_col:
        st.markdown("**🟢 Factors increasing forecast**")
        for d in pos_drivers[:3]:
            st.markdown(f"""
<div class="driver-card driver-up">
  <b>{d['description']}</b><br>
  <span class="badge badge-up">+€{abs(d['shap_value']):,.0f} impact</span>
  &nbsp; <span style="color:#8b949e;font-size:12px">value: {d['raw_value']}</span>
</div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown("**🔴 Factors decreasing forecast**")
        for d in neg_drivers[:3]:
            st.markdown(f"""
<div class="driver-card driver-down">
  <b>{d['description']}</b><br>
  <span class="badge badge-down">−€{abs(d['shap_value']):,.0f} impact</span>
  &nbsp; <span style="color:#8b949e;font-size:12px">value: {d['raw_value']}</span>
</div>""", unsafe_allow_html=True)

    # ── Weekly pattern ────────────────────────────────────────────────────────
    st.markdown("### Weekly sales pattern")
    dow_map = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
    weekly = store_df.groupby("DayOfWeek")["Sales"].mean().reset_index()
    weekly["Day"] = weekly["DayOfWeek"].map(dow_map)
    fig2 = px.bar(weekly, x="Day", y="Sales",
                  color="Sales", color_continuous_scale="Blues",
                  template="plotly_dark")
    fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                       height=260, margin=dict(l=0,r=0,t=10,b=0),
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAPE",           f"{metrics['test_mape']*100:.1f}%")
    m2.metric("Naive Baseline MAPE", f"{metrics['naive_mape']*100:.1f}%")
    m3.metric("Test RMSE",           f"€{metrics['test_rmse']:,.0f}")
    m4.metric("Model vs Naive",
              f"{(1 - metrics['test_mape']/metrics['naive_mape'])*100:.0f}% better")

    st.markdown("### Train / Validation / Test split")
    st.markdown(f"""
    | Split | Rows | Date range |
    |-------|------|------------|
    | Train | {metrics['n_train']:,} | Before {metrics['cutoff_val']} |
    | Validation | {metrics['n_val']:,} | {metrics['cutoff_val']} – {metrics['cutoff_test']} |
    | Test | {metrics['n_test']:,} | After {metrics['cutoff_test']} |
    """)
    st.info("**Time-based split** — no data from the future is used to train or validate. This avoids leakage that would inflate reported accuracy.")

    st.markdown("### Feature importance (top 15)")
    feat_imp = pd.DataFrame({
        "Feature": forecaster.feature_cols,
        "Importance": forecaster.model_median.feature_importances_
    }).sort_values("Importance", ascending=True).tail(15)

    fig3 = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Blues",
                  template="plotly_dark")
    fig3.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                       height=420, margin=dict(l=0,r=0,t=10,b=0),
                       coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Sales distribution across all stores")
    fig4 = px.histogram(df_all, x="Sales", nbins=80,
                        color_discrete_sequence=["#58a6ff"],
                        template="plotly_dark")
    fig4.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                       height=280, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Average sales by month (all stores)")
    monthly = df_all.copy()
    monthly["Month"] = monthly["Date"].dt.month
    monthly_avg = monthly.groupby("Month")["Sales"].mean().reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_avg["Month"] = monthly_avg["Month"].map(month_names)
    fig5 = px.line(monthly_avg, x="Month", y="Sales", markers=True,
                   color_discrete_sequence=["#58a6ff"], template="plotly_dark")
    fig5.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                       height=260, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Batch Upload
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Upload":
    st.markdown("# 📂 Batch Store Forecast")
    st.markdown("Upload a CSV of store records to score multiple stores at once. Click a row to see its SHAP drivers.")
    st.caption("Stores are shown alphabetically — not ranked by predicted sales.")

    uploaded = st.file_uploader("Upload CSV (same format as train.csv)", type="csv")

    if uploaded:
        batch_raw = pd.read_csv(uploaded, dtype={"StateHoliday": str}, low_memory=False)
        st.success(f"Loaded {len(batch_raw):,} rows.")

        # Try to merge store data if Store column present
        store_meta = pd.read_csv(ROOT / "data" / "raw" / "store.csv")
        if "Store" in batch_raw.columns:
            batch_raw = batch_raw.merge(store_meta, on="Store", how="left")

        batch_raw["Date"] = pd.to_datetime(batch_raw["Date"])
        batch_raw["StateHoliday"] = batch_raw["StateHoliday"].replace("0","None").fillna("None")
        for col in ["CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear",
                    "Promo2SinceWeek","Promo2SinceYear","PromoInterval"]:
            if col in batch_raw.columns:
                batch_raw[col] = batch_raw[col].fillna(0) if col != "PromoInterval" else batch_raw[col].fillna("None")

        for col in ["StoreType","Assortment","StateHoliday"]:
            if col in batch_raw.columns:
                batch_raw[col] = batch_raw[col].astype("category").cat.codes

        if "CompetitionAge" not in batch_raw.columns:
            batch_raw["CompetitionAge"] = 0
        if "Promo2Active" not in batch_raw.columns:
            batch_raw["Promo2Active"] = 0

        # Build lag/rolling with full history for context
        batch_feat = build_features(batch_raw, is_train=True)

        for c in forecaster.feature_cols:
            if c not in batch_feat.columns:
                batch_feat[c] = 0

        preds = forecaster.predict(batch_feat)
        batch_feat["Forecast"]  = preds["median"].round(0).astype(int)
        batch_feat["Low (Q10)"] = preds["q10"].round(0).astype(int)
        batch_feat["High (Q90)"]= preds["q90"].round(0).astype(int)

        display_cols = ["Store","Date","Forecast","Low (Q10)","High (Q90)","Promo","SchoolHoliday"]
        display_cols = [c for c in display_cols if c in batch_feat.columns]
        display_df   = batch_feat[display_cols].sort_values("Store").reset_index(drop=True)

        st.dataframe(display_df, use_container_width=True, height=360)

        st.markdown("### Forecast summary across uploaded stores")
        fig6 = px.box(batch_feat, x="Store", y="Forecast",
                      color_discrete_sequence=["#58a6ff"],
                      template="plotly_dark")
        fig6.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                           height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig6, use_container_width=True)

        csv_out = display_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download scored CSV", csv_out,
                           file_name="forecasts.csv", mime="text/csv")

    else:
        st.info("Upload a CSV file above to get started. You can use the provided `sample_batch.csv` from the repo.")
