"""
app.py
ValSafe ML — Streamlit Dashboard
Interactive dashboard for incident analysis and ML-powered predictions.

Run: streamlit run app.py
"""

import os
import joblib
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ValSafe ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #3B8BD4;
        margin-bottom: 8px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #6c757d; }
    .metric-card p  { margin: 4px 0 0; font-size: 26px; font-weight: 700; color: #1a1a2e; }
    .section-header {
        font-size: 16px; font-weight: 600;
        color: #1a1a2e; border-bottom: 2px solid #3B8BD4;
        padding-bottom: 6px; margin: 20px 0 14px;
    }
    .pred-box {
        border-radius: 10px; padding: 14px 18px;
        text-align: center; margin: 6px 0;
    }
    .pred-severity  { background:#fff3cd; border:1.5px solid #ffc107; }
    .pred-type      { background:#d1ecf1; border:1.5px solid #17a2b8; }
    .pred-status    { background:#d4edda; border:1.5px solid #28a745; }
    .pred-label { font-size:11px; color:#555; margin:0; }
    .pred-value { font-size:22px; font-weight:700; margin:4px 0 0; }
</style>
""", unsafe_allow_html=True)


# ── LOAD RESOURCES ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = "models"
    missing = [
        f for f in [
            "severity_model.pkl", "type_model.pkl", "status_model.pkl",
            "encoders.pkl", "feature_columns.pkl"
        ]
        if not os.path.exists(os.path.join(base, f))
    ]
    if missing:
        return None, None, None
    severity_model  = joblib.load(f"{base}/severity_model.pkl")
    type_model      = joblib.load(f"{base}/type_model.pkl")
    status_model    = joblib.load(f"{base}/status_model.pkl")
    encoders        = joblib.load(f"{base}/encoders.pkl")
    feature_cols    = joblib.load(f"{base}/feature_columns.pkl")
    return (
        {"severity": severity_model, "type": type_model, "status": status_model},
        encoders,
        feature_cols,
    )


@st.cache_data
def load_data():
    path = "data/incidents.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["occurred_at", "created_at"])
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce")
    df["month_label"] = df["occurred_at"].dt.strftime("%Y-%m")
    return df


def safe_encode(encoders, col, value):
    """Encode a value; return 0 if unseen label."""
    le = encoders[col]
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0


def predict_incident(models, encoders, feature_cols, inputs: dict) -> dict:
    row = pd.DataFrame([{
        "cat_id_enc":           safe_encode(encoders, "cat_id",       inputs["cat_id"]),
        "root_cause_enc":       safe_encode(encoders, "root_cause",   inputs["root_cause"]),
        "dept_id_enc":          safe_encode(encoders, "dept_id",       inputs["dept_id"]),
        "site_id_enc":          safe_encode(encoders, "site_id",       inputs["site_id"]),
        "media_status_enc":     safe_encode(encoders, "media_status",  inputs["media_status"]),
        "hour_of_day":          inputs["hour_of_day"],
        "day_of_week":          inputs["day_of_week"],
        "month":                inputs["month"],
        "days_since_occurred":  inputs["days_since_occurred"],
        "resolved_by_admin":    int(inputs["resolved_by_admin"]),
    }])[feature_cols]

    results = {}
    for target, model in models.items():
        enc_key = target + "_level" if target == "severity" else target
        pred_enc  = model.predict(row)[0]
        proba     = model.predict_proba(row)[0]
        label     = encoders[enc_key].inverse_transform([pred_enc])[0]
        classes   = encoders[enc_key].classes_
        results[target] = {
            "label":       label,
            "confidence":  round(float(proba.max()) * 100, 1),
            "all_proba":   dict(zip(classes, (proba * 100).round(1))),
        }
    return results


# ── LOAD ─────────────────────────────────────────────────────────────────────
models, encoders, feature_cols = load_models()
df = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("ValSafe ML")
    st.caption("Incident Intelligence Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔮 Predict Incident", "📈 Trends", "🗺️ Location Map", "📋 Data Explorer"],
        label_visibility="collapsed",
    )
    st.divider()

    if df is not None:
        st.markdown("**Dataset info**")
        st.markdown(f"- Records: **{len(df):,}**")
        st.markdown(f"- Date range: **{df['occurred_at'].min().strftime('%b %Y')}** → **{df['occurred_at'].max().strftime('%b %Y')}**")
    else:
        st.warning("No data found. Run `generate_data.py` first.")

    if models is None:
        st.warning("Models not found. Run `train_model.py` first.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Incident Overview")

    if df is None:
        st.error("No data loaded. Please run `generate_data.py` first.")
        st.stop()

    # KPI cards
    total        = len(df)
    high_sev     = int((df["severity_level"] == "high").sum())
    pending      = int((df["status"] == "Pending").sum())
    escalated    = int((df["status"] == "Escalated").sum())
    resolve_rate = round((df["status"] == "Resolved").mean() * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, color in [
        (c1, "Total incidents",    f"{total:,}",       "#3B8BD4"),
        (c2, "High severity",      f"{high_sev:,}",    "#E24B4A"),
        (c3, "Pending",            f"{pending:,}",     "#EF9F27"),
        (c4, "Escalated",          f"{escalated:,}",   "#D4537E"),
        (c5, "Resolution rate",    f"{resolve_rate}%", "#1D9E75"),
    ]:
        col.markdown(
            f'<div class="metric-card" style="border-left-color:{color}">'
            f'<h3>{label}</h3><p style="color:{color}">{value}</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Row 1: severity + type
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Incidents by severity</div>', unsafe_allow_html=True)
        sev_counts = df["severity_level"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"low": "#1D9E75", "medium": "#EF9F27", "high": "#E24B4A"}
        fig = px.bar(
            sev_counts, x="Severity", y="Count",
            color="Severity", color_discrete_map=color_map,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Incidents by type</div>', unsafe_allow_html=True)
        type_counts = df["type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig2 = px.pie(
            type_counts, names="Type", values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: status + category
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Status breakdown</div>', unsafe_allow_html=True)
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        stat_color = {"Pending": "#EF9F27", "Resolved": "#1D9E75", "Escalated": "#E24B4A"}
        fig3 = px.bar(
            status_counts, x="Count", y="Status",
            orientation="h", color="Status",
            color_discrete_map=stat_color, text="Count",
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Top incident categories</div>', unsafe_allow_html=True)
        cat_counts = df["cat_id"].value_counts().head(8).reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig4 = px.bar(
            cat_counts, x="Count", y="Category",
            orientation="h", text="Count",
            color_discrete_sequence=["#3B8BD4"],
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Severity × Type heatmap
    st.markdown('<div class="section-header">Severity vs incident type heatmap</div>', unsafe_allow_html=True)
    pivot = df.groupby(["type", "severity_level"]).size().unstack(fill_value=0)
    fig5 = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
    )
    fig5.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT INCIDENT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Incident":
    st.title("🔮 Predict Incident Outcome")
    st.caption("Fill in the incident details below. The ML models will predict severity, type, and status.")

    if models is None:
        st.error("Models not loaded. Please run `train_model.py` first.")
        st.stop()

    CATEGORIES = [
        "Fire Safety", "Chemical Exposure", "Electrical Hazard",
        "Slip and Fall", "Equipment Failure", "Ergonomics",
        "Security Breach", "Environmental",
    ]
    ROOT_CAUSES = [
        "Human Error", "Equipment Malfunction", "Inadequate Training",
        "Poor Housekeeping", "Design Flaw", "Fatigue",
        "Communication Failure", "Environmental Factor",
    ]
    DEPARTMENTS = [f"DEPT-{str(i).zfill(3)}" for i in range(1, 21)]
    SITES       = [f"SITE-{str(i).zfill(3)}" for i in range(1, 11)]
    MEDIA_OPTS  = ["no", "yes", "success", "failed"]
    DAYS        = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    with st.form("predict_form"):
        st.markdown('<div class="section-header">Incident details</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            cat_id      = st.selectbox("Category",          CATEGORIES)
            root_cause  = st.selectbox("Root cause",        ROOT_CAUSES)
            media_status= st.selectbox("Media attached?",   MEDIA_OPTS)
        with col2:
            dept_id     = st.selectbox("Department",        DEPARTMENTS)
            site_id     = st.selectbox("Site",              SITES)
            resolved_by_admin = st.checkbox("Resolved by admin?")
        with col3:
            hour_of_day       = st.slider("Hour of day",         0, 23, 9)
            day_of_week       = st.selectbox("Day of week",      DAYS)
            month             = st.slider("Month",               1, 12, 6)
            days_since        = st.number_input("Days since occurred", 0, 730, 3)

        submitted = st.form_submit_button("🔮 Run prediction", use_container_width=True)

    if submitted:
        inputs = {
            "cat_id":              cat_id,
            "root_cause":          root_cause,
            "dept_id":             dept_id,
            "site_id":             site_id,
            "media_status":        media_status,
            "hour_of_day":         hour_of_day,
            "day_of_week":         DAYS.index(day_of_week),
            "month":               month,
            "days_since_occurred": int(days_since),
            "resolved_by_admin":   resolved_by_admin,
        }

        with st.spinner("Running models..."):
            results = predict_incident(models, encoders, feature_cols, inputs)

        st.divider()
        st.markdown("### Predictions")

        r1, r2, r3 = st.columns(3)
        sev   = results["severity"]
        typ   = results["type"]
        stat  = results["status"]

        sev_colors  = {"low": "#1D9E75", "medium": "#EF9F27", "high": "#E24B4A"}
        stat_colors = {"Pending": "#EF9F27", "Resolved": "#1D9E75", "Escalated": "#E24B4A"}

        r1.markdown(
            f'<div class="pred-box pred-severity">'
            f'<p class="pred-label">SEVERITY</p>'
            f'<p class="pred-value" style="color:{sev_colors.get(sev["label"], "#333")}">'
            f'{sev["label"].upper()}</p>'
            f'<p class="pred-label">{sev["confidence"]}% confidence</p></div>',
            unsafe_allow_html=True,
        )
        r2.markdown(
            f'<div class="pred-box pred-type">'
            f'<p class="pred-label">INCIDENT TYPE</p>'
            f'<p class="pred-value" style="color:#17a2b8">'
            f'{typ["label"].replace("_"," ").title()}</p>'
            f'<p class="pred-label">{typ["confidence"]}% confidence</p></div>',
            unsafe_allow_html=True,
        )
        r3.markdown(
            f'<div class="pred-box pred-status">'
            f'<p class="pred-label">STATUS FORECAST</p>'
            f'<p class="pred-value" style="color:{stat_colors.get(stat["label"], "#333")}">'
            f'{stat["label"].upper()}</p>'
            f'<p class="pred-label">{stat["confidence"]}% confidence</p></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### Probability breakdown")

        pa, pb, pc = st.columns(3)
        for col, result, title, color in [
            (pa, sev,  "Severity probabilities",  "#EF9F27"),
            (pb, typ,  "Type probabilities",       "#17a2b8"),
            (pc, stat, "Status probabilities",     "#1D9E75"),
        ]:
            proba_df = pd.DataFrame(
                list(result["all_proba"].items()),
                columns=["Label", "Probability (%)"],
            ).sort_values("Probability (%)", ascending=True)

            fig = px.bar(
                proba_df, x="Probability (%)", y="Label",
                orientation="h", text="Probability (%)",
                color_discrete_sequence=[color],
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                title=title, height=260,
                margin=dict(t=30, b=10, l=10, r=40),
                xaxis=dict(range=[0, 110]),
            )
            col.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.title("📈 Incident Trends")

    if df is None:
        st.error("No data loaded.")
        st.stop()

    # Monthly trend
    st.markdown('<div class="section-header">Monthly incident volume</div>', unsafe_allow_html=True)
    monthly = (
        df.groupby(["month_label", "severity_level"])
        .size()
        .reset_index(name="count")
        .sort_values("month_label")
    )
    color_map = {"low": "#1D9E75", "medium": "#EF9F27", "high": "#E24B4A"}
    fig = px.line(
        monthly, x="month_label", y="count",
        color="severity_level", color_discrete_map=color_map,
        markers=True, labels={"month_label": "Month", "count": "Incidents"},
    )
    fig.update_layout(height=340, margin=dict(t=10, b=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Incidents by hour of day
    with col_a:
        st.markdown('<div class="section-header">Incidents by hour of day</div>', unsafe_allow_html=True)
        hourly = df.groupby("hour_of_day").size().reset_index(name="count")
        fig2 = px.bar(
            hourly, x="hour_of_day", y="count",
            labels={"hour_of_day": "Hour (0–23)", "count": "Incidents"},
            color_discrete_sequence=["#3B8BD4"],
        )
        fig2.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Incidents by day of week
    with col_b:
        st.markdown('<div class="section-header">Incidents by day of week</div>', unsafe_allow_html=True)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = df.groupby("day_of_week").size().reset_index(name="count")
        daily["day_name"] = daily["day_of_week"].map(lambda x: day_names[x])
        fig3 = px.bar(
            daily, x="day_name", y="count",
            labels={"day_name": "Day", "count": "Incidents"},
            color_discrete_sequence=["#D4537E"],
            category_orders={"day_name": day_names},
        )
        fig3.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # Resolution time analysis
    st.markdown('<div class="section-header">Incidents by root cause</div>', unsafe_allow_html=True)
    rc = df["root_cause"].value_counts().reset_index()
    rc.columns = ["Root cause", "Count"]
    fig4 = px.bar(
        rc, x="Count", y="Root cause", orientation="h",
        color="Count", color_continuous_scale="Blues", text="Count",
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    # Severity over months stacked
    st.markdown('<div class="section-header">Severity composition over time (stacked)</div>', unsafe_allow_html=True)
    pivot = (
        df.groupby(["month_label", "severity_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("month_label")
    )
    fig5 = go.Figure()
    for sev, color in [("low", "#1D9E75"), ("medium", "#EF9F27"), ("high", "#E24B4A")]:
        if sev in pivot.columns:
            fig5.add_trace(go.Bar(
                x=pivot["month_label"], y=pivot[sev],
                name=sev, marker_color=color,
            ))
    fig5.update_layout(
        barmode="stack", height=320,
        margin=dict(t=10, b=10), xaxis_tickangle=-45,
    )
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LOCATION MAP
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Location Map":
    st.title("🗺️ Incident Location Map")

    if df is None:
        st.error("No data loaded.")
        st.stop()

    map_df = df.copy()
    map_df["lat"] = pd.to_numeric(map_df["loc_lat"],  errors="coerce")
    map_df["lon"] = pd.to_numeric(map_df["loc_long"], errors="coerce")
    map_df = map_df.dropna(subset=["lat", "lon"])

    col1, col2, col3 = st.columns(3)
    sev_filter  = col1.multiselect("Severity",  ["low","medium","high"], default=["low","medium","high"])
    type_filter = col2.multiselect("Type",       df["type"].unique().tolist(), default=df["type"].unique().tolist())
    stat_filter = col3.multiselect("Status",    ["Pending","Resolved","Escalated"], default=["Pending","Resolved","Escalated"])

    filtered = map_df[
        map_df["severity_level"].isin(sev_filter) &
        map_df["type"].isin(type_filter) &
        map_df["status"].isin(stat_filter)
    ]

    st.caption(f"Showing **{len(filtered):,}** incidents")

    color_map = {"low": "#1D9E75", "medium": "#EF9F27", "high": "#E24B4A"}

    fig = px.scatter_geo(
        filtered,
        lat="lat", lon="lon",
        color="severity_level",
        color_discrete_map=color_map,
        hover_data={"type": True, "status": True, "cat_id": True, "lat": False, "lon": False},
        opacity=0.6,
        size_max=8,
    )
    fig.update_layout(
        height=520,
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        margin=dict(t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Department hotspots
    st.markdown('<div class="section-header">Incidents per department</div>', unsafe_allow_html=True)
    dept_counts = filtered["dept_id"].value_counts().head(10).reset_index()
    dept_counts.columns = ["Department", "Count"]
    fig2 = px.bar(
        dept_counts, x="Department", y="Count",
        color="Count", color_continuous_scale="Reds", text="Count",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer")

    if df is None:
        st.error("No data loaded.")
        st.stop()

    # Filters
    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)

    sev_f  = f1.multiselect("Severity",  df["severity_level"].unique(), default=list(df["severity_level"].unique()))
    typ_f  = f2.multiselect("Type",      df["type"].unique(),           default=list(df["type"].unique()))
    stat_f = f3.multiselect("Status",    df["status"].unique(),         default=list(df["status"].unique()))

    filtered = df[
        df["severity_level"].isin(sev_f) &
        df["type"].isin(typ_f) &
        df["status"].isin(stat_f)
    ]

    st.caption(f"**{len(filtered):,}** records match your filters")

    DISPLAY_COLS = [
        "id", "cat_id", "type", "severity_level", "status",
        "root_cause", "dept_id", "site_id", "media_status",
        "occurred_at", "days_since_occurred",
    ]
    st.dataframe(
        filtered[DISPLAY_COLS].reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    # Download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv,
        file_name=f"valsafe_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )