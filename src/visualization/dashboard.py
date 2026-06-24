import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from src.pipeline import SpaceWeatherPipeline
from src.data.solexs_loader import load_solexs_day
from src.data.hel1os_loader import load_hel1os_day
from src.data.goes_loader import load_goes_catalog
from src.data.sharp_loader import load_sharp_clean, compute_physics_features
from src.data.sunspot_loader import load_sunspot
from src.detection.wavelet_detector import WaveletFlareDetector
from src.catalog.master_catalog import match_events

st.set_page_config(page_title="Space Weather Intelligence System", layout="wide")
st.title("ISRO Aditya-L1 Solar Flare Intelligence System")

DATA_CACHE = {}


@st.cache_data(ttl=300)
def load_data(start, end):
    p = SpaceWeatherPipeline()
    p.load_all_data(start, end)
    return p


@st.cache_data(ttl=300)
def run_pipeline(start, end):
    p = load_data(start, end)
    p.run_detection()
    p.run_catalog_merge()
    return p


def plot_lightcurve(flux, title, events=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flux.index, y=flux.values,
        mode="lines", line=dict(width=1),
        name="Flux", hovertemplate="%{x}<br>Flux: %{y:.1f}"
    ))
    if events is not None and len(events) > 0:
        for _, ev in events.iterrows():
            fig.add_vrect(
                x0=ev["start_time"], x1=ev["end_time"],
                fillcolor="red", opacity=0.15, line_width=0
            )
            fig.add_vline(
                x=ev["peak_time"], line_dash="dot",
                line_color="red", opacity=0.5
            )
    fig.update_layout(
        title=title, height=350,
        xaxis_title="Time (UTC)", yaxis_title="Counts/s",
        hovermode="x unified"
    )
    return fig


def plot_cwt_spectrogram(flux, wavelet="morl", max_scale=64):
    import pywt
    scales = np.arange(1, max_scale + 1)
    coeffs, _ = pywt.cwt(flux.values.astype(float), scales, wavelet, sampling_period=1)
    power = np.abs(coeffs) ** 2
    fig = go.Figure(data=go.Heatmap(
        z=np.log1p(power),
        x=flux.index,
        y=scales,
        colorscale="Viridis",
        hovertemplate="Time: %{x}<br>Scale: %{y}<br>Log-Power: %{z:.2f}"
    ))
    fig.update_layout(
        title="CWT Scalogram (Morlet)", height=300,
        xaxis_title="Time (UTC)", yaxis_title="Scale"
    )
    return fig


st.sidebar.header("Controls")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime(2024, 11, 1), datetime(2024, 11, 5)),
    min_value=datetime(2024, 2, 1),
    max_value=datetime(2026, 5, 20),
)
if len(date_range) == 2:
    start_str = date_range[0].strftime("%Y%m%d")
    end_str = date_range[1].strftime("%Y%m%d")
else:
    start_str = date_range[0].strftime("%Y%m%d")
    end_str = date_range[0].strftime("%Y%m%d")

run_btn = st.sidebar.button("Run Pipeline", type="primary")
single_day = st.sidebar.date_input(
    "Single Day View", value=datetime(2024, 11, 17),
    min_value=datetime(2024, 2, 1), max_value=datetime(2026, 5, 20)
)

tab_data, tab_detection, tab_catalog, tab_forecast, tab_sharp = st.tabs(
    ["Data Overview", "Detection", "Catalog", "Forecasting", "SHARP / Sunspot"]
)

if run_btn:
    with st.spinner(f"Running pipeline {start_str} -> {end_str}..."):
        p = run_pipeline(start_str, end_str)
else:
    p = run_pipeline(start_str, end_str)

with tab_data:
    st.header("Data Availability")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("SoLEXS", f"{len(p.solexs_data):,}" if p.solexs_data is not None else "N/A")
    col2.metric("HEL1OS", f"{len(p.hel1os_data):,}" if p.hel1os_data is not None else "N/A")
    col3.metric("GOES Events", f"{len(p.goes_catalog):,}" if p.goes_catalog is not None else "N/A")
    col4.metric("SHARP Rows", f"{len(p.sharp_data):,}" if p.sharp_data is not None else "N/A")
    col5.metric("Sunspot Days", f"{len(p.sunspot_data):,}" if p.sunspot_data is not None else "N/A")

    day_str = single_day.strftime("%Y%m%d")
    col_l, col_r = st.columns(2)
    with col_l:
        try:
            s_df = load_solexs_day(day_str)
            st.plotly_chart(plot_lightcurve(s_df["flux"], f"SoLEXS SDD2 - {day_str}"), use_container_width=True)
        except Exception as e:
            st.warning(f"SoLEXS: {e}")
    with col_r:
        try:
            h_df = load_hel1os_day(day_str)
            st.plotly_chart(plot_lightcurve(h_df["flux"], f"HEL1OS CZT1 - {day_str}"), use_container_width=True)
        except Exception as e:
            st.warning(f"HEL1OS: {e}")

    if p.goes_catalog is not None:
        goes_sub = p.goes_catalog[
            (p.goes_catalog["start_time"] >= date_range[0])
            & (p.goes_catalog["start_time"] <= date_range[1])
        ]
        if len(goes_sub) > 0:
            fig = px.scatter(
                goes_sub, x="start_time", y="peak_flux", color="flare_class",
                hover_data=["peak_time"], log_y=True,
                title=f"GOES Flares ({len(goes_sub)} events)"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_detection:
    st.header("Flare Detection")
    det = WaveletFlareDetector(threshold_sigma=st.sidebar.slider("Detection σ", 3, 8, 5, 1))
    day_str = single_day.strftime("%Y%m%d")

    col1, col2 = st.columns(2)
    with col1:
        try:
            s_df = load_solexs_day(day_str)
            s_events = det.detect(s_df["flux"])
            st.metric("SoLEXS Events", len(s_events))
            st.plotly_chart(plot_lightcurve(s_df["flux"], "SoLEXS", s_events), use_container_width=True)
            if len(s_events) > 0:
                st.plotly_chart(plot_cwt_spectrogram(s_df["flux"]), use_container_width=True)
            with st.expander("SoLEXS Event Table"):
                st.dataframe(s_events.sort_values("peak_flux", ascending=False), use_container_width=True)
        except Exception as e:
            st.warning(f"SoLEXS: {e}")
    with col2:
        try:
            h_df = load_hel1os_day(day_str)
            h_events = det.detect(h_df["flux"])
            st.metric("HEL1OS Events", len(h_events))
            st.plotly_chart(plot_lightcurve(h_df["flux"], "HEL1OS", h_events), use_container_width=True)
            if len(h_events) > 0:
                with st.expander("HEL1OS Event Table"):
                    st.dataframe(h_events.sort_values("peak_flux", ascending=False), use_container_width=True)
        except Exception as e:
            st.warning(f"HEL1OS: {e}")

with tab_catalog:
    st.header("Master Catalog")
    if p.master_catalog is not None and len(p.master_catalog) > 0:
        cat = p.master_catalog.copy()
        cat["start_time"] = pd.to_datetime(cat["start_time"])
        fig = px.scatter(
            cat, x="start_time", y="peak_flux_solexs",
            color="source", hover_data=["duration_seconds", "confidence"],
            title=f"Master Catalog - {len(cat)} events"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Event Table")
        cols = [c for c in cat.columns if c != "master_id"]
        st.dataframe(
            cat[cols].sort_values("start_time"),
            use_container_width=True, height=400
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", len(cat))
        both = len(cat[cat["source"] == "both"]) if "source" in cat else 0
        col2.metric("SoLEXS+HEL1OS Matched", both)
        high_conf = len(cat[cat["confidence"] == "high"]) if "confidence" in cat else 0
        col3.metric("High Confidence", high_conf)

        if "peak_flux_solexs" in cat.columns:
            st.subheader("Peak Flux Distribution")
            fig = px.histogram(cat, x="peak_flux_solexs", log_y=True, nbins=50,
                               title="SoLEXS Peak Flux Distribution")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No catalog events for selected range")

with tab_forecast:
    st.header("Flare Forecasting")
    horizon = st.selectbox("Forecast Horizon", [15, 30, 60], index=0)
    if p.solexs_data is not None:
        flux = p.solexs_data["flux"]
        flux_1min = flux.resample("1min").mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flux_1min.index[-1000:], y=flux_1min.values[-1000:],
            mode="lines", name=f"SoLEXS (1-min)"
        ))
        if p.goes_catalog is not None:
            window = pd.Timedelta(minutes=horizon)
            now = flux_1min.index[-1]
            upcoming = p.goes_catalog[
                (p.goes_catalog["start_time"] >= now)
                & (p.goes_catalog["start_time"] < now + window)
            ]
            if len(upcoming) > 0:
                for _, e in upcoming.iterrows():
                    fig.add_vrect(
                        x0=e["start_time"], x1=e["end_time"],
                        fillcolor="orange", opacity=0.2, line_width=0,
                        annotation_text=e["flare_class"]
                    )
        fig.update_layout(title=f"Recent Flux + Upcoming GOES ({horizon}min window)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        from src.pipeline import XGBoostForecaster, compute_aligned_physics_features, get_flare_labels
        df = pd.DataFrame(index=flux_1min.index)
        df["flux"] = flux_1min.values
        sunspot = p.sunspot_data if p.sunspot_data is not None and not p.sunspot_data.empty else pd.DataFrame()
        if p.sharp_data is not None:
            aligned = compute_aligned_physics_features(flux_1min.index, p.sharp_data, sunspot)
            for c in aligned.columns:
                df[c] = aligned[c].values
        if p.goes_catalog is not None:
            labels = get_flare_labels(p.goes_catalog, flux_1min.index, lookback_minutes=horizon)
            df["flare_label"] = labels["flare_within_window"].values
        else:
            df["flare_label"] = 0

        with st.spinner("Training XGBoost..."):
            fc = XGBoostForecaster(horizon_minutes=horizon)
            fc.train(df)
            forecast = fc.forecast(df)

        cols = st.columns(4)
        cols[0].metric("P(Flare)", f"{forecast['flare_probability']:.1%}")
        cols[1].metric("P(B)", f"{forecast.get('p_b', 0):.1%}")
        cols[2].metric("P(C)", f"{forecast.get('p_c', 0):.1%}")
        cols[3].metric("P(M/X)", f"{forecast.get('p_m', 0) + forecast.get('p_x', 0):.1%}")

        importances = pd.DataFrame({
            "feature": fc.feature_cols[:20],
            "importance": fc.model.feature_importances_[:20]
        }).sort_values("importance", ascending=True)
        fig = px.bar(importances, x="importance", y="feature", orientation="h",
                     title="Top 20 Feature Importances")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No flux data available")

with tab_sharp:
    st.header("SHARP Magnetic Parameters")
    if p.sharp_data is not None:
        sd = p.sharp_data.copy()
        sd = sd.reset_index()
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["USFLUX (Total Unsigned Flux)",
                                            "TOTUSJH (Current Helicity)",
                                            "R_VALUE (Flare Potential)"])
        for row, col_name in enumerate(["USFLUX", "TOTUSJH", "R_VALUE"], 1):
            if col_name in sd.columns:
                fig.add_trace(
                    go.Scatter(x=sd["timestamp"], y=sd[col_name],
                               mode="markers", marker=dict(size=2),
                               name=col_name),
                    row=row, col=1
                )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Physics-derived Features")
        pf = compute_physics_features(p.sharp_data)
        st.dataframe(pf.describe(), use_container_width=True)

        for feat in ["magnetic_instability_index", "free_energy_proxy", "magnetic_twist"]:
            if feat in pf.columns:
                fig = px.line(
                    pf.reset_index(), x="timestamp", y=feat,
                    title=feat.replace("_", " ").title()
                )
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sunspot Numbers")
    if p.sunspot_data is not None:
        ss = p.sunspot_data.reset_index()
        fig = px.line(ss, x="timestamp", y="ssn", title="Daily Sunspot Number")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Sources**\n"
    "- SoLEXS SDD2 (1 Hz, FITS)\n"
    "- HEL1OS CZT1 (1 Hz, FITS)\n"
    "- GOES XRS Flare Catalog\n"
    "- SHARP (12 min, CSV)\n"
    "- Sunspot Number (daily)"
)

if __name__ == "__main__":
    pass
