import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from datetime import datetime

st.set_page_config(page_title="Fund Drawdown Analysis vs Nifty", layout="wide", page_icon="📉")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def classify_category(name):
    name_lower = name.lower()
    if "small cap" in name_lower or "smallcap" in name_lower:
        return "Small Cap"
    elif "mid cap" in name_lower or "midcap" in name_lower:
        return "Mid Cap"
    elif "large & mid" in name_lower or "large and mid" in name_lower or "large & midcap" in name_lower:
        return "Large & Mid Cap"
    elif "large cap" in name_lower or "largecap" in name_lower:
        return "Large Cap"
    elif "multi cap" in name_lower or "multicap" in name_lower:
        return "Multi Cap"
    elif "flexi cap" in name_lower or "flexicap" in name_lower:
        return "Flexi Cap"
    else:
        return "Other"

def compute_drawdown_series(prices: pd.Series) -> pd.Series:
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max * 100
    return drawdown

def get_nifty_drawdown_events(nifty_dd: pd.Series, threshold: float) -> pd.DataFrame:
    """Find contiguous drawdown periods where Nifty fell >= threshold %."""
    in_dd = nifty_dd <= -threshold
    events = []
    start = None
    for date, val in in_dd.items():
        if val and start is None:
            start = date
        elif not val and start is not None:
            # end of event
            segment = nifty_dd[start:date]
            peak_loss = segment.min()
            trough_date = segment.idxmin()
            events.append({"start": start, "end": date, "trough_date": trough_date, "nifty_max_dd": peak_loss})
            start = None
    if start is not None:
        segment = nifty_dd[start:]
        peak_loss = segment.min()
        trough_date = segment.idxmin()
        events.append({"start": start, "end": nifty_dd.index[-1], "trough_date": trough_date, "nifty_max_dd": peak_loss})
    return pd.DataFrame(events)

def fund_dd_during_event(fund_prices: pd.Series, event_start, event_end) -> float:
    segment = fund_prices[event_start:event_end].dropna()
    if len(segment) < 2:
        return np.nan
    peak = segment.iloc[0]
    trough = segment.min()
    return (trough - peak) / peak * 100

# ─────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    # ── Nifty ──
    nifty_raw = pd.read_csv("Nifty50_10Years_Data.csv")
    nifty_raw["Date"] = pd.to_datetime(nifty_raw["Date"], utc=True).dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    nifty_raw = nifty_raw.set_index("Date").sort_index()
    nifty = nifty_raw["Close"].rename("Nifty50")

    # ── Funds helper ──
    def parse_excel(path):
        raw = pd.read_excel(path, header=None)
        fund_names = raw.iloc[2, 1:].tolist()          # fund names on row index 2
        data_rows = raw.iloc[4:].copy()                 # actual data from row index 4
        data_rows.columns = ["Date"] + fund_names
        data_rows["Date"] = pd.to_datetime(data_rows["Date"], errors="coerce").dt.normalize()
        data_rows = data_rows.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in fund_names:
            data_rows[col] = pd.to_numeric(data_rows[col], errors="coerce")
        return data_rows

    f1 = parse_excel("funds1.xlsx")
    f2 = parse_excel("funds2.xlsx")
    all_funds = pd.concat([f1, f2], axis=1)

    # ── Filter last 6 years ──
    cutoff = nifty.index.max() - pd.DateOffset(years=6)
    nifty = nifty[nifty.index >= cutoff]
    all_funds = all_funds[all_funds.index >= cutoff]

    # Align on common dates
    common = nifty.index.intersection(all_funds.index)
    nifty = nifty.reindex(common)
    all_funds = all_funds.reindex(common)

    # Build category map
    category_map = {col: classify_category(col) for col in all_funds.columns}

    return nifty, all_funds, category_map

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("📉 Fund Drawdown Analysis vs Nifty50 (Last 6 Years)")
st.markdown("Identifies market drawdown events and ranks funds by **resilience** (least drawdown) in each category.")

with st.spinner("Loading data…"):
    nifty, all_funds, category_map = load_data()

# Sidebar controls
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Nifty Drawdown Trigger (%)", min_value=2.0, max_value=20.0, value=5.0, step=0.5,
                               help="Show events where Nifty fell at least this much from peak")
top_n = st.sidebar.slider("Top N resilient funds per category", min_value=3, max_value=15, value=5)
categories_all = sorted(set(category_map.values()))
selected_cats = st.sidebar.multiselect("Filter categories", categories_all, default=categories_all)

# ─────────────────────────────────────────────
# Compute Nifty drawdown & events
# ─────────────────────────────────────────────

nifty_dd = compute_drawdown_series(nifty)
events_df = get_nifty_drawdown_events(nifty_dd, threshold)

if events_df.empty:
    st.warning(f"No Nifty drawdown events ≥ {threshold}% found in last 6 years. Try lowering the threshold.")
    st.stop()

# ─────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Date Range", f"{nifty.index.min().strftime('%b %Y')} – {nifty.index.max().strftime('%b %Y')}")
col2.metric("📊 Total Funds", len(all_funds.columns))
col3.metric("⚡ Drawdown Events", len(events_df))
col4.metric("📉 Worst Nifty DD", f"{events_df['nifty_max_dd'].min():.1f}%")

st.divider()

# ─────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📆 Drawdown Events", "🏆 Fund Rankings by Category", "📈 Fund vs Nifty Chart", "🗂️ Raw Drawdown Table"])

# ─── Tab 1: Events overview ───────────────────
with tab1:
    st.subheader(f"Nifty50 Drawdown Events ≥ {threshold}%")

    fig_nifty = go.Figure()
    fig_nifty.add_trace(go.Scatter(x=nifty_dd.index, y=nifty_dd.values,
                                    fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
                                    line=dict(color="#ef5350", width=1.5),
                                    name="Nifty Drawdown %"))
    fig_nifty.add_hline(y=-threshold, line_dash="dash", line_color="orange",
                         annotation_text=f"−{threshold}% trigger", annotation_position="bottom right")

    for _, ev in events_df.iterrows():
        fig_nifty.add_vrect(x0=ev["start"], x1=ev["end"],
                             fillcolor="rgba(239,83,80,0.1)", layer="below", line_width=0)

    fig_nifty.update_layout(title="Nifty50 Rolling Drawdown (shaded = triggered events)",
                             xaxis_title="Date", yaxis_title="Drawdown %",
                             height=350, template="plotly_white",
                             yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig_nifty, use_container_width=True)

    # Events table
    ev_display = events_df.copy()
    ev_display["start"] = ev_display["start"].dt.strftime("%d %b %Y")
    ev_display["end"] = ev_display["end"].dt.strftime("%d %b %Y")
    ev_display["trough_date"] = ev_display["trough_date"].dt.strftime("%d %b %Y")
    ev_display["nifty_max_dd"] = ev_display["nifty_max_dd"].map("{:.2f}%".format)
    ev_display.columns = ["Event Start", "Event End", "Worst Day", "Nifty Max Drawdown"]
    ev_display.index = range(1, len(ev_display)+1)
    st.dataframe(ev_display, use_container_width=True)

# ─── Tab 2: Rankings per category ─────────────
with tab2:
    st.subheader("🏆 Most Resilient Funds During Nifty Drawdowns")

    # For each fund, compute avg drawdown during all events
    fund_event_dd = {}
    for fund in all_funds.columns:
        dds = []
        for _, ev in events_df.iterrows():
            dd = fund_dd_during_event(all_funds[fund].dropna(), ev["start"], ev["end"])
            if not np.isnan(dd):
                dds.append(dd)
        if dds:
            fund_event_dd[fund] = np.mean(dds)

    fund_dd_series = pd.Series(fund_event_dd).rename("avg_drawdown_pct")
    fund_df = fund_dd_series.reset_index()
    fund_df.columns = ["Fund", "Avg Drawdown (%)"]
    fund_df["Category"] = fund_df["Fund"].map(category_map)
    fund_df = fund_df[fund_df["Category"].isin(selected_cats)]
    fund_df = fund_df.sort_values("Avg Drawdown (%)", ascending=False)  # less negative = better

    for cat in sorted(selected_cats):
        cat_df = fund_df[fund_df["Category"] == cat].head(top_n).copy()
        if cat_df.empty:
            continue

        with st.expander(f"📂 {cat} — Top {top_n} Resilient Funds", expanded=True):
            fig_bar = go.Figure()
            colors = ["#26a69a" if v >= -threshold else "#ef9a9a" for v in cat_df["Avg Drawdown (%)"]]
            fig_bar.add_trace(go.Bar(
                x=cat_df["Avg Drawdown (%)"],
                y=cat_df["Fund"].apply(lambda x: x[:40] + "…" if len(x) > 40 else x),
                orientation="h",
                marker_color=colors,
                text=cat_df["Avg Drawdown (%)"].map("{:.2f}%".format),
                textposition="outside"
            ))
            fig_bar.add_vline(x=-threshold, line_dash="dash", line_color="orange",
                               annotation_text=f"−{threshold}% (trigger)", annotation_position="top right")
            fig_bar.update_layout(
                title=f"{cat}: Average Drawdown During Nifty Events (closer to 0 = more resilient)",
                xaxis_title="Avg Drawdown (%)", height=max(250, top_n * 50),
                template="plotly_white", margin=dict(l=250),
                xaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Table
            st.dataframe(
                cat_df.reset_index(drop=True)[["Fund", "Avg Drawdown (%)"]].assign(
                    **{"Avg Drawdown (%)": cat_df["Avg Drawdown (%)"].map("{:.2f}%".format).values}
                ),
                use_container_width=True, hide_index=True
            )

# ─── Tab 3: Fund vs Nifty chart ───────────────
with tab3:
    st.subheader("📈 Compare Fund NAV vs Nifty (Indexed to 100)")

    # Pick top 5 most resilient across all categories for default
    top5 = fund_df.nlargest(5, "Avg Drawdown (%)").sort_values("Avg Drawdown (%)", ascending=False)
    fund_options = all_funds.columns.tolist()
    selected_funds = st.multiselect("Select funds to compare", fund_options,
                                     default=top5["Fund"].tolist()[:5])

    if selected_funds:
        fig_line = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.65, 0.35],
                                  subplot_titles=("Indexed NAV (base=100)", "Drawdown (%)"))

        # Index Nifty
        nifty_idx = nifty / nifty.iloc[0] * 100
        fig_line.add_trace(go.Scatter(x=nifty_idx.index, y=nifty_idx.values,
                                       name="Nifty50", line=dict(color="black", width=2, dash="dot")), row=1, col=1)
        fig_line.add_trace(go.Scatter(x=nifty_dd.index, y=nifty_dd.values,
                                       name="Nifty DD", line=dict(color="black", width=1, dash="dot"),
                                       fill="tozeroy", fillcolor="rgba(0,0,0,0.05)"), row=2, col=1)

        palette = px.colors.qualitative.Bold
        for i, fund in enumerate(selected_funds):
            series = all_funds[fund].dropna()
            if series.empty:
                continue
            s_idx = series / series.iloc[0] * 100
            s_dd = compute_drawdown_series(series)
            color = palette[i % len(palette)]
            short_name = fund.split("-")[0][:30]
            fig_line.add_trace(go.Scatter(x=s_idx.index, y=s_idx.values, name=short_name,
                                           line=dict(color=color, width=1.5)), row=1, col=1)
            fig_line.add_trace(go.Scatter(x=s_dd.index, y=s_dd.values, name=f"{short_name} DD",
                                           line=dict(color=color, width=1, dash="dash"),
                                           showlegend=False), row=2, col=1)

        # Shade events
        for _, ev in events_df.iterrows():
            fig_line.add_vrect(x0=ev["start"], x1=ev["end"], fillcolor="rgba(255,152,0,0.1)",
                                layer="below", line_width=0, row="all", col=1)

        fig_line.update_layout(height=600, template="plotly_white",
                                legend=dict(orientation="h", y=-0.15))
        fig_line.update_yaxes(ticksuffix="%", row=2, col=1)
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("🟠 Shaded bands = Nifty drawdown events ≥ " + str(threshold) + "%")

# ─── Tab 4: Full table ─────────────────────────
with tab4:
    st.subheader("🗂️ Full Drawdown Table — All Events × All Funds")
    st.markdown("Each cell = fund's drawdown during that Nifty event (start-to-trough). **Green = outperformed Nifty, Red = underperformed.**")

    # Build matrix
    records = []
    for fund in all_funds.columns:
        row = {"Fund": fund, "Category": category_map[fund]}
        for i, ev in events_df.iterrows():
            label = f"Event {i+1}\n({ev['start'].strftime('%b %y')})"
            dd = fund_dd_during_event(all_funds[fund].dropna(), ev["start"], ev["end"])
            row[label] = round(dd, 2) if not np.isnan(dd) else None
        records.append(row)

    matrix_df = pd.DataFrame(records)
    matrix_df = matrix_df[matrix_df["Category"].isin(selected_cats)]

    cat_filter = st.selectbox("Filter by category", ["All"] + sorted(selected_cats))
    if cat_filter != "All":
        matrix_df = matrix_df[matrix_df["Category"] == cat_filter]

    event_cols = [c for c in matrix_df.columns if c.startswith("Event")]
    styled = matrix_df.set_index(["Category", "Fund"])[event_cols].style.background_gradient(
        cmap="RdYlGn", axis=None, vmin=-30, vmax=0
    ).format("{:.1f}%", na_rep="–")
    st.dataframe(styled, use_container_width=True, height=500)

st.divider()
st.caption("Data: Nifty50 (10-year daily close) • Mutual Fund NAV data from uploaded files • Drawdown = peak-to-trough decline")
