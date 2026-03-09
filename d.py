import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Fund Drawdown Analysis vs Nifty", layout="wide", page_icon="📉")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def classify_category(name):
    n = name.lower()
    # Check compound categories BEFORE simple ones to avoid mis-match
    if "flexi cap" in n or "flexicap" in n:
        return "Flexi Cap"
    elif "small cap" in n or "smallcap" in n:
        return "Small Cap"
    elif "multi cap" in n or "multicap" in n:
        return "Multi Cap"
    elif "large & mid" in n or "large and mid" in n or "large & midcap" in n or "large midcap" in n:
        return "Large & Mid Cap"
    elif "large cap" in n or "largecap" in n:
        return "Large Cap"
    elif "mid cap" in n or "midcap" in n:
        return "Mid Cap"
    else:
        return "Other"

def compute_drawdown_series(prices):
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max * 100

def get_nifty_drawdown_events(nifty_dd, threshold):
    in_dd = nifty_dd <= -threshold
    events = []
    start = None
    for date, val in in_dd.items():
        if val and start is None:
            start = date
        elif not val and start is not None:
            segment = nifty_dd[start:date]
            events.append({
                "start": start,
                "end": date,
                "trough_date": segment.idxmin(),
                "nifty_max_dd": segment.min()
            })
            start = None
    if start is not None:
        segment = nifty_dd[start:]
        events.append({
            "start": start,
            "end": nifty_dd.index[-1],
            "trough_date": segment.idxmin(),
            "nifty_max_dd": segment.min()
        })
    return pd.DataFrame(events)

def fund_dd_during_event(fund_prices, event_start, event_end):
    segment = fund_prices[event_start:event_end].dropna()
    if len(segment) < 2:
        return np.nan
    return (segment.min() - segment.iloc[0]) / segment.iloc[0] * 100

def dd_to_hex(val):
    """Convert drawdown % to (bg_hex, fg_hex) — pure Python, no matplotlib."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#eeeeee", "#888888"
    ratio = max(0.0, min(1.0, abs(val) / 30.0))
    r = int(220 * ratio)
    g = int(180 * (1.0 - ratio * 0.8))
    b = 50
    bg = f"#{r:02x}{g:02x}{b:02x}"
    fg = "#ffffff" if ratio > 0.55 else "#111111"
    return bg, fg

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    # Nifty
    nifty_raw = pd.read_csv("Nifty50_10Years_Data.csv")
    nifty_raw["Date"] = (
        pd.to_datetime(nifty_raw["Date"], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    nifty_raw = nifty_raw.set_index("Date").sort_index()
    nifty = nifty_raw["Close"].rename("Nifty50")

    # Funds parser
    def parse_excel(path):
        raw = pd.read_excel(path, header=None)
        fund_names = raw.iloc[2, 1:].tolist()          # row 2 = fund names
        data_rows = raw.iloc[4:].copy()                 # row 4+ = NAV data
        data_rows.columns = ["Date"] + fund_names
        data_rows["Date"] = (
            pd.to_datetime(data_rows["Date"], errors="coerce").dt.normalize()
        )
        data_rows = data_rows.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in fund_names:
            data_rows[col] = pd.to_numeric(data_rows[col], errors="coerce")
        return data_rows

    f1 = parse_excel("funds1.xlsx")
    f2 = parse_excel("funds2.xlsx")
    all_funds = pd.concat([f1, f2], axis=1, sort=True)

    # Last 6 years only
    cutoff = nifty.index.max() - pd.DateOffset(years=6)
    nifty = nifty[nifty.index >= cutoff]
    all_funds = all_funds[all_funds.index >= cutoff]

    # Align on common trading dates
    common = nifty.index.intersection(all_funds.index)
    nifty = nifty.reindex(common)
    all_funds = all_funds.reindex(common)

    category_map = {col: classify_category(col) for col in all_funds.columns}
    return nifty, all_funds, category_map

# ─────────────────────────────────────────────
# App UI
# ─────────────────────────────────────────────

st.title("📉 Fund Drawdown Analysis vs Nifty50 (Last 6 Years)")
st.markdown("Identifies Nifty market drawdown events and ranks mutual funds by **resilience** (least drawdown) in each category.")

with st.spinner("Loading data…"):
    nifty, all_funds, category_map = load_data()

# Sidebar
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Nifty Drawdown Trigger (%)", min_value=2.0, max_value=20.0, value=5.0, step=0.5,
    help="Flag events where Nifty fell at least this % from its peak"
)
top_n = st.sidebar.slider("Top N resilient funds per category", min_value=3, max_value=15, value=5)
categories_all = sorted(set(category_map.values()))
selected_cats = st.sidebar.multiselect("Filter categories", categories_all, default=categories_all)

# ─────────────────────────────────────────────
# Compute drawdowns & events
# ─────────────────────────────────────────────

nifty_dd = compute_drawdown_series(nifty)
events_df = get_nifty_drawdown_events(nifty_dd, threshold)

if events_df.empty:
    st.warning(f"No Nifty drawdown events ≥ {threshold}% found. Try lowering the threshold.")
    st.stop()

# ─────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("📅 Date Range", f"{nifty.index.min().strftime('%b %Y')} – {nifty.index.max().strftime('%b %Y')}")
c2.metric("📊 Total Funds", len(all_funds.columns))
c3.metric("⚡ Drawdown Events", len(events_df))
c4.metric("📉 Worst Nifty DD", f"{events_df['nifty_max_dd'].min():.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📆 Drawdown Events",
    "🏆 Fund Rankings by Category",
    "📈 Fund vs Nifty Chart",
    "🗂️ Drawdown Heatmap"
])

# ─── Tab 1 ───────────────────────────────────
with tab1:
    st.subheader(f"Nifty50 Drawdown Events ≥ {threshold}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nifty_dd.index, y=nifty_dd.values,
        fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
        line=dict(color="#ef5350", width=1.5),
        name="Nifty Drawdown %"
    ))
    fig.add_hline(y=-threshold, line_dash="dash", line_color="orange",
                  annotation_text=f"−{threshold}% trigger")
    for _, ev in events_df.iterrows():
        fig.add_vrect(x0=ev["start"], x1=ev["end"],
                      fillcolor="rgba(239,83,80,0.1)", layer="below", line_width=0)
    fig.update_layout(
        title="Nifty50 Rolling Drawdown (shaded = triggered events)",
        xaxis_title="Date", yaxis_title="Drawdown %",
        height=350, template="plotly_white",
        yaxis=dict(ticksuffix="%")
    )
    st.plotly_chart(fig, use_container_width=True)

    ev_disp = events_df.copy()
    ev_disp["start"] = ev_disp["start"].dt.strftime("%d %b %Y")
    ev_disp["end"] = ev_disp["end"].dt.strftime("%d %b %Y")
    ev_disp["trough_date"] = ev_disp["trough_date"].dt.strftime("%d %b %Y")
    ev_disp["nifty_max_dd"] = ev_disp["nifty_max_dd"].map("{:.2f}%".format)
    ev_disp.columns = ["Event Start", "Event End", "Worst Day", "Nifty Max Drawdown"]
    ev_disp.index = range(1, len(ev_disp) + 1)
    st.dataframe(ev_disp, use_container_width=True)

# ─── Tab 2 ───────────────────────────────────
with tab2:
    st.subheader("🏆 Most Resilient Funds During Nifty Drawdowns")

    fund_event_dd = {}
    for fund in all_funds.columns:
        dds = []
        for _, ev in events_df.iterrows():
            dd = fund_dd_during_event(all_funds[fund].dropna(), ev["start"], ev["end"])
            if not np.isnan(dd):
                dds.append(dd)
        if dds:
            fund_event_dd[fund] = np.mean(dds)

    fund_df = pd.Series(fund_event_dd).rename("avg_dd").reset_index()
    fund_df.columns = ["Fund", "Avg Drawdown (%)"]
    fund_df["Category"] = fund_df["Fund"].map(category_map)
    fund_df = fund_df[fund_df["Category"].isin(selected_cats)]
    fund_df = fund_df.sort_values("Avg Drawdown (%)", ascending=False)

    for cat in sorted(selected_cats):
        cat_df = fund_df[fund_df["Category"] == cat].head(top_n).copy()
        if cat_df.empty:
            continue
        with st.expander(f"📂 {cat}  —  Top {top_n} most resilient funds", expanded=True):
            colors = ["#26a69a" if v >= -threshold else "#ef9a9a"
                      for v in cat_df["Avg Drawdown (%)"]]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=cat_df["Avg Drawdown (%)"],
                y=cat_df["Fund"].apply(lambda x: x[:45] + "…" if len(x) > 45 else x),
                orientation="h",
                marker_color=colors,
                text=cat_df["Avg Drawdown (%)"].map("{:.2f}%".format),
                textposition="outside"
            ))
            fig_bar.add_vline(x=-threshold, line_dash="dash", line_color="orange",
                               annotation_text=f"−{threshold}% trigger")
            fig_bar.update_layout(
                title=f"{cat}: Avg Drawdown During Nifty Events (closer to 0 = more resilient)",
                xaxis_title="Avg Drawdown (%)",
                height=max(260, top_n * 55),
                template="plotly_white",
                margin=dict(l=260),
                xaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            tbl = cat_df.reset_index(drop=True)[["Fund", "Avg Drawdown (%)"]].copy()
            tbl["Avg Drawdown (%)"] = tbl["Avg Drawdown (%)"].map("{:.2f}%".format)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

# ─── Tab 3 ───────────────────────────────────
with tab3:
    st.subheader("📈 Compare Fund NAV vs Nifty (Indexed to 100)")

    top5 = fund_df.nlargest(5, "Avg Drawdown (%)") if "fund_df" in dir() else pd.DataFrame()
    default_funds = top5["Fund"].tolist() if not top5.empty else []
    selected_funds = st.multiselect("Select funds to compare",
                                    all_funds.columns.tolist(),
                                    default=default_funds[:5])

    if selected_funds:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.65, 0.35],
                              subplot_titles=("Indexed NAV (base=100)", "Drawdown (%)"))

        nifty_idx = nifty / nifty.iloc[0] * 100
        fig2.add_trace(go.Scatter(x=nifty_idx.index, y=nifty_idx.values,
                                   name="Nifty50",
                                   line=dict(color="black", width=2, dash="dot")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=nifty_dd.index, y=nifty_dd.values,
                                   name="Nifty DD",
                                   line=dict(color="black", width=1, dash="dot"),
                                   fill="tozeroy", fillcolor="rgba(0,0,0,0.05)"), row=2, col=1)

        palette = px.colors.qualitative.Bold
        for i, fund in enumerate(selected_funds):
            series = all_funds[fund].dropna()
            if series.empty:
                continue
            s_idx = series / series.iloc[0] * 100
            s_dd = compute_drawdown_series(series)
            color = palette[i % len(palette)]
            label = fund.split("-")[0][:35]
            fig2.add_trace(go.Scatter(x=s_idx.index, y=s_idx.values,
                                       name=label, line=dict(color=color, width=1.5)), row=1, col=1)
            fig2.add_trace(go.Scatter(x=s_dd.index, y=s_dd.values,
                                       name=label + " DD",
                                       line=dict(color=color, width=1, dash="dash"),
                                       showlegend=False), row=2, col=1)

        for _, ev in events_df.iterrows():
            fig2.add_vrect(x0=ev["start"], x1=ev["end"],
                           fillcolor="rgba(255,152,0,0.1)", layer="below",
                           line_width=0, row="all", col=1)

        fig2.update_layout(height=600, template="plotly_white",
                            legend=dict(orientation="h", y=-0.15))
        fig2.update_yaxes(ticksuffix="%", row=2, col=1)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"🟠 Shaded bands = Nifty drawdown events ≥ {threshold}%")

# ─── Tab 4 ───────────────────────────────────
with tab4:
    st.subheader("🗂️ Drawdown Heatmap — All Funds × All Events")
    st.markdown("Each cell = fund drawdown during that Nifty event. 🟢 Green = resilient, 🔴 Red = heavy drawdown.")

    # Build matrix rows
    records = []
    for fund in all_funds.columns:
        if category_map[fund] not in selected_cats:
            continue
        row = {"Category": category_map[fund], "Fund": fund}
        for i, ev in events_df.iterrows():
            label = f"Ev{i+1} {ev['start'].strftime('%b%y')}"
            dd = fund_dd_during_event(all_funds[fund].dropna(), ev["start"], ev["end"])
            row[label] = round(dd, 2) if not np.isnan(dd) else None
        records.append(row)

    matrix_df = pd.DataFrame(records)

    cat_filter = st.selectbox("Filter by category", ["All"] + sorted(selected_cats))
    if cat_filter != "All":
        matrix_df = matrix_df[matrix_df["Category"] == cat_filter]

    ev_cols = [c for c in matrix_df.columns if c.startswith("Ev")]

    # Build HTML table — NO pandas Styler, NO matplotlib dependency
    th_style = "style='padding:5px 8px;background:#1e3a5f;color:white;font-size:11px;white-space:nowrap;position:sticky;top:0'"
    header = f"<th {th_style}>Category</th><th {th_style}>Fund</th>"
    for col in ev_cols:
        header += f"<th {th_style}>{col}</th>"

    rows_html = ""
    for idx, row in matrix_df.iterrows():
        zebra = "#fafafa" if idx % 2 == 0 else "#ffffff"
        td_base = f"style='padding:4px 8px;font-size:11px;background:{zebra};white-space:nowrap'"
        cells = f"<td {td_base}><b>{row['Category']}</b></td>"
        cells += f"<td {td_base}>{row['Fund']}</td>"
        for col in ev_cols:
            val = row[col]
            bg, fg = dd_to_hex(val)
            txt = f"{val:.1f}%" if (val is not None and not (isinstance(val, float) and np.isnan(val))) else "–"
            cells += f"<td style='background:{bg};color:{fg};padding:4px 6px;text-align:center;font-size:11px'>{txt}</td>"
        rows_html += f"<tr>{cells}</tr>"

    html_out = f"""
    <div style='overflow-x:auto;overflow-y:auto;max-height:540px;
                border:1px solid #ddd;border-radius:6px;font-family:sans-serif'>
      <table style='border-collapse:collapse;width:100%'>
        <thead><tr>{header}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.html(html_out)

st.divider()
st.caption("Data: Nifty50 daily close • Mutual Fund NAV (funds1.xlsx + funds2.xlsx) • Drawdown = peak-to-trough % decline")
