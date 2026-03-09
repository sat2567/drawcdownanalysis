import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Fund Drawdown Analysis vs Nifty", layout="wide", page_icon="📉")

# ══════════════════════════════════════════════════════════════════
#  CORRECT LOGIC:
#  1. Find Nifty PEAK dates (rolling highs)
#  2. From each peak, detect when Nifty falls >= threshold%
#  3. Track to the lowest point = TROUGH
#  4. For each fund: return = (NAV on trough_date - NAV on peak_date)
#                             / NAV on peak_date * 100
#  5. Rank funds: closest to 0 = most resilient
# ══════════════════════════════════════════════════════════════════

def classify_category(name):
    n = name.lower()
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


def find_crash_events(nifty: pd.Series, threshold: float) -> pd.DataFrame:
    """
    Walk forward through Nifty prices.
    Track the rolling peak. When price falls >= threshold% from that peak,
    a crash starts. Continue until price recovers above peak*(1-threshold/2).
    The lowest point in the crash window = trough.
    Merge events < 10 days apart.
    """
    prices = nifty.values
    dates  = nifty.index
    n      = len(prices)
    events = []
    i = 0

    while i < n:
        peak_idx   = i
        peak_val   = prices[i]
        trough_idx = i
        trough_val = prices[i]
        in_crash   = False
        j = i + 1

        while j < n:
            pct_from_peak = (prices[j] - peak_val) / peak_val * 100

            if pct_from_peak <= -threshold:
                in_crash = True
                if prices[j] < trough_val:
                    trough_val = prices[j]
                    trough_idx = j
                j += 1
            elif in_crash:
                recovery_level = peak_val * (1 - threshold / 200)
                if prices[j] > recovery_level:
                    break
                else:
                    if prices[j] < trough_val:
                        trough_val = prices[j]
                        trough_idx = j
                    j += 1
            else:
                if prices[j] > peak_val:
                    peak_val   = prices[j]
                    peak_idx   = j
                    trough_idx = j
                    trough_val = prices[j]
                j += 1

        if in_crash:
            events.append({
                "peak_date"   : dates[peak_idx],
                "trough_date" : dates[trough_idx],
                "peak_value"  : round(float(peak_val), 2),
                "trough_value": round(float(trough_val), 2),
                "nifty_fall"  : round((float(trough_val) - float(peak_val)) / float(peak_val) * 100, 2),
            })
            i = trough_idx + 1
        else:
            break

    if not events:
        return pd.DataFrame()

    # Merge nearby events
    merged = [events[0]]
    for ev in events[1:]:
        gap = (ev["peak_date"] - merged[-1]["trough_date"]).days
        if gap <= 10:
            if ev["trough_value"] < merged[-1]["trough_value"]:
                merged[-1]["trough_date"]  = ev["trough_date"]
                merged[-1]["trough_value"] = ev["trough_value"]
                merged[-1]["nifty_fall"]   = round(
                    (merged[-1]["trough_value"] - merged[-1]["peak_value"])
                    / merged[-1]["peak_value"] * 100, 2)
        else:
            merged.append(ev)

    return pd.DataFrame(merged).reset_index(drop=True)


def fund_return_in_window(fund_nav: pd.Series, peak_date, trough_date):
    """
    % change = (NAV on trough_date - NAV on peak_date) / NAV on peak_date * 100
    Uses nearest available date if exact date has no NAV.
    """
    nav = fund_nav.dropna().sort_index()
    if nav.empty:
        return np.nan
    after_peak = nav[nav.index >= peak_date]
    if after_peak.empty:
        return np.nan
    nav_start = float(after_peak.iloc[0])
    before_trough = nav[nav.index <= trough_date]
    if before_trough.empty:
        return np.nan
    nav_end = float(before_trough.iloc[-1])
    if nav_start == 0 or np.isnan(nav_start):
        return np.nan
    return (nav_end - nav_start) / nav_start * 100


def dd_to_hex(val):
    """Pure-Python colour: green=0%, red=-30%+. No matplotlib needed."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#eeeeee", "#777777"
    ratio = max(0.0, min(1.0, abs(float(val)) / 30.0))
    r  = int(220 * ratio)
    g  = int(175 * (1.0 - ratio * 0.85))
    b  = 55
    bg = f"#{r:02x}{g:02x}{b:02x}"
    fg = "#ffffff" if ratio > 0.5 else "#111111"
    return bg, fg


def compute_drawdown_series(prices):
    return (prices - prices.cummax()) / prices.cummax() * 100


# ── Data loading ──────────────────────────────
@st.cache_data
def load_data():
    raw = pd.read_csv("Nifty50_10Years_Data.csv")
    raw["Date"] = (
        pd.to_datetime(raw["Date"], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    nifty = raw.set_index("Date").sort_index()["Close"].rename("Nifty50")

    def parse_excel(path):
        df    = pd.read_excel(path, header=None)
        names = df.iloc[2, 1:].tolist()
        data  = df.iloc[4:].copy()
        data.columns = ["Date"] + names
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
        data = data.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in names:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        return data

    funds = pd.concat(
        [parse_excel("funds1.xlsx"), parse_excel("funds2.xlsx")],
        axis=1, sort=True
    )
    cutoff = nifty.index.max() - pd.DateOffset(years=6)
    nifty  = nifty[nifty.index >= cutoff]
    funds  = funds[funds.index >= cutoff]
    common = nifty.index.intersection(funds.index)
    nifty  = nifty.reindex(common)
    funds  = funds.reindex(common)
    cat_map = {col: classify_category(col) for col in funds.columns}
    return nifty, funds, cat_map


# ── App ───────────────────────────────────────
st.title("📉 Fund Resilience During Nifty Crashes  —  Last 6 Years")
st.markdown(
    "For every Nifty crash ≥ threshold, measures **exactly how much each fund fell "
    "from the Nifty peak day to the Nifty trough day** and ranks by resilience."
)

with st.spinner("Loading data…"):
    nifty, all_funds, category_map = load_data()

st.sidebar.header("⚙️  Settings")
threshold     = st.sidebar.slider("Nifty crash trigger (%)", 2.0, 25.0, 9.0, 0.5)
top_n         = st.sidebar.slider("Top N resilient funds per category", 3, 15, 5)
all_cats      = sorted(set(category_map.values()))
selected_cats = st.sidebar.multiselect("Show categories", all_cats, default=all_cats)

events_df = find_crash_events(nifty, threshold)
nifty_dd  = compute_drawdown_series(nifty)

if events_df.empty:
    st.warning(f"No crash events >= {threshold}% found. Lower the threshold.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📅 Data Range", f"{nifty.index.min().strftime('%b %Y')} – {nifty.index.max().strftime('%b %Y')}")
c2.metric("📊 Funds", len(all_funds.columns))
c3.metric(f"⚡ Crashes ≥ {threshold}%", len(events_df))
c4.metric("📉 Worst Crash", f"{events_df['nifty_fall'].min():.1f}%")

st.divider()

# Pre-compute fund returns for all events
@st.cache_data
def build_fund_matrix(_all_funds, _events_df):
    result = {}
    for fund in _all_funds.columns:
        nav = _all_funds[fund]
        row = {}
        for i, ev in _events_df.iterrows():
            row[i] = fund_return_in_window(nav, ev["peak_date"], ev["trough_date"])
        result[fund] = row
    return result

fund_matrix = build_fund_matrix(all_funds, events_df)

avg_dd = {}
for fund, ev_dict in fund_matrix.items():
    vals = [v for v in ev_dict.values() if not np.isnan(v)]
    if vals:
        avg_dd[fund] = np.mean(vals)

fund_df = pd.DataFrame({
    "Fund": list(avg_dd.keys()),
    "Avg Return (%)": list(avg_dd.values()),
})
fund_df["Category"] = fund_df["Fund"].map(category_map)
fund_df = fund_df[fund_df["Category"].isin(selected_cats)]
fund_df = fund_df.sort_values("Avg Return (%)", ascending=False)
avg_nifty_fall = events_df["nifty_fall"].mean()


tab1, tab2, tab3, tab4 = st.tabs([
    "📆 Crash Events",
    "🏆 Fund Rankings",
    "📈 NAV Chart",
    "🗂️ Full Heatmap"
])

# ── Tab 1 ─────────────────────────────────────
with tab1:
    st.subheader(f"Nifty50 Crash Events ≥ {threshold}%")
    st.caption("Peak = Nifty all-time high before fall. Trough = lowest point. Fund returns measured between these exact dates.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nifty.index, y=nifty.values,
        line=dict(color="#546e7a", width=1.5), name="Nifty50"
    ))
    colours = px.colors.qualitative.Set2
    for i, ev in events_df.iterrows():
        col = colours[i % len(colours)]
        fig.add_vrect(x0=ev["peak_date"], x1=ev["trough_date"],
                      fillcolor=col, opacity=0.2, layer="below", line_width=0)
        fig.add_annotation(
            x=ev["trough_date"], y=ev["trough_value"],
            text=f"Ev{i+1}: {ev['nifty_fall']:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor=col,
            font=dict(size=10, color=col), ay=-35
        )
    fig.update_layout(
        title="Nifty50 — shaded regions = crash windows (peak to trough)",
        xaxis_title="Date", yaxis_title="Nifty50",
        height=400, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    ev_disp = events_df.copy()
    ev_disp["peak_date"]    = ev_disp["peak_date"].dt.strftime("%d %b %Y")
    ev_disp["trough_date"]  = ev_disp["trough_date"].dt.strftime("%d %b %Y")
    ev_disp["peak_value"]   = ev_disp["peak_value"].map("{:,.0f}".format)
    ev_disp["trough_value"] = ev_disp["trough_value"].map("{:,.0f}".format)
    ev_disp["nifty_fall"]   = ev_disp["nifty_fall"].map("{:.2f}%".format)
    ev_disp.index = range(1, len(ev_disp) + 1)
    ev_disp.columns = ["Peak Date", "Trough Date", "Nifty at Peak", "Nifty at Trough", "Nifty Fall %"]
    st.dataframe(ev_disp, use_container_width=True)

    st.info(
        f"💡 **Fund return formula:**  "
        f"(NAV on Trough Date − NAV on Peak Date) ÷ NAV on Peak Date × 100.  "
        f"If Nifty fell {threshold}% and a fund only fell 4%, that fund is very resilient."
    )

# ── Tab 2 ─────────────────────────────────────
with tab2:
    st.subheader("🏆 Most Resilient Funds — Avg Return During Crashes")
    st.caption(
        f"Averaged across all {len(events_df)} crash events. "
        "Closest to 0% (or positive) = most resilient. "
        f"Red dashed line = average Nifty fall ({avg_nifty_fall:.1f}%)."
    )
    show_breakdown = st.checkbox("Show per-event breakdown", value=False)

    for cat in sorted(selected_cats):
        cat_df = fund_df[fund_df["Category"] == cat].head(top_n).copy()
        if cat_df.empty:
            continue
        with st.expander(f"📂  {cat}  —  Top {top_n} most resilient", expanded=True):

            bar_colors = []
            for v in cat_df["Avg Return (%)"]:
                if v >= 0:               bar_colors.append("#00897b")
                elif v >= -threshold:    bar_colors.append("#66bb6a")
                else:                    bar_colors.append("#ef9a9a")

            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=cat_df["Avg Return (%)"],
                y=cat_df["Fund"].apply(lambda x: x[:50] + "…" if len(x) > 50 else x),
                orientation="h",
                marker_color=bar_colors,
                text=cat_df["Avg Return (%)"].map("{:.2f}%".format),
                textposition="outside"
            ))
            fig_b.add_vline(x=avg_nifty_fall, line_dash="dash", line_color="#e53935",
                             annotation_text=f"Avg Nifty ({avg_nifty_fall:.1f}%)",
                             annotation_position="bottom right")
            fig_b.add_vline(x=0, line_color="#aaaaaa", line_width=1)
            fig_b.update_layout(
                title=f"{cat} — avg % return during Nifty crash windows",
                xaxis_title="Avg % return (peak→trough)",
                height=max(280, top_n * 58),
                template="plotly_white",
                margin=dict(l=270),
                xaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(fig_b, use_container_width=True)

            if show_breakdown:
                tbl_rows = []
                for _, r in cat_df.iterrows():
                    fund = r["Fund"]
                    row_d = {"Fund": fund, "Avg (%)": f"{r['Avg Return (%)']:.2f}%"}
                    for i, ev in events_df.iterrows():
                        v = fund_matrix[fund][i]
                        row_d[f"Ev{i+1} {ev['peak_date'].strftime('%b%y')}"] = (
                            f"{v:.1f}%" if not np.isnan(v) else "–"
                        )
                    tbl_rows.append(row_d)
                st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)
            else:
                tbl = cat_df[["Fund", "Avg Return (%)"]].copy().reset_index(drop=True)
                tbl["vs Nifty Avg"] = (cat_df["Avg Return (%)"].values - avg_nifty_fall)
                tbl["vs Nifty Avg"] = tbl["vs Nifty Avg"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
                tbl["Avg Return (%)"] = tbl["Avg Return (%)"].map("{:.2f}%".format)
                st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Tab 3 ─────────────────────────────────────
with tab3:
    st.subheader("📈 NAV Chart — Zoomed into a Specific Crash")

    ev_labels = {
        f"Event {i+1}: {ev['peak_date'].strftime('%d %b %Y')} → "
        f"{ev['trough_date'].strftime('%d %b %Y')}  (Nifty {ev['nifty_fall']:.1f}%)": i
        for i, ev in events_df.iterrows()
    }
    chosen_label = st.selectbox("Select crash event", list(ev_labels.keys()))
    chosen_idx   = ev_labels[chosen_label]
    chosen_ev    = events_df.loc[chosen_idx]

    pad     = pd.Timedelta(days=20)
    w_start = chosen_ev["peak_date"]   - pad
    w_end   = chosen_ev["trough_date"] + pad

    default_funds = fund_df[fund_df["Category"].isin(selected_cats)].head(5)["Fund"].tolist()
    sel_funds = st.multiselect("Select funds to overlay", all_funds.columns.tolist(),
                               default=default_funds)

    if sel_funds:
        fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.65, 0.35],
                              subplot_titles=("NAV indexed to 100 at peak date",
                                              "% change from peak date"))

        nifty_w   = nifty[w_start:w_end]
        base_n    = float(nifty[nifty.index >= chosen_ev["peak_date"]].iloc[0])
        nifty_idx = nifty_w / base_n * 100
        nifty_pct = (nifty_w - base_n) / base_n * 100

        fig3.add_trace(go.Scatter(x=nifty_idx.index, y=nifty_idx.values,
                                   name="Nifty50",
                                   line=dict(color="black", width=2.5, dash="dot")), row=1, col=1)
        fig3.add_trace(go.Scatter(x=nifty_pct.index, y=nifty_pct.values,
                                   name="Nifty %",
                                   line=dict(color="black", width=1.5, dash="dot"),
                                   fill="tozeroy", fillcolor="rgba(0,0,0,0.06)"), row=2, col=1)

        palette = px.colors.qualitative.Bold
        for idx, fund in enumerate(sel_funds):
            nav_w = all_funds[fund][w_start:w_end].dropna()
            if nav_w.empty:
                continue
            ap = nav_w[nav_w.index >= chosen_ev["peak_date"]]
            if ap.empty:
                continue
            base_f  = float(ap.iloc[0])
            nav_idx = nav_w / base_f * 100
            nav_pct = (nav_w - base_f) / base_f * 100
            color   = palette[idx % len(palette)]
            label   = fund.split("-")[0][:35]
            ret     = fund_matrix[fund].get(chosen_idx, np.nan)
            leg     = f"{label}  ({ret:.1f}%)" if not np.isnan(ret) else label

            fig3.add_trace(go.Scatter(x=nav_idx.index, y=nav_idx.values,
                                       name=leg, line=dict(color=color, width=1.8)), row=1, col=1)
            fig3.add_trace(go.Scatter(x=nav_pct.index, y=nav_pct.values,
                                       name=leg, line=dict(color=color, width=1.2, dash="dash"),
                                       showlegend=False), row=2, col=1)

        fig3.add_vrect(x0=chosen_ev["peak_date"], x1=chosen_ev["trough_date"],
                       fillcolor="rgba(239,83,80,0.1)", layer="below", line_width=0, row="all", col=1)
        for rn in [1, 2]:
            fig3.add_vline(x=chosen_ev["peak_date"], line_dash="dot",
                           line_color="green", line_width=1.5, row=rn, col=1)
            fig3.add_vline(x=chosen_ev["trough_date"], line_dash="dot",
                           line_color="red", line_width=1.5, row=rn, col=1)

        fig3.update_layout(height=620, template="plotly_white",
                            legend=dict(orientation="h", y=-0.18))
        fig3.update_yaxes(ticksuffix="%", row=2, col=1)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("🟢 Green line = peak date  |  🔴 Red line = trough date  |  Legend shows each fund's return during this crash")

# ── Tab 4 ─────────────────────────────────────
with tab4:
    st.subheader("🗂️ Full Heatmap — Every Fund × Every Crash Event")
    st.markdown("Each cell = fund % return from Nifty peak → Nifty trough. 🟢 Green = resilient, 🔴 Red = heavy loss.")

    cat_filter = st.selectbox("Filter category", ["All"] + sorted(selected_cats))

    records = []
    for fund in all_funds.columns:
        if category_map[fund] not in selected_cats:
            continue
        if cat_filter != "All" and category_map[fund] != cat_filter:
            continue
        row = {"Category": category_map[fund], "Fund": fund}
        for i, ev in events_df.iterrows():
            lbl = f"Ev{i+1} {ev['peak_date'].strftime('%b%y')}"
            row[lbl] = fund_matrix[fund][i]
        records.append(row)

    matrix_df = pd.DataFrame(records)
    ev_cols   = [c for c in matrix_df.columns if c.startswith("Ev")]

    th = ("style='padding:5px 8px;background:#1a3a5c;color:white;"
          "font-size:10px;white-space:nowrap;text-align:center;"
          "position:sticky;top:0;z-index:2'")
    td_lbl = "style='padding:4px 8px;font-size:11px;white-space:nowrap;background:#f5f5f5'"

    hdr = f"<th {th}>Category</th><th {th}>Fund</th>"
    for col in ev_cols:
        hdr += f"<th {th}>{col}</th>"

    body = ""
    for ridx, row in matrix_df.iterrows():
        cells  = f"<td {td_lbl}><b>{row['Category']}</b></td>"
        cells += f"<td {td_lbl}>{row['Fund']}</td>"
        for col in ev_cols:
            val    = row[col]
            bg, fg = dd_to_hex(val)
            txt    = (f"{float(val):.1f}%"
                      if (val is not None and isinstance(val, float) and not np.isnan(val))
                      else "–")
            cells += (f"<td style='background:{bg};color:{fg};"
                      f"padding:5px 7px;text-align:center;font-size:11px'>{txt}</td>")
        body += f"<tr>{cells}</tr>"

    st.html(f"""
    <div style='overflow-x:auto;overflow-y:auto;max-height:560px;
                border:1px solid #ddd;border-radius:6px;font-family:sans-serif'>
      <table style='border-collapse:collapse;width:100%'>
        <thead><tr>{hdr}</tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """)

    # Nifty reference row
    st.markdown("**📌 Nifty reference fall per event:**")
    ref = {"Benchmark": "Nifty50"}
    for i, ev in events_df.iterrows():
        ref[f"Ev{i+1} {ev['peak_date'].strftime('%b%y')}"] = f"{ev['nifty_fall']:.1f}%"
    st.dataframe(pd.DataFrame([ref]), use_container_width=True, hide_index=True)

st.divider()
st.caption("Return = (NAV at Nifty trough − NAV at Nifty peak) / NAV at Nifty peak × 100  •  Last 6 years of data")
