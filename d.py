import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Fund Crash & Recovery Analysis",
    layout="wide", page_icon="📊"
)

# ══════════════════════════════════════════════════════════════
#  LOGIC SUMMARY
#
#  CRASH ANALYSIS
#  ─────────────
#  1. Find Nifty rolling peak dates
#  2. Detect when Nifty falls >= threshold% from peak  → crash starts
#  3. Track to the absolute lowest point              → trough
#  4. Fund drawdown = (NAV on trough_date - NAV on peak_date)
#                     / NAV on peak_date × 100
#  5. Rank: closest to 0 = most resilient
#
#  RECOVERY ANALYSIS
#  ─────────────────
#  1. Start from the trough date of each crash event
#  2. Recovery end = EITHER:
#       (a) Full recovery: date Nifty closes back at/above peak_value
#       (b) Fixed window:  trough + N days (for incomplete recoveries)
#  3. Fund recovery = (NAV on recovery_end - NAV on trough_date)
#                     / NAV on trough_date × 100
#  4. Rank: highest gain = fastest / strongest recovery
#
#  ALL-WEATHER SCORE
#  ─────────────────
#  Combined rank = drawdown_rank + recovery_rank (lower = better)
#  Best all-weather fund = fell least AND recovered most
# ══════════════════════════════════════════════════════════════


# ── Helpers ───────────────────────────────────────────────────

def classify_category(name):
    n = name.lower()
    if "flexi cap" in n or "flexicap" in n:       return "Flexi Cap"
    elif "small cap" in n or "smallcap" in n:      return "Small Cap"
    elif "multi cap" in n or "multicap" in n:      return "Multi Cap"
    elif ("large & mid" in n or "large and mid" in n
          or "large & midcap" in n or "large midcap" in n): return "Large & Mid Cap"
    elif "large cap" in n or "largecap" in n:      return "Large Cap"
    elif "mid cap" in n or "midcap" in n:          return "Mid Cap"
    else:                                           return "Other"


def classify_sector(name):
    """Map sector fund name → sector label."""
    n = name.lower()
    if any(x in n for x in ["banking", "financial", "fin serv", "bfsi"]):
        return "Banking & Finance"
    elif any(x in n for x in ["pharma", "healthcare", "health", "diagnostics"]):
        return "Pharma & Healthcare"
    elif any(x in n for x in ["tech", "digital", "teck", "it "]):
        return "Technology"
    elif any(x in n for x in ["infra", "infrastructure", "build india", "tiger", "t.i.g.e.r",
                               "power & infra", "resources & energy", "eco reform"]):
        return "Infrastructure"
    elif any(x in n for x in ["consumption", "consumer", "fmcg", "retail"]):
        return "Consumption"
    elif any(x in n for x in ["energy", "natural res", "power", "resources"]):
        return "Energy & Resources"
    elif any(x in n for x in ["auto", "automotive"]):
        return "Automotive"
    elif "services" in n:
        return "Services"
    else:
        return "Sector – Other"


def find_crash_events(nifty: pd.Series, threshold: float) -> pd.DataFrame:
    """
    Identify distinct crash events.
    Returns: peak_date, trough_date, peak_value, trough_value,
             nifty_fall (%), recovery_date (when Nifty returns to peak or NaT),
             recovery_days (int or NaN)
    """
    prices = nifty.values
    dates  = nifty.index
    N      = len(prices)
    events = []
    i = 0

    while i < N:
        peak_idx = i;  peak_val   = prices[i]
        trough_idx = i; trough_val = prices[i]
        in_crash = False
        j = i + 1

        while j < N:
            pct = (prices[j] - peak_val) / peak_val * 100
            if pct <= -threshold:
                in_crash = True
                if prices[j] < trough_val:
                    trough_val = prices[j]; trough_idx = j
                j += 1
            elif in_crash:
                if prices[j] > peak_val * (1 - threshold / 200):
                    break
                else:
                    if prices[j] < trough_val:
                        trough_val = prices[j]; trough_idx = j
                    j += 1
            else:
                if prices[j] > peak_val:
                    peak_val = prices[j]; peak_idx = j
                    trough_idx = j; trough_val = prices[j]
                j += 1

        if in_crash:
            # Find when Nifty fully recovers back to peak_value
            future = nifty[dates[trough_idx]:]
            recovered = future[future >= peak_val]
            rec_date  = recovered.index[0] if not recovered.empty else pd.NaT
            rec_days  = (rec_date - dates[trough_idx]).days if rec_date is not pd.NaT else np.nan

            events.append({
                "peak_date"    : dates[peak_idx],
                "trough_date"  : dates[trough_idx],
                "peak_value"   : round(float(peak_val), 2),
                "trough_value" : round(float(trough_val), 2),
                "nifty_fall"   : round((float(trough_val)-float(peak_val))/float(peak_val)*100, 2),
                "recovery_date": rec_date,
                "recovery_days": rec_days,
            })
            i = trough_idx + 1
        else:
            break

    if not events:
        return pd.DataFrame()

    # Merge events < 10 days apart
    merged = [events[0]]
    for ev in events[1:]:
        gap = (ev["peak_date"] - merged[-1]["trough_date"]).days
        if gap <= 10:
            if ev["trough_value"] < merged[-1]["trough_value"]:
                merged[-1].update({
                    "trough_date" : ev["trough_date"],
                    "trough_value": ev["trough_value"],
                    "nifty_fall"  : round((ev["trough_value"]-merged[-1]["peak_value"])
                                          /merged[-1]["peak_value"]*100, 2),
                    "recovery_date": ev["recovery_date"],
                    "recovery_days": ev["recovery_days"],
                })
        else:
            merged.append(ev)

    return pd.DataFrame(merged).reset_index(drop=True)


def fund_return_window(nav: pd.Series, start_date, end_date):
    """
    Generic: % change of fund NAV from start_date to end_date.
    Uses nearest available trading date if exact date missing.
    """
    s = nav.dropna().sort_index()
    if s.empty: return np.nan
    after  = s[s.index >= start_date]
    before = s[s.index <= end_date]
    if after.empty or before.empty: return np.nan
    v0 = float(after.iloc[0])
    v1 = float(before.iloc[-1])
    if v0 == 0 or np.isnan(v0): return np.nan
    return (v1 - v0) / v0 * 100


def fund_recovery(nav: pd.Series, trough_date, recovery_end_date,
                  fixed_days: int, last_date):
    """
    Recovery return from trough_date.
    If recovery_end_date is known (Nifty already recovered), use it.
    Otherwise use trough_date + fixed_days, capped at last_date.
    Returns (return_pct, end_date_used, is_full_recovery)
    """
    if pd.notna(recovery_end_date):
        ret = fund_return_window(nav, trough_date, recovery_end_date)
        return ret, recovery_end_date, True
    else:
        end = min(trough_date + pd.Timedelta(days=fixed_days), last_date)
        ret = fund_return_window(nav, trough_date, end)
        return ret, end, False


def dd_to_hex(val, vmin=-30):
    """Drawdown colouring: 0% = green, vmin% = red. No matplotlib."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#eeeeee", "#777777"
    ratio = max(0.0, min(1.0, abs(float(val)) / abs(vmin)))
    r = int(215 * ratio);  g = int(170 * (1 - ratio * 0.85));  b = 55
    return f"#{r:02x}{g:02x}{b:02x}", "#fff" if ratio > 0.5 else "#111"


def rec_to_hex(val, vmax=40):
    """Recovery colouring: vmax% = deep green, 0% = light. No matplotlib."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#eeeeee", "#777777"
    ratio = max(0.0, min(1.0, float(val) / vmax))
    r = int(30  + 190*(1-ratio))
    g = int(120 + 55*ratio)
    b = int(30  + 20*(1-ratio))
    return f"#{r:02x}{g:02x}{b:02x}", "#fff" if ratio > 0.6 else "#111"


def score_to_hex(val, vmax):
    """All-weather score: lower = greener."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#eeeeee", "#777777"
    ratio = max(0.0, min(1.0, float(val) / vmax))
    r = int(30 + 190*ratio); g = int(160 - 100*ratio); b = 55
    return f"#{r:02x}{g:02x}{b:02x}", "#fff" if ratio > 0.6 else "#111"


def html_table(df_rows, col_headers, cell_fn):
    """
    Build a scrollable HTML table.
    cell_fn(col_name, value) → (display_text, bg_hex, fg_hex)
    """
    th = ("style='padding:6px 9px;background:#1a3a5c;color:#fff;"
          "font-size:11px;white-space:nowrap;text-align:center;"
          "position:sticky;top:0;z-index:2'")
    hdr = "".join(f"<th {th}>{c}</th>" for c in col_headers)

    rows = ""
    for ridx, row in enumerate(df_rows):
        cells = ""
        for col in col_headers:
            val = row.get(col, None)
            txt, bg, fg = cell_fn(col, val)
            cells += (f"<td style='background:{bg};color:{fg};"
                      f"padding:5px 8px;font-size:11px;"
                      f"white-space:nowrap;text-align:center'>{txt}</td>")
        rows += f"<tr>{cells}</tr>"

    return f"""
    <div style='overflow:auto;max-height:560px;border:1px solid #ddd;
                border-radius:6px;font-family:sans-serif'>
      <table style='border-collapse:collapse;width:100%'>
        <thead><tr>{hdr}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = pd.read_csv("Nifty50_10Years_Data.csv")
    raw["Date"] = (pd.to_datetime(raw["Date"], utc=True)
                   .dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None))
    nifty = raw.set_index("Date").sort_index()["Close"].rename("Nifty50")

    def parse_excel(path):
        df = pd.read_excel(path, header=None)
        names = df.iloc[2, 1:].tolist()
        data  = df.iloc[4:].copy()
        data.columns = ["Date"] + names
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
        data = data.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in names:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        return data

    equity_funds  = pd.concat([parse_excel("funds1.xlsx"),
                                parse_excel("funds2.xlsx")], axis=1, sort=True)
    sector_funds  = parse_excel("sectorfunds.xlsx")

    # Build category maps before merging
    eq_cat_map  = {col: classify_category(col) for col in equity_funds.columns}
    sec_cat_map = {col: classify_sector(col)   for col in sector_funds.columns}
    cat_map     = {**eq_cat_map, **sec_cat_map}

    # Track which fund belongs to which universe
    fund_universe = (
        {col: "Equity" for col in equity_funds.columns} |
        {col: "Sector" for col in sector_funds.columns}
    )

    all_funds = pd.concat([equity_funds, sector_funds], axis=1, sort=True)

    # Return FULL data — year filtering happens in the UI based on user selection
    common = nifty.index.intersection(all_funds.index)
    nifty     = nifty.reindex(common)
    all_funds = all_funds.reindex(common)
    return nifty, all_funds, cat_map, fund_universe


# ── Pre-compute all fund × event matrices ─────────────────────
@st.cache_data
def build_matrices(_all_funds, _events_df, fixed_days, _last_date):
    crash_mat   = {}   # fund → {ev_idx: crash_return %}
    recovery_mat = {}  # fund → {ev_idx: recovery_return %}
    rec_end_mat  = {}  # fund → {ev_idx: (end_date, is_full)}

    for fund in _all_funds.columns:
        nav = _all_funds[fund]
        crash_mat[fund]    = {}
        recovery_mat[fund] = {}
        rec_end_mat[fund]  = {}
        for i, ev in _events_df.iterrows():
            # Crash: peak → trough
            crash_mat[fund][i] = fund_return_window(
                nav, ev["peak_date"], ev["trough_date"])
            # Recovery: trough → recovery_end (or fixed window)
            ret, end_d, is_full = fund_recovery(
                nav, ev["trough_date"], ev["recovery_date"],
                fixed_days, _last_date)
            recovery_mat[fund][i] = ret
            rec_end_mat[fund][i]  = (end_d, is_full)

    return crash_mat, recovery_mat, rec_end_mat


# ─────────────────────────────────────────────────────────────
#  APP UI
# ─────────────────────────────────────────────────────────────
st.title("📊 Fund Crash & Recovery Analysis")
st.markdown(
    "Ranks mutual funds on **two dimensions** during every Nifty crash:  \n"
    "📉 **Drawdown resilience** — which funds fell least  \n"
    "📈 **Recovery speed** — which funds bounced back hardest  \n"
    "🏅 **All-weather score** — which funds did best on both"
)

with st.spinner("Loading data…"):
    nifty_full, all_funds_full, category_map, fund_universe = load_data()

last_date  = nifty_full.index.max()
first_date = nifty_full.index.min()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️  Settings")

# Year range selector — how far back to analyse
st.sidebar.markdown("### 📅 Analysis Period")
years_back = st.sidebar.select_slider(
    "How many years back?",
    options=[1, 2, 3, 4, 5, 6],
    value=6,
    help="Choose how many years of history to include in the analysis"
)
cutoff = last_date - pd.DateOffset(years=years_back)
# Fund data starts Jan 2020 — clamp cutoff so we don't request data before that
cutoff = max(cutoff, first_date)

# Apply year filter
nifty     = nifty_full[nifty_full.index >= cutoff]
all_funds = all_funds_full[all_funds_full.index >= cutoff]

# Show what period is selected
period_start = nifty.index.min()
st.sidebar.info(
    f"📆 **{years_back} year{'s' if years_back > 1 else ''} of data**  \n"
    f"{period_start.strftime('%d %b %Y')} → {last_date.strftime('%d %b %Y')}  \n"
    f"({len(nifty)} trading days)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Crash Detection")
threshold = st.sidebar.slider(
    "Nifty crash trigger (%)", 2.0, 25.0, 9.0, 0.5,
    help="Detect events where Nifty fell ≥ this % from its rolling peak")
fixed_days = st.sidebar.slider(
    "Recovery window for incomplete recoveries (days)", 30, 365, 90, 10,
    help="If Nifty hasn't fully recovered yet, measure fund gain over this many days from trough")
st.sidebar.markdown("### 🎛️ Display")
top_n = st.sidebar.slider("Top N funds per category", 3, 15, 5)

# Fund universe toggle
fund_type = st.sidebar.radio(
    "Fund universe",
    ["All Funds", "Equity Funds Only", "Sector Funds Only"],
    index=0,
    help="Equity = diversified funds (Large Cap, Mid Cap, Flexi Cap etc.)  |  Sector = thematic funds (Banking, Pharma, Tech etc.)"
)

# Build category list based on universe selection
equity_cats = sorted({v for k,v in category_map.items() if fund_universe.get(k)=="Equity"})
sector_cats = sorted({v for k,v in category_map.items() if fund_universe.get(k)=="Sector"})
all_cats    = sorted(set(category_map.values()))

if fund_type == "Equity Funds Only":
    default_cats = equity_cats
elif fund_type == "Sector Funds Only":
    default_cats = sector_cats
else:
    default_cats = all_cats

selected_cats = st.sidebar.multiselect("Filter categories", all_cats, default=default_cats)

# ── Find crash events ─────────────────────────────────────────
events_df = find_crash_events(nifty, threshold)
if events_df.empty:
    st.warning(f"No crash events ≥ {threshold}% found. Lower the threshold.")
    st.stop()

# ── Build matrices ────────────────────────────────────────────
crash_mat, recovery_mat, rec_end_mat = build_matrices(
    all_funds, events_df, fixed_days, last_date)

# ── Aggregate averages ────────────────────────────────────────
avg_crash = {}
avg_rec   = {}
ev_indices = list(events_df.index)
for fund in all_funds.columns:
    if fund not in crash_mat:
        continue
    c_vals = [crash_mat[fund].get(i, np.nan)    for i in ev_indices]
    r_vals = [recovery_mat[fund].get(i, np.nan) for i in ev_indices]
    c_vals = [v for v in c_vals if isinstance(v, float) and not np.isnan(v)]
    r_vals = [v for v in r_vals if isinstance(v, float) and not np.isnan(v)]
    if c_vals: avg_crash[fund] = np.mean(c_vals)
    if r_vals: avg_rec[fund]   = np.mean(r_vals)

summary_df = pd.DataFrame({
    "Fund"           : list(avg_crash.keys()),
    "Avg Crash (%)"  : list(avg_crash.values()),
    "Avg Recovery (%)": [avg_rec.get(f, np.nan) for f in avg_crash.keys()],
})
summary_df["Category"] = summary_df["Fund"].map(category_map)
summary_df = summary_df[summary_df["Category"].isin(selected_cats)]


avg_nifty_crash = events_df["nifty_fall"].mean()

# ── Top-level metrics ─────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📅 Period",
          f"{nifty.index.min().strftime('%b %Y')} – {last_date.strftime('%b %Y')}",
          delta=f"{years_back} yr{'s' if years_back>1 else ''}")
c2.metric("📊 Funds", len(all_funds.columns))
c3.metric(f"⚡ Crashes ≥ {threshold}%", len(events_df))
c4.metric("📉 Worst Crash", f"{events_df['nifty_fall'].min():.1f}%")
fully = events_df["recovery_date"].notna().sum()
c5.metric("✅ Full Recoveries", f"{fully}/{len(events_df)}")

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab5 = st.tabs([
    "📆 Crash Events",
    "📉 Drawdown Rankings",
    "📈 Recovery Rankings",
    "🗂️ Full Heatmap",
])

# ─── TAB 1 — Events ───────────────────────────────────────────
with tab1:
    st.subheader(f"Nifty50 Crash Events ≥ {threshold}%")
    st.caption(
        "**Peak** = Nifty all-time high before crash.  "
        "**Trough** = lowest point.  "
        "**Recovery** = date Nifty closed back at/above peak (NaT = not yet recovered)."
    )

    # Price chart with crash + recovery bands
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nifty.index, y=nifty.values,
        line=dict(color="#37474f", width=1.5), name="Nifty50"))

    colours_crash = px.colors.qualitative.Set2
    colours_rec   = px.colors.qualitative.Pastel

    for i, ev in events_df.iterrows():
        cc = colours_crash[i % len(colours_crash)]
        # crash band
        fig.add_vrect(x0=ev["peak_date"], x1=ev["trough_date"],
                      fillcolor=cc, opacity=0.25, layer="below", line_width=0)
        fig.add_annotation(
            x=ev["trough_date"], y=ev["trough_value"],
            text=f"Ev{i+1} {ev['nifty_fall']:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor=cc,
            font=dict(size=9, color=cc), ay=-38)
        # recovery band (if available)
        if pd.notna(ev["recovery_date"]):
            cr = colours_rec[i % len(colours_rec)]
            fig.add_vrect(x0=ev["trough_date"], x1=ev["recovery_date"],
                          fillcolor=cr, opacity=0.18, layer="below", line_width=0)

    fig.update_layout(
        title="Nifty50 — 🔴 crash windows, 🟢 recovery windows",
        xaxis_title="Date", yaxis_title="Nifty50",
        height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Events table
    ev_disp = events_df.copy()
    ev_disp["peak_date"]    = ev_disp["peak_date"].dt.strftime("%d %b %Y")
    ev_disp["trough_date"]  = ev_disp["trough_date"].dt.strftime("%d %b %Y")
    ev_disp["peak_value"]   = ev_disp["peak_value"].map("{:,.0f}".format)
    ev_disp["trough_value"] = ev_disp["trough_value"].map("{:,.0f}".format)
    ev_disp["nifty_fall"]   = ev_disp["nifty_fall"].map("{:.2f}%".format)
    ev_disp["recovery_date"]= ev_disp["recovery_date"].apply(
        lambda x: x.strftime("%d %b %Y") if pd.notna(x) else "Not yet recovered")
    ev_disp["recovery_days"]= ev_disp["recovery_days"].apply(
        lambda x: f"{int(x)} days" if not np.isnan(x) else f">{fixed_days}d window used")
    ev_disp.index = range(1, len(ev_disp)+1)
    ev_disp.columns = ["Peak Date","Trough Date","Nifty at Peak",
                        "Nifty at Trough","Nifty Fall %",
                        "Nifty Recovery Date","Recovery Duration"]
    st.dataframe(ev_disp, use_container_width=True)

    st.info(
        "💡 **How fund returns are calculated:**\n\n"
        f"📉 **Crash** = (NAV on Trough Date − NAV on Peak Date) / NAV on Peak Date × 100\n\n"
        f"📈 **Recovery** = (NAV on Recovery End − NAV on Trough Date) / NAV on Trough Date × 100  "
        f"*(Recovery End = date Nifty returns to peak, or trough + {fixed_days} days if not yet recovered)*"
    )


# ─── TAB 2 — Drawdown Rankings ────────────────────────────────
with tab2:
    st.subheader("📉 Drawdown Resilience — Which Funds Fell Least?")
    st.caption(
        f"Avg crash return across all {len(events_df)} events. "
        "Closer to 0 = fell least = most resilient. "
        f"Red line = avg Nifty fall ({avg_nifty_crash:.1f}%)."
    )
    show_ev2 = st.checkbox("Show per-event breakdown", key="ev2", value=False)

    for cat in sorted(selected_cats):
        cat_df = (summary_df[summary_df["Category"]==cat]
                  .sort_values("Avg Crash (%)", ascending=False)
                  .head(top_n).copy())
        if cat_df.empty: continue

        with st.expander(f"📂  {cat}  —  Top {top_n} most resilient", expanded=True):
            bar_col = ["#00897b" if v >= -threshold else "#66bb6a" if v >= avg_nifty_crash else "#ef9a9a"
                       for v in cat_df["Avg Crash (%)"]]
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=cat_df["Avg Crash (%)"],
                y=cat_df["Fund"].apply(lambda x: x[:50]+"…" if len(x)>50 else x),
                orientation="h", marker_color=bar_col,
                text=cat_df["Avg Crash (%)"].map("{:.2f}%".format),
                textposition="outside"))
            fig_b.add_vline(x=avg_nifty_crash, line_dash="dash", line_color="#e53935",
                             annotation_text=f"Avg Nifty ({avg_nifty_crash:.1f}%)")
            fig_b.add_vline(x=0, line_color="#aaa", line_width=1)
            fig_b.update_layout(
                title=f"{cat} — avg % return peak→trough",
                xaxis_title="Avg % return", height=max(280, top_n*58),
                template="plotly_white", margin=dict(l=270),
                xaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_b, use_container_width=True)

            if show_ev2:
                rows = []
                for _, r in cat_df.iterrows():
                    d = {"Fund": r["Fund"], "Avg": f"{r['Avg Crash (%)']:.2f}%"}
                    for i, ev in events_df.iterrows():
                        v = crash_mat[r["Fund"]][i]
                        d[f"Ev{i+1} {ev['peak_date'].strftime('%b%y')}"] = (
                            f"{v:.1f}%" if not np.isnan(v) else "–")
                    rows.append(d)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                tbl = cat_df[["Fund","Avg Crash (%)"]].copy().reset_index(drop=True)
                tbl["vs Nifty"] = (cat_df["Avg Crash (%)"].values - avg_nifty_crash)
                tbl["vs Nifty"] = tbl["vs Nifty"].map(lambda x: f"+{x:.2f}%" if x>=0 else f"{x:.2f}%")
                tbl["Avg Crash (%)"] = tbl["Avg Crash (%)"].map("{:.2f}%".format)
                st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─── TAB 3 — Recovery Rankings ────────────────────────────────
with tab3:
    st.subheader("📈 Recovery Speed — Which Funds Bounced Back Hardest?")
    st.caption(
        f"Recovery = fund % gain from Nifty trough to Nifty recovery date "
        f"(or trough + {fixed_days} days if Nifty not yet back to peak). "
        "Higher = recovered more = stronger bounce."
    )
    show_ev3 = st.checkbox("Show per-event breakdown", key="ev3", value=False)

    # Nifty recovery reference: how much did Nifty itself gain from trough back to peak
    nifty_rec_refs = {}
    for i, ev in events_df.iterrows():
        if pd.notna(ev["recovery_date"]):
            r = fund_return_window(nifty, ev["trough_date"], ev["recovery_date"])
        else:
            end = min(ev["trough_date"] + pd.Timedelta(days=fixed_days), last_date)
            r = fund_return_window(nifty, ev["trough_date"], end)
        nifty_rec_refs[i] = r if r is not None else np.nan
    avg_nifty_rec = np.nanmean(list(nifty_rec_refs.values()))

    for cat in sorted(selected_cats):
        cat_df = (summary_df[summary_df["Category"]==cat]
                  .sort_values("Avg Recovery (%)", ascending=False)
                  .head(top_n).copy())
        if cat_df.empty: continue

        with st.expander(f"📂  {cat}  —  Top {top_n} fastest recovery", expanded=True):
            bar_col = ["#1565c0" if v >= avg_nifty_rec * 1.1
                       else "#42a5f5" if v >= avg_nifty_rec
                       else "#ffcc80"
                       for v in cat_df["Avg Recovery (%)"]]
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(
                x=cat_df["Avg Recovery (%)"],
                y=cat_df["Fund"].apply(lambda x: x[:50]+"…" if len(x)>50 else x),
                orientation="h", marker_color=bar_col,
                text=cat_df["Avg Recovery (%)"].map("{:.2f}%".format),
                textposition="outside"))
            fig_r.add_vline(x=avg_nifty_rec, line_dash="dash", line_color="#1565c0",
                             annotation_text=f"Avg Nifty recovery ({avg_nifty_rec:.1f}%)")
            fig_r.update_layout(
                title=f"{cat} — avg % gain trough→recovery end",
                xaxis_title="Avg % recovery", height=max(280, top_n*58),
                template="plotly_white", margin=dict(l=270),
                xaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_r, use_container_width=True)

            if show_ev3:
                rows = []
                for _, r in cat_df.iterrows():
                    d = {"Fund": r["Fund"], "Avg": f"{r['Avg Recovery (%)']:.2f}%"}
                    for i, ev in events_df.iterrows():
                        v = recovery_mat[r["Fund"]][i]
                        end_d, is_full = rec_end_mat[r["Fund"]][i]
                        suffix = "✅" if is_full else f"({fixed_days}d)"
                        d[f"Ev{i+1} {ev['trough_date'].strftime('%b%y')} {suffix}"] = (
                            f"{v:.1f}%" if not np.isnan(v) else "–")
                    rows.append(d)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                tbl = cat_df[["Fund","Avg Recovery (%)"]].copy().reset_index(drop=True)
                tbl["vs Nifty"] = (cat_df["Avg Recovery (%)"].values - avg_nifty_rec)
                tbl["vs Nifty"] = tbl["vs Nifty"].map(lambda x: f"+{x:.2f}%" if x>=0 else f"{x:.2f}%")
                tbl["Avg Recovery (%)"] = tbl["Avg Recovery (%)"].map("{:.2f}%".format)
                st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─── TAB 5 — Full Heatmap ─────────────────────────────────────
with tab5:
    st.subheader("🗂️ Complete Heatmap — Crash + Recovery for Every Fund × Every Event")
    st.markdown(
        "Each cell shows the fund's % return during that event window. "
        "The **Nifty50 row** is pinned at the top of the table as a benchmark reference."
    )

    col_vm, col_cf = st.columns([1, 1])
    with col_vm:
        view_mode = st.radio("Show", ["📉 Crash Returns", "📈 Recovery Returns"],
                             horizontal=True, key="hm_mode")
    with col_cf:
        cat_filter = st.selectbox("Filter category", ["All"] + sorted(selected_cats),
                                  key="hm_cat")

    is_crash = "Crash" in view_mode

    # ── Build Nifty reference values ──────────────────────────
    nifty_row_vals = {}
    for i, ev in events_df.iterrows():
        if is_crash:
            nifty_row_vals[i] = ev["nifty_fall"]
        else:
            nr = nifty_rec_refs.get(i, np.nan)
            nifty_row_vals[i] = nr

    # ── Build fund rows ───────────────────────────────────────
    records = []
    for fund in all_funds.columns:
        cat = category_map[fund]
        if cat not in selected_cats: continue
        if cat_filter != "All" and cat != cat_filter: continue
        row = {"_cat": cat, "_fund": fund, "_is_nifty": False}
        for i, ev in events_df.iterrows():
            row[i] = crash_mat.get(fund, {}).get(i, np.nan) if is_crash else recovery_mat.get(fund, {}).get(i, np.nan)
        records.append(row)

    # Sort by category then by avg performance
    records.sort(key=lambda r: (
        r["_cat"],
        -np.nanmean([v for k,v in r.items()
                     if isinstance(k, int) and not np.isnan(float(v)) if v is not None])
    ))

    # ── Column header labels ──────────────────────────────────
    ev_indices = list(events_df.index)

    # ── CSS constants ─────────────────────────────────────────
    # Header row style
    TH_BASE  = ("padding:7px 10px;font-size:11px;white-space:nowrap;"
                "text-align:center;position:sticky;top:0;z-index:3;")
    TH_DARK  = f"style='{TH_BASE}background:#1a3a5c;color:#ffffff;'"
    TH_EVENT = f"style='{TH_BASE}background:#1a3a5c;color:#ffffff;'"

    # Category/Fund cell — dark background, white text, always readable
    TD_CAT   = ("style='padding:5px 10px;font-size:11px;font-weight:700;"
                "white-space:nowrap;background:#2c3e50;color:#ffffff;"
                "border-right:1px solid #4a5568;'")
    TD_FUND  = ("style='padding:5px 10px;font-size:11px;"
                "white-space:nowrap;background:#f8f9fa;color:#1a1a2e;"
                "border-right:2px solid #dee2e6;max-width:240px;'")

    # Nifty benchmark row
    TD_NIFTY_CAT  = ("style='padding:6px 10px;font-size:11px;font-weight:700;"
                     "white-space:nowrap;background:#7f1d1d;color:#ffffff;"
                     "border-right:1px solid #991b1b;position:sticky;top:28px;z-index:2;'")
    TD_NIFTY_FUND = ("style='padding:6px 10px;font-size:12px;font-weight:700;"
                     "white-space:nowrap;background:#7f1d1d;color:#ffffff;"
                     "border-right:2px solid #991b1b;position:sticky;top:28px;z-index:2;'")

    # ── Build header ──────────────────────────────────────────
    hdr = f"<th {TH_DARK}>Category</th><th {TH_DARK}>Fund Name</th>"
    for i, ev in events_df.iterrows():
        crash_str = f"▼ {abs(ev['nifty_fall']):.1f}%"
        date_str  = (f"Ev{i+1}<br>"
                     f"{ev['peak_date'].strftime('%d %b %y')} →<br>"
                     f"{ev['trough_date'].strftime('%d %b %y')}<br>"
                     f"<span style='color:#ff6b6b;font-weight:700'>{crash_str}</span>")
        hdr += f"<th {TH_EVENT}>{date_str}</th>"

    # ── Build Nifty benchmark row (pinned second row) ─────────
    nifty_cells = f"<td {TD_NIFTY_CAT}>📊 BENCHMARK</td>"
    nifty_cells += f"<td {TD_NIFTY_FUND}>Nifty 50</td>"
    for i in ev_indices:
        val = nifty_row_vals.get(i, np.nan)
        if is_crash:
            bg, fg = dd_to_hex(val)
        else:
            bg, fg = rec_to_hex(val)
        txt = f"{float(val):.1f}%" if (val is not None and isinstance(val, float) and not np.isnan(val)) else "–"
        nifty_cells += (
            f"<td style='background:{bg};color:{fg};"
            f"padding:6px 8px;text-align:center;font-size:12px;"
            f"font-weight:700;border:1px solid rgba(255,255,255,0.2);"
            f"position:sticky;top:28px;z-index:2'>{txt}</td>"
        )
    nifty_row_html = f"<tr>{nifty_cells}</tr>"

    # ── Build fund rows ───────────────────────────────────────
    body = ""
    prev_cat = None
    for ridx, row in enumerate(records):
        cat  = row["_cat"]
        fund = row["_fund"]

        # Category label cell — new category gets accent background
        cat_changed = cat != prev_cat
        if cat_changed:
            cat_bg   = "#2c3e50"
            cat_text = f"<b>{cat}</b>"
            prev_cat = cat
        else:
            cat_bg   = "#3d5166"
            cat_text = ""   # blank for same-category rows to reduce visual noise

        td_cat_dyn = (f"style='padding:5px 10px;font-size:11px;font-weight:700;"
                      f"white-space:nowrap;background:{cat_bg};color:#ffffff;"
                      f"border-right:1px solid #4a5568;'")

        # Fund name — alternate row shading
        row_bg   = "#ffffff" if ridx % 2 == 0 else "#f0f4f8"
        td_fund_dyn = (f"style='padding:5px 10px;font-size:11px;"
                       f"white-space:nowrap;background:{row_bg};color:#1a1a2e;"
                       f"border-right:2px solid #dee2e6;'")

        cells  = f"<td {td_cat_dyn}>{cat_text}</td>"
        cells += f"<td {td_fund_dyn}>{fund}</td>"

        for i in ev_indices:
            val = row.get(i)
            if is_crash:
                bg, fg = dd_to_hex(val)
            else:
                bg, fg = rec_to_hex(val)
            txt = (f"{float(val):.1f}%"
                   if (val is not None and isinstance(val, float) and not np.isnan(val))
                   else "–")
            # Show outperformance vs Nifty as a small badge
            nifty_val = nifty_row_vals.get(i, np.nan)
            badge = ""
            if (val is not None and isinstance(val, float) and not np.isnan(val)
                    and not np.isnan(nifty_val)):
                diff = val - nifty_val
                if is_crash:
                    # Crash: fund diff > 0 means fund fell LESS than Nifty = good
                    if diff > 1.0:
                        badge = f"<br><span style='font-size:9px;font-weight:700;color:#ffffff;background:rgba(0,0,0,0.32);border-radius:3px;padding:0 3px'>▲{diff:.1f}%</span>"
                    elif diff < -1.0:
                        badge = f"<br><span style='font-size:9px;font-weight:700;color:#ffffff;background:rgba(0,0,0,0.32);border-radius:3px;padding:0 3px'>▼{abs(diff):.1f}%</span>"
                else:
                    # Recovery: fund diff > 0 means fund recovered MORE than Nifty = good
                    if diff > 1.0:
                        badge = f"<br><span style='font-size:9px;font-weight:700;color:#ffffff;background:rgba(0,0,0,0.32);border-radius:3px;padding:0 3px'>+{diff:.1f}%</span>"
                    elif diff < -1.0:
                        badge = f"<br><span style='font-size:9px;font-weight:700;color:#ffffff;background:rgba(0,0,0,0.32);border-radius:3px;padding:0 3px'>{diff:.1f}%</span>"

            cells += (f"<td style='background:{bg};color:{fg};"
                      f"padding:5px 7px;text-align:center;font-size:11px;"
                      f"border:1px solid rgba(0,0,0,0.05)'>{txt}{badge}</td>")
        body += f"<tr>{cells}</tr>"

    # ── Legend — rendered OUTSIDE the scrollable table div ──────
    if is_crash:
        legend_items = [
            ("<span style='display:inline-block;width:14px;height:14px;"
             "background:#00aa37;border-radius:3px;vertical-align:middle'></span>",
             "<b style='color:#1a1a1a'>Green</b> = small loss (resilient)"),
            ("<span style='display:inline-block;width:14px;height:14px;"
             "background:#dc3737;border-radius:3px;vertical-align:middle'></span>",
             "<b style='color:#1a1a1a'>Red</b> = large loss"),
            ("<b style='color:#15803d;font-size:13px'>▲X%</b>",
             "<span style='color:#1a1a1a'>= beat Nifty by X%</span>"),
            ("<b style='color:#b91c1c;font-size:13px'>▼X%</b>",
             "<span style='color:#1a1a1a'>= worse than Nifty by X%</span>"),
        ]
    else:
        legend_items = [
            ("<span style='display:inline-block;width:14px;height:14px;"
             "background:#1eaf1e;border-radius:3px;vertical-align:middle'></span>",
             "<b style='color:#1a1a1a'>Dark green</b> = strong recovery"),
            ("<span style='display:inline-block;width:14px;height:14px;"
             "background:#dc7832;border-radius:3px;vertical-align:middle'></span>",
             "<b style='color:#1a1a1a'>Orange/Yellow</b> = weak recovery"),
            ("<b style='color:#15803d;font-size:13px'>+X%</b>",
             "<span style='color:#1a1a1a'>= recovered more than Nifty</span>"),
            ("<b style='color:#b91c1c;font-size:13px'>−X%</b>",
             "<span style='color:#1a1a1a'>= recovered less than Nifty</span>"),
        ]

    legend_spans = "".join(
        f"<span style='display:flex;align-items:center;gap:6px;"
        f"padding:6px 14px;background:#ffffff;border:1px solid #e2e8f0;"
        f"border-radius:20px;white-space:nowrap;font-size:12px'>"
        f"{icon}&nbsp;{label}</span>"
        for icon, label in legend_items
    )

    st.html(f"""
    <div style='overflow:auto;max-height:600px;border:1px solid #dee2e6;
                border-radius:8px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
      <table style='border-collapse:collapse;width:100%'>
        <thead>
          <tr>{hdr}</tr>
        </thead>
        <tbody>
          {nifty_row_html}
          {body}
        </tbody>
      </table>
    </div>""")

    # Legend rendered outside scroll container so it's always visible
    st.html(f"""
    <div style='display:flex;flex-wrap:wrap;gap:8px;padding:10px 4px;
                margin-top:8px;font-family:-apple-system,BlinkMacSystemFont,sans-serif'>
      <span style='font-size:12px;font-weight:600;color:#374151;
                   align-self:center;padding-right:4px'>Legend:</span>
      {legend_spans}
    </div>""")

    # ── Summary stats below heatmap ───────────────────────────
    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**📌 Nifty50 reference values per event:**")
        ref_rows = []
        for i, ev in events_df.iterrows():
            crash_val = ev["nifty_fall"]
            rec_val   = nifty_rec_refs.get(i, np.nan)
            ref_rows.append({
                "Event"        : f"Ev{i+1}",
                "Period"       : f"{ev['peak_date'].strftime('%d %b %Y')} → {ev['trough_date'].strftime('%d %b %Y')}",
                "Nifty Crash %" : f"{crash_val:.2f}%",
                "Nifty Recovery %": f"{rec_val:.2f}%" if not np.isnan(rec_val) else "In progress",
            })
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
    with col_s2:
        st.markdown("**🏅 Funds that beat Nifty in most events:**")
        beat_counts = []
        for fund in all_funds.columns:
            if category_map[fund] not in selected_cats: continue
            if cat_filter != "All" and category_map[fund] != cat_filter: continue
            beats = 0
            total = 0
            for i in ev_indices:
                nv = nifty_row_vals.get(i, np.nan)
                fv = crash_mat.get(fund, {}).get(i, np.nan) if is_crash else recovery_mat.get(fund, {}).get(i, np.nan)
                if not np.isnan(nv) and not np.isnan(fv):
                    total += 1
                    if is_crash and fv > nv: beats += 1
                    elif not is_crash and fv > nv: beats += 1
            if total > 0:
                beat_counts.append({
                    "Fund"    : fund,
                    "Category": category_map[fund],
                    "Beat Nifty": f"{beats}/{total} events",
                    "Hit Rate": beats/total
                })
        if beat_counts:
            bc_df = (pd.DataFrame(beat_counts)
                     .sort_values("Hit Rate", ascending=False)
                     .head(10)
                     .reset_index(drop=True))
            bc_df["Hit Rate"] = bc_df["Hit Rate"].map("{:.0%}".format)
            bc_df = bc_df.drop(columns=["Hit Rate"])
            st.dataframe(bc_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "📉 Crash = (NAV at trough − NAV at peak) / NAV at peak × 100  •  "
    f"📈 Recovery = (NAV at recovery end − NAV at trough) / NAV at trough × 100  •  "
    f"Recovery end = Nifty back to peak, or trough + {fixed_days} days if incomplete  •  "
    "Last 6 years of data"
)
