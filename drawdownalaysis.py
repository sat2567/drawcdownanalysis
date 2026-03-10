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
def build_matrices(_all_funds, _events_df, fixed_days, _last_date, _fund_keys=None):
    # _fund_keys is a tuple of column names — used as an explicit cache key
    # so switching universe (equity vs sector) always triggers a fresh compute
    crash_mat    = {}
    recovery_mat = {}
    rec_end_mat  = {}

    for fund in _all_funds.columns:
        nav = _all_funds[fund]
        crash_mat[fund]    = {}
        recovery_mat[fund] = {}
        rec_end_mat[fund]  = {}
        for i, ev in _events_df.iterrows():
            crash_mat[fund][i] = fund_return_window(
                nav, ev["peak_date"], ev["trough_date"])
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

with st.spinner("Loading data…"):
    nifty_full, all_funds_full, category_map, fund_universe = load_data()

last_date  = nifty_full.index.max()
first_date = nifty_full.index.min()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️  Settings")

years_back = st.sidebar.select_slider(
    "Years of history", options=[1, 2, 3, 4, 5, 6], value=6)
cutoff = max(last_date - pd.DateOffset(years=years_back), first_date)
nifty     = nifty_full[nifty_full.index >= cutoff]
all_funds = all_funds_full[all_funds_full.index >= cutoff]

threshold  = st.sidebar.slider("Crash threshold (%)", 2.0, 25.0, 9.0, 0.5,
    help="Detect Nifty falls ≥ this % from rolling peak")
fixed_days = st.sidebar.slider("Recovery window (days)", 30, 365, 90, 10,
    help="Days from trough to measure recovery if Nifty hasn't fully recovered")
top_n      = st.sidebar.slider("Top N funds shown", 3, 15, 5)

fund_type = st.sidebar.radio(
    "Fund universe",
    ["All Funds", "Equity Only", "Sector Only"],
    help="Equity = diversified (Large/Mid/Flexi Cap)  |  Sector = thematic (Banking, Pharma, Tech…)"
)

if fund_type == "Equity Only":
    active_funds = [f for f in all_funds.columns if fund_universe.get(f) == "Equity"]
elif fund_type == "Sector Only":
    active_funds = [f for f in all_funds.columns if fund_universe.get(f) == "Sector"]
else:
    active_funds = list(all_funds.columns)

all_funds     = all_funds[active_funds]
all_cats      = sorted({category_map[f] for f in active_funds})
selected_cats = st.sidebar.multiselect("Categories", all_cats, default=all_cats,
                                        key=f"cats_{fund_type}")

# ── Crash events ──────────────────────────────────────────────
events_df = find_crash_events(nifty, threshold)
if events_df.empty:
    st.warning(f"No crash events ≥ {threshold}% found. Lower the threshold.")
    st.stop()

# ── Matrices ──────────────────────────────────────────────────
crash_mat, recovery_mat, rec_end_mat = build_matrices(
    all_funds, events_df, fixed_days, last_date,
    _fund_keys=tuple(all_funds.columns))

# ── Nifty recovery refs (needed by tabs) ──────────────────────
nifty_rec_refs = {}
for i, ev in events_df.iterrows():
    if pd.notna(ev["recovery_date"]):
        r = fund_return_window(nifty, ev["trough_date"], ev["recovery_date"])
    else:
        end = min(ev["trough_date"] + pd.Timedelta(days=fixed_days), last_date)
        r   = fund_return_window(nifty, ev["trough_date"], end)
    nifty_rec_refs[i] = r if r is not None else np.nan
avg_nifty_rec   = np.nanmean(list(nifty_rec_refs.values()))

# ── Averages ──────────────────────────────────────────────────
avg_crash = {}
avg_rec   = {}
ev_indices = list(events_df.index)
for fund in all_funds.columns:
    if fund not in crash_mat: continue
    c_vals = [crash_mat[fund].get(i, np.nan)    for i in ev_indices]
    r_vals = [recovery_mat[fund].get(i, np.nan) for i in ev_indices]
    c_vals = [v for v in c_vals if isinstance(v, float) and not np.isnan(v)]
    r_vals = [v for v in r_vals if isinstance(v, float) and not np.isnan(v)]
    if c_vals: avg_crash[fund] = np.mean(c_vals)
    if r_vals: avg_rec[fund]   = np.mean(r_vals)

summary_df = pd.DataFrame({
    "Fund"            : list(avg_crash.keys()),
    "Avg Crash (%)"   : list(avg_crash.values()),
    "Avg Recovery (%)": [avg_rec.get(f, np.nan) for f in avg_crash.keys()],
})
summary_df["Category"] = summary_df["Fund"].map(category_map)
summary_df = summary_df[summary_df["Category"].isin(selected_cats)]
avg_nifty_crash = events_df["nifty_fall"].mean()

# ── Top metrics ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📅 Period",
          f"{nifty.index.min().strftime('%b %Y')} – {last_date.strftime('%b %Y')}")
c2.metric("📊 Funds", len(all_funds.columns))
c3.metric(f"⚡ Crashes ≥ {threshold}%", len(events_df))
c4.metric("📉 Worst Nifty Crash", f"{events_df['nifty_fall'].min():.1f}%")

# Crash events summary — compact expander instead of full tab
with st.expander(f"📆 View crash events ({len(events_df)} detected)", expanded=False):
    ev_disp = events_df[["peak_date","trough_date","nifty_fall","recovery_date","recovery_days"]].copy()
    ev_disp["peak_date"]     = ev_disp["peak_date"].dt.strftime("%d %b %Y")
    ev_disp["trough_date"]   = ev_disp["trough_date"].dt.strftime("%d %b %Y")
    ev_disp["nifty_fall"]    = ev_disp["nifty_fall"].map("{:.2f}%".format)
    ev_disp["recovery_date"] = ev_disp["recovery_date"].apply(
        lambda x: x.strftime("%d %b %Y") if pd.notna(x) else "Not yet")
    ev_disp["recovery_days"] = ev_disp["recovery_days"].apply(
        lambda x: f"{int(x)}d" if not np.isnan(x) else "—")
    ev_disp.index = range(1, len(ev_disp)+1)
    ev_disp.columns = ["Peak Date", "Trough Date", "Nifty Fall", "Nifty Recovered", "Days"]
    st.dataframe(ev_disp, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABS  (3 tabs only)
# ══════════════════════════════════════════════════════════════
tab_crash, tab_rec, tab_heat = st.tabs([
    "📉 Drawdown Rankings",
    "📈 Recovery Rankings",
    "🗂️ Heatmap",
])


# ─── TAB: Drawdown Rankings ───────────────────────────────────
with tab_crash:
    st.caption(
        f"Average % fall per fund from Nifty peak → trough across all {len(events_df)} crash events. "
        f"Closer to 0 = fell least.  Red line = Nifty avg ({avg_nifty_crash:.1f}%)."
    )

    for cat in sorted(selected_cats):
        cat_df = (summary_df[summary_df["Category"] == cat]
                  .sort_values("Avg Crash (%)", ascending=False)
                  .head(top_n).copy())
        if cat_df.empty: continue

        with st.expander(f"**{cat}**", expanded=True):
            bar_col = ["#00897b" if v >= -threshold
                       else "#66bb6a" if v >= avg_nifty_crash
                       else "#ef9a9a"
                       for v in cat_df["Avg Crash (%)"]]
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=cat_df["Avg Crash (%)"],
                y=cat_df["Fund"].apply(lambda x: x[:55]+"…" if len(x)>55 else x),
                orientation="h", marker_color=bar_col,
                text=cat_df["Avg Crash (%)"].map("{:.1f}%".format),
                textposition="outside"))
            fig_b.add_vline(x=avg_nifty_crash, line_dash="dash", line_color="#e53935",
                            annotation_text=f"Nifty avg {avg_nifty_crash:.1f}%",
                            annotation_font_size=10)
            fig_b.add_vline(x=0, line_color="#ccc", line_width=1)
            fig_b.update_layout(
                height=max(260, top_n * 52),
                template="plotly_white",
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(ticksuffix="%", title=""),
                yaxis=dict(title=""),
                showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True)

            # Simple table: fund | avg crash | vs nifty
            tbl = cat_df[["Fund", "Avg Crash (%)"]].copy().reset_index(drop=True)
            tbl["vs Nifty"] = (cat_df["Avg Crash (%)"].values - avg_nifty_crash)
            tbl["vs Nifty"] = tbl["vs Nifty"].map(
                lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
            tbl["Avg Crash (%)"] = tbl["Avg Crash (%)"].map("{:.1f}%".format)
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─── TAB: Recovery Rankings ───────────────────────────────────
with tab_rec:
    st.caption(
        f"Average % gain from Nifty trough → recovery date across all {len(events_df)} events. "
        f"Higher = stronger bounce.  Blue line = Nifty avg ({avg_nifty_rec:.1f}%)."
    )

    for cat in sorted(selected_cats):
        cat_df = (summary_df[summary_df["Category"] == cat]
                  .sort_values("Avg Recovery (%)", ascending=False)
                  .head(top_n).copy())
        if cat_df.empty: continue

        with st.expander(f"**{cat}**", expanded=True):
            bar_col = ["#1565c0" if v >= avg_nifty_rec * 1.1
                       else "#42a5f5" if v >= avg_nifty_rec
                       else "#ffcc80"
                       for v in cat_df["Avg Recovery (%)"]]
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(
                x=cat_df["Avg Recovery (%)"],
                y=cat_df["Fund"].apply(lambda x: x[:55]+"…" if len(x)>55 else x),
                orientation="h", marker_color=bar_col,
                text=cat_df["Avg Recovery (%)"].map("{:.1f}%".format),
                textposition="outside"))
            fig_r.add_vline(x=avg_nifty_rec, line_dash="dash", line_color="#1565c0",
                            annotation_text=f"Nifty avg {avg_nifty_rec:.1f}%",
                            annotation_font_size=10)
            fig_r.update_layout(
                height=max(260, top_n * 52),
                template="plotly_white",
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(ticksuffix="%", title=""),
                yaxis=dict(title=""),
                showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)

            tbl = cat_df[["Fund", "Avg Recovery (%)"]].copy().reset_index(drop=True)
            tbl["vs Nifty"] = (cat_df["Avg Recovery (%)"].values - avg_nifty_rec)
            tbl["vs Nifty"] = tbl["vs Nifty"].map(
                lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
            tbl["Avg Recovery (%)"] = tbl["Avg Recovery (%)"].map("{:.1f}%".format)
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─── TAB: Heatmap ─────────────────────────────────────────────
with tab_heat:
    col_vm, col_cf = st.columns([1, 1])
    with col_vm:
        view_mode = st.radio("Show", ["📉 Crash Returns", "📈 Recovery Returns"],
                             horizontal=True, key="hm_mode")
    with col_cf:
        cat_filter = st.selectbox("Category", ["All"] + sorted(selected_cats), key="hm_cat")

    is_crash = "Crash" in view_mode

    # Nifty reference per event
    nifty_row_vals = {}
    for i, ev in events_df.iterrows():
        nifty_row_vals[i] = ev["nifty_fall"] if is_crash else nifty_rec_refs.get(i, np.nan)

    # Fund rows
    records = []
    for fund in all_funds.columns:
        cat = category_map[fund]
        if cat not in selected_cats: continue
        if cat_filter != "All" and cat != cat_filter: continue
        row = {"_cat": cat, "_fund": fund}
        for i in ev_indices:
            row[i] = crash_mat.get(fund, {}).get(i, np.nan) if is_crash                      else recovery_mat.get(fund, {}).get(i, np.nan)
        records.append(row)

    records.sort(key=lambda r: (
        r["_cat"],
        -np.nanmean([v for k, v in r.items()
                     if isinstance(k, int) and isinstance(v, float) and not np.isnan(v)]
                    or [0])
    ))

    # ── Build HTML table ──────────────────────────────────────
    TH = ("padding:6px 10px;font-size:11px;white-space:nowrap;text-align:center;"
          "position:sticky;top:0;z-index:3;background:#1a3a5c;color:#ffffff;")

    hdr = f"<th style='{TH}'>Category</th><th style='{TH}'>Fund</th>"
    for i, ev in events_df.iterrows():
        fall_str = f"<br><span style='color:#ff6b6b;font-weight:700'>▼{abs(ev['nifty_fall']):.1f}%</span>"
        hdr += (f"<th style='{TH}'>"
                f"Ev{i+1} {ev['peak_date'].strftime('%b%y')}{fall_str}</th>")

    # Nifty row
    nc = "padding:6px 8px;text-align:center;font-size:12px;font-weight:700;"          "border:1px solid rgba(255,255,255,0.2);position:sticky;top:28px;z-index:2;"
    nifty_cells = (f"<td style='background:#7f1d1d;color:#fff;padding:5px 10px;"
                   f"font-size:11px;font-weight:700;white-space:nowrap'>BENCHMARK</td>"
                   f"<td style='background:#7f1d1d;color:#fff;padding:5px 10px;"
                   f"font-size:12px;font-weight:700;white-space:nowrap'>Nifty 50</td>")
    for i in ev_indices:
        val = nifty_row_vals.get(i, np.nan)
        bg, fg = (dd_to_hex(val) if is_crash else rec_to_hex(val))
        txt = f"{float(val):.1f}%" if isinstance(val, float) and not np.isnan(val) else "–"
        nifty_cells += f"<td style='{nc}background:{bg};color:{fg}'>{txt}</td>"

    # Fund rows
    body = ""
    prev_cat = None
    for ridx, row in enumerate(records):
        cat  = row["_cat"]
        fund = row["_fund"]
        cat_bg   = "#2c3e50" if cat != prev_cat else "#3d5166"
        cat_text = f"<b>{cat}</b>" if cat != prev_cat else ""
        prev_cat = cat
        row_bg   = "#ffffff" if ridx % 2 == 0 else "#f0f4f8"

        cells  = (f"<td style='background:{cat_bg};color:#fff;padding:5px 10px;"
                  f"font-size:11px;font-weight:700;white-space:nowrap'>{cat_text}</td>")
        cells += (f"<td style='background:{row_bg};color:#1a1a2e;padding:5px 10px;"
                  f"font-size:11px;white-space:nowrap'>{fund}</td>")

        for i in ev_indices:
            val = row.get(i, np.nan)
            bg, fg = (dd_to_hex(val) if is_crash else rec_to_hex(val))
            txt = f"{float(val):.1f}%" if isinstance(val, float) and not np.isnan(val) else "–"
            nv  = nifty_row_vals.get(i, np.nan)
            badge = ""
            if isinstance(val, float) and not np.isnan(val) and not np.isnan(nv):
                diff = val - nv
                if diff > 1:
                    badge = (f"<br><span style='font-size:9px;font-weight:700;"
                             f"color:#fff;background:rgba(0,0,0,0.30);"
                             f"border-radius:3px;padding:0 3px'>▲{diff:.1f}%</span>")
                elif diff < -1:
                    badge = (f"<br><span style='font-size:9px;font-weight:700;"
                             f"color:#fff;background:rgba(0,0,0,0.30);"
                             f"border-radius:3px;padding:0 3px'>▼{abs(diff):.1f}%</span>")
            cells += (f"<td style='background:{bg};color:{fg};padding:5px 7px;"
                      f"text-align:center;font-size:11px;"
                      f"border:1px solid rgba(0,0,0,0.05)'>{txt}{badge}</td>")
        body += f"<tr>{cells}</tr>"

    # Legend
    if is_crash:
        legend_items = [
            ("<span style='display:inline-block;width:12px;height:12px;background:#00aa37;border-radius:2px'></span>",
             "<b style='color:#111'>Green</b> = small loss"),
            ("<span style='display:inline-block;width:12px;height:12px;background:#dc3737;border-radius:2px'></span>",
             "<b style='color:#111'>Red</b> = large loss"),
            ("<b style='color:#15803d'>▲X%</b>", "<span style='color:#111'>beat Nifty</span>"),
            ("<b style='color:#b91c1c'>▼X%</b>", "<span style='color:#111'>worse than Nifty</span>"),
        ]
    else:
        legend_items = [
            ("<span style='display:inline-block;width:12px;height:12px;background:#1eaf1e;border-radius:2px'></span>",
             "<b style='color:#111'>Green</b> = strong recovery"),
            ("<span style='display:inline-block;width:12px;height:12px;background:#dc7832;border-radius:2px'></span>",
             "<b style='color:#111'>Orange</b> = weak recovery"),
            ("<b style='color:#15803d'>+X%</b>", "<span style='color:#111'>recovered more than Nifty</span>"),
            ("<b style='color:#b91c1c'>−X%</b>", "<span style='color:#111'>recovered less than Nifty</span>"),
        ]
    legend_spans = "".join(
        f"<span style='display:flex;align-items:center;gap:5px;padding:5px 12px;"
        f"background:#fff;border:1px solid #e2e8f0;border-radius:20px;"
        f"font-size:12px'>{icon}&nbsp;{label}</span>"
        for icon, label in legend_items
    )

    st.html(f"""
    <div style='overflow:auto;max-height:600px;border:1px solid #dee2e6;
                border-radius:8px;font-family:-apple-system,sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,0.07)'>
      <table style='border-collapse:collapse;width:100%'>
        <thead><tr>{hdr}</tr></thead>
        <tbody><tr>{nifty_cells}</tr>{body}</tbody>
      </table>
    </div>""")

    st.html(f"""
    <div style='display:flex;flex-wrap:wrap;gap:8px;padding:10px 4px;margin-top:6px;
                font-family:-apple-system,sans-serif'>
      <span style='font-size:12px;font-weight:600;color:#374151;align-self:center'>Legend:</span>
      {legend_spans}
    </div>""")

st.caption(
    f"📉 Crash = (NAV at trough − NAV at peak) / NAV at peak × 100  •  "
    f"📈 Recovery = (NAV at recovery end − NAV at trough) / NAV at trough × 100"
)
