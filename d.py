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

    funds = pd.concat([parse_excel("funds1.xlsx"),
                       parse_excel("funds2.xlsx")], axis=1, sort=True)
    cutoff = nifty.index.max() - pd.DateOffset(years=6)
    nifty  = nifty[nifty.index >= cutoff]
    funds  = funds[funds.index >= cutoff]
    common = nifty.index.intersection(funds.index)
    nifty  = nifty.reindex(common)
    funds  = funds.reindex(common)
    cat_map = {col: classify_category(col) for col in funds.columns}
    return nifty, funds, cat_map


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
st.title("📊 Fund Crash & Recovery Analysis  —  Last 6 Years")
st.markdown(
    "Ranks mutual funds on **two dimensions** during every Nifty crash:  \n"
    "📉 **Drawdown resilience** — which funds fell least  \n"
    "📈 **Recovery speed** — which funds bounced back hardest  \n"
    "🏅 **All-weather score** — which funds did best on both"
)

with st.spinner("Loading data…"):
    nifty, all_funds, category_map = load_data()

last_date = nifty.index.max()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️  Settings")
threshold = st.sidebar.slider(
    "Nifty crash trigger (%)", 2.0, 25.0, 9.0, 0.5,
    help="Detect events where Nifty fell ≥ this % from its rolling peak")
fixed_days = st.sidebar.slider(
    "Recovery window for incomplete recoveries (days)", 30, 365, 90, 10,
    help="If Nifty hasn't fully recovered yet, measure fund gain over this many days from trough")
top_n = st.sidebar.slider("Top N funds per category", 3, 15, 5)
all_cats      = sorted(set(category_map.values()))
selected_cats = st.sidebar.multiselect("Show categories", all_cats, default=all_cats)

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
for fund in all_funds.columns:
    c_vals = [v for v in crash_mat[fund].values()    if not np.isnan(v)]
    r_vals = [v for v in recovery_mat[fund].values() if not np.isnan(v)]
    if c_vals: avg_crash[fund] = np.mean(c_vals)
    if r_vals: avg_rec[fund]   = np.mean(r_vals)

summary_df = pd.DataFrame({
    "Fund"           : list(avg_crash.keys()),
    "Avg Crash (%)"  : list(avg_crash.values()),
    "Avg Recovery (%)": [avg_rec.get(f, np.nan) for f in avg_crash.keys()],
})
summary_df["Category"] = summary_df["Fund"].map(category_map)
summary_df = summary_df[summary_df["Category"].isin(selected_cats)]

# Ranks within category (lower = better for both)
def add_ranks(df):
    df = df.copy()
    df["Crash Rank"]    = df.groupby("Category")["Avg Crash (%)"].rank(
        ascending=False, method="min")   # less negative = rank 1
    df["Recovery Rank"] = df.groupby("Category")["Avg Recovery (%)"].rank(
        ascending=False, method="min")   # higher = rank 1
    df["All-Weather Score"] = df["Crash Rank"] + df["Recovery Rank"]  # lower = better
    df["All-Weather Rank"]  = df.groupby("Category")["All-Weather Score"].rank(
        ascending=True, method="min")
    return df

summary_df = add_ranks(summary_df)
avg_nifty_crash = events_df["nifty_fall"].mean()

# ── Top-level metrics ─────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📅 Data Range",
          f"{nifty.index.min().strftime('%b %Y')} – {last_date.strftime('%b %Y')}")
c2.metric("📊 Funds", len(all_funds.columns))
c3.metric(f"⚡ Crashes ≥ {threshold}%", len(events_df))
c4.metric("📉 Worst Crash", f"{events_df['nifty_fall'].min():.1f}%")
fully = events_df["recovery_date"].notna().sum()
c5.metric("✅ Full Recoveries", f"{fully}/{len(events_df)}")

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📆 Crash Events",
    "📉 Drawdown Rankings",
    "📈 Recovery Rankings",
    "🏅 All-Weather Rankings",
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


# ─── TAB 4 — All-Weather Rankings ─────────────────────────────
with tab4:
    st.subheader("🏅 All-Weather Rankings — Best on Both Crash + Recovery")
    st.markdown(
        "**All-Weather Score = Crash Rank + Recovery Rank** within each category.  \n"
        "Lower score = better. Score of 2 = ranked #1 on both crash AND recovery (perfect fund).  \n"
        "🟦 Dark blue = excellent, 🟨 Yellow = average."
    )

    for cat in sorted(selected_cats):
        cat_df = (summary_df[summary_df["Category"]==cat]
                  .sort_values("All-Weather Score", ascending=True)
                  .head(top_n).copy())
        if cat_df.empty: continue

        with st.expander(f"📂  {cat}  —  Top {top_n} all-weather funds", expanded=True):
            # Scatter: x = avg crash, y = avg recovery, size = all-weather score inverted
            cat_all = summary_df[summary_df["Category"]==cat].copy()
            max_score = cat_all["All-Weather Score"].max()
            cat_all["bubble_size"] = (max_score - cat_all["All-Weather Score"] + 1) * 4

            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=cat_all["Avg Crash (%)"],
                y=cat_all["Avg Recovery (%)"],
                mode="markers+text",
                marker=dict(
                    size=cat_all["bubble_size"],
                    color=cat_all["All-Weather Score"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Score<br>(lower=better)"),
                    line=dict(width=1, color="white")
                ),
                text=cat_all["Fund"].apply(lambda x: x.split("-")[0][:25]),
                textposition="top center",
                textfont=dict(size=9),
                customdata=cat_all[["Fund","All-Weather Score",
                                     "Avg Crash (%)","Avg Recovery (%)"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Score: %{customdata[1]:.0f}<br>"
                    "Crash: %{customdata[2]:.2f}%<br>"
                    "Recovery: %{customdata[3]:.2f}%<extra></extra>"
                )
            ))
            # Quadrant lines
            fig_s.add_vline(x=avg_nifty_crash, line_dash="dot",
                             line_color="red", opacity=0.4)
            fig_s.add_hline(y=avg_nifty_rec,  line_dash="dot",
                             line_color="blue", opacity=0.4)
            # Ideal quadrant label
            fig_s.add_annotation(
                x=cat_all["Avg Crash (%)"].max()*0.7,
                y=cat_all["Avg Recovery (%)"].max()*0.95,
                text="⭐ Ideal: low crash, high recovery",
                font=dict(size=9, color="#555"), showarrow=False)

            fig_s.update_layout(
                title=f"{cat} — Crash vs Recovery (top-right = best)",
                xaxis_title="Avg Crash Return % (→ closer to 0 = better)",
                yaxis_title="Avg Recovery % (→ higher = better)",
                height=480, template="plotly_white")
            st.plotly_chart(fig_s, use_container_width=True)

            # Ranking table
            tbl = cat_df[["Fund","Avg Crash (%)","Avg Recovery (%)",
                           "Crash Rank","Recovery Rank","All-Weather Score",
                           "All-Weather Rank"]].copy().reset_index(drop=True)
            tbl["Avg Crash (%)"]   = tbl["Avg Crash (%)"].map("{:.2f}%".format)
            tbl["Avg Recovery (%)"]= tbl["Avg Recovery (%)"].map("{:.2f}%".format)
            tbl["Crash Rank"]      = tbl["Crash Rank"].map("{:.0f}".format)
            tbl["Recovery Rank"]   = tbl["Recovery Rank"].map("{:.0f}".format)
            tbl["All-Weather Score"] = tbl["All-Weather Score"].map("{:.0f}".format)
            tbl["All-Weather Rank"]  = tbl["All-Weather Rank"].map("{:.0f}".format)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Cross-category NAV chart for best all-weather fund per category
    st.divider()
    st.subheader("📈 Best All-Weather Fund per Category — NAV Comparison")
    best_per_cat = (summary_df.sort_values("All-Weather Score")
                    .groupby("Category").first().reset_index())
    best_funds   = best_per_cat[best_per_cat["Category"].isin(selected_cats)]["Fund"].tolist()

    ev_labels2 = {
        f"Ev{i+1}: {ev['peak_date'].strftime('%d %b %Y')} → "
        f"{ev['trough_date'].strftime('%d %b %Y')} (Nifty {ev['nifty_fall']:.1f}%)": i
        for i, ev in events_df.iterrows()
    }
    chosen_label2 = st.selectbox("Select crash event to zoom", list(ev_labels2.keys()),
                                  key="aw_ev")
    chosen_idx2   = ev_labels2[chosen_label2]
    chosen_ev2    = events_df.loc[chosen_idx2]

    pad   = pd.Timedelta(days=15)
    # window: before peak → after recovery (or fixed)
    rec_end_global = chosen_ev2["recovery_date"]
    if pd.isna(rec_end_global):
        rec_end_global = chosen_ev2["trough_date"] + pd.Timedelta(days=fixed_days)
    w_start2 = chosen_ev2["peak_date"]   - pad
    w_end2   = min(rec_end_global + pad, last_date)

    fig_aw = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35],
                            subplot_titles=("NAV indexed to 100 at crash peak",
                                            "% change from crash peak"))
    nifty_w2  = nifty[w_start2:w_end2]
    base_n2   = float(nifty[nifty.index >= chosen_ev2["peak_date"]].iloc[0])
    fig_aw.add_trace(go.Scatter(x=nifty_w2.index,
                                 y=(nifty_w2/base_n2*100).values,
                                 name="Nifty50",
                                 line=dict(color="black",width=2.5,dash="dot")), row=1, col=1)
    fig_aw.add_trace(go.Scatter(x=nifty_w2.index,
                                 y=((nifty_w2-base_n2)/base_n2*100).values,
                                 name="Nifty %",
                                 line=dict(color="black",width=1.5,dash="dot"),
                                 fill="tozeroy", fillcolor="rgba(0,0,0,0.05)"), row=2, col=1)

    palette = px.colors.qualitative.Bold
    for idx, fund in enumerate(best_funds):
        nav_w = all_funds[fund][w_start2:w_end2].dropna()
        if nav_w.empty: continue
        ap2 = nav_w[nav_w.index >= chosen_ev2["peak_date"]]
        if ap2.empty: continue
        base_f2 = float(ap2.iloc[0])
        color   = palette[idx % len(palette)]
        label   = f"{fund.split('-')[0][:30]} [{category_map[fund]}]"
        cr_ret  = crash_mat[fund].get(chosen_idx2, np.nan)
        rc_ret  = recovery_mat[fund].get(chosen_idx2, np.nan)
        leg_lbl = f"{label} | 📉{cr_ret:.1f}% 📈{rc_ret:.1f}%"

        fig_aw.add_trace(go.Scatter(x=nav_w.index, y=(nav_w/base_f2*100).values,
                                     name=leg_lbl, line=dict(color=color, width=1.8)),
                          row=1, col=1)
        fig_aw.add_trace(go.Scatter(x=nav_w.index,
                                     y=((nav_w-base_f2)/base_f2*100).values,
                                     name=leg_lbl, line=dict(color=color,width=1.2,dash="dash"),
                                     showlegend=False), row=2, col=1)

    for rn in [1, 2]:
        fig_aw.add_vline(x=chosen_ev2["peak_date"], line_dash="dot",
                          line_color="green", line_width=1.5, row=rn, col=1)
        fig_aw.add_vline(x=chosen_ev2["trough_date"], line_dash="dot",
                          line_color="red", line_width=1.5, row=rn, col=1)
        if pd.notna(chosen_ev2["recovery_date"]):
            fig_aw.add_vline(x=chosen_ev2["recovery_date"], line_dash="dot",
                              line_color="blue", line_width=1.5, row=rn, col=1)

    fig_aw.update_layout(height=640, template="plotly_white",
                          legend=dict(orientation="h", y=-0.2, font=dict(size=9)))
    fig_aw.update_yaxes(ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig_aw, use_container_width=True)
    st.caption("🟢 Peak  |  🔴 Trough  |  🔵 Full recovery  |  Legend: 📉 crash return, 📈 recovery return")


# ─── TAB 5 — Full Heatmap ─────────────────────────────────────
with tab5:
    st.subheader("🗂️ Complete Heatmap — Crash + Recovery for Every Fund × Every Event")

    view_mode  = st.radio("Show", ["📉 Crash Returns", "📈 Recovery Returns"],
                           horizontal=True, key="hm_mode")
    cat_filter = st.selectbox("Filter category", ["All"] + sorted(selected_cats),
                               key="hm_cat")

    records = []
    for fund in all_funds.columns:
        cat = category_map[fund]
        if cat not in selected_cats: continue
        if cat_filter != "All" and cat != cat_filter: continue
        row = {"Category": cat, "Fund": fund}
        for i, ev in events_df.iterrows():
            ev_lbl = f"Ev{i+1} {ev['peak_date'].strftime('%b%y')}"
            if "Crash" in view_mode:
                row[ev_lbl] = crash_mat[fund][i]
            else:
                row[ev_lbl] = recovery_mat[fund][i]
        records.append(row)

    matrix_df = pd.DataFrame(records)
    ev_cols   = [c for c in matrix_df.columns if c.startswith("Ev")]

    is_crash = "Crash" in view_mode
    th = ("style='padding:5px 8px;background:#1a3a5c;color:white;"
          "font-size:10px;white-space:nowrap;text-align:center;"
          "position:sticky;top:0;z-index:2'")
    td_lbl = "style='padding:4px 8px;font-size:11px;white-space:nowrap;background:#f5f5f5'"

    hdr = f"<th {th}>Category</th><th {th}>Fund</th>"
    for col in ev_cols:
        hdr += f"<th {th}>{col}</th>"

    body = ""
    for _, row in matrix_df.iterrows():
        cells  = f"<td {td_lbl}><b>{row['Category']}</b></td>"
        cells += f"<td {td_lbl}>{row['Fund']}</td>"
        for col in ev_cols:
            val = row[col]
            if is_crash:
                bg, fg = dd_to_hex(val)
            else:
                bg, fg = rec_to_hex(val)
            txt = (f"{float(val):.1f}%"
                   if (val is not None and isinstance(val, float) and not np.isnan(val))
                   else "–")
            cells += (f"<td style='background:{bg};color:{fg};"
                      f"padding:5px 7px;text-align:center;font-size:11px'>{txt}</td>")
        body += f"<tr>{cells}</tr>"

    st.html(f"""
    <div style='overflow:auto;max-height:560px;border:1px solid #ddd;
                border-radius:6px;font-family:sans-serif'>
      <table style='border-collapse:collapse;width:100%'>
        <thead><tr>{hdr}</tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>""")

    # Nifty reference row
    st.markdown("---")
    st.markdown("**📌 Nifty reference per event:**")
    ref = {"Metric": "Nifty50"}
    for i, ev in events_df.iterrows():
        lbl = f"Ev{i+1} {ev['peak_date'].strftime('%b%y')}"
        if is_crash:
            ref[lbl] = f"{ev['nifty_fall']:.1f}%"
        else:
            nr = nifty_rec_refs.get(i, np.nan)
            ref[lbl] = f"{nr:.1f}%" if not np.isnan(nr) else "–"
    st.dataframe(pd.DataFrame([ref]), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "📉 Crash = (NAV at trough − NAV at peak) / NAV at peak × 100  •  "
    f"📈 Recovery = (NAV at recovery end − NAV at trough) / NAV at trough × 100  •  "
    f"Recovery end = Nifty back to peak, or trough + {fixed_days} days if incomplete  •  "
    "Last 6 years of data"
)
