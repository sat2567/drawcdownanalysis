import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fund Crash & Recovery Analysis", layout="wide", page_icon="📉")

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_data(_cache_buster=None):
    """_cache_buster is derived from file mtimes — changing any data file auto-invalidates cache."""
    raw = pd.read_csv("Nifty50_10Years_Data.csv")
    raw["Date"] = (pd.to_datetime(raw["Date"], utc=True)
                   .dt.tz_convert("Asia/Kolkata")
                   .dt.normalize()
                   .dt.tz_localize(None))
    nifty = raw.set_index("Date").sort_index()["Close"]

    def parse_excel(path):
        df    = pd.read_excel(path, header=None)
        names = df.iloc[2, 1:].tolist()
        data  = df.iloc[4:].copy()
        data.columns = ["Date"] + names
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
        data  = data.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in names:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        return data

    f1         = parse_excel("funds1.xlsx")
    f2         = parse_excel("funds2.xlsx")
    flexi      = parse_excel("flexi.xlsx")
    sec1       = parse_excel("sector1.xlsx")
    sec2       = parse_excel("sector2.xlsx")
    sec3       = parse_excel("sector3.xlsx")
    multiasset = parse_excel("multiasset.xlsx")
    funds = pd.concat([f1, f2, flexi, sec1, sec2, sec3, multiasset], axis=1, sort=True)

    def classify(name):
        n = name.lower()
        if "flexi cap"   in n or "flexicap"   in n: return "Flexi Cap"
        if "small cap"   in n or "smallcap"   in n: return "Small Cap"
        if "multi cap"   in n or "multicap"   in n: return "Multi Cap"
        if "large & mid" in n or "large and mid" in n or "large midcap" in n: return "Large & Mid Cap"
        if "large cap"   in n or "largecap"   in n: return "Large Cap"
        if "mid cap"     in n or "midcap"     in n: return "Mid Cap"
        if "multi asset" in n or "multi-asset" in n or "multiasset" in n: return "Multi Asset"
        if any(x in n for x in ["infra", "psu", "manufactur", "engineer", "build", "tiger", "t.i.g.e.r", "mfg"]): return "Infrastructure"
        if any(x in n for x in ["bank", "fin serv", "financial serv", "finserv", "bfsi"]): return "Banking & Finance"
        if any(x in n for x in ["pharma", "health", "medic"]): return "Pharma & Healthcare"
        if any(x in n for x in ["tech", "teck", "informat", "it fund", "digital", "services", "export"]): return "Technology"
        if any(x in n for x in ["consum", "fmcg", "retail"]): return "Consumption"
        if any(x in n for x in ["energy", "power", "resource", "oil", "gas"]): return "Energy"
        if any(x in n for x in ["auto", "mobil", "transport"]): return "Automotive"
        return "Other"

    cat_map  = {col: classify(col) for col in funds.columns}
    # Track equity vs sector so UI can treat them differently
    type_map = {}
    for col in f1.columns:         type_map[col] = "Equity"
    for col in f2.columns:         type_map[col] = "Equity"
    for col in flexi.columns:      type_map[col] = "Equity"
    for col in sec1.columns:       type_map[col] = "Sector"
    for col in sec2.columns:       type_map[col] = "Sector"
    for col in sec3.columns:       type_map[col] = "Sector"
    for col in multiasset.columns: type_map[col] = "MultiAsset"
    return nifty, funds, cat_map, type_map


# ─────────────────────────────────────────────────────────────
#  CRASH DETECTION
# ─────────────────────────────────────────────────────────────

def find_crashes(nifty, threshold):
    prices = nifty.values
    dates  = nifty.index
    N = len(prices)
    events = []
    i = 0
    while i < N:
        peak_idx   = i
        peak_val   = prices[i]
        trough_idx = i
        trough_val = prices[i]
        in_crash   = False
        j = i + 1
        while j < N:
            pct = (prices[j] - peak_val) / peak_val * 100
            if pct <= -threshold:
                in_crash = True
                if prices[j] < trough_val:
                    trough_val = prices[j]
                    trough_idx = j
                j += 1
            elif in_crash:
                if prices[j] > peak_val * (1 - threshold / 200):
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
            # Find Nifty full recovery date (first close >= peak_val after trough)
            after_trough = nifty[nifty.index > dates[trough_idx]]
            recovered    = after_trough[after_trough >= peak_val]
            rec_date     = recovered.index[0] if not recovered.empty else None

            events.append({
                "peak_date":     dates[peak_idx],
                "trough_date":   dates[trough_idx],
                "peak_val":      round(float(peak_val), 2),
                "trough_val":    round(float(trough_val), 2),
                "nifty_fall":    round((trough_val - peak_val) / peak_val * 100, 2),
                "recovery_date": rec_date,
            })
            i = trough_idx + 1
        else:
            break
    return pd.DataFrame(events)


# ─────────────────────────────────────────────────────────────
#  FUND RETURNS IN A WINDOW — direct NAV lookup
# ─────────────────────────────────────────────────────────────

def fund_returns_in_window(funds, cat_map, start_date, end_date, max_gap_days=7, use_fund_peak=False):
    """
    % change in fund NAV from start_date to end_date.
    Excludes funds whose NAV starts more than max_gap_days after start_date
    or ends more than max_gap_days before end_date (incomplete period data).

    use_fund_peak: if True, use the fund's own maximum NAV in the window
                   [start_date, end_date] as v0 instead of the NAV on start_date.
                   This corrects for funds that peaked after the Nifty peak date.
    """
    rows = []
    for fund in funds.columns:
        nav = funds[fund].dropna()
        s   = nav[nav.index >= start_date]
        e   = nav[nav.index <= end_date]
        if s.empty or e.empty:
            continue
        # Fund started too late — missed the beginning of the period
        if (s.index[0] - start_date).days > max_gap_days:
            continue
        # Fund data ended too early — missed the end of the period
        if (end_date - e.index[-1]).days > max_gap_days:
            continue
        # NAV in full window
        window_nav = nav[(nav.index >= start_date) & (nav.index <= end_date)]
        if use_fund_peak:
            # Use the fund's own peak ONLY if it occurs before the midpoint of the window.
            # This corrects funds that peaked after Nifty but before the crash bottom.
            # If the fund kept rising all the way to the trough, use Nifty-date NAV instead
            # (those funds genuinely held up or gained — don't distort their return).
            # Use fund's own peak if it occurred in the first 2/3 of the crash window.
            # Funds peaking in the last 1/3 genuinely held up — use Nifty-date NAV.
            cutoff = start_date + (end_date - start_date) * 2 / 3
            peak_idx = window_nav.idxmax()
            if peak_idx <= cutoff:
                v0 = float(window_nav.max())
            else:
                v0 = float(s.iloc[0])
        else:
            v0 = float(s.iloc[0])
        v1 = float(e.iloc[-1])
        if v0 <= 0 or np.isnan(v0) or np.isnan(v1):
            continue
        rows.append({
            "Fund":     fund,
            "Category": cat_map.get(fund, "Other"),
            "Return":   round((v1 - v0) / v0 * 100, 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  CHART HELPER
# ─────────────────────────────────────────────────────────────

def draw_bar_chart(cat_df, nifty_ref, annotation, top_n, mode):
    """
    mode = 'crash'    → lower bar = better, red bars are bad
    mode = 'recovery' → higher bar = better, blue bars are good
    """
    plot_df = (cat_df
               .sort_values("Return", ascending=False)
               .head(top_n)
               .sort_values("Return", ascending=True)   # Plotly renders bottom→top
               .copy())
    if plot_df.empty:
        return None, None

    if mode == "crash":
        colors = [
            "#2196f3" if v >= nifty_ref * 0.5
            else "#66bb6a" if v >= nifty_ref
            else "#ef9a9a"
            for v in plot_df["Return"]
        ]
        vline_color = "red"
    else:
        colors = [
            "#1565c0" if v >= nifty_ref * 1.1
            else "#42a5f5" if v >= nifty_ref
            else "#ffcc80"
            for v in plot_df["Return"]
        ]
        vline_color = "#42a5f5"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["Return"],
        y=plot_df["Fund"].apply(lambda x: x[:60] + "…" if len(x) > 60 else x),
        orientation="h",
        marker_color=colors,
        text=plot_df["Return"].map("{:.1f}%".format),
        textposition="outside",
    ))
    fig.add_vline(
        x=nifty_ref,
        line_dash="dash", line_color=vline_color, line_width=2,
        annotation_text=annotation,
        annotation_font_size=11, annotation_font_color=vline_color,
    )
    fig.add_vline(x=0, line_color="#aaa", line_width=1)
    fig.update_layout(
        height=max(300, len(plot_df) * 50),
        template="plotly_dark",
        margin=dict(l=10, r=90, t=10, b=10),
        xaxis=dict(ticksuffix="%", title=""),
        yaxis=dict(title=""),
        showlegend=False,
    )

    tbl = plot_df[["Fund", "Return"]].copy().reset_index(drop=True)
    tbl["vs Nifty"] = (plot_df["Return"].values - nifty_ref)
    tbl["vs Nifty"] = tbl["vs Nifty"].map(lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
    tbl["Return"]   = tbl["Return"].map("{:.1f}%".format)
    return fig, tbl


# ─────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────

st.title("📉 Fund Crash & Recovery Analysis")

# Bust cache whenever any data file changes
import os, hashlib
_data_files = ["funds1.xlsx","funds2.xlsx","flexi.xlsx",
               "sector1.xlsx","sector2.xlsx","sector3.xlsx",
               "multiasset.xlsx","Nifty50_10Years_Data.csv"]
_cache_key = hashlib.md5(
    b"".join(str(os.path.getmtime(f)).encode() for f in _data_files if os.path.exists(f))
).hexdigest()

with st.spinner("Loading data…"):
    nifty, funds, cat_map, type_map = load_all_data(_cache_key)

# Sidebar
threshold = st.sidebar.slider("Nifty crash threshold (%)", 5.0, 40.0, 15.0, 1.0)
top_n     = st.sidebar.slider("Top N funds shown per category", 3, 20, 10)
use_fund_peak = st.sidebar.toggle(
    "Use fund's own peak (crash tab)",
    value=True,
    help="ON: measures fall from the fund's own highest NAV within the crash window (more accurate). "
         "OFF: measures from Nifty's peak date NAV."
)

crashes = find_crashes(nifty, threshold)

if crashes.empty:
    st.warning(f"No Nifty crashes ≥ {threshold:.0f}% found.")
    st.stop()

# Event selector
ev_labels = [
    f"Ev{i+1}:  {r['peak_date'].strftime('%d %b %Y')}  →  "
    f"{r['trough_date'].strftime('%d %b %Y')}   |   Nifty {r['nifty_fall']:.1f}%"
    for i, r in crashes.iterrows()
]

st.subheader(f"{len(crashes)} Nifty crash(es) ≥ {threshold:.0f}% detected")
sel_label = st.selectbox("Select crash event:", ev_labels, index=len(ev_labels) - 1)
sel_idx   = ev_labels.index(sel_label)
ev        = crashes.iloc[sel_idx]

peak_dt    = ev["peak_date"]
trough_dt  = ev["trough_date"]
nifty_fall = ev["nifty_fall"]
rec_date   = ev["recovery_date"]

# Recovery window for this event
if rec_date is not None:
    recovery_end  = rec_date
    recovery_label = f"Full recovery ({rec_date.strftime('%d %b %Y')})"
    is_full        = True
else:
    # Not yet recovered — use slider for partial window
    fixed_days    = st.sidebar.slider("Recovery window (days, since not recovered)", 30, 730, 180, 10)
    recovery_end  = min(trough_dt + pd.Timedelta(days=fixed_days), nifty.index.max())
    recovery_label = f"Partial ({fixed_days}d from trough)"
    is_full        = False

# Nifty recovery return
nifty_rec_s = nifty[nifty.index >= trough_dt]
nifty_rec_e = nifty[nifty.index <= recovery_end]
nifty_rec_v0 = float(nifty_rec_s.iloc[0]) if not nifty_rec_s.empty else np.nan
nifty_rec_v1 = float(nifty_rec_e.iloc[-1]) if not nifty_rec_e.empty else np.nan
nifty_rec_pct = (nifty_rec_v1 - nifty_rec_v0) / nifty_rec_v0 * 100

# Header metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak Date",    peak_dt.strftime("%d %b %Y"))
c2.metric("Trough Date",  trough_dt.strftime("%d %b %Y"))
c3.metric("Recovery End", recovery_end.strftime("%d %b %Y"),
          delta="Full ✅" if is_full else "Partial ⏳")
c4.metric("Nifty Fall",   f"{nifty_fall:.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────
#  EQUITY CATEGORIES  (individual fund breakdown per category)
#  SECTOR CATEGORIES  (combined: one bar per sector = avg of all funds in it)
# ─────────────────────────────────────────────────────────────

EQUITY_CATS = {"Flexi Cap", "Small Cap", "Multi Cap", "Large & Mid Cap",
               "Large Cap", "Mid Cap", "Multi Asset"}
SECTOR_CATS = {"Banking & Finance", "Pharma & Healthcare", "Technology",
               "Infrastructure", "Consumption", "Energy", "Automotive", "Other"}

tab_crash, tab_rec, tab_consistent = st.tabs(["📉 Crash Period", "📈 Recovery Period", "🏅 Consistent Performers"])


def render_equity_section(df, nifty_ref, ref_label, mode, col_label):
    """One expander per equity category showing top N individual funds."""
    eq_df = df[df["Category"].isin(EQUITY_CATS)]
    if eq_df.empty:
        return
    st.markdown("### 📂 Equity Funds — by Category")

    # Summary table: avg + median + best fund per category
    agg = (eq_df.groupby("Category")["Return"]
           .agg(Count="count", Avg="mean", Median="median", Best="max", Worst="min")
           .reset_index()
           .sort_values("Avg", ascending=False))
    best_fund = (eq_df.sort_values("Return", ascending=False)
                 .groupby("Category", sort=False).first()
                 .reset_index()[["Category", "Fund", "Return"]]
                 .rename(columns={"Fund": "Best Fund", "Return": "Best Return"}))
    summary = agg.merge(best_fund, on="Category")
    summary["vs Nifty (Avg)"] = (summary["Avg"] - nifty_ref).map(
        lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
    summary["Best Return"] = summary["Best Return"].map("{:.1f}%".format)
    summary["Worst"]       = summary["Worst"].map("{:.1f}%".format)
    summary["Avg"]         = summary["Avg"].map("{:.1f}%".format)
    summary["Median"]      = summary["Median"].map("{:.1f}%".format)
    summary = summary[["Category", "Count", "Avg", "Median", "Best Fund", "Best Return", "Worst", "vs Nifty (Avg)"]]
    summary.columns = ["Category", "# Funds", "Avg Return", "Median Return", "Best Fund", "Best Return", "Worst Return", "vs Nifty (Avg)"]
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.markdown("")

    for cat in sorted(eq_df["Category"].unique()):
        cat_df = eq_df[eq_df["Category"] == cat]
        total  = len(cat_df)
        beat_n = (cat_df["Return"] > nifty_ref).sum()
        avg    = cat_df["Return"].mean()
        median = cat_df["Return"].median()
        fig, tbl = draw_bar_chart(cat_df, nifty_ref, ref_label, top_n, mode)
        if fig is None:
            continue
        with st.expander(
            f"**{cat}**  —  {beat_n}/{total} beat Nifty  |  Avg: {avg:.1f}%  |  Median: {median:.1f}%",
            expanded=True
        ):
            st.plotly_chart(fig, use_container_width=True)
            tbl.columns = ["Fund", col_label, "vs Nifty"]
            st.dataframe(tbl, use_container_width=True, hide_index=True)


def render_sector_section(df, nifty_ref, ref_label, mode, col_label):
    """One bar per sector = average return of all funds in that sector."""
    sec_df = df[df["Category"].isin(SECTOR_CATS)]
    if sec_df.empty:
        return
    st.markdown("### 🏭 Sector Funds — Category Averages")

    # Aggregate: mean + median + count per sector
    agg = (sec_df.groupby("Category")["Return"]
           .agg(Avg="mean", Median="median", Count="count", Best="max", Worst="min")
           .reset_index()
           .sort_values("Avg", ascending=False))

    # Summary table
    tbl = agg.copy()
    tbl["vs Nifty (Avg)"] = (tbl["Avg"] - nifty_ref).map(
        lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
    tbl["Best"]   = tbl["Best"].map("{:.1f}%".format)
    tbl["Worst"]  = tbl["Worst"].map("{:.1f}%".format)
    tbl["Avg"]    = tbl["Avg"].map("{:.1f}%".format)
    tbl["Median"] = tbl["Median"].map("{:.1f}%".format)
    tbl = tbl[["Category", "Count", "Avg", "Median", "Best", "Worst", "vs Nifty (Avg)"]]
    tbl.columns = ["Sector", "# Funds", "Avg Return", "Median Return", "Best Fund Return", "Worst Fund Return", "vs Nifty (Avg)"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.markdown("")

    # Single bar chart — one bar per sector
    plot = agg.sort_values("Avg", ascending=True)  # Plotly bottom→top = best at top

    if mode == "crash":
        colors = ["#2196f3" if v >= nifty_ref * 0.5
                  else "#66bb6a" if v >= nifty_ref
                  else "#ef9a9a"
                  for v in plot["Avg"]]
        vline_color = "red"
    else:
        colors = ["#1565c0" if v >= nifty_ref * 1.1
                  else "#42a5f5" if v >= nifty_ref
                  else "#ffcc80"
                  for v in plot["Avg"]]
        vline_color = "#42a5f5"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot["Avg"],
        y=plot["Category"],
        orientation="h",
        marker_color=colors,
        text=plot["Avg"].map("{:.1f}%".format),
        textposition="outside",
        customdata=plot["Count"],
        hovertemplate="%{y}<br>Avg: %{x:.1f}%<br>Funds: %{customdata}<extra></extra>",
    ))
    fig.add_vline(x=nifty_ref, line_dash="dash", line_color=vline_color, line_width=2,
                  annotation_text=ref_label,
                  annotation_font_size=11, annotation_font_color=vline_color)
    fig.add_vline(x=0, line_color="#aaa", line_width=1)
    fig.update_layout(
        height=max(300, len(plot) * 55),
        template="plotly_dark",
        margin=dict(l=10, r=90, t=10, b=10),
        xaxis=dict(ticksuffix="%", title="Average return of all funds in sector"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Expandable detail per sector
    with st.expander("See individual funds within each sector", expanded=False):
        for cat in sorted(sec_df["Category"].unique()):
            cat_df = sec_df[sec_df["Category"] == cat].sort_values("Return", ascending=False)
            st.markdown(f"**{cat}** ({len(cat_df)} funds)")
            t = cat_df[["Fund", "Return"]].copy()
            t["vs Nifty"] = (t["Return"] - nifty_ref).map(
                lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
            t["Return"] = t["Return"].map("{:.1f}%".format)
            t.columns = ["Fund", col_label, "vs Nifty"]
            st.dataframe(t, use_container_width=True, hide_index=True)


# ══════════════  CRASH TAB  ══════════════
with tab_crash:
    st.caption(
        f"Fund NAV change from **{peak_dt.strftime('%d %b %Y')}** (Nifty peak) "
        f"to **{trough_dt.strftime('%d %b %Y')}** (Nifty trough).  "
        f"Nifty fell **{nifty_fall:.1f}%** in this period."
    )
    crash_df = fund_returns_in_window(funds, cat_map, peak_dt, trough_dt, use_fund_peak=use_fund_peak)
    if crash_df.empty:
        st.warning("No fund data for this period.")
    else:
        render_equity_section(crash_df, nifty_fall,
                              f"Nifty {nifty_fall:.1f}%", "crash", "Fall in Period")
        st.divider()
        render_sector_section(crash_df, nifty_fall,
                              f"Nifty {nifty_fall:.1f}%", "crash", "Fall in Period")


# ══════════════  RECOVERY TAB  ══════════════
with tab_rec:
    st.caption(
        f"Fund NAV change from **{trough_dt.strftime('%d %b %Y')}** (Nifty trough) "
        f"to **{recovery_end.strftime('%d %b %Y')}** ({recovery_label}).  "
        f"Nifty recovered **{nifty_rec_pct:.1f}%** in this period."
    )
    rec_df = fund_returns_in_window(funds, cat_map, trough_dt, recovery_end)
    if rec_df.empty:
        st.warning("No fund data for this recovery period.")
    else:
        render_equity_section(rec_df, nifty_rec_pct,
                              f"Nifty {nifty_rec_pct:.1f}%", "recovery", "Recovery Gain")
        st.divider()
        render_sector_section(rec_df, nifty_rec_pct,
                              f"Nifty {nifty_rec_pct:.1f}%", "recovery", "Recovery Gain")


# ══════════════  CONSISTENT PERFORMERS TAB  ══════════════
with tab_consistent:
    st.caption(
        "Funds that beat or matched Nifty50 **in every crash event ≥ 15%** where they had complete data.  "
        "Only funds present in **at least 2 events** are shown."
    )

    CONSISTENT_THRESHOLD = 15.0
    all_crashes = find_crashes(nifty, CONSISTENT_THRESHOLD)

    if all_crashes.empty:
        st.warning("No crash events ≥ 15% found.")
    else:
        # Build per-fund per-event return table
        records = []
        for idx, ev_row in all_crashes.iterrows():
            ev_peak   = ev_row["peak_date"]
            ev_trough = ev_row["trough_date"]
            ev_fall   = ev_row["nifty_fall"]
            ev_label  = f"Ev{idx+1}: {ev_peak.strftime('%b %Y')} ({ev_fall:.1f}%)"
            df_ev = fund_returns_in_window(
                funds, cat_map, ev_peak, ev_trough,
                use_fund_peak=use_fund_peak
            )
            for _, row in df_ev.iterrows():
                records.append({
                    "Fund":        row["Fund"],
                    "Category":    row["Category"],
                    "Event":       ev_label,
                    "Nifty_Fall":  ev_fall,
                    "Return":      row["Return"],
                    "Beat_Nifty":  row["Return"] > ev_fall,
                })

        if not records:
            st.warning("No fund data found across events.")
        else:
            all_ev_df = pd.DataFrame(records)
            ev_labels_all = all_ev_df["Event"].unique().tolist()

            # Pivot: Fund × Event → Return
            pivot = all_ev_df.pivot_table(
                index=["Fund", "Category"], columns="Event",
                values="Return", aggfunc="first"
            ).reset_index()

            beat_pivot = all_ev_df.pivot_table(
                index=["Fund", "Category"], columns="Event",
                values="Beat_Nifty", aggfunc="first"
            ).reset_index()

            # Count events each fund participated in and how many it beat Nifty
            ret_cols = [c for c in pivot.columns if c not in ["Fund", "Category"]]
            pivot["Events_Participated"] = pivot[ret_cols].notna().sum(axis=1)
            beat_cols = [c for c in beat_pivot.columns if c not in ["Fund", "Category"]]
            pivot["Events_Beat"] = beat_pivot[beat_cols].sum(axis=1, min_count=1).fillna(0).astype(int)
            pivot["Beat_Rate"]   = (pivot["Events_Beat"] / pivot["Events_Participated"] * 100).round(1)

            # Filter: at least 2 events participated
            pivot = pivot[pivot["Events_Participated"] >= 2].copy()

            # Avg return across all events participated
            pivot["Avg_Return"] = pivot[ret_cols].mean(axis=1).round(2)

            # All categories together
            all_cats = sorted(pivot["Category"].unique())

            st.subheader("🏆 Funds by Beat Rate across all crash events")
            st.markdown("*Beat Rate = % of events where fund fell less than Nifty50*")

            for cat in all_cats:
                cat_p = pivot[pivot["Category"] == cat].copy()
                if cat_p.empty:
                    continue

                # Sort by beat rate desc, then avg return desc
                cat_p = cat_p.sort_values(["Beat_Rate", "Avg_Return"], ascending=[False, False])

                display = cat_p[["Fund", "Events_Participated", "Events_Beat", "Beat_Rate", "Avg_Return"]].copy()
                # Add per-event return columns
                for ev in ev_labels_all:
                    if ev in cat_p.columns:
                        nifty_fall_ev = all_ev_df[all_ev_df["Event"] == ev]["Nifty_Fall"].iloc[0]
                        display[ev] = cat_p[ev].map(
                            lambda x, nf=nifty_fall_ev: (
                                f"{x:.1f}% ✅" if pd.notna(x) and x > nf
                                else f"{x:.1f}% ❌" if pd.notna(x)
                                else "—"
                            )
                        )

                display["Beat_Rate"]  = display["Beat_Rate"].map("{:.0f}%".format)
                display["Avg_Return"] = display["Avg_Return"].map("{:.1f}%".format)
                display.columns = (
                    ["Fund", "# Events", "# Beat", "Beat Rate", "Avg Return"] +
                    [e for e in ev_labels_all if e in cat_p.columns]
                )

                total = len(display)
                always_beat = (cat_p["Beat_Rate"] == 100).sum()
                with st.expander(
                    f"**{cat}**  —  {total} funds  |  {always_beat} beat Nifty in ALL events",
                    expanded=True
                ):
                    st.dataframe(display, use_container_width=True, hide_index=True)
