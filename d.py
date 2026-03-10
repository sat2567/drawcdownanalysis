import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fund Crash Analysis", layout="wide", page_icon="📉")

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_all_data():
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

    f1  = parse_excel("funds1.xlsx")
    f2  = parse_excel("funds2.xlsx")
    sec = parse_excel("sectorfunds.xlsx")
    funds = pd.concat([f1, f2, sec], axis=1, sort=True)

    def classify(name):
        n = name.lower()
        if "flexi cap"   in n or "flexicap"   in n: return "Flexi Cap"
        if "small cap"   in n or "smallcap"   in n: return "Small Cap"
        if "multi cap"   in n or "multicap"   in n: return "Multi Cap"
        if "large & mid" in n or "large and mid" in n or "large midcap" in n: return "Large & Mid Cap"
        if "large cap"   in n or "largecap"   in n: return "Large Cap"
        if "mid cap"     in n or "midcap"     in n: return "Mid Cap"
        if any(x in n for x in ["bank", "fin serv", "financial serv", "finserv"]): return "Banking & Finance"
        if any(x in n for x in ["pharma", "health", "medic"]): return "Pharma & Healthcare"
        if any(x in n for x in ["tech", "informat", "it fund", "digital"]): return "Technology"
        if any(x in n for x in ["infra", "psu", "manufactur", "engineer"]): return "Infrastructure"
        if any(x in n for x in ["consum", "fmcg", "retail"]): return "Consumption"
        if any(x in n for x in ["energy", "power", "resource", "oil", "gas"]): return "Energy"
        if any(x in n for x in ["auto", "mobil", "transport"]): return "Automotive"
        return "Other"

    cat_map = {col: classify(col) for col in funds.columns}
    return nifty, funds, cat_map


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
            events.append({
                "peak_date":   dates[peak_idx],
                "trough_date": dates[trough_idx],
                "peak_val":    round(float(peak_val), 2),
                "trough_val":  round(float(trough_val), 2),
                "nifty_fall":  round((trough_val - peak_val) / peak_val * 100, 2),
            })
            i = trough_idx + 1
        else:
            break
    return pd.DataFrame(events)


# ─────────────────────────────────────────────────────────────
#  FUND RETURNS IN WINDOW — direct NAV lookup, no cache
# ─────────────────────────────────────────────────────────────

def fund_returns_in_window(funds, cat_map, start_date, end_date):
    rows = []
    for fund in funds.columns:
        nav = funds[fund].dropna()
        s   = nav[nav.index >= start_date]
        e   = nav[nav.index <= end_date]
        if s.empty or e.empty:
            continue
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
#  APP
# ─────────────────────────────────────────────────────────────

st.title("📉 Fund Returns During Nifty Crashes")

with st.spinner("Loading data…"):
    nifty, funds, cat_map = load_all_data()

# Sidebar
threshold = st.sidebar.slider("Nifty crash threshold (%)", 5.0, 40.0, 15.0, 1.0)
top_n     = st.sidebar.slider("Top N funds shown per category", 3, 20, 10)

crashes = find_crashes(nifty, threshold)

if crashes.empty:
    st.warning(f"No Nifty crashes ≥ {threshold:.0f}% found.")
    st.stop()

# Event selector
ev_labels = [
    f"Ev{i+1}:  {r['peak_date'].strftime('%d %b %Y')}  →  {r['trough_date'].strftime('%d %b %Y')}   |   Nifty {r['nifty_fall']:.1f}%"
    for i, r in crashes.iterrows()
]

st.subheader(f"{len(crashes)} Nifty crash(es) ≥ {threshold:.0f}% detected")

sel_label = st.selectbox("Select crash event:", ev_labels, index=len(ev_labels) - 1)
sel_idx   = ev_labels.index(sel_label)
ev        = crashes.iloc[sel_idx]

peak_dt   = ev["peak_date"]
trough_dt = ev["trough_date"]
nifty_ret = ev["nifty_fall"]

c1, c2, c3 = st.columns(3)
c1.metric("Peak Date",   peak_dt.strftime("%d %b %Y"))
c2.metric("Trough Date", trough_dt.strftime("%d %b %Y"))
c3.metric("Nifty Fall",  f"{nifty_ret:.1f}%")

st.caption(
    f"Each fund's NAV change from **{peak_dt.strftime('%d %b %Y')}** to "
    f"**{trough_dt.strftime('%d %b %Y')}** — the exact period Nifty fell {nifty_ret:.1f}%."
)
st.divider()

# Compute returns directly from NAV data
returns_df = fund_returns_in_window(funds, cat_map, peak_dt, trough_dt)

if returns_df.empty:
    st.warning("No fund data available for this period.")
    st.stop()

# Category filter
all_cats      = sorted(returns_df["Category"].unique())
selected_cats = st.sidebar.multiselect("Categories", all_cats, default=all_cats)
returns_df    = returns_df[returns_df["Category"].isin(selected_cats)]

# Best fund per category summary
st.subheader("🏆 Best Fund per Category (fell least)")
best = (returns_df
        .sort_values("Return", ascending=False)
        .groupby("Category", sort=False)
        .first()
        .reset_index()[["Category", "Fund", "Return"]])
best["vs Nifty"] = (best["Return"] - nifty_ret).map(lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
best["Return"]   = best["Return"].map("{:.1f}%".format)
st.dataframe(best, use_container_width=True, hide_index=True)
st.divider()

# Per-category charts
st.subheader(f"📊 Top {top_n} Most Resilient Funds per Category")

for cat in sorted(selected_cats):
    cat_df = (returns_df[returns_df["Category"] == cat]
              .sort_values("Return", ascending=False)
              .head(top_n)
              .sort_values("Return", ascending=True)  # reversed: best at top in Plotly
              .copy())
    if cat_df.empty:
        continue

    total_in_cat = len(returns_df[returns_df["Category"] == cat])
    beat_nifty   = (returns_df[returns_df["Category"] == cat]["Return"] > nifty_ret).sum()

    with st.expander(
        f"**{cat}**  —  {beat_nifty}/{total_in_cat} funds beat Nifty",
        expanded=True
    ):
        colors = [
            "#2196f3" if v >= nifty_ret * 0.5
            else "#66bb6a" if v >= nifty_ret
            else "#ef9a9a"
            for v in cat_df["Return"]
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat_df["Return"],
            y=cat_df["Fund"].apply(lambda x: x[:60] + "…" if len(x) > 60 else x),
            orientation="h",
            marker_color=colors,
            text=cat_df["Return"].map("{:.1f}%".format),
            textposition="outside",
        ))
        fig.add_vline(
            x=nifty_ret,
            line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"Nifty {nifty_ret:.1f}%",
            annotation_font_size=11, annotation_font_color="red",
        )
        fig.add_vline(x=0, line_color="#aaa", line_width=1)
        fig.update_layout(
            height=max(300, len(cat_df) * 50),
            template="plotly_dark",
            margin=dict(l=10, r=90, t=10, b=10),
            xaxis=dict(ticksuffix="%", title=""),
            yaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        tbl = cat_df[["Fund", "Return"]].copy().reset_index(drop=True)
        tbl["vs Nifty"] = (cat_df["Return"].values - nifty_ret)
        tbl["vs Nifty"] = tbl["vs Nifty"].map(lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
        tbl["Return"]   = tbl["Return"].map("{:.1f}%".format)
        tbl.columns     = ["Fund", "Return in Period", "vs Nifty"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)
