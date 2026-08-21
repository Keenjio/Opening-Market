"""
Indian Stock Market Pre-Market Analysis Dashboard — Streamlit App
====================================================================
Run with:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Tabs:
  A. Live Dashboard
     1. US Markets (Nasdaq & Dow Jones)
     2. India VIX (volatility regime)
     3. Nifty 50 market breadth (advances vs declines)
     4. Global cue / GIFT Nifty proxy
     5. Combined pre-market verdict
  B. History & Analysis
     - Date-wise log of the 09:00 and 16:00 "official" scores
     - Hour-by-hour intraday log (09:30-15:30 grid, captured via
       GitHub Actions -> data/score_log.csv)
     - Correlation check: did the 09:00-today score line up with
       where the market actually stood at 16:00-the previous day,
       and how did the day's score evolve intraday?

Historical data is captured by a separate headless script
(capture_score.py) run on a schedule by GitHub Actions, NOT by this
app. Streamlit Community Cloud has no background scheduler and only
runs code when a page is loaded, so exact-time capture at 09:00/16:00
etc. cannot happen from inside the app itself. See:
    .github/workflows/capture.yml
    capture_score.py
This app only *reads* data/score_log.csv for the History tab.
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from market_data import (
    NIFTY_50_SYMBOLS,
    VIX_THRESHOLD,
    compute_snapshot,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pre-Market Dashboard",
    page_icon="📈",
    layout="wide",
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "score_log.csv")
PRIMARY_SLOTS = ("09:00", "16:00")

# ---------------------------------------------------------------------------
# Cached wrapper around the shared snapshot function
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def get_live_snapshot():
    snap = compute_snapshot()
    # breadth_df isn't picklable-safe to cache alongside scalars in all
    # streamlit versions if it contains weird dtypes, but plain DataFrames
    # are fine - kept as-is for the live view.
    return snap


@st.cache_data(ttl=120, show_spinner=False)
def load_history():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return df
    df["date_ist"] = pd.to_datetime(df["date_ist"]).dt.date
    return df


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def color_change(val):
    if pd.isna(val):
        return ""
    color = "#16a34a" if val >= 0 else "#dc2626"
    return f"color: {color}; font-weight: 600;"


def status_color(val):
    if val == "Advance":
        return "color: #16a34a; font-weight: 600;"
    if val == "Decline":
        return "color: #dc2626; font-weight: 600;"
    return "color: #6b7280;"


def verdict_color(val):
    if val == "GREEN":
        return "color: #16a34a; font-weight: 700;"
    if val == "RED":
        return "color: #dc2626; font-weight: 700;"
    if val == "FLAT":
        return "color: #d97706; font-weight: 700;"
    return "color: #6b7280;"


def styled_table(df, extra_cols=None):
    fmt = {"Last": "{:.2f}", "Change %": "{:+.2f}%"}
    styler = df.style.format(fmt, na_rep="—")
    apply_fn = "map" if hasattr(styler, "map") else "applymap"
    if "Change %" in df.columns:
        styler = getattr(styler, apply_fn)(color_change, subset=["Change %"])
    if "Status" in df.columns:
        styler = getattr(styler, apply_fn)(status_color, subset=["Status"])
    return styler


def slot_sort_key(slot):
    h, m = map(int, slot.split(":"))
    return h * 60 + m


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("📈 Indian Market Pre-Open Dashboard")
st.caption(datetime.now().strftime("%A, %d %B %Y — %H:%M:%S"))

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

tab_live, tab_history = st.tabs(["🔴 Live Dashboard", "🗂️ History & Analysis"])

# ===========================================================================
# TAB A: LIVE DASHBOARD
# ===========================================================================
with tab_live:
    with st.spinner("Fetching live market data..."):
        snap = get_live_snapshot()

    breadth_df = snap["_breadth_df"]
    advances, declines, unchanged = snap["advances"], snap["declines"], snap["unchanged"]
    total = advances + declines + unchanged

    # --- 1. US Markets ---
    st.subheader("1. US Markets (Previous Close)")
    us_cols = st.columns(2)
    us_pairs = [
        ("Nasdaq Composite", snap["nasdaq_change_pct"]),
        ("Dow Jones", snap["dow_change_pct"]),
    ]
    for col, (name, change) in zip(us_cols, us_pairs):
        with col:
            st.metric(label=name, value="—", delta=f"{change:+.2f}%" if change is not None else None)

    st.divider()

    # --- 2. India VIX ---
    st.subheader("2. India VIX (Volatility Gauge)")
    vix = snap["india_vix"]
    vix_col1, vix_col2 = st.columns([1, 2])
    with vix_col1:
        st.metric("India VIX", f"{vix:.2f}" if vix is not None else "N/A")
    with vix_col2:
        if vix is not None:
            if vix > VIX_THRESHOLD:
                st.error(f"⚠️ Market Condition: **VOLATILE** (VIX {vix:.2f} > {VIX_THRESHOLD})")
            else:
                st.success(f"✅ Market Condition: **STABLE** (VIX {vix:.2f} ≤ {VIX_THRESHOLD})")
        else:
            st.warning("India VIX data unavailable")

    st.divider()

    # --- 3. Market Breadth ---
    st.subheader(f"3. Market Breadth (Nifty sample, {len(NIFTY_50_SYMBOLS)} stocks)")
    if total > 0:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Advances", advances)
        b2.metric("Declines", declines)
        b3.metric("Unchanged", unchanged)
        sentiment = snap["breadth_sentiment"]
        color = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "orange"}.get(sentiment, "gray")
        b4.markdown(f"**Breadth Sentiment**  \n:{color}[**{sentiment}**]")

        st.bar_chart(
            pd.DataFrame({"Advances": [advances], "Declines": [declines], "Unchanged": [unchanged]}).T,
            use_container_width=True,
        )

        st.markdown("#### Advances vs Declines — Full List")
        adv_col, dec_col = st.columns(2)
        with adv_col:
            st.markdown(f"**🟢 Advances ({advances})**")
            adv_table = breadth_df[breadth_df["Status"] == "Advance"].sort_values("Change %", ascending=False)
            if not adv_table.empty:
                st.dataframe(styled_table(adv_table[["Symbol", "Last", "Change %"]]), hide_index=True,
                             use_container_width=True, height=min(38 * (len(adv_table) + 1), 400))
            else:
                st.caption("No advancing stocks in this sample.")
        with dec_col:
            st.markdown(f"**🔴 Declines ({declines})**")
            dec_table = breadth_df[breadth_df["Status"] == "Decline"].sort_values("Change %", ascending=True)
            if not dec_table.empty:
                st.dataframe(styled_table(dec_table[["Symbol", "Last", "Change %"]]), hide_index=True,
                             use_container_width=True, height=min(38 * (len(dec_table) + 1), 400))
            else:
                st.caption("No declining stocks in this sample.")

        if unchanged > 0:
            unch_table = breadth_df[breadth_df["Status"] == "Unchanged"][["Symbol", "Last", "Change %"]]
            with st.expander(f"⚪ Unchanged ({unchanged})"):
                st.dataframe(styled_table(unch_table), hide_index=True, use_container_width=True)

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("**Top Gainers**")
            st.dataframe(styled_table(breadth_df.sort_values("Change %", ascending=False).head(3)),
                         hide_index=True, use_container_width=True)
        with gc2:
            st.markdown("**Top Losers**")
            st.dataframe(styled_table(breadth_df.sort_values("Change %", ascending=True).head(3)),
                         hide_index=True, use_container_width=True)

        with st.expander("Full breadth table (all stocks, with status)"):
            st.dataframe(
                styled_table(breadth_df[["Symbol", "Last", "Change %", "Status"]].sort_values("Change %", ascending=False)),
                hide_index=True, use_container_width=True,
            )
    else:
        st.warning("Breadth data unavailable")

    st.divider()

    # --- 4. Global Cue ---
    st.subheader("4. Global Cue / GIFT Nifty Proxy")
    gc_change = snap["global_cue_change_pct"]
    if gc_change is not None:
        gcol1, gcol2 = st.columns([1, 2])
        with gcol1:
            label = "Nifty Futures" if not snap["global_cue_is_fallback"] else snap["global_cue_source"]
            st.metric(label=label, value="—", delta=f"{gc_change:+.2f}%")
        with gcol2:
            if snap["global_cue_is_fallback"]:
                st.info(
                    "⚠️ Live GIFT Nifty futures not available via this data source. "
                    f"Showing a fallback estimate: **{snap['global_cue_source']}**. "
                    "Cross-check the official NSE IX GIFT Nifty page before trading."
                )
            else:
                st.write(f"Source ticker: `{snap['global_cue_source']}`")
    else:
        st.warning("No global cue data available")

    st.divider()

    # --- 5. Combined Verdict ---
    st.subheader("5. Combined Pre-Market Verdict")
    if snap["score"] is not None:
        score, n_signals, verdict = snap["score"], snap["n_signals"], snap["verdict"]
        if verdict == "GREEN":
            st.success(f"### 🟢 LIKELY GREEN OPEN — Signal Score: {score}/{n_signals}")
        elif verdict == "RED":
            st.error(f"### 🔴 LIKELY RED OPEN — Signal Score: {score}/{n_signals}")
        else:
            st.warning(f"### 🟡 MIXED / FLAT OPEN — Signal Score: {score}/{n_signals}")
    else:
        st.warning("Insufficient data for a verdict")

    st.caption("Disclaimer: For informational purposes only. Not investment advice.")

# ===========================================================================
# TAB B: HISTORY & ANALYSIS
# ===========================================================================
with tab_history:
    st.subheader("🗂️ Historical Score Log")
    st.caption(
        "Snapshots are captured automatically by a scheduled GitHub Action "
        "(not by this app) at 09:00, hourly from 10:00-15:00, 15:30, and "
        "16:00 IST on trading days, and committed to `data/score_log.csv`. "
        "This tab only reads that file."
    )

    hist = load_history()

    if hist.empty:
        st.info(
            "No historical data yet. Once the GitHub Actions workflow "
            "(`.github/workflows/capture.yml`) has run at least once, "
            "snapshots will appear here."
        )
    else:
        hist = hist.sort_values(["date_ist", "slot_ist"], key=lambda s: s.map(slot_sort_key) if s.name == "slot_ist" else s)
        hist_sorted = hist.sort_values(["date_ist"]).copy()
        hist_sorted["_slot_order"] = hist_sorted["slot_ist"].map(slot_sort_key)
        hist_sorted = hist_sorted.sort_values(["date_ist", "_slot_order"]).drop(columns="_slot_order")

        # -------------------------------------------------------------
        # 1. Date-wise 09:00 / 16:00 official score table
        # -------------------------------------------------------------
        st.markdown("### 📅 Date-wise Official Scores (09:00 & 16:00)")
        official = hist_sorted[hist_sorted["slot_ist"].isin(PRIMARY_SLOTS)].copy()
        if official.empty:
            st.caption("No 09:00/16:00 snapshots logged yet.")
        else:
            pivot = official.pivot_table(
                index="date_ist", columns="slot_ist",
                values=["score", "verdict"], aggfunc="first",
            )
            # Flatten multiindex columns into readable names
            pivot.columns = [f"{slot} {metric}" for metric, slot in pivot.columns]
            pivot = pivot.reset_index().sort_values("date_ist", ascending=False)
            st.dataframe(pivot, hide_index=True, use_container_width=True)

        st.divider()

        # -------------------------------------------------------------
        # 2. Hour-wise intraday tracker (09:30-15:30 grid + score evolution)
        # -------------------------------------------------------------
        st.markdown("### ⏱️ Hour-wise Intraday Score Tracker")
        available_dates = sorted(hist_sorted["date_ist"].unique(), reverse=True)
        pick_date = st.selectbox(
            "Select a trading day",
            options=available_dates,
            format_func=lambda d: d.strftime("%A, %d %b %Y"),
        )
        day_df = hist_sorted[hist_sorted["date_ist"] == pick_date].copy()
        day_df["_slot_order"] = day_df["slot_ist"].map(slot_sort_key)
        day_df = day_df.sort_values("_slot_order").drop(columns="_slot_order")

        if day_df.empty:
            st.caption("No snapshots for this date.")
        else:
            display_cols = [
                "slot_ist", "score", "verdict", "india_vix", "vix_condition",
                "advances", "declines", "unchanged", "breadth_sentiment",
                "nasdaq_change_pct", "dow_change_pct", "global_cue_change_pct",
            ]
            show_df = day_df[display_cols].rename(columns={"slot_ist": "Time (IST)"})
            styled = show_df.style
            apply_fn = "map" if hasattr(styled, "map") else "applymap"
            styled = getattr(styled, apply_fn)(verdict_color, subset=["verdict"])
            st.dataframe(styled, hide_index=True, use_container_width=True)

            st.markdown("**Score evolution through the day**")
            chart_df = day_df.set_index("slot_ist")[["score"]]
            st.line_chart(chart_df, use_container_width=True)

        st.divider()

        # -------------------------------------------------------------
        # 3. Correlation: 09:00-today vs 16:00-previous-day
        # -------------------------------------------------------------
        st.markdown("### 🔁 Correlation: 09:00 (today) vs 16:00 (previous day)")
        st.caption(
            "For each trading day, this compares the verdict generated from "
            "data available at 09:00 that day against the verdict generated "
            "from data available at 16:00 the prior trading day — i.e. did "
            "the overnight/pre-open signal agree with where the market had "
            "settled the evening before?"
        )

        if official.empty or official["slot_ist"].nunique() < 2:
            st.caption("Not enough 09:00/16:00 pairs logged yet to compute correlation.")
        else:
            am = official[official["slot_ist"] == "09:00"][["date_ist", "score", "verdict"]].rename(
                columns={"score": "score_0900", "verdict": "verdict_0900"}
            )
            pm = official[official["slot_ist"] == "16:00"][["date_ist", "score", "verdict"]].rename(
                columns={"score": "score_1600", "verdict": "verdict_1600"}
            )
            pm_shifted = pm.copy()
            pm_shifted = pm_shifted.sort_values("date_ist")
            # Map each date's 16:00 row to the *next* available trading date's 09:00 row.
            all_dates = sorted(set(am["date_ist"]) | set(pm_shifted["date_ist"]))
            pm_shifted["next_date"] = pm_shifted["date_ist"].apply(
                lambda d: next((x for x in all_dates if x > d), None)
            )
            merged = pm_shifted.merge(
                am, left_on="next_date", right_on="date_ist",
                suffixes=("_prevclose", "_today"),
            )
            merged = merged.rename(columns={"date_ist_prevclose": "prev_close_date", "date_ist_today": "next_open_date"})
            merged["agree"] = merged["verdict_1600"] == merged["verdict_0900"]

            if merged.empty:
                st.caption("Not enough consecutive trading days logged yet.")
            else:
                agree_pct = 100 * merged["agree"].mean()
                st.metric("Agreement rate (prev-close verdict = next-open verdict)", f"{agree_pct:.0f}%")

                out = merged[[
                    "prev_close_date", "verdict_1600", "score_1600",
                    "next_open_date", "verdict_0900", "score_0900", "agree",
                ]].sort_values("next_open_date", ascending=False)
                out = out.rename(columns={
                    "prev_close_date": "Prev-day date",
                    "verdict_1600": "16:00 verdict (prev day)",
                    "score_1600": "16:00 score",
                    "next_open_date": "Next trading day",
                    "verdict_0900": "09:00 verdict (next day)",
                    "score_0900": "09:00 score",
                    "agree": "Agreed?",
                })
                st.dataframe(out, hide_index=True, use_container_width=True)

        st.divider()
        with st.expander("Full raw log (all slots, all dates)"):
            st.dataframe(hist_sorted, hide_index=True, use_container_width=True)
